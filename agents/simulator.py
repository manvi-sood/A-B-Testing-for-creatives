"""
AGENT 2 — A/B Simulation Agent
================================
Takes Agent 1's scored variations and simulates real-world ad performance.

Responsibilities:
  1. Convert OpenCLIP scores → synthetic CTR and ROAS
  2. Inject realistic noise (platform-specific variance)
  3. Run statistical comparison to declare a winner
  4. Return confidence score + simulation summary
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
#  Platform-specific performance baselines
#  (reflects real-world CTR/ROAS ranges per platform)
# ─────────────────────────────────────────────

PLATFORM_PROFILES = {
    "Meta Feed": {
        "ctr_multiplier":  8.0,
        "roas_multiplier": 15.0,
        "ctr_noise":       0.35,
        "roas_noise":      0.6,
        "ctr_floor":       0.4,    # minimum realistic CTR %
        "roas_floor":      1.2,
    },
    "Meta Reels": {
        "ctr_multiplier":  10.0,   # Reels tends to have higher engagement
        "roas_multiplier": 12.0,
        "ctr_noise":       0.5,    # but also more variance
        "roas_noise":      0.8,
        "ctr_floor":       0.6,
        "roas_floor":      1.0,
    },
    "Amazon": {
        "ctr_multiplier":  6.0,
        "roas_multiplier": 20.0,   # Amazon has higher purchase intent
        "ctr_noise":       0.25,
        "roas_noise":      0.5,
        "ctr_floor":       0.3,
        "roas_floor":      2.0,
    },
    "Google Display": {
        "ctr_multiplier":  4.0,    # Display CTRs are typically lower
        "roas_multiplier": 10.0,
        "ctr_noise":       0.2,
        "roas_noise":      0.4,
        "ctr_floor":       0.1,
        "roas_floor":      0.8,
    },
}

DEFAULT_PROFILE = PLATFORM_PROFILES["Meta Feed"]

# Simulated budget split across variations (in INR)
DEFAULT_BUDGET_PER_VARIATION = 500   # ₹500 each = very lean test


# ─────────────────────────────────────────────
#  Core simulation
# ─────────────────────────────────────────────

def _simulate_single(
    score: float,
    profile: dict,
    n_simulations: int = 1000,
    seed: Optional[int] = None,
) -> dict:
    """
    Run Monte Carlo simulation for a single variation.
    Returns mean CTR, mean ROAS, and distribution arrays.
    """
    rng = np.random.default_rng(seed)

    # Base values from score
    base_ctr  = score * profile["ctr_multiplier"]
    base_roas = score * profile["roas_multiplier"]

    # Gaussian noise around the base
    ctrs  = rng.normal(base_ctr,  profile["ctr_noise"],  n_simulations)
    roass = rng.normal(base_roas, profile["roas_noise"], n_simulations)

    # Apply floors (no negative CTR/ROAS)
    ctrs  = np.clip(ctrs,  profile["ctr_floor"],  None)
    roass = np.clip(roass, profile["roas_floor"], None)

    return {
        "mean_ctr":   float(np.mean(ctrs)),
        "std_ctr":    float(np.std(ctrs)),
        "mean_roas":  float(np.mean(roass)),
        "std_roas":   float(np.std(roass)),
        "ctr_dist":   ctrs.tolist(),
        "roas_dist":  roass.tolist(),
        "p95_ctr":    float(np.percentile(ctrs, 95)),
        "p05_ctr":    float(np.percentile(ctrs, 5)),
        "p95_roas":   float(np.percentile(roass, 95)),
        "p05_roas":   float(np.percentile(roass, 5)),
    }


def _compute_confidence(winner_dist: list, runner_up_dist: list) -> float:
    """
    Bootstrap confidence: % of simulations where winner beats runner-up.
    Returns a value between 0.5 and 1.0.
    """
    w = np.array(winner_dist)
    r = np.array(runner_up_dist)
    min_len = min(len(w), len(r))
    wins = np.sum(w[:min_len] > r[:min_len])
    return float(wins / min_len)


def _effect_size(mean_a: float, std_a: float, mean_b: float, std_b: float) -> float:
    """Cohen's d effect size between winner and runner-up."""
    pooled_std = math.sqrt((std_a**2 + std_b**2) / 2)
    if pooled_std == 0:
        return 0.0
    return abs(mean_a - mean_b) / pooled_std


# ─────────────────────────────────────────────
#  Main Entry Point
# ─────────────────────────────────────────────

def run_agent2(agent1_output: dict, budget_per_variation: int = DEFAULT_BUDGET_PER_VARIATION) -> dict:
    """
    Takes Agent 1's output dict and returns full simulation results.

    Returns:
    {
      "simulated_variations": [
          {
            variation_id, label, type, score,
            mean_ctr, std_ctr, mean_roas, std_roas,
            ctr_dist, roas_dist,
            estimated_clicks, estimated_revenue,
            rank
          }
      ],
      "winner": dict,
      "runner_up": dict,
      "confidence": float,        # 0-1, how often winner beats runner-up
      "effect_size": float,       # Cohen's d
      "significance": str,        # "Strong" / "Moderate" / "Weak"
      "summary_df": pd.DataFrame,
      "platform": str,
    }
    """
    platform = agent1_output.get("platform", "Meta Feed")
    profile  = PLATFORM_PROFILES.get(platform, DEFAULT_PROFILE)

    # Flatten all variations from all source images
    all_variations = []
    for result in agent1_output["results"]:
        for var in result["variations"]:
            all_variations.append({
                **var,
                "source_filename": result["source_filename"],
            })

    # ── Simulate each variation ──────────────────────────────────────
    simulated = []
    for i, var in enumerate(all_variations):
        sim = _simulate_single(
            score=var["score"],
            profile=profile,
            n_simulations=1000,
            seed=i,  # deterministic per variation
        )

        # Estimated clicks and revenue given budget
        cpm_estimate = 80  # ₹80 CPM as Indian market baseline
        impressions  = (budget_per_variation / cpm_estimate) * 1000
        est_clicks   = impressions * (sim["mean_ctr"] / 100)
        est_revenue  = est_clicks * sim["mean_roas"]  # rough revenue proxy

        simulated.append({
            "variation_id":       var["variation_id"],
            "label":              var["label"],
            "type":               var["type"],
            "source_filename":    var.get("source_filename", ""),
            "clip_score":         round(var["score"], 4),
            "mean_ctr":           round(sim["mean_ctr"], 2),
            "std_ctr":            round(sim["std_ctr"], 2),
            "mean_roas":          round(sim["mean_roas"], 2),
            "std_roas":           round(sim["std_roas"], 2),
            "p95_ctr":            round(sim["p95_ctr"], 2),
            "p05_ctr":            round(sim["p05_ctr"], 2),
            "p95_roas":           round(sim["p95_roas"], 2),
            "p05_roas":           round(sim["p05_roas"], 2),
            "ctr_dist":           sim["ctr_dist"],
            "roas_dist":          sim["roas_dist"],
            "est_impressions":    round(impressions),
            "est_clicks":         round(est_clicks),
            "est_revenue_inr":    round(est_revenue),
            "budget_inr":         budget_per_variation,
            "image":              var.get("image"),
        })

    # ── Sort by ROAS (primary), CTR (secondary) ─────────────────────
    simulated.sort(key=lambda x: (x["mean_roas"], x["mean_ctr"]), reverse=True)
    for i, v in enumerate(simulated):
        v["rank"] = i + 1

    winner    = simulated[0]
    runner_up = simulated[1] if len(simulated) > 1 else winner

    # ── Statistical analysis ─────────────────────────────────────────
    confidence  = _compute_confidence(winner["roas_dist"], runner_up["roas_dist"])
    effect      = _effect_size(
        winner["mean_roas"], winner["std_roas"],
        runner_up["mean_roas"], runner_up["std_roas"],
    )

    if confidence >= 0.85 and effect >= 0.5:
        significance = "Strong"
    elif confidence >= 0.70:
        significance = "Moderate"
    else:
        significance = "Weak — consider more data"

    # ── Summary DataFrame (for display) ─────────────────────────────
    df_rows = []
    for v in simulated:
        df_rows.append({
            "Rank":           v["rank"],
            "Variation":      v["label"],
            "Type":           v["type"].title(),
            "Source":         v["source_filename"],
            "CLIP Score":     v["clip_score"],
            "Est. CTR (%)":   v["mean_ctr"],
            "CTR Range":      f"{v['p05_ctr']}–{v['p95_ctr']}",
            "Est. ROAS":      v["mean_roas"],
            "ROAS Range":     f"{v['p05_roas']}–{v['p95_roas']}",
            "Est. Clicks":    v["est_clicks"],
            "Est. Revenue ₹": v["est_revenue_inr"],
        })
    summary_df = pd.DataFrame(df_rows)

    return {
        "simulated_variations": simulated,
        "winner":               winner,
        "runner_up":            runner_up,
        "confidence":           round(confidence, 3),
        "effect_size":          round(effect, 3),
        "significance":         significance,
        "summary_df":           summary_df,
        "platform":             platform,
        "budget_per_variation": budget_per_variation,
        "total_budget_inr":     budget_per_variation * len(simulated),
        "scoring_prompts":      agent1_output.get("scoring_prompts", []),
        "state":                agent1_output.get("state", ""),
        "product_category":     agent1_output.get("product_category", ""),
    }
