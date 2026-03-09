"""
<<<<<<< HEAD
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
=======
MONTE CARLO EXPERIMENT SIMULATOR (v2)
=======================================
Multi-arm simulator. Replaces the old head-to-head Agent 2.

Takes Bandit Agent's budget allocations and runs weighted Monte Carlo
simulations across ALL top-K creatives simultaneously.

Key differences from v1:
  - Runs N simulations per creative where N = bandit allocation (not fixed 1000)
  - Computes pairwise confidence: every creative vs every other (not just winner vs #2)
  - Returns full leaderboard with rank, confidence bands, and bandit context
  - Statistical significance assessed across the full field, not just top-2

Pipeline:
  1. Per-creative Monte Carlo  — CTR + ROAS distributions using bandit n_simulations
  2. Pairwise confidence matrix — bootstrap win-rate for each pair
  3. Overall winner confidence  — P(winner beats ALL others)
  4. Leaderboard + summary DataFrame

Output schema (per creative in leaderboard):
  {
    creative_id, label, filename, image,
    clip_score, rank,
    mean_ctr, std_ctr, p05_ctr, p95_ctr,
    mean_roas, std_roas, p05_roas, p95_roas,
    ctr_dist (list), roas_dist (list),
    est_impressions, est_clicks, est_revenue_inr,
    n_simulations,      # from bandit allocation
    win_rate,           # from bandit
    vs_field_confidence # P(this creative beats all others in field)
  }
"""

import math
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Platform performance profiles
>>>>>>> feature-V2-bandit
# ─────────────────────────────────────────────

PLATFORM_PROFILES = {
    "Meta Feed": {
        "ctr_multiplier":  8.0,
        "roas_multiplier": 15.0,
        "ctr_noise":       0.35,
<<<<<<< HEAD
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
=======
        "roas_noise":      0.60,
        "ctr_floor":       0.40,
        "roas_floor":      1.20,
    },
    "Meta Reels": {
        "ctr_multiplier":  10.0,
        "roas_multiplier": 12.0,
        "ctr_noise":       0.50,
        "roas_noise":      0.80,
        "ctr_floor":       0.60,
        "roas_floor":      1.00,
    },
    "Amazon": {
        "ctr_multiplier":  6.0,
        "roas_multiplier": 20.0,
        "ctr_noise":       0.25,
        "roas_noise":      0.50,
        "ctr_floor":       0.30,
        "roas_floor":      2.00,
    },
    "Google Display": {
        "ctr_multiplier":  4.0,
        "roas_multiplier": 10.0,
        "ctr_noise":       0.20,
        "roas_noise":      0.40,
        "ctr_floor":       0.10,
        "roas_floor":      0.80,
    },
}

DEFAULT_PROFILE     = PLATFORM_PROFILES["Meta Feed"]
CPM_INR             = 80    # Rs.80 CPM — Indian market baseline
DEFAULT_BUDGET_INR  = 500   # Rs.500 per creative default test budget


# ─────────────────────────────────────────────
#  1. Per-creative Monte Carlo
# ─────────────────────────────────────────────

def _simulate_creative(
    clip_score:   float,
    profile:      dict,
    n_sims:       int,
    seed:         int = 0,
) -> dict:
    """
    Run Monte Carlo for a single creative.
    n_sims is the bandit-allocated budget for this arm.
    """
    rng = np.random.default_rng(seed)

    base_ctr  = clip_score * profile["ctr_multiplier"]
    base_roas = clip_score * profile["roas_multiplier"]

    ctrs  = np.clip(
        rng.normal(base_ctr,  profile["ctr_noise"],  n_sims),
        profile["ctr_floor"], None
    )
    roass = np.clip(
        rng.normal(base_roas, profile["roas_noise"], n_sims),
        profile["roas_floor"], None
    )

    return {
        "mean_ctr":  float(np.mean(ctrs)),
        "std_ctr":   float(np.std(ctrs)),
        "p05_ctr":   float(np.percentile(ctrs,  5)),
        "p95_ctr":   float(np.percentile(ctrs, 95)),
        "mean_roas": float(np.mean(roass)),
        "std_roas":  float(np.std(roass)),
        "p05_roas":  float(np.percentile(roass,  5)),
        "p95_roas":  float(np.percentile(roass, 95)),
        "ctr_dist":  ctrs.tolist(),
        "roas_dist": roass.tolist(),
    }


# ─────────────────────────────────────────────
#  2. Statistical helpers
# ─────────────────────────────────────────────

def _pairwise_confidence(dist_a: list, dist_b: list) -> float:
    """Bootstrap: P(A beats B) — fraction of matched draws where A > B."""
    a = np.array(dist_a)
    b = np.array(dist_b)
    n = min(len(a), len(b))
    return float(np.sum(a[:n] > b[:n]) / n)


def _vs_field_confidence(winner_dist: list, all_dists: list) -> float:
    """
    P(winner beats ALL other creatives simultaneously).
    Truncates all dists to shortest length — bandit gives different n_simulations
    per creative so arrays are unequal length, causing numpy broadcast errors.
    """
    if not all_dists:
        return 1.0
    # Truncate everything to the shortest distribution
    n = min(len(winner_dist), min(len(d) for d in all_dists))
    w       = np.array(winner_dist[:n])
    others  = [np.array(d[:n]) for d in all_dists]
    stacked   = np.stack(others, axis=0)                     # (K-1, n)
    beats_all = np.all(w[np.newaxis, :] > stacked, axis=0)  # (n,)
    return float(np.mean(beats_all))


def _cohens_d(mean_a: float, std_a: float, mean_b: float, std_b: float) -> float:
    pooled = math.sqrt((std_a ** 2 + std_b ** 2) / 2)
    return abs(mean_a - mean_b) / pooled if pooled > 0 else 0.0


def _significance_label(confidence: float, effect: float) -> str:
    if confidence >= 0.85 and effect >= 0.50:
        return "Strong"
    elif confidence >= 0.70:
        return "Moderate"
    else:
        return "Weak — consider more diverse creatives"


# ─────────────────────────────────────────────
#  3. Main Entry Point
# ─────────────────────────────────────────────

def run_simulator(
    agent1_output:      dict,
    bandit_output:      dict,
    budget_per_creative: int = DEFAULT_BUDGET_INR,
) -> dict:
    """
    Multi-arm Monte Carlo simulator.

    Accepts:
      agent1_output  — from run_agent1()
      bandit_output  — from run_bandit()

    Returns:
    {
      "leaderboard":          list of simulated creative dicts (ranked),
      "winner":               dict,
      "runner_up":            dict,
      "top_5":                list (up to 5),
      "confidence":           float,   P(winner beats runner-up)
      "vs_field_confidence":  float,   P(winner beats ALL others)
      "effect_size":          float,   Cohen's d winner vs runner-up
      "significance":         str,
      "pairwise_matrix":      dict,    {id_a: {id_b: confidence}}
      "summary_df":           pd.DataFrame,
      "platform":             str,
      "product_category":     str,
      "scoring_prompts":      list,
      "budget_per_creative":  int,
      "bandit_note":          str,
>>>>>>> feature-V2-bandit
    }
    """
    platform = agent1_output.get("platform", "Meta Feed")
    profile  = PLATFORM_PROFILES.get(platform, DEFAULT_PROFILE)

<<<<<<< HEAD
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
=======
    # ── Build lookup: creative_id → allocation dict ───────────────────
    alloc_map = {
        a["creative_id"]: a
        for a in bandit_output.get("allocations", [])
    }

    # ── Use top_k_creatives from Agent 1 ─────────────────────────────
    creatives = agent1_output.get("top_k_creatives", agent1_output.get("all_creatives", []))

    if len(creatives) < 2:
        return {"error": "Need at least 2 creatives for simulation."}

    # ── Step 1: Simulate each creative ───────────────────────────────
    simulated = []
    for i, c in enumerate(creatives):
        cid      = c["creative_id"]
        alloc    = alloc_map.get(cid, {})
        n_sims   = alloc.get("n_simulations", 500)  # fallback if bandit missed it

        sim = _simulate_creative(
            clip_score = c.get("score", c.get("combined_score", 0.25)),
            profile    = profile,
            n_sims     = n_sims,
            seed       = i,
        )

        impressions  = (budget_per_creative / CPM_INR) * 1000
        est_clicks   = impressions * (sim["mean_ctr"] / 100)
        est_revenue  = est_clicks * sim["mean_roas"]

        simulated.append({
            # Identity
            "creative_id":   cid,
            "label":         c.get("label", c.get("filename", f"Creative {i+1}")),
            "filename":      c.get("filename", ""),
            "image":         c.get("image"),
            # Scores
            "clip_score":    round(c.get("score", c.get("combined_score", 0.0)), 4),
            "image_score":   round(c.get("image_score", 0.0), 4),
            "copy_score":    round(c.get("copy_score", 0.0), 4),
            # CTR
            "mean_ctr":      round(sim["mean_ctr"],  2),
            "std_ctr":       round(sim["std_ctr"],   2),
            "p05_ctr":       round(sim["p05_ctr"],   2),
            "p95_ctr":       round(sim["p95_ctr"],   2),
            # ROAS
            "mean_roas":     round(sim["mean_roas"], 2),
            "std_roas":      round(sim["std_roas"],  2),
            "p05_roas":      round(sim["p05_roas"],  2),
            "p95_roas":      round(sim["p95_roas"],  2),
            # Distributions (for charting + confidence calc)
            "ctr_dist":      sim["ctr_dist"],
            "roas_dist":     sim["roas_dist"],
            # Financials
            "est_impressions":  round(impressions),
            "est_clicks":       round(est_clicks),
            "est_revenue_inr":  round(est_revenue),
            "budget_inr":       budget_per_creative,
            # Bandit metadata
            "n_simulations": n_sims,
            "bandit_wins":   alloc.get("bandit_wins", 0),
            "win_rate":      alloc.get("win_rate", 0.0),
            # Copy fields for brief context
            "headline":      c.get("headline", ""),
            "primary_text":  c.get("primary_text", ""),
            "cta":           c.get("cta", ""),
        })

    # ── Step 2: Rank by ROAS (primary), CTR (secondary) ──────────────
>>>>>>> feature-V2-bandit
    simulated.sort(key=lambda x: (x["mean_roas"], x["mean_ctr"]), reverse=True)
    for i, v in enumerate(simulated):
        v["rank"] = i + 1

    winner    = simulated[0]
<<<<<<< HEAD
    runner_up = simulated[1] if len(simulated) > 1 else winner

    # ── Statistical analysis ─────────────────────────────────────────
    confidence  = _compute_confidence(winner["roas_dist"], runner_up["roas_dist"])
    effect      = _effect_size(
=======
    runner_up = simulated[1]

    # ── Step 3: Pairwise confidence matrix ───────────────────────────
    pairwise = {}
    for va in simulated:
        pairwise[va["creative_id"]] = {}
        for vb in simulated:
            if va["creative_id"] == vb["creative_id"]:
                pairwise[va["creative_id"]][vb["creative_id"]] = 1.0
            else:
                pairwise[va["creative_id"]][vb["creative_id"]] = round(
                    _pairwise_confidence(va["roas_dist"], vb["roas_dist"]), 3
                )

    # ── Step 4: Winner statistics ─────────────────────────────────────
    confidence = _pairwise_confidence(winner["roas_dist"], runner_up["roas_dist"])

    other_dists = [v["roas_dist"] for v in simulated[1:]]
    vs_field    = _vs_field_confidence(winner["roas_dist"], other_dists)

    effect = _cohens_d(
>>>>>>> feature-V2-bandit
        winner["mean_roas"], winner["std_roas"],
        runner_up["mean_roas"], runner_up["std_roas"],
    )

<<<<<<< HEAD
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
=======
    # Attach vs_field_confidence to each creative
    for v in simulated:
        others = [x["roas_dist"] for x in simulated if x["creative_id"] != v["creative_id"]]
        v["vs_field_confidence"] = round(
            _vs_field_confidence(v["roas_dist"], others), 3
        )

    # ── Step 5: Summary DataFrame ─────────────────────────────────────
    rows = []
    for v in simulated:
        rows.append({
            "Rank":              v["rank"],
            "Creative":          v["label"],
            "File":              v["filename"],
            "CLIP Score":        v["clip_score"],
            "Bandit Runs":       v["n_simulations"],
            "Est. CTR (%)":      v["mean_ctr"],
            "CTR Range":         f"{v['p05_ctr']}–{v['p95_ctr']}",
            "Est. ROAS":         v["mean_roas"],
            "ROAS Range":        f"{v['p05_roas']}–{v['p95_roas']}",
            "vs Field Conf.":    f"{v['vs_field_confidence']*100:.1f}%",
            "Est. Clicks":       v["est_clicks"],
            "Est. Revenue Rs.":  v["est_revenue_inr"],
        })

    summary_df = pd.DataFrame(rows)

    return {
        "leaderboard":           simulated,
        "winner":                winner,
        "runner_up":             runner_up,
        "top_5":                 simulated[:5],
        "confidence":            round(confidence, 3),
        "vs_field_confidence":   round(vs_field, 3),
        "effect_size":           round(effect, 3),
        "significance":          _significance_label(confidence, effect),
        "pairwise_matrix":       pairwise,
        "summary_df":            summary_df,
        "platform":              platform,
        "product_category":      agent1_output.get("product_category", ""),
        "audience_desc":         agent1_output.get("audience_desc", ""),
        "scoring_prompts":       agent1_output.get("scoring_prompts", []),
        "budget_per_creative":   budget_per_creative,
        "total_budget_inr":      budget_per_creative * len(simulated),
        "bandit_note":           bandit_output.get("exploration_note", ""),
>>>>>>> feature-V2-bandit
    }
