"""
BANDIT DECISION AGENT (v2) — NEW
==================================
Thompson Sampling over top-K creatives to allocate Monte Carlo simulation budget.

Instead of giving every creative equal simulation runs, the Bandit Agent:
  - Treats each creative as an "arm" of a multi-armed bandit
  - Initialises Beta(alpha, beta) priors seeded from Agent 1 CLIP scores
  - Samples from each arm's Beta distribution N_ROUNDS times
  - Tallies how many rounds each arm "won" → proportional budget allocation
  - Returns simulation_budget: {creative_id: n_simulations}

Why this matters:
  - A clearly dominant creative gets more simulation runs → tighter confidence intervals
  - A weak creative wastes fewer compute cycles
  - Uncertain creatives (similar scores) get explored fairly
  - Mimics how a real ad auction would shift spend toward winners

Beta prior seeding from CLIP score:
  score ∈ [0.20, 0.35] typical range for ViT-B-32
  We map score → (alpha, beta) so higher-scoring creatives start with
  a stronger prior, but the sampling still has meaningful variance.

  alpha = 1 + score * PRIOR_STRENGTH   (successes proxy)
  beta  = 1 + (1 - score) * PRIOR_STRENGTH  (failures proxy)

  PRIOR_STRENGTH = 10 gives a gentle lean without overwhelming exploration.

Output:
  {
    "allocations": [
      {
        "creative_id":    str,
        "label":          str,
        "clip_score":     float,
        "alpha":          float,   # Beta prior alpha (successes)
        "beta":           float,   # Beta prior beta  (failures)
        "bandit_wins":    int,     # rounds this arm won
        "win_rate":       float,   # bandit_wins / N_ROUNDS
        "n_simulations":  int,     # Monte Carlo budget allocated
      }
    ],
    "total_simulations": int,
    "n_rounds":          int,
    "exploration_note":  str,      # human-readable summary
  }
"""

import numpy as np


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

N_ROUNDS        = 2000   # Thompson Sampling rounds — more = smoother allocation
MIN_SIMS        = 200    # minimum simulations guaranteed to every creative
MAX_SIMS        = 2000   # cap per creative to avoid runaway compute
TOTAL_SIM_BUDGET = 5000  # total Monte Carlo runs to distribute across all creatives
PRIOR_STRENGTH  = 10     # controls how strongly CLIP score seeds the Beta prior


# ─────────────────────────────────────────────
#  Core Thompson Sampling
# ─────────────────────────────────────────────

def _seed_prior(clip_score: float) -> tuple:
    """
    Convert a CLIP score into (alpha, beta) Beta distribution parameters.

    CLIP scores for ViT-B-32 typically range 0.20–0.35.
    We normalise to [0, 1] loosely and seed the prior.

    alpha = 1 + score * PRIOR_STRENGTH
    beta  = 1 + (1 - normalised_score) * PRIOR_STRENGTH

    A score of 0.30 → alpha=4.0, beta=2.0 (modest positive lean)
    A score of 0.20 → alpha=3.0, beta=3.0 (neutral, maximum uncertainty)
    A score of 0.35 → alpha=4.5, beta=1.5 (stronger positive lean)
    """
    # Normalise score to [0, 1] using typical ViT-B-32 range
    normalised = np.clip((clip_score - 0.15) / (0.40 - 0.15), 0.0, 1.0)
    alpha = 1.0 + normalised * PRIOR_STRENGTH
    beta  = 1.0 + (1.0 - normalised) * PRIOR_STRENGTH
    return round(alpha, 3), round(beta, 3)


def run_bandit(
    top_k_creatives: list,
    n_rounds:        int  = N_ROUNDS,
    total_budget:    int  = TOTAL_SIM_BUDGET,
    min_sims:        int  = MIN_SIMS,
    max_sims:        int  = MAX_SIMS,
    seed:            int  = 42,
) -> dict:
    """
    Thompson Sampling budget allocator.

    Parameters
    ----------
    top_k_creatives : list of creative dicts from Agent 1
                      must have keys: creative_id, label, score (combined_score)
    n_rounds        : number of Thompson Sampling rounds
    total_budget    : total Monte Carlo simulation runs to distribute
    min_sims        : floor — every creative gets at least this many runs
    max_sims        : ceiling — no creative gets more than this many runs
    seed            : RNG seed for reproducibility

    Returns
    -------
    dict with allocations list + metadata
    """
    rng = np.random.default_rng(seed)
    k   = len(top_k_creatives)

    if k == 0:
        return {"error": "No creatives passed to Bandit Agent."}

    if k == 1:
        # Edge case: only one creative — give it all budget
        c = top_k_creatives[0]
        return {
            "allocations": [{
                **_creative_allocation(c, n_rounds, n_rounds, total_budget),
            }],
            "total_simulations": total_budget,
            "n_rounds":          n_rounds,
            "exploration_note":  "Only one creative — full budget allocated.",
        }

    # ── Seed Beta priors from CLIP scores ────────────────────────────
    # Use 'score' field (combined_score from Agent 1: 0.6*image + 0.4*copy)
    alphas = []
    betas  = []
    for c in top_k_creatives:
        a, b = _seed_prior(c.get("score", c.get("combined_score", 0.25)))
        alphas.append(a)
        betas.append(b)

    alphas = np.array(alphas, dtype=float)
    betas  = np.array(betas,  dtype=float)

    # ── Thompson Sampling rounds ──────────────────────────────────────
    win_counts = np.zeros(k, dtype=int)

    for _ in range(n_rounds):
        # Sample one value from each arm's Beta distribution
        samples = rng.beta(alphas, betas)
        winner  = int(np.argmax(samples))
        win_counts[winner] += 1

        # Bayesian update: winner gets +1 success, others get +1 failure
        # (soft update — prevents one arm from dominating too early)
        alphas[winner] += 1.0
        for j in range(k):
            if j != winner:
                betas[j] += 0.5   # softer failure signal

    # ── Convert win counts → simulation budget ────────────────────────
    win_rates    = win_counts / n_rounds
    raw_budgets  = win_rates * total_budget

    # Apply floor + ceiling
    budgets = np.clip(raw_budgets, min_sims, max_sims).astype(int)

    # Redistribute any budget freed by the ceiling cap
    total_allocated = int(budgets.sum())
    remainder       = total_budget - total_allocated
    if remainder > 0:
        # Give leftover to arms that didn't hit their ceiling
        under_cap = np.where(budgets < max_sims)[0]
        if len(under_cap) > 0:
            extra_each = remainder // len(under_cap)
            budgets[under_cap] += extra_each

    # ── Build output ──────────────────────────────────────────────────
    # Re-read final alpha/beta values after all updates
    allocations = []
    for i, c in enumerate(top_k_creatives):
        allocations.append({
            "creative_id":   c["creative_id"],
            "label":         c.get("label", c.get("filename", f"Creative {i+1}")),
            "filename":      c.get("filename", ""),
            "clip_score":    c.get("score", c.get("combined_score", 0.0)),
            "alpha":         round(float(alphas[i]), 2),
            "beta":          round(float(betas[i]),  2),
            "bandit_wins":   int(win_counts[i]),
            "win_rate":      round(float(win_rates[i]), 4),
            "n_simulations": int(budgets[i]),
        })

    # Sort by n_simulations descending for readability
    allocations.sort(key=lambda x: x["n_simulations"], reverse=True)

    # Exploration note
    top_arm     = allocations[0]
    bottom_arm  = allocations[-1]
    score_range = max(c.get("score", 0) for c in top_k_creatives) - \
                  min(c.get("score", 0) for c in top_k_creatives)

    if score_range < 0.005:
        note = (
            f"Scores are very close (gap: {score_range:.4f}). "
            "Budget distributed nearly equally — high exploration mode."
        )
    elif score_range < 0.015:
        note = (
            f"Moderate score gap ({score_range:.4f}). "
            f"'{top_arm['label']}' leads with {top_arm['n_simulations']} runs."
        )
    else:
        note = (
            f"Strong score gap ({score_range:.4f}). "
            f"'{top_arm['label']}' dominates with {top_arm['n_simulations']} runs "
            f"vs '{bottom_arm['label']}' with {bottom_arm['n_simulations']}."
        )

    return {
        "allocations":       allocations,
        "total_simulations": int(budgets.sum()),
        "n_rounds":          n_rounds,
        "exploration_note":  note,
    }


# ─────────────────────────────────────────────
#  Helper (internal)
# ─────────────────────────────────────────────

def _creative_allocation(creative: dict, wins: int, total_rounds: int, budget: int) -> dict:
    a, b = _seed_prior(creative.get("score", creative.get("combined_score", 0.25)))
    return {
        "creative_id":   creative["creative_id"],
        "label":         creative.get("label", creative.get("filename", "")),
        "filename":      creative.get("filename", ""),
        "clip_score":    creative.get("score", creative.get("combined_score", 0.0)),
        "alpha":         round(a, 2),
        "beta":          round(b, 2),
        "bandit_wins":   wins,
        "win_rate":      round(wins / total_rounds, 4) if total_rounds > 0 else 1.0,
        "n_simulations": budget,
    }
