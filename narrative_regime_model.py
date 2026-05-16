"""
narrative_regime_model.py — Cross-Asset Narrative Regime Model
==============================================================
Uses the 8-state cross-asset narrative history (already computed by
tactical_macro_prep.py) to answer a single actionable question:

    Given where the macro narrative has been and where it is today,
    what is the probability of buying the dip, buying the rip, or
    staying out entirely?

LOGIC
-----
States 1–4  (SPX ↑):  risk-on  — Goldilocks, Broad Risk-On, US Exceptionalism, Soft Landing
States 5–8  (SPX ↓):  risk-off — Flight to Safety, Hawkish Squeeze, Growth Scare, Supply Shock

An 8×8 empirical transition matrix is estimated from the 60-day narrative
history with Laplace smoothing (prevents zero probabilities from sparse
windows).

Given the current state, one-step-ahead transition probabilities are summed
across risk-on vs risk-off destination states:

    P(→ risk-on next)  = Σ T[current, s] for s in {1,2,3,4}
    P(→ risk-off next) = Σ T[current, s] for s in {5,6,7,8}

Posture classification:
    BUY_RIP  — currently risk-on  AND P(→ risk-on) ≥ RIP_THRESHOLD
    BUY_DIP  — currently risk-off AND P(→ risk-on) ≥ DIP_THRESHOLD
    AVOID    — currently risk-off AND P(→ risk-on) < AVOID_CEILING
    NEUTRAL  — everything else (uncertain transition, mid-regime)

Regime momentum (independent signal):
    momentum_10d = % of last 10 days in risk-on states
    momentum_20d = % of last 20 days in risk-on states

Composite regime score (0–1):
    regime_score = 0.5 × P(→ risk-on next)
                 + 0.3 × momentum_10d
                 + 0.2 × momentum_20d

OUTPUT
------
data/narrative_regime.parquet  — one row (current snapshot), columns below
data/narrative_regime_history.parquet — rolling daily record (appended)

Columns:
    generated_at, current_state_id, current_state_name, current_class
    p_risk_on_next, p_risk_off_next
    p_buy_rip, p_buy_dip, p_avoid
    posture, posture_confidence
    momentum_10d, momentum_20d
    regime_score
    streak_days, streak_class
    transition_matrix_json  (8×8 as JSON string for debug/display)

FUTURE EXTENSION (Layer 1 from probabilistic stack)
----------------------------------------------------
Once the HMM breadth model (hmm_sp500_breadth.py) is integrated into the
pipeline, its P(risk-on) can be composed with regime_score here as a second
independent signal:

    composite = 0.6 × regime_score + 0.4 × hmm_p_risk_on

That keeps this model as the intraday/narrative signal and the HMM as the
slower structural filter, matching the stack architecture in the research brief.

Usage:
    python narrative_regime_model.py        # standalone
    from narrative_regime_model import run  # called by tactical_macro_prep.py
"""

import os
import json
import logging
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.makedirs("data", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

NARRATIVE_HISTORY_PATH = "data/narrative_history.parquet"
OUTPUT_SNAPSHOT_PATH   = "data/narrative_regime.parquet"
OUTPUT_HISTORY_PATH    = "data/narrative_regime_history.parquet"

# State classification — SPX direction determines risk-on vs risk-off
RISK_ON_STATES  = {1, 2, 3, 4}   # SPX ↑: Goldilocks, Broad Risk-On, US Exceptionalism, Soft Landing
RISK_OFF_STATES = {5, 6, 7, 8}   # SPX ↓: Flight to Safety, Hawkish Squeeze, Growth Scare, Supply Shock

STATE_NAMES = {
    1: "Goldilocks",
    2: "Broad Risk-On",
    3: "US Exceptionalism",
    4: "Soft Landing",
    5: "Flight to Safety",
    6: "Hawkish Squeeze",
    7: "Growth Scare",
    8: "Supply Shock",
}

# Posture thresholds
RIP_THRESHOLD   = 0.60   # in risk-on:  P(→ risk-on) ≥ this → BUY_RIP
DIP_THRESHOLD   = 0.55   # in risk-off: P(→ risk-on) ≥ this → BUY_DIP
AVOID_CEILING   = 0.38   # in risk-off: P(→ risk-on) < this → AVOID

# Laplace smoothing (prevents zero probabilities from 60-day sparse windows)
LAPLACE_ALPHA = 0.5

# Composite score weights
W_TRANSITION   = 0.50
W_MOMENTUM_10D = 0.30
W_MOMENTUM_20D = 0.20


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSITION MATRIX ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_transition_matrix(
    state_sequence: list[int],
    n_states: int = 8,
    alpha: float = LAPLACE_ALPHA,
) -> np.ndarray:
    """
    Estimate the n_states × n_states row-stochastic transition matrix from a
    sequence of observed states (1-indexed: 1..8).

    Laplace smoothing adds `alpha` pseudo-counts to every cell before
    normalising, ensuring no zero probabilities from sparse windows.

    Returns:
        T: (n_states, n_states) ndarray where T[i, j] = P(→ j+1 | from i+1)
           (0-indexed internally; state ID - 1 = matrix index)
    """
    counts = np.full((n_states, n_states), alpha)   # Laplace prior

    for i in range(len(state_sequence) - 1):
        from_s = state_sequence[i] - 1      # convert to 0-indexed
        to_s   = state_sequence[i + 1] - 1
        if 0 <= from_s < n_states and 0 <= to_s < n_states:
            counts[from_s, to_s] += 1

    # Row-normalize
    row_sums = counts.sum(axis=1, keepdims=True)
    T = counts / np.where(row_sums > 0, row_sums, 1.0)
    return T


# ═══════════════════════════════════════════════════════════════════════════════
#  POSTURE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_posture(
    current_state_id: int,
    p_risk_on_next: float,
    momentum_10d: float,
) -> tuple[str, float]:
    """
    Return (posture_label, confidence) based on current state, transition
    probability, and short-term momentum.

    Posture labels:
        BUY_RIP  — continuation buy (risk-on → risk-on)
        BUY_DIP  — recovery buy (risk-off → risk-on transition expected)
        AVOID    — stay out (risk-off → risk-off continuation expected)
        NEUTRAL  — uncertain; wait for clarity

    Confidence = the dominant probability driving the decision, so the number
    always has a natural interpretation in the dashboard.
    """
    in_risk_on = current_state_id in RISK_ON_STATES

    if in_risk_on:
        if p_risk_on_next >= RIP_THRESHOLD:
            return "BUY_RIP", round(p_risk_on_next, 3)
        elif p_risk_on_next < (1 - RIP_THRESHOLD):
            # Risk-on state but high probability of flipping — caution
            return "NEUTRAL", round(0.5 + abs(p_risk_on_next - 0.5), 3)
        else:
            return "NEUTRAL", round(p_risk_on_next, 3)
    else:
        # Currently risk-off
        if p_risk_on_next >= DIP_THRESHOLD:
            # Strong recovery signal — but only if short-term momentum agrees
            if momentum_10d >= 0.40:
                return "BUY_DIP", round(p_risk_on_next, 3)
            else:
                # Transition likely but market hasn't confirmed — wait
                return "NEUTRAL", round((p_risk_on_next + momentum_10d) / 2, 3)
        elif p_risk_on_next <= AVOID_CEILING:
            return "AVOID", round(1 - p_risk_on_next, 3)
        else:
            return "NEUTRAL", round(p_risk_on_next, 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAK DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_streak(state_sequence: list[int]) -> tuple[int, str]:
    """
    Count how many consecutive days (from the end) the market has been in the
    same risk class (risk-on or risk-off).

    Returns (streak_days, streak_class) where streak_class ∈ {"risk-on", "risk-off"}.
    Longer streaks = higher conviction in the current regime.
    """
    if not state_sequence:
        return 0, "unknown"

    current_class = "risk-on" if state_sequence[-1] in RISK_ON_STATES else "risk-off"
    streak = 1

    for s in reversed(state_sequence[:-1]):
        s_class = "risk-on" if s in RISK_ON_STATES else "risk-off"
        if s_class == current_class:
            streak += 1
        else:
            break

    return streak, current_class


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_regime_posture(history_path: str = NARRATIVE_HISTORY_PATH) -> Optional[dict]:
    """
    Load narrative history, compute transition matrix, and return the full
    regime posture snapshot as a dict.

    Returns None if the history file is missing or has insufficient data.
    """
    if not os.path.exists(history_path):
        log.warning(f"Narrative history not found: {history_path}")
        return None

    df = pd.read_parquet(history_path)
    if df.empty:
        log.warning("Narrative history is empty.")
        return None

    # Normalise column names — tactical_data_layer may use either convention
    df.columns = [c.lower() for c in df.columns]

    # Find state_id column (may be 'state_id', 'narrative_state_id', etc.)
    sid_col = next(
        (c for c in df.columns if "state_id" in c and "narrative" not in c),
        next((c for c in df.columns if "state_id" in c), None)
    )
    name_col = next(
        (c for c in df.columns if "state_name" in c and "narrative" not in c),
        next((c for c in df.columns if "state_name" in c), None)
    )

    if sid_col is None:
        log.warning(f"No state_id column found. Columns: {list(df.columns)}")
        return None

    # Sort chronologically
    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col:
        df = df.sort_values(date_col)

    state_seq = df[sid_col].dropna().astype(int).tolist()

    if len(state_seq) < 5:
        log.warning(f"Insufficient history ({len(state_seq)} rows) for transition matrix.")
        return None

    # ── Transition matrix ──────────────────────────────────────────────────
    T = estimate_transition_matrix(state_seq)

    current_state_id = state_seq[-1]
    current_idx      = current_state_id - 1   # 0-indexed

    p_row = T[current_idx]   # transition distribution from current state

    p_risk_on_next  = float(sum(p_row[s - 1] for s in RISK_ON_STATES))
    p_risk_off_next = float(sum(p_row[s - 1] for s in RISK_OFF_STATES))

    # ── Regime momentum ────────────────────────────────────────────────────
    def risk_on_pct(n: int) -> float:
        window = state_seq[-n:] if len(state_seq) >= n else state_seq
        return sum(1 for s in window if s in RISK_ON_STATES) / len(window)

    momentum_10d = risk_on_pct(10)
    momentum_20d = risk_on_pct(20)

    # ── Posture classification ─────────────────────────────────────────────
    posture, posture_confidence = classify_posture(
        current_state_id, p_risk_on_next, momentum_10d
    )

    # ── Regime score (0–1 composite) ───────────────────────────────────────
    regime_score = (
        W_TRANSITION   * p_risk_on_next
        + W_MOMENTUM_10D * momentum_10d
        + W_MOMENTUM_20D * momentum_20d
    )

    # ── Streak ─────────────────────────────────────────────────────────────
    streak_days, streak_class = compute_streak(state_seq)

    # ── Per-posture probabilities for dashboard display ────────────────────
    # These are derived from the transition probability and momentum,
    # not raw frequencies, so they sum close to 1 but aren't forced to.
    in_risk_on = current_state_id in RISK_ON_STATES

    if in_risk_on:
        p_buy_rip = round(p_risk_on_next * ((momentum_10d + momentum_20d) / 2 + 0.5) / 1.5, 3)
        p_buy_dip = 0.0   # not applicable when already in risk-on
        p_avoid   = round((1 - p_risk_on_next) * (1 - momentum_10d), 3)
    else:
        p_buy_dip = round(p_risk_on_next * ((momentum_10d + momentum_20d) / 2 + 0.5) / 1.5, 3)
        p_buy_rip = 0.0   # not applicable when in risk-off
        p_avoid   = round(p_risk_off_next * (1 - (momentum_10d + momentum_20d) / 2), 3)

    # ── State name lookup ──────────────────────────────────────────────────
    current_state_name = STATE_NAMES.get(current_state_id, f"State {current_state_id}")
    if name_col and not df[name_col].empty:
        last_name = df[name_col].iloc[-1]
        if pd.notna(last_name) and str(last_name).strip():
            current_state_name = str(last_name)

    # ── Most likely next state ─────────────────────────────────────────────
    next_state_idx    = int(np.argmax(p_row))
    next_state_id     = next_state_idx + 1
    next_state_name   = STATE_NAMES.get(next_state_id, f"State {next_state_id}")
    next_state_prob   = round(float(p_row[next_state_idx]), 3)

    # ── Top 3 most likely transitions ─────────────────────────────────────
    top3_idxs = np.argsort(p_row)[::-1][:3]
    top3_transitions = [
        {
            "state_id":   int(idx + 1),
            "state_name": STATE_NAMES.get(int(idx + 1), f"State {int(idx + 1)}"),
            "class":      "risk-on" if (idx + 1) in RISK_ON_STATES else "risk-off",
            "probability": round(float(p_row[idx]), 3),
        }
        for idx in top3_idxs
    ]

    return {
        "generated_at":        datetime.now().isoformat(),
        "history_days":        len(state_seq),
        # Current state
        "current_state_id":    current_state_id,
        "current_state_name":  current_state_name,
        "current_class":       "risk-on" if in_risk_on else "risk-off",
        # One-step transition probabilities
        "p_risk_on_next":      round(p_risk_on_next, 3),
        "p_risk_off_next":     round(p_risk_off_next, 3),
        # Per-posture probabilities
        "p_buy_rip":           p_buy_rip,
        "p_buy_dip":           p_buy_dip,
        "p_avoid":             p_avoid,
        # Top-level posture signal
        "posture":             posture,
        "posture_confidence":  posture_confidence,
        # Regime momentum
        "momentum_10d":        round(momentum_10d, 3),
        "momentum_20d":        round(momentum_20d, 3),
        "regime_score":        round(regime_score, 3),
        # Streak
        "streak_days":         streak_days,
        "streak_class":        streak_class,
        # Most likely next state
        "next_state_id":       next_state_id,
        "next_state_name":     next_state_name,
        "next_state_prob":     next_state_prob,
        "top3_transitions":    top3_transitions,
        # Full transition row for debug/display (from current state only)
        "transition_row_json": json.dumps(
            {STATE_NAMES.get(i + 1, f"State {i+1}"): round(float(p), 3)
             for i, p in enumerate(p_row)}
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSIST
# ═══════════════════════════════════════════════════════════════════════════════

def save_regime_snapshot(result: dict) -> None:
    """Write current snapshot parquet + append to rolling history."""
    snap_df = pd.DataFrame([result])
    snap_df.to_parquet(OUTPUT_SNAPSHOT_PATH, index=False)
    log.info(f"✅ {OUTPUT_SNAPSHOT_PATH}")

    # Rolling history — append today's row
    if os.path.exists(OUTPUT_HISTORY_PATH):
        try:
            hist = pd.read_parquet(OUTPUT_HISTORY_PATH)
            # Drop any existing row for the same date (idempotent)
            today = datetime.now().strftime("%Y-%m-%d")
            hist = hist[~hist["generated_at"].str.startswith(today)]
            hist = pd.concat([hist, snap_df], ignore_index=True)
        except Exception:
            hist = snap_df
    else:
        hist = snap_df

    # Keep last 180 days
    if "generated_at" in hist.columns:
        hist = hist.sort_values("generated_at").tail(180)

    hist.to_parquet(OUTPUT_HISTORY_PATH, index=False)
    log.info(f"✅ {OUTPUT_HISTORY_PATH} ({len(hist)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def run(history_path: str = NARRATIVE_HISTORY_PATH) -> Optional[dict]:
    """
    Called by tactical_macro_prep.py after narrative_history.parquet is written.
    Returns the result dict (for embedding in the macro state parquet) and
    also writes its own output files.
    """
    result = compute_regime_posture(history_path)
    if result is None:
        log.error("Narrative regime model returned no result.")
        return None

    save_regime_snapshot(result)
    return result


def _print_summary(result: dict) -> None:
    state_bar = "█" * int(result["regime_score"] * 20)
    state_bar = f"{state_bar:<20}"

    p_on  = result["p_risk_on_next"]
    p_off = result["p_risk_off_next"]
    bar_on  = "▓" * int(p_on  * 20)
    bar_off = "▓" * int(p_off * 20)

    print(f"\n  Current state  : [{result['current_state_id']}] {result['current_state_name']}"
          f"  ({result['current_class']})")
    print(f"  Streak         : {result['streak_days']} days in {result['streak_class']}")
    print(f"\n  P(→ risk-on)   : {p_on:.3f}  {bar_on}")
    print(f"  P(→ risk-off)  : {p_off:.3f}  {bar_off}")
    print(f"\n  Momentum 10d   : {result['momentum_10d']:.1%}  of days risk-on")
    print(f"  Momentum 20d   : {result['momentum_20d']:.1%}  of days risk-on")
    print(f"  Regime score   : {result['regime_score']:.3f}  {state_bar}")
    print(f"\n  ──────────────────────────────────────────")
    print(f"  POSTURE        : {result['posture']}  (confidence {result['posture_confidence']:.1%})")
    if result["current_class"] == "risk-on":
        print(f"  P(buy the rip) : {result['p_buy_rip']:.3f}")
    else:
        print(f"  P(buy the dip) : {result['p_buy_dip']:.3f}")
    print(f"  P(avoid)       : {result['p_avoid']:.3f}")
    print(f"\n  Top 3 next states:")
    for t in result["top3_transitions"]:
        bar = "▓" * int(t["probability"] * 20)
        arrow = "↗" if t["class"] == "risk-on" else "↘"
        print(f"    {arrow} [{t['state_id']}] {t['state_name']:<22}"
              f"  {t['probability']:.3f}  {bar}")


def main():
    print("=" * 60)
    print("NARRATIVE REGIME MODEL — Cross-Asset Posture Signal")
    print("=" * 60)

    result = run()
    if result:
        _print_summary(result)
        print(f"\n  ✅ Written to {OUTPUT_SNAPSHOT_PATH}")
        print(f"  ✅ History appended to {OUTPUT_HISTORY_PATH}")
    else:
        print("  ❌ No result — check that narrative_history.parquet exists.")
        print("     Run tactical_macro_prep.py first.")

    print("=" * 60)


if __name__ == "__main__":
    main()
