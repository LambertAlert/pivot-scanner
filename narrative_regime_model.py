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
    momentum_5d  = risk_on_pct(5)

    # ── Narrative acceleration ─────────────────────────────────────────────
    # Capital Flows insight: rate of change matters as much as level.
    # Acceleration = short-window momentum minus longer-window baseline.
    # Positive = regime improving faster than the trend (add exposure).
    # Negative = regime deteriorating relative to trend (reduce exposure).
    narrative_acceleration = round(momentum_10d - momentum_20d, 3)
    if   narrative_acceleration >= 0.20:  acceleration_label = "ACCELERATING"
    elif narrative_acceleration >= 0.08:  acceleration_label = "IMPROVING"
    elif narrative_acceleration <= -0.20: acceleration_label = "DECELERATING"
    elif narrative_acceleration <= -0.08: acceleration_label = "DETERIORATING"
    else:                                  acceleration_label = "STABLE"

    # ── Posture classification ─────────────────────────────────────────────
    # Acceleration modulates confidence: ACCELERATING boosts BUY_RIP conviction,
    # DECELERATING lowers it and can flip NEUTRAL → early AVOID warning.
    posture, posture_confidence = classify_posture(
        current_state_id, p_risk_on_next, momentum_10d
    )
    # Confidence modifier from acceleration
    if acceleration_label in ("ACCELERATING", "IMPROVING") and posture == "BUY_RIP":
        posture_confidence = min(1.0, round(posture_confidence + 0.06, 3))
    elif acceleration_label in ("DECELERATING", "DETERIORATING") and posture in ("BUY_RIP", "NEUTRAL"):
        posture_confidence = max(0.0, round(posture_confidence - 0.06, 3))

    # ── Regime score (0–1 composite) ───────────────────────────────────────
    # Acceleration adds a directional modifier: improving regimes score
    # higher, deteriorating regimes score lower, capped at ±0.08.
    accel_bonus = max(-0.08, min(0.08, narrative_acceleration * 0.4))
    regime_score = round(min(1.0, max(0.0,
        W_TRANSITION   * p_risk_on_next
        + W_MOMENTUM_10D * momentum_10d
        + W_MOMENTUM_20D * momentum_20d
        + accel_bonus
    )), 3)

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
        "momentum_5d":         round(momentum_5d, 3),
        "regime_score":        regime_score,
        # Narrative acceleration (Capital Flows: velocity matters as much as level)
        "narrative_acceleration": narrative_acceleration,
        "acceleration_label":     acceleration_label,
        # Streak
        "streak_days":         streak_days,
        "streak_class":        streak_class,
        # Most likely next state
        "next_state_id":       next_state_id,
        "next_state_name":     next_state_name,
        "next_state_prob":     next_state_prob,
        "top3_transitions":    top3_transitions,
        # Capital flows signals — injected by tactical_macro_prep.py after FRED/FX fetch.
        # None until injection; dashboard renders gracefully as "awaiting data".
        "real_rate_10y":       None,   # DFII10: 10Y TIPS real yield
        "real_rate_direction": None,   # "FALLING" / "RISING" / "FLAT"
        "real_rate_label":     None,   # "NEGATIVE" / "NEAR ZERO" / "POSITIVE" / "ELEVATED"
        "carry_jpy_5d":        None,   # USDJPY=X 5-day % change (carry proxy)
        "carry_signal":        None,   # "UNWIND" / "STABLE" / "EXPANSION"
        # Curve regime — injected by tactical_macro_prep.py from FRED T10Y2Y
        "curve_regime":        None,   # "BEAR_STEEPENER" / "BULL_STEEPENER" / "FLATTENING" / "INVERTED" / "NEUTRAL"
        "spread_2s10s":        None,   # 10Y-2Y spread in % (T10Y2Y)
        "spread_5s30s":        None,   # 30Y-5Y spread in % (T30Y5Y)
        "curve_direction":     None,   # "STEEPENING" / "FLATTENING" / "FLAT"
        # Entry mode — computed by tactical_macro_prep.py after all injections.
        # This is the single actionable output synthesizing all capital flows signals.
        "entry_mode":          None,   # "CONTINUATION" / "TIGHT_MA" / "ANTICIPATION_ONLY" / "CASH"
        "entry_mode_reason":   None,   # one-sentence plain-English rationale
        # Full transition row for debug/display (from current state only)
        "transition_row_json": json.dumps(
            {STATE_NAMES.get(i + 1, f"State {i+1}"): round(float(p), 3)
             for i, p in enumerate(p_row)}
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY MODE SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

# Thresholds matching Capital Flows framework from war room spec
REAL_RATE_RESTRICTIVE = 1.50   # above this = tight liquidity, no chase
CARRY_UNWIND_THRESHOLD = -1.5  # USDJPY 5d % drop ≥ this = reduce size 30-50%

ENTRY_MODE_ICONS = {
    "CONTINUATION":      "▲",
    "TIGHT_MA":          "◆",
    "ANTICIPATION_ONLY": "◈",
    "CASH":              "■",
}

ENTRY_MODE_COLORS = {
    "CONTINUATION":      "green",
    "TIGHT_MA":          "amber",
    "ANTICIPATION_ONLY": "blue",
    "CASH":              "red",
}


def compute_entry_mode(regime: dict) -> tuple[str, str]:
    """
    Synthesize posture × acceleration × real rate × carry × curve into a single
    actionable entry mode string and one-sentence reason.

    Called by tactical_macro_prep.py AFTER real_rate and carry are injected
    (those fields are None in compute_regime_posture output until injection).

    Entry modes
    -----------
    CONTINUATION      — BUY_RIP confirmed, no liquidity headwinds; full-size
                        continuation entries valid.
    TIGHT_MA          — Valid longs exist but one or more liquidity constraints
                        active; wait for 10/21 EMA pullback, no chasing extension.
    ANTICIPATION_ONLY — BUY_DIP in progress or NEUTRAL with improving acceleration;
                        early positioning only, half-size max, await confirmation.
    CASH              — AVOID posture OR carry unwind active; stand aside entirely.

    Returns
    -------
    (entry_mode: str, entry_mode_reason: str)
    """
    posture       = str(regime.get("posture", "NEUTRAL"))
    accel_label   = str(regime.get("acceleration_label", "STABLE"))
    carry_signal  = str(regime.get("carry_signal") or "STABLE")
    curve_regime  = str(regime.get("curve_regime") or "UNKNOWN")

    real_rate_raw = regime.get("real_rate_10y")
    real_rate     = float(real_rate_raw) if real_rate_raw is not None else None

    # ── Hard stops — override everything else ──────────────────────────────
    if carry_signal == "UNWIND":
        carry_5d = regime.get("carry_jpy_5d")
        carry_str = f" (JPY {carry_5d:+.1f}%)" if carry_5d is not None else ""
        return (
            "CASH",
            f"Carry unwind active{carry_str} — reduce size 30-50%, no new longs until JPY stabilises",
        )

    if posture == "AVOID":
        return (
            "CASH",
            "Regime posture AVOID — narrative state deteriorating, stay out of longs",
        )

    # ── Liquidity constraint flags ─────────────────────────────────────────
    rr_restrictive  = (real_rate is not None and real_rate > REAL_RATE_RESTRICTIVE)
    bear_steepener  = (curve_regime == "BEAR_STEEPENER")
    liquidity_tight = rr_restrictive or bear_steepener

    def _rr_str():
        if real_rate is not None:
            return f"real rate {real_rate:+.2f}% (ELEVATED > {REAL_RATE_RESTRICTIVE:.1f}%)"
        return "real rate elevated"

    def _curve_str():
        return "curve bear steepening (multiple compression risk)"

    def _constraint_reasons():
        parts = []
        if rr_restrictive:
            parts.append(_rr_str())
        if bear_steepener:
            parts.append(_curve_str())
        return " + ".join(parts)

    # ── BUY_RIP branch ─────────────────────────────────────────────────────
    if posture == "BUY_RIP":
        if liquidity_tight:
            return (
                "TIGHT_MA",
                f"BUY_RIP but {_constraint_reasons()} — wait for 10/21 EMA test, no chase",
            )
        if accel_label in ("DECELERATING", "DETERIORATING"):
            return (
                "TIGHT_MA",
                f"BUY_RIP but momentum {accel_label.lower()} — wait for MA pullback before adding",
            )
        # Clean BUY_RIP — acceleration neutral-to-positive, no liquidity headwinds
        accel_note = (
            "acceleration confirmed"
            if accel_label in ("ACCELERATING", "IMPROVING")
            else "momentum stable"
        )
        rr_note = (
            f"real rate {real_rate:+.2f}% (below restrictive zone)"
            if real_rate is not None
            else "real rate data pending"
        )
        return (
            "CONTINUATION",
            f"BUY_RIP + {accel_note}, {rr_note}, carry {carry_signal.lower()} — full continuation entries valid",
        )

    # ── BUY_DIP branch ─────────────────────────────────────────────────────
    if posture == "BUY_DIP":
        if accel_label in ("ACCELERATING", "IMPROVING") and not liquidity_tight:
            return (
                "ANTICIPATION_ONLY",
                f"BUY_DIP forming, acceleration turning positive — early positioning (half-size), await follow-through",
            )
        # BUY_DIP but still messy
        constraints = []
        if liquidity_tight:
            constraints.append(_constraint_reasons())
        if accel_label in ("DECELERATING", "DETERIORATING"):
            constraints.append(f"acceleration {accel_label.lower()}")
        constraint_note = "; ".join(constraints) if constraints else "conditions not yet confirmed"
        return (
            "ANTICIPATION_ONLY",
            f"BUY_DIP signal but {constraint_note} — watch only, no full size yet",
        )

    # ── NEUTRAL fallback ───────────────────────────────────────────────────
    # Check if we're trending toward a BUY_DIP setup
    if accel_label in ("ACCELERATING", "IMPROVING"):
        return (
            "ANTICIPATION_ONLY",
            f"NEUTRAL posture but acceleration {accel_label.lower()} — watch for BUY_DIP confirmation, no full commitment",
        )
    return (
        "CASH",
        "NEUTRAL regime — await posture clarity before committing capital",
    )


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

    accel = result["narrative_acceleration"]
    accel_arrow = "▲" if accel > 0.08 else ("▼" if accel < -0.08 else "→")

    print(f"\n  Current state  : [{result['current_state_id']}] {result['current_state_name']}"
          f"  ({result['current_class']})")
    print(f"  Streak         : {result['streak_days']} days in {result['streak_class']}")
    print(f"\n  P(→ risk-on)   : {p_on:.3f}  {bar_on}")
    print(f"  P(→ risk-off)  : {p_off:.3f}  {bar_off}")
    print(f"\n  Momentum  5d   : {result['momentum_5d']:.1%}")
    print(f"  Momentum 10d   : {result['momentum_10d']:.1%}")
    print(f"  Momentum 20d   : {result['momentum_20d']:.1%}")
    print(f"  Acceleration   : {accel:+.3f}  {accel_arrow}  {result['acceleration_label']}")
    print(f"  Regime score   : {result['regime_score']:.3f}  {state_bar}")
    print(f"\n  ──────────────────────────────────────────")
    print(f"  POSTURE        : {result['posture']}  (confidence {result['posture_confidence']:.1%})")
    if result["current_class"] == "risk-on":
        print(f"  P(buy the rip) : {result['p_buy_rip']:.3f}")
    else:
        print(f"  P(buy the dip) : {result['p_buy_dip']:.3f}")
    print(f"  P(avoid)       : {result['p_avoid']:.3f}")
    if result.get("real_rate_10y") is not None:
        print(f"\n  Real Rate 10Y  : {result['real_rate_10y']:.2f}%  [{result['real_rate_label']}]"
              f"  {result['real_rate_direction']}")
    if result.get("carry_signal") is not None:
        print(f"  Carry (JPY 5d) : {result['carry_jpy_5d']:+.2f}%  [{result['carry_signal']}]")
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
