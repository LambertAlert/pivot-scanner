"""
radar_prep.py — Setups Radar Scoring
=====================================
Computes IN MOTION and LOADING scores for every ticker on the daily watchlist.

IN MOTION = moving right now (chase or hold)
LOADING   = coiled but not yet triggered (watch for entry)

Pulls daily bars + intraday data, runs full enrichment:
  • ATR contraction (5d ATR / 20d ATR)
  • Distance from 5-day high
  • Volume profile (RVOL, dry-up, spike)
  • Today's % change
  • Range position in candle
  • EMA proximity (9, 21)
  • Higher lows count

Saves: data/radar_data.json
Runs: as part of the daily_screener pipeline (after daily_screener.py)
       OR can be triggered standalone

Dependencies: pip install yfinance pandas numpy
"""

import os
import json
import time
import logging
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from data_layer import get_latest_daily_watchlist, get_today_triggers

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_JSON      = "data/radar_data.json"
RATE_LIMIT_PAUSE = 1.0
os.makedirs("data", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_daily(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """Pull 6 months of daily OHLCV — enough history for 20-day metrics + buffer."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if df.empty or len(df) < 25:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        log.warning(f"[{ticker}] daily fetch error: {e}")
        return None


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — Wilder's smoothing."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  IN MOTION SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def calc_in_motion_score(df: pd.DataFrame, recent_trigger: bool = False) -> dict:
    """
    Score 0-100 capturing "moving right now."

    Components (weighted):
      1. Today's % change ≥ +2%                           (25%)
      2. Volume vs 20-day avg ≥ 1.5×                     (20%)
      3. Recent 30-min pivot trigger (last 1–2 bars)    (25%)
      4. Breaking out of 10-day range                    (15%)
      5. Strong daily candle (close in upper 60% range)  (15%)
    """
    if len(df) < 25:
        return {"score": 0, "components": {}, "reasons": []}

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    today_close = close.iloc[-1]
    prev_close  = close.iloc[-2]
    today_high  = high.iloc[-1]
    today_low   = low.iloc[-1]
    today_vol   = vol.iloc[-1]

    # ── 1. Today's % change ────────────────────────────────────────────────
    pct_chg = ((today_close / prev_close) - 1) * 100
    if   pct_chg >= 5.0:  c1, lbl1 = 100, f"Up {pct_chg:.1f}% today"
    elif pct_chg >= 3.0:  c1, lbl1 = 80,  f"Up {pct_chg:.1f}% today"
    elif pct_chg >= 2.0:  c1, lbl1 = 60,  f"Up {pct_chg:.1f}% today"
    elif pct_chg >= 1.0:  c1, lbl1 = 40,  f"Up {pct_chg:.1f}% today"
    elif pct_chg >= 0:    c1, lbl1 = 20,  f"Flat to slightly up"
    else:                 c1, lbl1 = 0,   f"Down {pct_chg:.1f}% today"

    # ── 2. Volume vs 20-day avg ────────────────────────────────────────────
    avg_vol_20 = vol.iloc[-20:].mean()
    rvol = today_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
    if   rvol >= 3.0:  c2, lbl2 = 100, f"Volume {rvol:.1f}× — heavy"
    elif rvol >= 2.0:  c2, lbl2 = 80,  f"Volume {rvol:.1f}× — strong"
    elif rvol >= 1.5:  c2, lbl2 = 60,  f"Volume {rvol:.1f}× — elevated"
    elif rvol >= 1.0:  c2, lbl2 = 30,  f"Volume {rvol:.1f}× — normal"
    else:              c2, lbl2 = 0,   f"Volume {rvol:.1f}× — light"

    # ── 3. Recent 30-min pivot trigger ─────────────────────────────────────
    if recent_trigger:
        c3, lbl3 = 100, "30-min pivot triggered today"
    else:
        c3, lbl3 = 0, ""

    # ── 4. Breaking 10-day range ───────────────────────────────────────────
    high_10d_prev = high.iloc[-11:-1].max()
    breaking_out  = today_close > high_10d_prev
    if breaking_out:
        c4, lbl4 = 100, f"Breaking 10-day high (${high_10d_prev:.2f})"
    else:
        dist_pct = ((today_close / high_10d_prev) - 1) * 100
        if dist_pct >= -1:  c4, lbl4 = 60, "Within 1% of 10-day high"
        elif dist_pct >= -3: c4, lbl4 = 30, "Approaching 10-day high"
        else:               c4, lbl4 = 0, ""

    # ── 5. Candle position (close in upper portion of range) ───────────────
    candle_range = today_high - today_low
    if candle_range > 0:
        pos_in_range = (today_close - today_low) / candle_range
    else:
        pos_in_range = 0.5
    if   pos_in_range >= 0.85: c5, lbl5 = 100, "Closed near high — strong candle"
    elif pos_in_range >= 0.70: c5, lbl5 = 70,  "Closed in upper third"
    elif pos_in_range >= 0.50: c5, lbl5 = 40,  "Mid-range close"
    else:                      c5, lbl5 = 0,   "Closed weak in lower half"

    # ── Weighted composite ─────────────────────────────────────────────────
    weights = {"pct_chg": 0.25, "rvol": 0.20, "trigger": 0.25,
               "breakout": 0.15, "candle": 0.15}
    components = {
        "pct_chg":  round(c1, 1),
        "rvol":     round(c2, 1),
        "trigger":  round(c3, 1),
        "breakout": round(c4, 1),
        "candle":   round(c5, 1),
    }
    score = sum(components[k] * weights[k] for k in weights)

    reasons = [r for r in [lbl1, lbl2, lbl3, lbl4, lbl5] if r]
    return {
        "score":       round(score, 1),
        "components":  components,
        "reasons":     reasons,
        "today_pct":   round(pct_chg, 2),
        "rvol":        round(rvol, 2),
        "today_close": round(float(today_close), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  LOADING SCORE  (pre-trigger, coiled)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_loading_score(df: pd.DataFrame, pivot_8w_tier: str = "NONE",
                        daily_bbuw: float = 0) -> dict:
    """
    Score 0-100 capturing "loaded but not yet moving."

    Components (weighted):
      1. 8W Pivot tier (PROXIMITY/STANDARD = high)         (20%)
      2. Daily BBUW ≥ 70                                   (15%)
      3. ATR contraction (5d ATR / 20d ATR)               (15%)
      4. Volume drying up (5d vol / 20d vol)              (15%)
      5. On 21 EMA (within 2%)                             (15%)
      6. Distance from 5-day high (within 3%)             (10%)
      7. Higher lows trend (last 5–8 bars)                 (10%)
    """
    if len(df) < 25:
        return {"score": 0, "components": {}, "reasons": []}

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    today_close = close.iloc[-1]

    # ── 1. 8W Pivot tier (passed in) ───────────────────────────────────────
    tier_score = {
        "STRONG":    100,   # already pivoted with confirmation
        "STANDARD":  90,    # solid pivot setup
        "PROXIMITY": 100,   # this is the sweet spot for LOADING — within 3% of EMA
        "WEAK":      40,
        "NONE":      0,
    }.get(pivot_8w_tier, 0)
    c1, lbl1 = tier_score, f"8W Pivot: {pivot_8w_tier}" if pivot_8w_tier != "NONE" else ""

    # ── 2. Daily BBUW ──────────────────────────────────────────────────────
    if   daily_bbuw >= 80: c2, lbl2 = 100, f"BBUW {daily_bbuw:.0f} (elite)"
    elif daily_bbuw >= 70: c2, lbl2 = 80,  f"BBUW {daily_bbuw:.0f}"
    elif daily_bbuw >= 60: c2, lbl2 = 50,  f"BBUW {daily_bbuw:.0f}"
    else:                  c2, lbl2 = 20,  f"BBUW {daily_bbuw:.0f}"

    # ── 3. ATR contraction ────────────────────────────────────────────────
    atr_series = calc_atr(df, 14)
    atr_5d  = atr_series.iloc[-5:].mean()
    atr_20d = atr_series.iloc[-20:].mean()
    atr_ratio = atr_5d / atr_20d if atr_20d > 0 else 1.0
    if   atr_ratio < 0.50: c3, lbl3 = 100, f"ATR squeezed to {atr_ratio:.2f} of baseline"
    elif atr_ratio < 0.65: c3, lbl3 = 80,  f"ATR contracting ({atr_ratio:.2f})"
    elif atr_ratio < 0.80: c3, lbl3 = 50,  f"Mild ATR contraction"
    else:                  c3, lbl3 = 10,  ""

    # ── 4. Volume dry-up ───────────────────────────────────────────────────
    vol_5d  = vol.iloc[-5:].mean()
    vol_20d = vol.iloc[-20:].mean()
    vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0
    if   vol_ratio < 0.55: c4, lbl4 = 100, f"Volume dry-up ({vol_ratio:.2f}×)"
    elif vol_ratio < 0.70: c4, lbl4 = 80,  f"Volume contracting ({vol_ratio:.2f}×)"
    elif vol_ratio < 0.90: c4, lbl4 = 40,  f"Mildly quiet volume"
    else:                  c4, lbl4 = 10,  ""

    # ── 5. On 21 EMA ───────────────────────────────────────────────────────
    ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    ema_dist_pct = abs((today_close - ema_21) / ema_21) * 100
    if   ema_dist_pct < 1.0: c5, lbl5 = 100, f"Sitting on 21 EMA ({ema_dist_pct:.1f}%)"
    elif ema_dist_pct < 2.0: c5, lbl5 = 80,  f"Within 2% of 21 EMA"
    elif ema_dist_pct < 4.0: c5, lbl5 = 40,  f"Within 4% of 21 EMA"
    else:                    c5, lbl5 = 10,  ""

    # ── 6. Distance from 5-day high ────────────────────────────────────────
    high_5d = high.iloc[-5:].max()
    dist_high_pct = ((today_close - high_5d) / high_5d) * 100
    if   -1 <= dist_high_pct <= 0:  c6, lbl6 = 100, "At 5-day high"
    elif -3 <= dist_high_pct <  -1: c6, lbl6 = 80,  f"Within 3% of 5-day high"
    elif -5 <= dist_high_pct <  -3: c6, lbl6 = 40,  f"5-7% pullback from 5-day high"
    else:                           c6, lbl6 = 10,  ""

    # ── 7. Higher lows trend ───────────────────────────────────────────────
    recent_lows = low.iloc[-7:].values
    hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] >= recent_lows[i-1])
    hl_pct = hl_count / (len(recent_lows) - 1) if len(recent_lows) > 1 else 0
    if   hl_pct >= 0.85: c7, lbl7 = 100, f"Higher lows {hl_count}/{len(recent_lows)-1}"
    elif hl_pct >= 0.65: c7, lbl7 = 70,  f"Mostly higher lows"
    elif hl_pct >= 0.50: c7, lbl7 = 40,  f"Mixed but constructive"
    else:                c7, lbl7 = 10,  ""

    # ── Weighted composite ─────────────────────────────────────────────────
    weights = {
        "pivot_8w":     0.20,
        "bbuw":         0.15,
        "atr_contract": 0.15,
        "vol_dryup":    0.15,
        "ema_21":       0.15,
        "high_5d":      0.10,
        "higher_lows":  0.10,
    }
    components = {
        "pivot_8w":     round(c1, 1),
        "bbuw":         round(c2, 1),
        "atr_contract": round(c3, 1),
        "vol_dryup":    round(c4, 1),
        "ema_21":       round(c5, 1),
        "high_5d":      round(c6, 1),
        "higher_lows":  round(c7, 1),
    }
    score = sum(components[k] * weights[k] for k in weights)

    reasons = [r for r in [lbl1, lbl2, lbl3, lbl4, lbl5, lbl6, lbl7] if r]
    return {
        "score":      round(score, 1),
        "components": components,
        "reasons":    reasons,
        "atr_ratio":  round(atr_ratio, 2),
        "vol_ratio":  round(vol_ratio, 2),
        "ema21_dist_pct": round(ema_dist_pct, 2),
        "dist_high_5d_pct": round(dist_high_pct, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_state(in_motion: float, loading: float) -> str:
    """
    Determine the dominant state for a name.
    A name can score high on both — those are the gold (already moving + still loaded).
    """
    if in_motion >= 70 and loading >= 60: return "MOMENTUM_PRIMED"   # ideal — both
    if in_motion >= 70:                   return "IN_MOTION"
    if loading >= 70:                     return "LOADED"
    if loading >= 50:                     return "DEVELOPING"
    if in_motion >= 50:                   return "RUNNING"
    return "QUIET"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting RADAR prep...")

    daily_data    = get_latest_daily_watchlist()
    daily_entries = daily_data.get("entries", [])
    if not daily_entries:
        log.error("No daily watchlist found. Run daily_screener.py first.")
        return

    # Get today's triggers — used for IN MOTION score
    today_triggers = get_today_triggers()
    triggered_today = {t.get("ticker") for t in today_triggers}

    log.info(f"Scoring {len(daily_entries)} tickers across radar dimensions...")

    radar_records = []
    for i, entry in enumerate(daily_entries, 1):
        ticker = entry["ticker"]
        log.info(f"[{i}/{len(daily_entries)}] {ticker}")

        df = fetch_daily(ticker)
        if df is None:
            continue

        recent_trigger = ticker in triggered_today

        in_motion = calc_in_motion_score(df, recent_trigger=recent_trigger)
        loading   = calc_loading_score(
            df,
            pivot_8w_tier=entry.get("pivot_8w_tier", "NONE"),
            daily_bbuw=entry.get("daily_bbuw", 0),
        )

        state = classify_state(in_motion["score"], loading["score"])

        radar_records.append({
            "ticker":         ticker,
            "conviction":     entry.get("conviction", "LOW"),
            "theme":          entry.get("theme", "Unclassified"),
            "theme_rank":     entry.get("theme_rank", 99),
            "pivot_8w_tier":  entry.get("pivot_8w_tier", "NONE"),
            "weekly_stage":   entry.get("weekly_stage"),
            "daily_stage":    entry.get("daily_stage"),
            "trend_template": entry.get("trend_template"),
            "weekly_bbuw":    entry.get("weekly_bbuw"),
            "daily_bbuw":     entry.get("daily_bbuw"),
            "ema8":           entry.get("ema8"),
            "pct_from_ema8":  entry.get("pct_from_ema8"),

            # Radar-specific
            "state":              state,
            "in_motion_score":    in_motion["score"],
            "in_motion_components": in_motion["components"],
            "in_motion_reasons":  in_motion["reasons"],
            "loading_score":      loading["score"],
            "loading_components": loading["components"],
            "loading_reasons":    loading["reasons"],

            # Raw values for display
            "today_close":        in_motion.get("today_close"),
            "today_pct":          in_motion.get("today_pct"),
            "rvol":               in_motion.get("rvol"),
            "atr_ratio":          loading.get("atr_ratio"),
            "vol_ratio":          loading.get("vol_ratio"),
            "ema21_dist_pct":     loading.get("ema21_dist_pct"),
            "dist_high_5d_pct":   loading.get("dist_high_5d_pct"),
            "had_trigger_today":  recent_trigger,
        })

        time.sleep(RATE_LIMIT_PAUSE)

    # ── Write output ─────────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now().isoformat(),
        "count":        len(radar_records),
        "entries":      radar_records,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Summary log
    n_motion  = sum(1 for r in radar_records if r["state"] in ("IN_MOTION","MOMENTUM_PRIMED","RUNNING"))
    n_loading = sum(1 for r in radar_records if r["state"] in ("LOADED","DEVELOPING","MOMENTUM_PRIMED"))
    n_primed  = sum(1 for r in radar_records if r["state"] == "MOMENTUM_PRIMED")
    log.info(f"\n  RADAR COMPLETE — {len(radar_records)} scored")
    log.info(f"    🔥 In motion : {n_motion}")
    log.info(f"    👀 Loaded    : {n_loading}")
    log.info(f"    ⭐ Primed    : {n_primed}  (high in both)")


if __name__ == "__main__":
    main()
