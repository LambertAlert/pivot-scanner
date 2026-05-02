"""
WEEKLY SCREENER (v2) — writes to data_layer (JSON + SQLite) instead of txt files.
"""

import os
import csv
import time
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf

from data_layer import save_weekly_watchlist, save_industry_ranks


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_CSV         = "watchlist_input.csv"
THEMES_CSV        = "sector_themes.csv"   # needed for industry rank computation
RATE_LIMIT_PAUSE  = 1.0
LOG_LEVEL         = logging.INFO
MIN_BBUW_SCORE    = 40
QUALIFYING_STAGES = [1, 2]

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_weekly_bars(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1wk", auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        log.error(f"[{ticker}] Weekly fetch error: {e}")
        return None


def fetch_daily_bars_for_template(ticker: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 200:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        log.error(f"[{ticker}] Daily fetch error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_weinstein_stage(df: pd.DataFrame) -> int:
    close = df["close"]
    ma_30w = close.rolling(window=30).mean()
    if len(df) < 30 or pd.isna(ma_30w.iloc[-1]):
        return 0
    ma_now = ma_30w.iloc[-1]
    ma_10w_ago = ma_30w.iloc[-10] if len(ma_30w) >= 10 else ma_30w.iloc[0]
    ma_slope_pct = ((ma_now - ma_10w_ago) / ma_10w_ago) * 100
    price = close.iloc[-1]
    above_ma = price > ma_now
    pct_from_ma = ((price - ma_now) / ma_now) * 100
    if above_ma and ma_slope_pct > 1.0:
        return 2
    if not above_ma and ma_slope_pct < -1.0:
        return 4
    if abs(pct_from_ma) < 5 and abs(ma_slope_pct) < 1.0:
        return 1
    if above_ma and ma_slope_pct < 1.0:
        return 3
    return 1


def minervini_trend_template(df_daily: pd.DataFrame) -> dict:
    if len(df_daily) < 200:
        return {"score": 0, "criteria": {}}
    close = df_daily["close"]
    sma_50 = close.rolling(50).mean()
    sma_150 = close.rolling(150).mean()
    sma_200 = close.rolling(200).mean()
    price = close.iloc[-1]
    high_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()
    low_52w = close.iloc[-252:].min() if len(close) >= 252 else close.min()
    sma_200_trending_up = sma_200.iloc[-1] > sma_200.iloc[-21] if len(sma_200) >= 21 else False
    criteria = {
        "1_above_150sma": bool(price > sma_150.iloc[-1]),
        "2_above_200sma": bool(price > sma_200.iloc[-1]),
        "3_150_above_200": bool(sma_150.iloc[-1] > sma_200.iloc[-1]),
        "4_200_trending_up": bool(sma_200_trending_up),
        "5_50_above_150_and_200": bool(sma_50.iloc[-1] > sma_150.iloc[-1] and sma_50.iloc[-1] > sma_200.iloc[-1]),
        "6_above_50sma": bool(price > sma_50.iloc[-1]),
        "7_above_low_30pct": bool(price >= low_52w * 1.30),
        "8_within_25pct_high": bool(price >= high_52w * 0.75),
    }
    return {"score": sum(criteria.values()), "criteria": criteria}


# ═══════════════════════════════════════════════════════════════════════════════
#  BBUW (WEEKLY)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_bbuw_weekly(df: pd.DataFrame, df_spy: pd.DataFrame = None) -> dict:
    if len(df) < 30:
        return {"score": 0, "components": {}}
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    weekly_range = high - low
    recent_range = weekly_range.iloc[-4:].mean()
    baseline_range = weekly_range.iloc[-16:-4].mean()
    range_ratio = recent_range / baseline_range if baseline_range > 0 else 1.0
    range_score = max(0, min(100, (1.0 - range_ratio) * 200))
    recent_vol = vol.iloc[-4:].mean()
    baseline_vol = vol.iloc[-16:-4].mean()
    vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1.0
    vol_score = max(0, min(100, (1.0 - vol_ratio) * 200))
    recent_lows = low.iloc[-6:].values
    hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] >= recent_lows[i-1])
    hl_score = (hl_count / 5) * 100
    avg_range_20 = weekly_range.iloc[-20:].mean()
    coil_weeks = sum(1 for r in weekly_range.iloc[-8:] if r < avg_range_20 * 0.6)
    coil_score = min(100, coil_weeks * 15)
    rs_score = 50
    if df_spy is not None and len(df_spy) >= 26 and len(df) >= 26:
        try:
            stock_perf = (close.iloc[-1] / close.iloc[-26]) - 1
            spy_perf = (df_spy["close"].iloc[-1] / df_spy["close"].iloc[-26]) - 1
            rs_score = max(0, min(100, 50 + (stock_perf - spy_perf) * 200))
        except Exception:
            pass
    high_20w = high.iloc[-20:].max()
    pullback_pct = ((close.iloc[-1] / high_20w) - 1) * 100
    if -15 <= pullback_pct <= -5:
        pb_score = 100
    elif -25 <= pullback_pct < -15:
        pb_score = 70
    elif -5 < pullback_pct <= 0:
        pb_score = 60
    else:
        pb_score = 30
    weights = {"range_contraction": 0.25, "volume_contraction": 0.15, "higher_lows": 0.15,
               "coil_duration": 0.10, "rs_vs_spy": 0.20, "pullback_depth": 0.15}
    components = {
        "range_contraction": round(range_score, 1),
        "volume_contraction": round(vol_score, 1),
        "higher_lows": round(hl_score, 1),
        "coil_duration": round(coil_score, 1),
        "rs_vs_spy": round(rs_score, 1),
        "pullback_depth": round(pb_score, 1),
    }
    composite = sum(components[k] * weights[k] for k in weights)
    return {"score": round(composite, 1), "components": components}


# ═══════════════════════════════════════════════════════════════════════════════
#  8-WEEK PIVOT DETECTION  (@1ChartMaster / Elite Swing Traders)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Concept (from @1ChartMaster):
#   Buy momentum stocks pulling back to the 8-week EMA — a faster, stricter
#   dynamic support level than the 50-day SMA (≈10W). The pivot fires when
#   price has dipped to the EMA and the weekly bar closes back above it,
#   showing the bounce/reversal happening NOW.
#
# Detection logic (current week's pivot):
#   1. Weekly Low  ≤ 8-week EMA(close)         → price touched/dipped to support
#   2. Weekly Close > 8-week EMA(close)        → bullish pivot, closed above
#   3. (context) Prior week Close > prior 8EMA → in uptrend, this is a pullback
#                                                 not a fresh breakdown
#
# Strength tiers:
#   STRONG  = pivot fires + EMA itself rising + bull volume > 1.5× 20W avg
#   STANDARD = pivot fires + EMA rising
#   PROXIMITY = within 3% of 8EMA from above (pullback in progress, not yet pivoted)
#   NONE    = no setup
# ═══════════════════════════════════════════════════════════════════════════════

def detect_8week_pivot(df: pd.DataFrame) -> dict:
    """
    Detect 8-week pivot setup on the current weekly bar.
    Returns dict with:
      pivot_fired       : bool — true pivot triggered THIS WEEK
      pivot_tier        : "STRONG" | "STANDARD" | "PROXIMITY" | "NONE"
      ema8              : current 8-week EMA value
      pct_from_ema8     : % distance from EMA (negative = below, positive = above)
      ema_rising        : EMA slope is up (more EMA(8)[-1] > EMA(8)[-3])
      prior_above_ema   : prior week was above EMA (uptrend context)
      bull_volume_spike : current week volume > 1.5× 20W avg AND green candle
      week_low          : low of current week
      week_close        : close of current week
    """
    if len(df) < 12:
        return {"pivot_fired": False, "pivot_tier": "NONE"}

    close = df["close"]
    low   = df["low"]
    vol   = df["volume"]

    # 8-week EMA on close
    ema8_series = close.ewm(span=8, adjust=False).mean()

    ema_now    = ema8_series.iloc[-1]
    ema_prev_3 = ema8_series.iloc[-3]
    week_close = close.iloc[-1]
    week_low   = low.iloc[-1]
    prev_close = close.iloc[-2]
    prev_ema   = ema8_series.iloc[-2]

    pct_from_ema = ((week_close - ema_now) / ema_now) * 100

    # Core pivot conditions
    cond_low_touched_ema  = week_low <= ema_now              # dipped to EMA
    cond_close_above_ema  = week_close > ema_now             # closed back above
    cond_prior_above_ema  = prev_close > prev_ema            # uptrend context
    ema_rising            = ema_now > ema_prev_3             # EMA itself trending up

    pivot_fired = bool(
        cond_low_touched_ema
        and cond_close_above_ema
        and cond_prior_above_ema
    )

    # Volume confirmation — current week volume vs 20W average, on green close
    avg_vol_20w = vol.iloc[-20:].mean() if len(vol) >= 20 else vol.mean()
    is_green_week = week_close > df["open"].iloc[-1]
    bull_volume_spike = bool(is_green_week and avg_vol_20w > 0
                              and vol.iloc[-1] > 1.5 * avg_vol_20w)

    # Tier classification
    if pivot_fired and ema_rising and bull_volume_spike:
        tier = "STRONG"
    elif pivot_fired and ema_rising:
        tier = "STANDARD"
    elif pivot_fired:
        tier = "WEAK"            # pivot but EMA flat/falling — caution
    elif (
        not pivot_fired
        and 0 < pct_from_ema < 3      # within 3% above EMA
        and cond_prior_above_ema
        and ema_rising
    ):
        tier = "PROXIMITY"            # pullback approaching, not yet pivoted
    else:
        tier = "NONE"

    return {
        "pivot_fired":       pivot_fired,
        "pivot_tier":        tier,
        "ema8":              round(float(ema_now), 4),
        "pct_from_ema8":     round(float(pct_from_ema), 2),
        "ema_rising":        bool(ema_rising),
        "prior_above_ema":   bool(cond_prior_above_ema),
        "bull_volume_spike": bull_volume_spike,
        "week_low":          round(float(week_low), 4),
        "week_close":        round(float(week_close), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  INDUSTRY RANKING  — computed from the full ticker universe
# ═══════════════════════════════════════════════════════════════════════════════

def compute_industry_ranks(all_results: list) -> list:
    """
    Rank every industry in the watchlist by how strongly its members
    are performing right now.  Uses ALL scanned tickers (qualified + unqualified)
    so every industry has enough representation to score fairly.

    Composite score (0-100) per industry — weighted components:
      • avg BBUW score             (30%) — compression / setup quality
      • avg RS vs SPY component    (25%) — relative strength signal
      • % in Stage 1 or 2         (20%) — structural trend health
      • % trend template ≥ 5/8    (15%) — Minervini criteria
      • % with active 8W pivot     (10%) — STRONG / STANDARD / PROXIMITY

    Returns list of dicts sorted rank 1..N.
    """
    import csv as csv_mod
    from collections import defaultdict

    # Load industry mapping from sector_themes.csv
    ticker_to_industry = {}
    if os.path.exists(THEMES_CSV):
        with open(THEMES_CSV, "r", newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                sym = (row.get("Symbol") or "").strip().upper()
                ind = (row.get("Industry") or "").strip()
                if sym and ind:
                    ticker_to_industry[sym] = ind

    # Group scanned results by industry
    industry_groups = defaultdict(list)
    for r in all_results:
        industry = ticker_to_industry.get(r["ticker"], "Unclassified")
        if industry and industry != "Unclassified":
            industry_groups[industry].append(r)

    # Score each industry
    scored = []
    for industry, members in industry_groups.items():
        n = len(members)
        if n < 2:
            continue   # skip single-ticker industries

        # 1. Avg BBUW
        avg_bbuw = sum(m.get("bbuw_score", 0) or 0 for m in members) / n

        # 2. Avg RS vs SPY — from the rs_vs_spy component inside bbuw_components
        rs_vals = [
            m.get("bbuw_components", {}).get("rs_vs_spy", 50) or 50
            for m in members
        ]
        avg_rs_component = sum(rs_vals) / len(rs_vals)   # 0-100 scale, 50 = neutral

        # 3. % in Stage 1 or 2
        pct_stage12 = sum(1 for m in members if m.get("stage") in [1, 2]) / n * 100

        # 4. % trend template score ≥ 5
        pct_trend = sum(
            1 for m in members
            if (m.get("trend_template_score") or 0) >= 5
        ) / n * 100

        # 5. % with active 8W pivot tier
        pct_8w = sum(
            1 for m in members
            if m.get("pivot_8w_tier") in ("STRONG", "STANDARD", "PROXIMITY")
        ) / n * 100

        # Weighted composite — all inputs already on 0-100 scale
        composite = (
            avg_bbuw           * 0.30 +
            avg_rs_component   * 0.25 +
            pct_stage12        * 0.20 +
            pct_trend          * 0.15 +
            pct_8w             * 0.10
        )

        scored.append({
            "industry":           industry,
            "ticker_count":       n,
            "avg_bbuw":           round(avg_bbuw, 1),
            "avg_rs_component":   round(avg_rs_component, 1),
            "pct_stage12":        round(pct_stage12, 1),
            "pct_trend_template": round(pct_trend, 1),
            "pct_8w_active":      round(pct_8w, 1),
            "composite_score":    round(composite, 1),
            "rank":               0,   # set below
        })

    # Sort descending by composite and assign rank
    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, row in enumerate(scored, 1):
        row["rank"] = i

    log.info(f"\n  INDUSTRY RANKS — {len(scored)} industries scored")
    for r in scored[:5]:
        log.info(f"    #{r['rank']:>2}  {r['industry']:<42}  "
                 f"score={r['composite_score']:.1f}  n={r['ticker_count']}")

    return scored


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def load_input_tickers(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        log.error(f"Input CSV not found: {csv_path}")
        return []
    tickers = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row.get("Symbol", "").strip().upper()
            if sym:
                tickers.append(sym)
    return tickers


def main():
    log.info("Starting WEEKLY screener (v2)...")
    tickers = load_input_tickers(INPUT_CSV)
    if not tickers:
        log.error("No tickers loaded. Exiting.")
        return

    log.info(f"Loaded {len(tickers)} tickers. Fetching SPY...")
    df_spy = fetch_weekly_bars("SPY")

    qualified  = []
    full_results = []   # ALL scanned tickers — used for industry ranking
    for i, ticker in enumerate(tickers, 1):
        log.info(f"[{i}/{len(tickers)}] {ticker}")
        df_weekly = fetch_weekly_bars(ticker)
        if df_weekly is None:
            continue
        df_daily = fetch_daily_bars_for_template(ticker)
        if df_daily is None:
            continue

        stage = classify_weinstein_stage(df_weekly)
        template = minervini_trend_template(df_daily)
        bbuw = calc_bbuw_weekly(df_weekly, df_spy)
        pivot_8w = detect_8week_pivot(df_weekly)

        # Every scanned ticker goes into full_results for industry ranking
        result = {
            "ticker":                ticker,
            "stage":                 stage,
            "trend_template_score":  template["score"],
            "trend_template_criteria": template["criteria"],
            "bbuw_score":            bbuw["score"],
            "bbuw_components":       bbuw["components"],
            "pivot_8w_fired":        pivot_8w["pivot_fired"],
            "pivot_8w_tier":         pivot_8w["pivot_tier"],
            "ema8":                  pivot_8w.get("ema8"),
            "pct_from_ema8":         pivot_8w.get("pct_from_ema8"),
            "ema8_rising":           pivot_8w.get("ema_rising", False),
            "pivot_8w_volume_spike": pivot_8w.get("bull_volume_spike", False),
        }
        full_results.append(result)

        if stage in QUALIFYING_STAGES and bbuw["score"] >= MIN_BBUW_SCORE:
            qualified.append(result)
            tier_emoji = {
                "STRONG":    "🔥",
                "STANDARD":  "✅",
                "WEAK":      "⚠️",
                "PROXIMITY": "👀",
                "NONE":      "  ",
            }.get(pivot_8w["pivot_tier"], "  ")
            log.info(f"  ✅ {ticker}: Stage {stage}, Trend {template['score']}/8, "
                     f"BBUW {bbuw['score']}, 8W-Pivot {tier_emoji} {pivot_8w['pivot_tier']}")

        time.sleep(RATE_LIMIT_PAUSE)

    # Sort: STRONG/STANDARD 8W pivots first, then by BBUW score
    tier_order = {"STRONG": 0, "STANDARD": 1, "WEAK": 2, "PROXIMITY": 3, "NONE": 4}
    qualified.sort(key=lambda x: (
        tier_order.get(x.get("pivot_8w_tier", "NONE"), 4),
        -x["bbuw_score"]
    ))
    save_weekly_watchlist(qualified)

    # ── Industry ranking — uses ALL scanned results, not just qualified ───────
    # Pass full_results so unqualified tickers still contribute to their industry score
    industry_ranks = compute_industry_ranks(full_results)
    save_industry_ranks(industry_ranks)

    # Tier summary
    tier_counts = {}
    for q in qualified:
        t = q.get("pivot_8w_tier", "NONE")
        tier_counts[t] = tier_counts.get(t, 0) + 1

    log.info(f"\n  WEEKLY SCREEN COMPLETE — {len(qualified)} qualified out of {len(tickers)}")
    log.info(f"    8W Pivots: 🔥 STRONG={tier_counts.get('STRONG', 0)}  "
             f"✅ STANDARD={tier_counts.get('STANDARD', 0)}  "
             f"⚠️ WEAK={tier_counts.get('WEAK', 0)}  "
             f"👀 PROXIMITY={tier_counts.get('PROXIMITY', 0)}")


if __name__ == "__main__":
    main()
