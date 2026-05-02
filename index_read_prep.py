"""
index_read_prep.py — Rule-Based Kell-Style Technical Structure Analysis
=========================================================================
Generates a structured "Index Read" for major indices and sector ETFs in
Oliver Kell's analytical framework (without quoting Kell himself).

Detects:
  • Weekly structure: HH/HL/LH/LL, base position, distance from 52w high
  • Daily structure: 10/21 EMA position, crossback events, 5d volume profile
  • Cycle of Price Action signals:
      - Reversal Extension     (extended move + hesitation candle)
      - EMA Crossback          (re-test of 10/21 EMA after extension)
      - Wedge Break            (converging trendlines resolving)
      - Bear Trap / Bull Trap  (false breakout/down that reverses)
      - Base Breakout          (flat base resolution)
  • Bull case / Bear case scenarios with key levels
  • Overall bias

Output: data/index_read.json
Schedule: runs as part of the macro_theme job (after market close)

Dependencies: yfinance, pandas, numpy
"""

import os
import json
import logging
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_JSON = "data/index_read.json"
os.makedirs("data", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  COVERAGE — indices + sector ETFs
# ═══════════════════════════════════════════════════════════════════════════════

INDICES = [
    {"ticker": "QQQ", "name": "Nasdaq 100",          "category": "index"},
    {"ticker": "SPY", "name": "S&P 500",              "category": "index"},
    {"ticker": "IWM", "name": "Russell 2000",         "category": "index"},
    {"ticker": "DIA", "name": "Dow Jones Industrial", "category": "index"},
    {"ticker": "MDY", "name": "S&P Mid-Cap 400",      "category": "index"},
]

SECTOR_ETFS = [
    {"ticker": "XLK",  "name": "Technology",          "category": "sector"},
    {"ticker": "XLV",  "name": "Health Care",          "category": "sector"},
    {"ticker": "XLF",  "name": "Financials",           "category": "sector"},
    {"ticker": "XLE",  "name": "Energy",               "category": "sector"},
    {"ticker": "XLI",  "name": "Industrials",          "category": "sector"},
    {"ticker": "XLY",  "name": "Consumer Discretionary","category": "sector"},
    {"ticker": "XLP",  "name": "Consumer Staples",     "category": "sector"},
    {"ticker": "XLB",  "name": "Materials",            "category": "sector"},
    {"ticker": "XLU",  "name": "Utilities",            "category": "sector"},
    {"ticker": "XLC",  "name": "Communication Svcs",   "category": "sector"},
    {"ticker": "XLRE", "name": "Real Estate",          "category": "sector"},
]

ALL_TICKERS = INDICES + SECTOR_ETFS


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_daily(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Pull 2 years of daily bars."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        log.warning(f"[{ticker}] daily fetch error: {e}")
        return None


def fetch_weekly(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    """Pull 5 years of weekly bars."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1wk", auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        log.warning(f"[{ticker}] weekly fetch error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  WEEKLY STRUCTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_weekly_structure(dfw: pd.DataFrame) -> dict:
    """Analyze weekly chart structure."""
    if dfw is None or len(dfw) < 30:
        return {}

    close  = dfw["close"]
    high   = dfw["high"]
    low    = dfw["low"]
    volume = dfw["volume"]

    # 20-week MA — Weinstein's reference
    ma_20w = close.rolling(20).mean()
    ma_30w = close.rolling(30).mean()
    ma_now = ma_20w.iloc[-1]
    ma_5w_ago = ma_20w.iloc[-5] if len(ma_20w) >= 5 else ma_20w.iloc[0]
    ma_slope = "rising" if ma_now > ma_5w_ago * 1.005 else "falling" if ma_now < ma_5w_ago * 0.995 else "flat"

    cur_close = close.iloc[-1]
    above_20w = cur_close > ma_now
    pct_from_20w = ((cur_close - ma_now) / ma_now) * 100

    # 52-week high / low context
    high_52w = high.iloc[-52:].max() if len(high) >= 52 else high.max()
    low_52w  = low.iloc[-52:].min()  if len(low) >= 52  else low.min()
    pct_from_52h = ((cur_close - high_52w) / high_52w) * 100
    pct_from_52l = ((cur_close - low_52w) / low_52w) * 100

    # Higher highs / higher lows (last 12 weeks vs prior 12 weeks)
    recent_high = high.iloc[-12:].max()
    prior_high  = high.iloc[-24:-12].max() if len(high) >= 24 else recent_high
    recent_low  = low.iloc[-12:].min()
    prior_low   = low.iloc[-24:-12].min() if len(low) >= 24 else recent_low

    higher_high = recent_high > prior_high
    higher_low  = recent_low  > prior_low
    lower_high  = recent_high < prior_high
    lower_low   = recent_low  < prior_low

    # Trend classification
    if higher_high and higher_low:
        trend = "Uptrend (HH/HL confirmed)"
    elif lower_high and lower_low:
        trend = "Downtrend (LH/LL confirmed)"
    elif higher_high and lower_low:
        trend = "Expanding range (volatile)"
    elif lower_high and higher_low:
        trend = "Contracting range (coiling)"
    else:
        trend = "Choppy / sideways"

    # Recent base detection — last 8 weeks
    last_8 = dfw.tail(8)
    base_high = last_8["high"].max()
    base_low  = last_8["low"].min()
    base_depth_pct = ((base_high - base_low) / base_high) * 100
    is_tight_base = base_depth_pct < 10  # Kell typically watches tight 5-10% bases
    base_position = "right side of base" if cur_close > (base_high + base_low) / 2 else "left side of base"

    return {
        "trend":             trend,
        "higher_high":       higher_high,
        "higher_low":        higher_low,
        "lower_high":        lower_high,
        "lower_low":         lower_low,
        "ma_20w":            round(float(ma_now), 2),
        "ma_20w_slope":      ma_slope,
        "above_20w":         bool(above_20w),
        "pct_from_20w":      round(float(pct_from_20w), 2),
        "high_52w":          round(float(high_52w), 2),
        "low_52w":           round(float(low_52w), 2),
        "pct_from_52w_high": round(float(pct_from_52h), 2),
        "pct_from_52w_low":  round(float(pct_from_52l), 2),
        "base_depth_pct":    round(float(base_depth_pct), 2),
        "is_tight_base":     bool(is_tight_base),
        "base_position":     base_position,
        "base_high":         round(float(base_high), 2),
        "base_low":          round(float(base_low), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DAILY STRUCTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_daily_structure(df: pd.DataFrame) -> dict:
    """Analyze daily chart structure."""
    if df is None or len(df) < 60:
        return {}

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    volume = df["volume"]

    # Working levels — Kell's 10 and 21 EMA
    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    sma_50  = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()

    cur_close = close.iloc[-1]
    cur_low   = low.iloc[-1]
    cur_high  = high.iloc[-1]

    # Position vs working levels
    above_10  = cur_close > ema_10.iloc[-1]
    above_21  = cur_close > ema_21.iloc[-1]
    above_50  = cur_close > sma_50.iloc[-1]
    above_200 = cur_close > sma_200.iloc[-1]

    pct_from_10  = ((cur_close - ema_10.iloc[-1])  / ema_10.iloc[-1])  * 100
    pct_from_21  = ((cur_close - ema_21.iloc[-1])  / ema_21.iloc[-1])  * 100
    pct_from_50  = ((cur_close - sma_50.iloc[-1])  / sma_50.iloc[-1])  * 100
    pct_from_200 = ((cur_close - sma_200.iloc[-1]) / sma_200.iloc[-1]) * 100

    # Volume profile — last 5 days vs 20-day avg
    avg_vol_20d  = volume.iloc[-20:].mean()
    recent_vol_5 = volume.iloc[-5:].mean()
    vol_ratio    = recent_vol_5 / avg_vol_20d if avg_vol_20d > 0 else 1.0

    # Up-day vs down-day volume balance
    returns_5d = close.pct_change().iloc[-5:]
    up_days   = (returns_5d > 0).sum()
    down_days = (returns_5d < 0).sum()

    # Higher lows count last 5 sessions
    recent_lows = low.iloc[-5:].values
    hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] >= recent_lows[i-1])

    # Today's % change + candle structure
    today_pct = ((cur_close - close.iloc[-2]) / close.iloc[-2]) * 100
    candle_range = cur_high - cur_low
    candle_position = (cur_close - cur_low) / candle_range if candle_range > 0 else 0.5

    return {
        "ema_10":              round(float(ema_10.iloc[-1]), 2),
        "ema_21":              round(float(ema_21.iloc[-1]), 2),
        "sma_50":              round(float(sma_50.iloc[-1]), 2),
        "sma_200":             round(float(sma_200.iloc[-1]), 2),
        "above_10ema":         bool(above_10),
        "above_21ema":         bool(above_21),
        "above_50sma":         bool(above_50),
        "above_200sma":        bool(above_200),
        "pct_from_10ema":      round(float(pct_from_10), 2),
        "pct_from_21ema":      round(float(pct_from_21), 2),
        "pct_from_50sma":      round(float(pct_from_50), 2),
        "pct_from_200sma":     round(float(pct_from_200), 2),
        "vol_ratio_5d":        round(float(vol_ratio), 2),
        "up_days_5":           int(up_days),
        "down_days_5":         int(down_days),
        "higher_lows_5":       int(hl_count),
        "today_pct":           round(float(today_pct), 2),
        "candle_position":     round(float(candle_position), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CYCLE OF PRICE ACTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_cycle_signals(df: pd.DataFrame, dfw: pd.DataFrame) -> dict:
    """
    Detect Kell's Cycle of Price Action signals on the most recent bar.
    Returns dict of signal name → bool/details.
    """
    signals = {}

    if df is None or len(df) < 30:
        return signals

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    volume = df["volume"]

    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()

    cur_close = close.iloc[-1]
    prev_close = close.iloc[-2]

    # ── 1. EMA Crossback (10 EMA) ─────────────────────────────────────────
    # Price re-tests the 10 EMA from above and holds (bullish)
    # Price re-tests the 10 EMA from below and rejects (bearish)
    e10_now = ema_10.iloc[-1]
    today_low = low.iloc[-1]
    today_high = high.iloc[-1]

    # Bullish crossback: low touched/dipped below 10 EMA, close back above
    bull_10_crossback = (today_low <= e10_now * 1.005) and (cur_close > e10_now) and (close.iloc[-2] > ema_10.iloc[-2])
    # Bearish crossback: high touched/poked above 10 EMA, close back below
    bear_10_crossback = (today_high >= e10_now * 0.995) and (cur_close < e10_now) and (close.iloc[-2] < ema_10.iloc[-2])

    signals["bull_10ema_crossback"] = bool(bull_10_crossback)
    signals["bear_10ema_crossback"] = bool(bear_10_crossback)

    # ── 2. EMA Crossback (21 EMA) — deeper pullback ───────────────────────
    e21_now = ema_21.iloc[-1]
    bull_21_crossback = (today_low <= e21_now * 1.01) and (cur_close > e21_now) and (close.iloc[-3:].min() > ema_21.iloc[-3:].min() * 0.99)
    bear_21_crossback = (today_high >= e21_now * 0.99) and (cur_close < e21_now)

    signals["bull_21ema_crossback"] = bool(bull_21_crossback)
    signals["bear_21ema_crossback"] = bool(bear_21_crossback)

    # ── 3. Reversal Extension ──────────────────────────────────────────────
    # Move extended (>5% in 10 days) followed by hesitation candle (small range)
    move_10d = ((close.iloc[-1] / close.iloc[-11]) - 1) * 100 if len(close) >= 11 else 0
    today_range = high.iloc[-1] - low.iloc[-1]
    avg_range_10 = (high.iloc[-11:-1] - low.iloc[-11:-1]).mean()
    is_hesitation = today_range < avg_range_10 * 0.7

    signals["bull_reversal_extension"] = bool(move_10d > 5 and is_hesitation)
    signals["bear_reversal_extension"] = bool(move_10d < -5 and is_hesitation)

    # ── 4. Bear Trap / Bull Trap ──────────────────────────────────────────
    # Bear trap: today's low broke a 10-day low but closed back above it
    low_10d = low.iloc[-11:-1].min()
    bear_trap = (today_low < low_10d * 0.999) and (cur_close > low_10d)
    signals["bear_trap"] = bool(bear_trap)

    # Bull trap: today's high broke 10-day high but closed back below
    high_10d = high.iloc[-11:-1].max()
    bull_trap = (today_high > high_10d * 1.001) and (cur_close < high_10d)
    signals["bull_trap"] = bool(bull_trap)

    # ── 5. Base Breakout / Base Breakdown ─────────────────────────────────
    # Flat base last 4 weeks (~20 trading days), today closes outside
    base_high = high.iloc[-21:-1].max()
    base_low  = low.iloc[-21:-1].min()
    base_depth = ((base_high - base_low) / base_high) * 100

    if base_depth < 12:  # tight base
        signals["base_breakout"] = bool(cur_close > base_high * 1.001)
        signals["base_breakdown"] = bool(cur_close < base_low * 0.999)
    else:
        signals["base_breakout"] = False
        signals["base_breakdown"] = False

    # ── 6. Wedge Detection — converging trendlines ────────────────────────
    # Linear regression on highs and lows over last 20 days
    n = 20
    if len(df) >= n:
        x = np.arange(n)
        high_slope = np.polyfit(x, high.iloc[-n:].values, 1)[0]
        low_slope  = np.polyfit(x, low.iloc[-n:].values, 1)[0]

        # Rising wedge: both slopes positive but highs flattening (low slope > high slope)
        # Falling wedge: both slopes negative, lows flattening (high slope < low slope, both negative)
        if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
            signals["rising_wedge"] = True
        elif high_slope < 0 and low_slope < 0 and high_slope < low_slope:
            signals["falling_wedge"] = True
        else:
            signals["rising_wedge"] = False
            signals["falling_wedge"] = False
    else:
        signals["rising_wedge"] = False
        signals["falling_wedge"] = False

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
#  CYCLE OF PRICE ACTION — PHASE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_cycle_phase(weekly: dict, daily: dict, signals: dict) -> dict:
    """
    Classify the current bar into one of four Cycle of Price Action phases:

      🌱 WEDGE_POP_BASE_BREAK  — Bottoming / reversal forming
                                 (falling wedge, base nearing breakout, bear trap)
      ↗️ EMA_CROSSBACK         — Low-risk entry zone after pullback
                                 (price re-testing 10/20 EMA from above and holding)
      🚀 TREND_CONTINUATION    — Riding the trend
                                 (above 10/20 EMA, working levels intact, HH/HL)
      ⚠️ EXHAUSTION            — End of cycle / distribution risk
                                 (extension, rising wedge, bull trap, lost 10 EMA)

    Returns dict with phase code, label, emoji, action note, and color.
    """
    # ── Score each phase based on signals + structure ─────────────────────
    score_wedge = 0
    score_crossback = 0
    score_continuation = 0
    score_exhaustion = 0

    # WEDGE POP / BASE n' BREAK signals
    if signals.get("falling_wedge"):       score_wedge += 3
    if signals.get("bear_trap"):           score_wedge += 3
    if signals.get("base_breakout"):       score_wedge += 4   # primary signal
    if signals.get("bull_reversal_extension"): score_wedge += 1

    # If still below the 20 SMA, this is a base/reversal context (not continuation)
    if not daily.get("above_50sma") and signals.get("bull_10ema_crossback"):
        score_wedge += 2

    # EMA CROSSBACK signals — primary low-risk entry phase
    if signals.get("bull_10ema_crossback"): score_crossback += 4
    if signals.get("bull_21ema_crossback"): score_crossback += 5  # deeper test, stronger signal
    # Crossback is most meaningful in an established uptrend
    if signals.get("bull_10ema_crossback") and weekly.get("higher_high") and weekly.get("higher_low"):
        score_crossback += 2
    if signals.get("bull_21ema_crossback") and daily.get("above_50sma"):
        score_crossback += 1

    # TREND CONTINUATION — riding the trend
    if daily.get("above_10ema") and daily.get("above_21ema") and daily.get("above_50sma"):
        score_continuation += 3
    if weekly.get("higher_high") and weekly.get("higher_low"):
        score_continuation += 2
    if weekly.get("above_20w") and weekly.get("ma_20w_slope") == "rising":
        score_continuation += 2
    # But only if we're not extended — distance from 10 EMA matters
    if 0 < daily.get("pct_from_10ema", 0) < 4:
        score_continuation += 1
    if daily.get("higher_lows_5", 0) >= 3:
        score_continuation += 1

    # EXHAUSTION / END OF CYCLE
    if signals.get("rising_wedge"):              score_exhaustion += 4
    if signals.get("bull_trap"):                 score_exhaustion += 4
    if signals.get("bear_reversal_extension"):   score_exhaustion += 2
    if signals.get("base_breakdown"):            score_exhaustion += 3
    if signals.get("bear_10ema_crossback"):      score_exhaustion += 3
    if signals.get("bear_21ema_crossback"):      score_exhaustion += 2
    # Lost a key working level after being above
    if not daily.get("above_10ema") and weekly.get("higher_high"):
        score_exhaustion += 2
    # Far extended above 10 EMA — overextension
    if daily.get("pct_from_10ema", 0) > 8:
        score_exhaustion += 2
    # Lower highs/lows on weekly = clear distribution
    if weekly.get("lower_high") and weekly.get("lower_low"):
        score_exhaustion += 3

    # ── Pick the highest-scoring phase ────────────────────────────────────
    scores = {
        "WEDGE_POP_BASE_BREAK": score_wedge,
        "EMA_CROSSBACK":        score_crossback,
        "TREND_CONTINUATION":   score_continuation,
        "EXHAUSTION":           score_exhaustion,
    }
    best_phase = max(scores, key=scores.get)
    best_score = scores[best_phase]

    # If everything is low (no clear signals), default to NEUTRAL
    if best_score < 2:
        best_phase = "NEUTRAL"

    # ── Phase metadata ────────────────────────────────────────────────────
    phase_meta = {
        "WEDGE_POP_BASE_BREAK": {
            "emoji":  "🌱",
            "label":  "Wedge Pop / Base n' Break",
            "color":  "blue",
            "action": "Bottoming / reversal forming. Watch for confirmation; prepare entries near breakout level.",
        },
        "EMA_CROSSBACK": {
            "emoji":  "↗️",
            "label":  "EMA Crossback",
            "color":  "green",
            "action": "Low-risk entry zone. Price tested 10/20 EMA and held — preferred Kell entry setup.",
        },
        "TREND_CONTINUATION": {
            "emoji":  "🚀",
            "label":  "Trend Continuation",
            "color":  "green",
            "action": "Riding the trend. Hold above 10/20 EMA; let it run. Avoid re-entering extended.",
        },
        "EXHAUSTION": {
            "emoji":  "⚠️",
            "label":  "Exhaustion / End of Cycle",
            "color":  "red",
            "action": "Distribution risk. Tighten stops; loss of 10 EMA = scale out. Avoid new longs.",
        },
        "NEUTRAL": {
            "emoji":  "·",
            "label":  "Neutral / Transition",
            "color":  "amber",
            "action": "No clear phase signal. Wait for structure to develop.",
        },
    }

    meta = phase_meta[best_phase]
    return {
        "phase":         best_phase,
        "phase_label":   meta["label"],
        "phase_emoji":   meta["emoji"],
        "phase_color":   meta["color"],
        "phase_action":  meta["action"],
        "phase_scores":  scores,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NARRATIVE GENERATION (Kell-style vocabulary, no attribution)
# ═══════════════════════════════════════════════════════════════════════════════

def build_narrative(weekly: dict, daily: dict, signals: dict, ticker: str, name: str) -> dict:
    """
    Generate a structured technical narrative using Kell's vocabulary
    (working levels, EMA Crossback, weekly structure, etc.) WITHOUT
    attributing anything to Kell himself.
    """

    # ── Daily structure read ──────────────────────────────────────────────
    daily_lines = []
    if daily.get("above_10ema"):
        daily_lines.append(f"Holding above the 10 EMA (working level intact, {daily['pct_from_10ema']:+.1f}% above).")
    else:
        daily_lines.append(f"Below the 10 EMA ({daily['pct_from_10ema']:+.1f}% — working level lost).")

    if daily.get("above_21ema"):
        daily_lines.append(f"Above 21 EMA ({daily['pct_from_21ema']:+.1f}%).")
    else:
        daily_lines.append(f"Below 21 EMA ({daily['pct_from_21ema']:+.1f}%) — deeper structural break.")

    if daily.get("above_200sma"):
        daily_lines.append(f"Above the 200 SMA ({daily['pct_from_200sma']:+.1f}%) — long-term trend intact.")
    else:
        daily_lines.append(f"Below the 200 SMA ({daily['pct_from_200sma']:+.1f}%) — long-term trend compromised.")

    if daily.get("vol_ratio_5d", 1.0) > 1.3:
        daily_lines.append(f"Volume elevated ({daily['vol_ratio_5d']:.2f}× 20d avg).")
    elif daily.get("vol_ratio_5d", 1.0) < 0.75:
        daily_lines.append(f"Volume contracting ({daily['vol_ratio_5d']:.2f}× — quiet pullback or accumulation).")

    if daily.get("higher_lows_5", 0) >= 3:
        daily_lines.append(f"Higher lows {daily['higher_lows_5']}/4 last 5 sessions — buyers stepping up.")

    # ── Weekly structure read ─────────────────────────────────────────────
    weekly_lines = []
    if weekly.get("above_20w"):
        weekly_lines.append(f"Above 20-week MA ({weekly['pct_from_20w']:+.1f}%, slope {weekly.get('ma_20w_slope','flat')}).")
    else:
        weekly_lines.append(f"Below 20-week MA ({weekly['pct_from_20w']:+.1f}%, slope {weekly.get('ma_20w_slope','flat')}).")

    weekly_lines.append(f"Trend read: {weekly.get('trend','—')}.")

    if weekly.get("is_tight_base"):
        weekly_lines.append(f"Tight base detected — last 8 weeks contained in a {weekly['base_depth_pct']:.1f}% range "
                            f"(${weekly['base_low']:.2f} to ${weekly['base_high']:.2f}). On the {weekly['base_position']}.")
    else:
        weekly_lines.append(f"8-week range: {weekly['base_depth_pct']:.1f}% deep — not yet a tight base.")

    if abs(weekly.get("pct_from_52w_high", -99)) < 5:
        weekly_lines.append(f"Within 5% of 52-week high — extended structurally.")
    elif weekly.get("pct_from_52w_high", 0) > -10:
        weekly_lines.append(f"Within 10% of 52-week high — constructive structural position.")

    # ── Cycle of Price Action read ─────────────────────────────────────────
    cycle_signals_active = []
    if signals.get("bull_10ema_crossback"):  cycle_signals_active.append("✅ Bullish 10 EMA Crossback")
    if signals.get("bear_10ema_crossback"):  cycle_signals_active.append("⚠️ Bearish 10 EMA Crossback")
    if signals.get("bull_21ema_crossback"):  cycle_signals_active.append("✅ Bullish 21 EMA Crossback (deeper test)")
    if signals.get("bear_21ema_crossback"):  cycle_signals_active.append("⚠️ Bearish 21 EMA Crossback")
    if signals.get("bull_reversal_extension"): cycle_signals_active.append("✅ Bullish Reversal Extension complete")
    if signals.get("bear_reversal_extension"): cycle_signals_active.append("⚠️ Bearish Reversal Extension complete")
    if signals.get("bear_trap"):  cycle_signals_active.append("✅ Bear Trap identified — false breakdown reversed")
    if signals.get("bull_trap"):  cycle_signals_active.append("⚠️ Bull Trap identified — failed breakout")
    if signals.get("base_breakout"):  cycle_signals_active.append("✅ Base Breakout fired")
    if signals.get("base_breakdown"): cycle_signals_active.append("⚠️ Base Breakdown fired")
    if signals.get("rising_wedge"):   cycle_signals_active.append("⚠️ Rising Wedge developing — bearish exhaustion pattern")
    if signals.get("falling_wedge"):  cycle_signals_active.append("✅ Falling Wedge developing — bullish reversal pattern")

    # ── Bull case / Bear case ──────────────────────────────────────────────
    bull_levels = []
    bear_levels = []

    if daily.get("ema_10"):
        bull_levels.append(f"Hold above 10 EMA (${daily['ema_10']:.2f})")
    if weekly.get("base_high"):
        bull_levels.append(f"Break and hold above ${weekly['base_high']:.2f} (8-week base high)")
    if weekly.get("high_52w"):
        bull_levels.append(f"Target ${weekly['high_52w']:.2f} (52-week high)")

    if daily.get("ema_21"):
        bear_levels.append(f"Loss of 21 EMA (${daily['ema_21']:.2f})")
    if weekly.get("base_low"):
        bear_levels.append(f"Loss of ${weekly['base_low']:.2f} (8-week base low) flips structure")
    if daily.get("sma_200"):
        bear_levels.append(f"Loss of 200 SMA (${daily['sma_200']:.2f}) compromises long-term trend")

    # ── Overall bias scoring ──────────────────────────────────────────────
    bull_score = 0
    bear_score = 0

    if daily.get("above_10ema"):  bull_score += 2
    else:                         bear_score += 1
    if daily.get("above_21ema"):  bull_score += 2
    else:                         bear_score += 2
    if daily.get("above_50sma"):  bull_score += 1
    else:                         bear_score += 1
    if daily.get("above_200sma"): bull_score += 2
    else:                         bear_score += 3
    if weekly.get("above_20w"):   bull_score += 2
    else:                         bear_score += 2
    if weekly.get("higher_high") and weekly.get("higher_low"): bull_score += 3
    if weekly.get("lower_high")  and weekly.get("lower_low"):  bear_score += 3
    if weekly.get("ma_20w_slope") == "rising": bull_score += 1
    if weekly.get("ma_20w_slope") == "falling": bear_score += 1

    # Cycle signal contributions
    bull_signals_count = sum(1 for k, v in signals.items() if v and k.startswith("bull")) + \
                         (1 if signals.get("bear_trap") else 0) + \
                         (1 if signals.get("base_breakout") else 0) + \
                         (1 if signals.get("falling_wedge") else 0)
    bear_signals_count = sum(1 for k, v in signals.items() if v and k.startswith("bear") and k != "bear_trap") + \
                         (1 if signals.get("bull_trap") else 0) + \
                         (1 if signals.get("base_breakdown") else 0) + \
                         (1 if signals.get("rising_wedge") else 0)

    bull_score += bull_signals_count
    bear_score += bear_signals_count

    if bull_score >= bear_score + 4:
        bias = "STRONGLY CONSTRUCTIVE"
        bias_color = "green"
    elif bull_score >= bear_score + 1:
        bias = "CONSTRUCTIVE"
        bias_color = "green"
    elif bear_score >= bull_score + 4:
        bias = "DISTRIBUTION RISK"
        bias_color = "red"
    elif bear_score >= bull_score + 1:
        bias = "CAUTIOUS"
        bias_color = "amber"
    else:
        bias = "NEUTRAL / TRANSITION"
        bias_color = "amber"

    return {
        "daily_read":     daily_lines,
        "weekly_read":    weekly_lines,
        "cycle_signals":  cycle_signals_active,
        "bull_levels":    bull_levels,
        "bear_levels":    bear_levels,
        "bias":           bias,
        "bias_color":     bias_color,
        "bull_score":     bull_score,
        "bear_score":     bear_score,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-TICKER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_ticker(meta: dict) -> Optional[dict]:
    """Run full Index Read for one ticker."""
    ticker = meta["ticker"]
    log.info(f"Analyzing {ticker} ({meta['name']})...")

    df  = fetch_daily(ticker)
    dfw = fetch_weekly(ticker)
    if df is None or dfw is None:
        log.warning(f"[{ticker}] insufficient data — skipping")
        return None

    weekly  = analyze_weekly_structure(dfw)
    daily   = analyze_daily_structure(df)
    signals = detect_cycle_signals(df, dfw)
    phase   = classify_cycle_phase(weekly, daily, signals)
    narr    = build_narrative(weekly, daily, signals, ticker, meta["name"])
    # Merge phase data into narrative for clean dashboard rendering
    narr.update(phase)

    return {
        "ticker":           ticker,
        "name":             meta["name"],
        "category":         meta["category"],
        "current_price":    round(float(df["close"].iloc[-1]), 2),
        "today_pct":        daily.get("today_pct", 0),
        "weekly":           weekly,
        "daily":            daily,
        "cycle_signals":    signals,
        "phase":            phase,
        "narrative":        narr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting INDEX READ analysis (Kell-style structure)...")

    results = []
    for meta in ALL_TICKERS:
        r = analyze_ticker(meta)
        if r:
            results.append(r)

    # Sort: indices first, then sectors, both by ticker
    results.sort(key=lambda r: (r["category"] != "index", r["ticker"]))

    output = {
        "generated_at": datetime.now().isoformat(),
        "count":        len(results),
        "indices":      [r for r in results if r["category"] == "index"],
        "sectors":      [r for r in results if r["category"] == "sector"],
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Summary log
    bull = sum(1 for r in results if r["narrative"]["bias_color"] == "green")
    bear = sum(1 for r in results if r["narrative"]["bias_color"] == "red")
    neut = sum(1 for r in results if r["narrative"]["bias_color"] == "amber")

    # Phase distribution
    from collections import Counter
    phase_counts = Counter(r["phase"]["phase"] for r in results)

    log.info(f"\n  INDEX READ COMPLETE — {len(results)} tickers analyzed")
    log.info(f"    🟢 Constructive : {bull}")
    log.info(f"    🟡 Cautious     : {neut}")
    log.info(f"    🔴 Distribution : {bear}")
    log.info(f"    Cycle Phases:")
    log.info(f"      🌱 Wedge Pop / Base n' Break : {phase_counts.get('WEDGE_POP_BASE_BREAK', 0)}")
    log.info(f"      ↗️  EMA Crossback             : {phase_counts.get('EMA_CROSSBACK', 0)}")
    log.info(f"      🚀 Trend Continuation        : {phase_counts.get('TREND_CONTINUATION', 0)}")
    log.info(f"      ⚠️  Exhaustion                : {phase_counts.get('EXHAUSTION', 0)}")
    log.info(f"      ·  Neutral                   : {phase_counts.get('NEUTRAL', 0)}")


if __name__ == "__main__":
    main()
