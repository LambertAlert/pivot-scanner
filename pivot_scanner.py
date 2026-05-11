"""
INTRADAY PIVOT SCANNER (v4) — R-ratio, key levels, composite score ranking.

Enhancements over v3:
  • ATR(14) daily → R-ratio = (trigger_close - stop) / ATR
  • Trigger bar RVOL vs 20-period intraday average
  • Distance from 8W EMA (from daily watchlist)
  • Distance from intraday session high (daily high so far)
  • Composite trigger score for ranking in dashboard
  • Deduplication: same ticker on both timeframes → keep 65min
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from data_layer import save_trigger, get_latest_daily_watchlist


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MIN_STREAK       = 3
RATE_LIMIT_PAUSE = 2
SCAN_30MIN       = True
SCAN_65MIN       = True
ALERT_TIERS      = ["HIGH", "MED"]

# R-ratio thresholds for scoring
R_ELITE   = 3.0   # +2 score pts
R_GOOD    = 2.0   # +1 score pt
R_MINIMUM = 1.0   # trigger still valid but no bonus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_intraday_bars(ticker: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period="60d", interval=interval, auto_adjust=True)
        if df.empty:
            return None
        df.index = df.index.tz_convert("UTC") if df.index.tzinfo else df.index.tz_localize("UTC")
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        df = df[df["close"] > 0]
        return df.tail(100)
    except Exception as e:
        log.error(f"[{ticker}] {interval} fetch error: {e}")
        return None


def fetch_daily_bars(ticker: str) -> Optional[pd.DataFrame]:
    """Pull daily bars for ATR and session high calculation."""
    try:
        df = yf.Ticker(ticker).history(period="60d", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 14:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df = df[df["close"] > 0]
        return df
    except Exception as e:
        log.warning(f"[{ticker}] Daily fetch error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  ATR + R-RATIO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df_daily: pd.DataFrame, period: int = 14) -> float:
    """ATR(14) on daily bars — Wilder's smoothing."""
    if df_daily is None or len(df_daily) < period:
        return np.nan
    h = df_daily["high"]
    l = df_daily["low"]
    c = df_daily["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else np.nan


def compute_r_ratio(trigger_close: float, stop_level: float,
                    direction: str, atr: float) -> Optional[float]:
    """
    R-ratio = risk-normalised distance from entry to stop.
    Bullish: (trigger_close - stop_level) / ATR
    Bearish: (stop_level - trigger_close) / ATR
    Positive = risk is within ATR, negative = already beyond 1 ATR of risk.
    """
    if np.isnan(atr) or atr <= 0:
        return None
    if direction == "bullish":
        risk = trigger_close - stop_level
    else:
        risk = stop_level - trigger_close
    return round(risk / atr, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  KEY LEVEL DISTANCES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_session_high_distance(df_daily: pd.DataFrame,
                                   trigger_close: float) -> Optional[float]:
    """
    Distance of trigger_close from today's session high (daily bar high).
    Negative = below session high (most common for longs mid-day).
    Near 0 = breaking out to new day high — strongest signal.
    """
    if df_daily is None or df_daily.empty:
        return None
    today_high = float(df_daily["high"].iloc[-1])
    if today_high <= 0:
        return None
    return round((trigger_close / today_high - 1) * 100, 2)


def compute_ema8w_distance(trigger_close: float,
                            ema8: Optional[float]) -> Optional[float]:
    """Distance from 8-week EMA (from daily watchlist entry)."""
    if ema8 is None or ema8 <= 0:
        return None
    return round((trigger_close / ema8 - 1) * 100, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTRADAY VOLUME RVOL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_intraday_rvol(df_intraday: pd.DataFrame) -> float:
    """
    Relative volume of the trigger bar vs trailing 20-bar average.
    > 2.0 = strong institutional activity on the trigger bar.
    """
    if len(df_intraday) < 5:
        return 1.0
    avg_vol = df_intraday["volume"].iloc[-21:-1].mean()
    trigger_vol = df_intraday["volume"].iloc[-1]
    if avg_vol <= 0:
        return 1.0
    return round(float(trigger_vol / avg_vol), 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPOSITE TRIGGER SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trigger_score(conviction: str, streak_len: int,
                           r_ratio: Optional[float], rvol: float,
                           timeframe: str, dist_session_high: Optional[float],
                           direction: str) -> int:
    """
    Composite score for ranking triggers in the dashboard.
    Higher = more actionable. Max ~15 points.

    Points:
      Conviction   HIGH=4, MED=2
      Streak       3=1, 4=2, 5+=3
      R-ratio      ≥3.0=3, ≥2.0=2, ≥1.0=1, <1.0=0
      RVOL         ≥2.0=2, ≥1.5=1
      Timeframe    65-MIN=2, 30-MIN=1  (higher TF = more reliable)
      Session high Bullish near/above session high = +1
    """
    score = 0

    # Conviction
    score += {"HIGH": 4, "MED": 2}.get(conviction, 0)

    # Streak length
    if streak_len >= 5:   score += 3
    elif streak_len >= 4: score += 2
    else:                  score += 1

    # R-ratio
    if r_ratio is not None:
        if r_ratio >= R_ELITE:   score += 3
        elif r_ratio >= R_GOOD:  score += 2
        elif r_ratio >= R_MINIMUM: score += 1

    # RVOL on trigger bar
    if rvol >= 2.0:   score += 2
    elif rvol >= 1.5: score += 1

    # Timeframe quality
    score += 2 if "65" in timeframe else 1

    # Near/above session high (bullish) or near/below session low (bearish)
    if dist_session_high is not None:
        if direction == "bullish" and dist_session_high >= -1.0:
            score += 1   # breaking out near session high — strong
        elif direction == "bearish" and dist_session_high <= -5.0:
            score += 1   # far from highs, trend day down

    return score


# ═══════════════════════════════════════════════════════════════════════════════
#  PIVOT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def is_green(row): return row["close"] > row["open"]
def is_red(row):   return row["close"] <= row["open"]


def detect_pivot(df: pd.DataFrame,
                 min_streak: int = MIN_STREAK) -> Optional[dict]:
    if len(df) < min_streak + 2:
        return None
    trigger_bar = df.iloc[-1]
    pivot_bar   = df.iloc[-2]

    # Bullish
    if is_green(pivot_bar):
        streak = 0
        for i in range(3, len(df) + 1):
            if is_red(df.iloc[-i]):
                streak += 1
            else:
                break
        if streak >= min_streak and trigger_bar["close"] > pivot_bar["high"]:
            return {
                "direction":   "bullish",
                "streak_len":  streak,
                "pivot_bar":   pivot_bar,
                "trigger_bar": trigger_bar,
                "pivot_time":  df.index[-2],
                "trigger_time":df.index[-1],
            }

    # Bearish
    if is_red(pivot_bar):
        streak = 0
        for i in range(3, len(df) + 1):
            if is_green(df.iloc[-i]):
                streak += 1
            else:
                break
        if streak >= min_streak and trigger_bar["close"] < pivot_bar["low"]:
            return {
                "direction":   "bearish",
                "streak_len":  streak,
                "pivot_bar":   pivot_bar,
                "trigger_bar": trigger_bar,
                "pivot_time":  df.index[-2],
                "trigger_time":df.index[-1],
            }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSIST TRIGGER
# ═══════════════════════════════════════════════════════════════════════════════

def persist_trigger(entry: dict, timeframe: str, result: dict,
                    df_daily: Optional[pd.DataFrame],
                    df_intraday: pd.DataFrame):
    """Build full trigger record with R-ratio, key levels, score and save."""
    pb = result["pivot_bar"]
    tb = result["trigger_bar"]
    d  = result["direction"]

    stop_level = float(pb["low"]) if d == "bullish" else float(pb["high"])
    trigger_close = float(tb["close"])

    # ATR + R-ratio
    atr     = compute_atr(df_daily)
    r_ratio = compute_r_ratio(trigger_close, stop_level, d, atr)

    # Intraday RVOL on trigger bar
    rvol = compute_intraday_rvol(df_intraday)

    # Key level distances
    dist_session_high = compute_session_high_distance(df_daily, trigger_close)
    dist_ema8w        = compute_ema8w_distance(trigger_close, entry.get("ema8"))

    # Composite score
    score = compute_trigger_score(
        conviction=entry["conviction"],
        streak_len=result["streak_len"],
        r_ratio=r_ratio,
        rvol=rvol,
        timeframe=timeframe,
        dist_session_high=dist_session_high,
        direction=d,
    )

    entry_note = "ITM/ATM weekly calls" if d == "bullish" else "ITM/ATM weekly puts"

    trigger = {
        # Identity
        "ticker":              entry["ticker"],
        "timeframe":           timeframe,
        "direction":           d,
        "conviction":          entry["conviction"],
        # Context from daily watchlist
        "theme":               entry.get("theme", ""),
        "theme_rank":          entry.get("theme_rank", 99),
        "industry":            entry.get("industry", ""),
        "industry_rank":       entry.get("industry_rank", 99),
        "weekly_stage":        entry.get("weekly_stage"),
        "daily_stage":         entry.get("daily_stage"),
        "trend_template":      entry.get("trend_template"),
        "weekly_bbuw":         entry.get("weekly_bbuw"),
        "daily_bbuw":          entry.get("daily_bbuw"),
        "ep_tier":             entry.get("ep_tier", "NONE"),
        "rs_rating":           entry.get("rs_rating"),
        # Pivot bars
        "streak_len":          result["streak_len"],
        "pivot_open":          float(pb["open"]),
        "pivot_high":          float(pb["high"]),
        "pivot_low":           float(pb["low"]),
        "pivot_close":         float(pb["close"]),
        "pivot_time":          str(result["pivot_time"]),
        "trigger_open":        float(tb["open"]),
        "trigger_high":        float(tb["high"]),
        "trigger_low":         float(tb["low"]),
        "trigger_close":       trigger_close,
        "trigger_time":        str(result["trigger_time"]),
        # Risk metrics
        "stop_level":          stop_level,
        "atr_14d":             round(atr, 4) if pd.notna(atr) else None,
        "r_ratio":             r_ratio,
        "rvol_trigger":        rvol,
        # Key level distances
        "dist_session_high_%": dist_session_high,
        "dist_ema8w_%":        dist_ema8w,
        # Ranking
        "trigger_score":       score,
        "entry_note":          entry_note,
    }

    save_trigger(trigger)
    r_str = f"  R={r_ratio:.1f}" if r_ratio is not None else ""
    log.info(f"  💾 {entry['ticker']} {timeframe} {d.upper()}"
             f"  score={score}{r_str}  rvol={rvol:.1f}×"
             f"  streak={result['streak_len']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting INTRADAY pivot scanner v4 (R-ratio + scoring)...")

    daily_data = get_latest_daily_watchlist()
    entries    = daily_data.get("entries", [])
    entries    = [e for e in entries if e["conviction"] in ALERT_TIERS]

    if not entries:
        log.error("No entries in daily watchlist. Run daily_screener.py first.")
        return

    log.info(f"Scanning {len(entries)} tickers (tiers: {ALERT_TIERS})")

    # Track which tickers already fired to deduplicate 30/65min
    # Rule: if same ticker fires on both, keep the 65-min (higher TF wins)
    fired_tickers_30min = set()
    fired_tickers_65min = set()
    trigger_count = 0

    for i, entry in enumerate(entries, 1):
        ticker = entry["ticker"]
        log.info(f"[{i}/{len(entries)}] {ticker} ({entry['conviction']})")

        # Fetch daily bars once per ticker for ATR + session high
        df_daily = fetch_daily_bars(ticker)

        # ── 30-MIN ──────────────────────────────────────────────────────────
        if SCAN_30MIN:
            df_30 = fetch_intraday_bars(ticker, "30m")
            if df_30 is not None:
                r = detect_pivot(df_30)
                if r:
                    fired_tickers_30min.add(ticker)
                    persist_trigger(entry, "30-MIN", r, df_daily, df_30)
                    trigger_count += 1

        # ── 65-MIN (60m bars — 65min not available in yfinance) ─────────────
        if SCAN_65MIN:
            df_60 = fetch_intraday_bars(ticker, "60m")
            if df_60 is not None:
                r = detect_pivot(df_60)
                if r:
                    fired_tickers_65min.add(ticker)
                    persist_trigger(entry, "65-MIN", r, df_daily, df_60)
                    trigger_count += 1

        if i < len(entries):
            time.sleep(RATE_LIMIT_PAUSE)

    # Deduplication log — both timeframes fired on same ticker
    dupes = fired_tickers_30min & fired_tickers_65min
    if dupes:
        log.info(f"  ⚠️ Both TF fired on: {', '.join(sorted(dupes))} — dashboard will show 65-MIN as primary")

    log.info(f"\n  SCAN COMPLETE — {trigger_count} trigger(s) persisted")
    log.info(f"    30-MIN: {len(fired_tickers_30min)}  |  65-MIN: {len(fired_tickers_65min)}")


if __name__ == "__main__":
    main()
