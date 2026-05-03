"""
volume_surge_prep.py — Volume Surge / Climax Detection
========================================================
Scans Stage 1/2 tickers from the weekly watchlist for:
  • Daily surge  — today's volume = highest day in last 60 trading days
                   AND ≥ 2.0× the 20-day average
  • Weekly surge — this week's volume = highest week in last 52 weeks
                   AND ≥ 1.5× the 20-week average

Philosophy:
  A volume surge/climax in a constructive setup is a LAGGING but HIGH-CONFIDENCE
  signal of institutional accumulation. It is NOT a buy trigger on its own.
  It creates a "loading dock" entry — watch for the BBUW coil and 30-min pivot
  to fire on these names in the following days/weeks.

  Key tracking fields:
    • Price at surge vs price now (did it continue or break down?)
    • Days/weeks since surge (freshness)
    • Current BBUW, stage, 8W pivot tier (is the setup intact?)

Output: data/volume_surges.json
Schedule: runs as part of the daily job (after daily_screener.py)

Dependencies: yfinance, pandas, numpy
"""

import os
import csv
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from data_layer import (
    get_latest_weekly_watchlist,
    save_volume_surges,
    get_volume_surges,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

THEMES_CSV        = "sector_themes.csv"
RATE_LIMIT_PAUSE  = 1.0

# Daily surge thresholds
DAILY_LOOKBACK    = 60     # trading days to look back for the "highest day"
DAILY_RVOL_MIN    = 2.0    # must be at least 2× the 20-day average to qualify

# Weekly surge thresholds
WEEKLY_LOOKBACK   = 52     # weeks to look back for the "highest week"
WEEKLY_RVOL_MIN   = 1.5    # must be at least 1.5× the 20-week average

# How many days a surge stays "fresh" in the log before marked as stale
DAILY_STALE_DAYS  = 20     # ~4 weeks of trading
WEEKLY_STALE_DAYS = 60     # ~12 weeks


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_daily(ticker: str) -> Optional[pd.DataFrame]:
    """Pull 6 months of daily OHLCV — covers 60-day lookback + current metrics."""
    try:
        df = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        log.warning(f"[{ticker}] daily fetch: {e}")
        return None


def fetch_weekly(ticker: str) -> Optional[pd.DataFrame]:
    """Pull 2 years of weekly OHLCV — covers 52-week lookback."""
    try:
        df = yf.Ticker(ticker).history(period="2y", interval="1wk", auto_adjust=True)
        if df.empty or len(df) < 20:
            return None
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        log.warning(f"[{ticker}] weekly fetch: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  DAILY SURGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_daily_surge(df: pd.DataFrame) -> Optional[dict]:
    """
    Find the most recent daily volume surge in the trailing DAILY_LOOKBACK days.
    Returns None if no qualifying surge exists.

    A daily surge qualifies if:
      1. That day's volume is the highest in the trailing DAILY_LOOKBACK bars
      2. That day's volume ≥ DAILY_RVOL_MIN × 20-day average volume
      3. The surge happened on a UP day (close > open) — buying climax, not panic selling
         OR on a DOWN day that reversed intraday (close > low by > 50% of range) — absorption

    Returns the most recent qualifying surge only.
    """
    if len(df) < max(DAILY_LOOKBACK, 25):
        return None

    vol   = df["volume"]
    close = df["close"]
    open_ = df["open"]
    high  = df["high"]
    low   = df["low"]

    # Look at the last DAILY_LOOKBACK bars
    window = df.tail(DAILY_LOOKBACK).copy()
    window_vol = window["volume"]

    # Highest volume day in the window
    max_vol_idx = window_vol.idxmax()
    max_vol     = window_vol.max()

    # 20-day average at the time of the surge
    loc = df.index.get_loc(max_vol_idx)
    avg_vol_20 = vol.iloc[max(0, loc - 20):loc].mean()
    if avg_vol_20 <= 0:
        return None

    rvol = max_vol / avg_vol_20

    # Must meet minimum RVOL threshold
    if rvol < DAILY_RVOL_MIN:
        return None

    # Candle quality — bullish absorption or buying climax
    surge_close = close.loc[max_vol_idx]
    surge_open  = open_.loc[max_vol_idx]
    surge_high  = high.loc[max_vol_idx]
    surge_low   = low.loc[max_vol_idx]
    surge_range = surge_high - surge_low

    is_up_day  = surge_close > surge_open
    # Absorption: even if it opened and sold off, it closed back up off the lows
    candle_pos = (surge_close - surge_low) / surge_range if surge_range > 0 else 0.5
    is_absorption = candle_pos > 0.5   # closed in upper half of range

    if not (is_up_day or is_absorption):
        return None   # pure distribution day — not what we want

    # Current price context
    current_close = close.iloc[-1]
    price_change_since = ((current_close / surge_close) - 1) * 100

    # Days since surge
    today = df.index[-1]
    surge_date = pd.Timestamp(max_vol_idx)
    days_since = (today - surge_date).days

    # Current metrics
    ema_21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    pct_from_21ema = ((current_close - ema_21) / ema_21) * 100

    # Is still fresh?
    is_fresh = days_since <= DAILY_STALE_DAYS

    return {
        "surge_type":         "DAILY",
        "surge_date":         surge_date.strftime("%Y-%m-%d"),
        "surge_volume":       int(max_vol),
        "avg_vol_20d":        int(avg_vol_20),
        "rvol":               round(rvol, 2),
        "surge_price":        round(float(surge_close), 2),
        "surge_open":         round(float(surge_open), 2),
        "surge_high":         round(float(surge_high), 2),
        "surge_low":          round(float(surge_low), 2),
        "candle_position":    round(float(candle_pos), 2),
        "is_up_day":          bool(is_up_day),
        "current_price":      round(float(current_close), 2),
        "price_chg_since_pct": round(float(price_change_since), 2),
        "days_since_surge":   int(days_since),
        "is_fresh":           bool(is_fresh),
        "pct_from_21ema":     round(float(pct_from_21ema), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  WEEKLY SURGE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_weekly_surge(dfw: pd.DataFrame) -> Optional[dict]:
    """
    Find the most recent weekly volume surge in the trailing WEEKLY_LOOKBACK weeks.
    Returns None if no qualifying surge exists.

    A weekly surge qualifies if:
      1. That week's volume is the highest in the trailing WEEKLY_LOOKBACK weeks
      2. That week's volume ≥ WEEKLY_RVOL_MIN × 20-week average volume
      3. The week closed up (bullish absorption or buying into strength)
    """
    if len(dfw) < max(WEEKLY_LOOKBACK, 25):
        return None

    vol   = dfw["volume"]
    close = dfw["close"]
    open_ = dfw["open"]
    high  = dfw["high"]
    low   = dfw["low"]

    window     = dfw.tail(WEEKLY_LOOKBACK).copy()
    window_vol = window["volume"]

    max_vol_idx = window_vol.idxmax()
    max_vol     = window_vol.max()

    loc = dfw.index.get_loc(max_vol_idx)
    avg_vol_20w = vol.iloc[max(0, loc - 20):loc].mean()
    if avg_vol_20w <= 0:
        return None

    rvol = max_vol / avg_vol_20w

    if rvol < WEEKLY_RVOL_MIN:
        return None

    surge_close = close.loc[max_vol_idx]
    surge_open  = open_.loc[max_vol_idx]
    surge_high  = high.loc[max_vol_idx]
    surge_low   = low.loc[max_vol_idx]
    surge_range = surge_high - surge_low

    is_up_week = surge_close > surge_open
    candle_pos = (surge_close - surge_low) / surge_range if surge_range > 0 else 0.5
    is_absorption = candle_pos > 0.5

    if not (is_up_week or is_absorption):
        return None

    current_close = close.iloc[-1]
    price_change_since = ((current_close / surge_close) - 1) * 100

    today = dfw.index[-1]
    surge_date = pd.Timestamp(max_vol_idx)
    days_since = (today - surge_date).days
    weeks_since = days_since // 7

    is_fresh = days_since <= WEEKLY_STALE_DAYS

    # Weekly EMA 21
    ema_21w = close.ewm(span=21, adjust=False).mean().iloc[-1]
    pct_from_21ema = ((current_close - ema_21w) / ema_21w) * 100

    return {
        "surge_type":          "WEEKLY",
        "surge_date":          surge_date.strftime("%Y-%m-%d"),
        "surge_volume":        int(max_vol),
        "avg_vol_20w":         int(avg_vol_20w),
        "rvol":                round(rvol, 2),
        "surge_price":         round(float(surge_close), 2),
        "surge_open":          round(float(surge_open), 2),
        "surge_high":          round(float(surge_high), 2),
        "surge_low":           round(float(surge_low), 2),
        "candle_position":     round(float(candle_pos), 2),
        "is_up_week":          bool(is_up_week),
        "current_price":       round(float(current_close), 2),
        "price_chg_since_pct": round(float(price_change_since), 2),
        "days_since_surge":    int(days_since),
        "weeks_since_surge":   int(weeks_since),
        "is_fresh":            bool(is_fresh),
        "pct_from_21ema":      round(float(pct_from_21ema), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTOR THEME LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

def load_themes(csv_path: str) -> dict:
    """Returns {ticker: {theme, industry, theme_rank}}."""
    result = {}
    if not os.path.exists(csv_path):
        return result
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = (row.get("Symbol") or "").strip().upper()
            if sym:
                result[sym] = {
                    "theme":    (row.get("Theme") or "").strip(),
                    "industry": (row.get("Industry") or "").strip(),
                }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting VOLUME SURGE scanner...")

    # Load Stage 1/2 names from weekly watchlist
    weekly_data = get_latest_weekly_watchlist()
    weekly_entries = weekly_data.get("entries", [])

    if not weekly_entries:
        log.error("No weekly watchlist found. Run weekly_screener.py first.")
        return

    log.info(f"Scanning {len(weekly_entries)} Stage 1/2 tickers for volume surges...")
    themes = load_themes(THEMES_CSV)

    # Load existing surge log to preserve history (append-only)
    existing = get_volume_surges()
    existing_events = existing.get("events", [])

    # Build a set of (ticker, surge_type, surge_date) already logged
    already_logged = {
        (e["ticker"], e["surge_type"], e["surge_date"])
        for e in existing_events
    }

    new_events    = []
    daily_count   = 0
    weekly_count  = 0

    for i, entry in enumerate(weekly_entries, 1):
        ticker = entry["ticker"]
        log.info(f"[{i}/{len(weekly_entries)}] {ticker}")

        # Shared context from weekly screen
        context = {
            "ticker":          ticker,
            "stage":           entry.get("stage"),
            "bbuw_score":      entry.get("bbuw_score"),
            "trend_template":  entry.get("trend_template_score"),
            "pivot_8w_tier":   entry.get("pivot_8w_tier", "NONE"),
            "theme":           themes.get(ticker, {}).get("theme", "Unclassified"),
            "industry":        themes.get(ticker, {}).get("industry", ""),
        }

        # ── Daily surge ────────────────────────────────────────────────────
        df_daily = fetch_daily(ticker)
        if df_daily is not None:
            daily_surge = detect_daily_surge(df_daily)
            if daily_surge:
                key = (ticker, "DAILY", daily_surge["surge_date"])
                if key not in already_logged:
                    event = {**context, **daily_surge,
                             "logged_at": datetime.now().isoformat()}
                    new_events.append(event)
                    already_logged.add(key)
                    daily_count += 1
                    log.info(f"  📊 DAILY SURGE: {ticker} | "
                             f"{daily_surge['rvol']:.1f}× avg | "
                             f"{daily_surge['surge_date']} | "
                             f"{'✅ fresh' if daily_surge['is_fresh'] else '⏳ aging'}")

        time.sleep(RATE_LIMIT_PAUSE * 0.5)   # shorter pause — we need weekly too

        # ── Weekly surge ───────────────────────────────────────────────────
        df_weekly = fetch_weekly(ticker)
        if df_weekly is not None:
            weekly_surge = detect_weekly_surge(df_weekly)
            if weekly_surge:
                key = (ticker, "WEEKLY", weekly_surge["surge_date"])
                if key not in already_logged:
                    event = {**context, **weekly_surge,
                             "logged_at": datetime.now().isoformat()}
                    new_events.append(event)
                    already_logged.add(key)
                    weekly_count += 1
                    log.info(f"  📊 WEEKLY SURGE: {ticker} | "
                             f"{weekly_surge['rvol']:.1f}× avg | "
                             f"{weekly_surge['surge_date']} | "
                             f"{'✅ fresh' if weekly_surge['is_fresh'] else '⏳ aging'}")

        time.sleep(RATE_LIMIT_PAUSE * 0.5)

    # ── Merge new events with existing log ────────────────────────────────
    # Update current_price and staleness on existing events using today's data
    # (simple approach: just prepend new, keep all history up to 180 days)
    all_events = new_events + existing_events

    # Deduplicate by (ticker, surge_type, surge_date) — keep first occurrence
    seen = set()
    deduped = []
    for e in all_events:
        key = (e["ticker"], e["surge_type"], e["surge_date"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    # Trim to last 180 days
    cutoff = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    deduped = [e for e in deduped if e.get("surge_date", "0000") >= cutoff]

    # Sort: freshest first, then by RVOL
    deduped.sort(key=lambda e: (
        0 if e.get("is_fresh") else 1,
        e.get("surge_date", ""),
        -e.get("rvol", 0),
    ), reverse=False)
    deduped.sort(key=lambda e: e.get("surge_date", ""), reverse=True)

    save_volume_surges(deduped)

    log.info(f"\n  VOLUME SURGE SCAN COMPLETE")
    log.info(f"    New daily surges  : {daily_count}")
    log.info(f"    New weekly surges : {weekly_count}")
    log.info(f"    Total in log      : {len(deduped)}")
    log.info(f"    (Log retains 180 days of history)")


if __name__ == "__main__":
    main()
