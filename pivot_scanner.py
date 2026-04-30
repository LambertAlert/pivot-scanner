"""
INTRADAY PIVOT SCANNER (v3) — writes triggers to data_layer (no email).
Reads daily_watchlist.json, runs 30-min and 65-min pivot detection,
persists every trigger to SQLite + JSON for the Streamlit dashboard.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

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
ALERT_TIERS      = ["HIGH", "MED"]    # only persist these tiers

LOG_LEVEL        = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
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
        return df.tail(100)
    except Exception as e:
        log.error(f"[{ticker}] {interval} fetch error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  PIVOT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def is_green(row): return row["close"] > row["open"]
def is_red(row):   return row["close"] <= row["open"]


def detect_pivot(df: pd.DataFrame, min_streak: int = MIN_STREAK) -> Optional[dict]:
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
                "direction": "bullish", "streak_len": streak,
                "pivot_bar": pivot_bar, "trigger_bar": trigger_bar,
                "pivot_time": df.index[-2], "trigger_time": df.index[-1],
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
                "direction": "bearish", "streak_len": streak,
                "pivot_bar": pivot_bar, "trigger_bar": trigger_bar,
                "pivot_time": df.index[-2], "trigger_time": df.index[-1],
            }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSIST TRIGGER
# ═══════════════════════════════════════════════════════════════════════════════

def persist_trigger(entry: dict, timeframe: str, result: dict):
    """Build full trigger record and save via data_layer."""
    pb = result["pivot_bar"]
    tb = result["trigger_bar"]
    d  = result["direction"]

    if d == "bullish":
        stop_level = float(pb["low"])
        entry_note = "ITM/ATM weekly calls"
    else:
        stop_level = float(pb["high"])
        entry_note = "ITM/ATM weekly puts"

    trigger = {
        "ticker":         entry["ticker"],
        "timeframe":      timeframe,
        "direction":      d,
        "conviction":     entry["conviction"],
        "theme":          entry["theme"],
        "theme_rank":     entry["theme_rank"],
        "weekly_stage":   entry["weekly_stage"],
        "daily_stage":    entry["daily_stage"],
        "trend_template": entry["trend_template"],
        "weekly_bbuw":    entry["weekly_bbuw"],
        "daily_bbuw":     entry["daily_bbuw"],
        "streak_len":     result["streak_len"],
        "pivot_open":     float(pb["open"]),
        "pivot_high":     float(pb["high"]),
        "pivot_low":      float(pb["low"]),
        "pivot_close":    float(pb["close"]),
        "pivot_time":     str(result["pivot_time"]),
        "trigger_open":   float(tb["open"]),
        "trigger_high":   float(tb["high"]),
        "trigger_low":    float(tb["low"]),
        "trigger_close":  float(tb["close"]),
        "trigger_time":   str(result["trigger_time"]),
        "stop_level":     stop_level,
        "entry_note":     entry_note,
    }

    save_trigger(trigger)
    log.info(f"  💾 Persisted: {entry['ticker']} {timeframe} {d.upper()}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting INTRADAY pivot scanner (v3 — dashboard mode)...")

    daily_data = get_latest_daily_watchlist()
    entries = daily_data.get("entries", [])
    entries = [e for e in entries if e["conviction"] in ALERT_TIERS]

    if not entries:
        log.error("No entries in daily watchlist. Run daily_screener.py first.")
        return

    log.info(f"Scanning {len(entries)} tickers (tiers: {ALERT_TIERS})")

    trigger_count = 0
    for i, entry in enumerate(entries, 1):
        ticker = entry["ticker"]
        log.info(f"[{i}/{len(entries)}] {ticker} ({entry['conviction']})")

        if SCAN_30MIN:
            df = fetch_intraday_bars(ticker, "30m")
            if df is not None:
                r = detect_pivot(df)
                if r:
                    persist_trigger(entry, "30-MIN", r)
                    trigger_count += 1

        if SCAN_65MIN:
            df = fetch_intraday_bars(ticker, "60m")
            if df is not None:
                r = detect_pivot(df)
                if r:
                    persist_trigger(entry, "65-MIN", r)
                    trigger_count += 1

        if i < len(entries):
            time.sleep(RATE_LIMIT_PAUSE)

    log.info(f"\n  SCAN COMPLETE — {trigger_count} trigger(s) persisted to data_layer")


if __name__ == "__main__":
    main()
