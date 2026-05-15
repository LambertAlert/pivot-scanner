"""
volume_surge_prep.py — Weekly Volume Surge / Climax Detection (v2)
==================================================================
Scans Stage 1/2 tickers from the weekly watchlist for weekly volume surges only.

v2 changes:
  - Daily surge detection removed — daily volume is noisy and false-signal-prone.
    Weekly volume is the Weinstein confirmation signal (institutional accumulation).
  - Replaced serial fetch_weekly() per ticker with a single batch yf.download()
    call for all tickers.  439 API calls → 1 batch call.

Philosophy:
  A WEEKLY volume surge/climax in a constructive Stage 1/2 setup is a lagging
  but high-confidence signal of institutional accumulation. Not a buy trigger
  on its own — it creates a "loading dock" entry to watch for BBUW coil and
  30-min pivot in the following days/weeks.

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

THEMES_CSV         = "sector_themes.csv"
BATCH_CHUNK        = 200    # tickers per yf.download() call

# Weekly surge thresholds
WEEKLY_LOOKBACK    = 52     # weeks to look back for the "highest week"
WEEKLY_RVOL_MIN    = 1.5    # must be ≥ 1.5× the 20-week average
WEEKLY_STALE_DAYS  = 60     # ~12 weeks before marked stale in the log

# History retention
LOG_RETENTION_DAYS = 180


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def batch_fetch_weekly(tickers: list) -> dict:
    """
    Download 2 years of weekly bars for all tickers in one yf.download() call.
    Returns {ticker: DataFrame} — tickers with < 20 usable rows are absent.
    """
    result = {}
    tickers = list(set(tickers))

    for i in range(0, len(tickers), BATCH_CHUNK):
        chunk = tickers[i: i + BATCH_CHUNK]
        try:
            raw = yf.download(
                chunk,
                period="2y",
                interval="1wk",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                log.warning(f"Weekly batch chunk {i//BATCH_CHUNK + 1}: empty result")
                continue

            for tk in chunk:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        df = raw.xs(tk, axis=1, level=1).copy()
                    else:
                        df = raw.copy()

                    df.columns = [c.lower() for c in df.columns]
                    needed = [c for c in ["open", "high", "low", "close", "volume"]
                              if c in df.columns]
                    df = df[needed].dropna(how="all")

                    # Strip timezone so date comparisons stay clean
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df.index = pd.to_datetime(df.index)

                    df = df[df["close"] > 0]
                    if len(df) >= 20:
                        result[tk] = df
                except Exception:
                    pass

        except Exception as e:
            log.warning(f"Weekly batch fetch error: {e}")

        if i + BATCH_CHUNK < len(tickers):
            time.sleep(2)

    log.info(f"Weekly batch fetch: {len(result)}/{len(tickers)} tickers with ≥20 rows")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  WEEKLY SURGE DETECTION — unchanged logic
# ═══════════════════════════════════════════════════════════════════════════════

def detect_weekly_surge(dfw: pd.DataFrame) -> Optional[dict]:
    """
    Find the most recent weekly volume surge in the trailing WEEKLY_LOOKBACK weeks.

    Qualifies if:
      1. That week's volume is the highest in the trailing WEEKLY_LOOKBACK weeks
      2. Volume ≥ WEEKLY_RVOL_MIN × 20-week average
      3. Week closed constructively (up OR absorbed — closed in upper half of range)

    Returns the most recent qualifying surge, or None.
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

    is_up_week    = surge_close > surge_open
    candle_pos    = (surge_close - surge_low) / surge_range if surge_range > 0 else 0.5
    is_absorption = candle_pos > 0.5

    if not (is_up_week or is_absorption):
        return None   # distribution week — skip

    current_close      = close.iloc[-1]
    price_change_since = ((current_close / surge_close) - 1) * 100

    last_bar   = pd.Timestamp(dfw.index[-1]).normalize()
    surge_date = pd.Timestamp(max_vol_idx)
    days_since  = (last_bar - surge_date).days
    weeks_since = days_since // 7

    is_fresh = days_since <= WEEKLY_STALE_DAYS

    ema_21w        = close.ewm(span=21, adjust=False).mean().iloc[-1]
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
#  SECTOR THEME LOOKUP — unchanged
# ═══════════════════════════════════════════════════════════════════════════════

def load_themes(csv_path: str) -> dict:
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
    log.info("Starting VOLUME SURGE scanner (v2 — weekly only, batch download)...")

    weekly_data    = get_latest_weekly_watchlist()
    weekly_entries = weekly_data.get("entries", [])

    if not weekly_entries:
        log.error("No weekly watchlist found. Run weekly_screener.py first.")
        return

    log.info(f"Scanning {len(weekly_entries)} Stage 1/2 tickers for weekly volume surges...")
    themes = load_themes(THEMES_CSV)

    # Load existing log to preserve history (append-only)
    existing        = get_volume_surges()
    existing_events = existing.get("events", [])
    already_logged  = {
        (e["ticker"], e["surge_type"], e["surge_date"])
        for e in existing_events
    }

    # ── One batch download for all weekly bars ─────────────────────────────
    tickers = [e["ticker"] for e in weekly_entries]
    log.info(f"Batch downloading weekly bars for {len(tickers)} tickers...")
    weekly_bars = batch_fetch_weekly(tickers)

    # ── Per-ticker detection (no API calls in loop) ────────────────────────
    new_events   = []
    weekly_count = 0

    for entry in weekly_entries:
        ticker = entry["ticker"]
        dfw    = weekly_bars.get(ticker)
        if dfw is None:
            continue

        context = {
            "ticker":         ticker,
            "stage":          entry.get("stage"),
            "bbuw_score":     entry.get("bbuw_score"),
            "trend_template": entry.get("trend_template_score"),
            "pivot_8w_tier":  entry.get("pivot_8w_tier", "NONE"),
            "theme":          themes.get(ticker, {}).get("theme", "Unclassified"),
            "industry":       themes.get(ticker, {}).get("industry", ""),
        }

        surge = detect_weekly_surge(dfw)
        if not surge:
            continue

        key = (ticker, "WEEKLY", surge["surge_date"])
        if key in already_logged:
            continue

        event = {**context, **surge, "logged_at": datetime.now().isoformat()}
        new_events.append(event)
        already_logged.add(key)
        weekly_count += 1
        log.info(
            f"  📊 WEEKLY SURGE: {ticker} | "
            f"{surge['rvol']:.1f}× avg | "
            f"{surge['surge_date']} | "
            f"{'✅ fresh' if surge['is_fresh'] else '⏳ aging'}"
        )

    # ── Merge, deduplicate, trim, sort ─────────────────────────────────────
    all_events = new_events + existing_events

    seen, deduped = set(), []
    for e in all_events:
        key = (e["ticker"], e["surge_type"], e["surge_date"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    cutoff = (datetime.now() - timedelta(days=LOG_RETENTION_DAYS)).strftime("%Y-%m-%d")
    deduped = [e for e in deduped if e.get("surge_date", "0000") >= cutoff]
    deduped.sort(key=lambda e: e.get("surge_date", ""), reverse=True)

    save_volume_surges(deduped)

    log.info(f"\n  VOLUME SURGE SCAN COMPLETE")
    log.info(f"    New weekly surges : {weekly_count}")
    log.info(f"    Total in log      : {len(deduped)}")
    log.info(f"    (Log retains {LOG_RETENTION_DAYS} days of history)")


if __name__ == "__main__":
    main()
