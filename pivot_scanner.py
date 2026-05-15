"""
INTRADAY PIVOT SCANNER (v4.3) — 30-min pivot only, batch downloads.

Changes over v4.2:
  - ORB removed (noise reduction).
  - 65-min scan already removed in v4.2.
  - 2 batch downloads per run: daily bars + 30m bars.

Trigger: Classic 30-min pivot (3+ bar streak reversal, HIGH + MED conviction).
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
ALERT_TIERS      = ["HIGH", "MED"]

# R-ratio thresholds
R_ELITE   = 3.0
R_GOOD    = 2.0
R_MINIMUM = 1.0

# Batch download chunk size
BATCH_CHUNK = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH DATA FETCHING  (replaces per-ticker serial calls)
# ═══════════════════════════════════════════════════════════════════════════════

def _slice_ticker(raw: pd.DataFrame, ticker: str, min_rows: int = 5) -> Optional[pd.DataFrame]:
    """
    Extract a single ticker from a yf.download() MultiIndex result.
    Returns None if the ticker has insufficient data.
    """
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw.xs(ticker, axis=1, level=1).copy()
        else:
            df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        needed = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[needed].dropna(how="all")
        df = df[df["close"] > 0]
        return df if len(df) >= min_rows else None
    except Exception:
        return None


def batch_fetch_daily(tickers: list) -> dict:
    """
    Download daily bars for all tickers in one call.
    Returns {ticker: df} — tickers with insufficient data are absent.
    """
    result = {}
    for i in range(0, len(tickers), BATCH_CHUNK):
        chunk = tickers[i: i + BATCH_CHUNK]
        try:
            raw = yf.download(
                chunk, period="60d", interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )
            if raw.empty:
                continue
            for tk in chunk:
                df = _slice_ticker(raw, tk, min_rows=14)
                if df is not None:
                    result[tk] = df
        except Exception as e:
            log.warning(f"batch_fetch_daily chunk error: {e}")
        if i + BATCH_CHUNK < len(tickers):
            time.sleep(RATE_LIMIT_PAUSE)
    log.info(f"  Daily batch: {len(result)}/{len(tickers)} tickers fetched")
    return result


def batch_fetch_intraday(tickers: list, interval: str) -> dict:
    """
    Download intraday bars for all tickers in one call.
    Returns {ticker: df} — strips tz to UTC, tails to 100 bars.
    """
    result = {}
    for i in range(0, len(tickers), BATCH_CHUNK):
        chunk = tickers[i: i + BATCH_CHUNK]
        try:
            raw = yf.download(
                chunk, period="60d", interval=interval,
                auto_adjust=True, progress=False, threads=True,
            )
            if raw.empty:
                continue
            # Normalise timezone
            if raw.index.tzinfo is None:
                raw.index = raw.index.tz_localize("UTC")
            else:
                raw.index = raw.index.tz_convert("UTC")
            for tk in chunk:
                df = _slice_ticker(raw, tk, min_rows=5)
                if df is not None:
                    result[tk] = df.sort_index().tail(100)
        except Exception as e:
            log.warning(f"batch_fetch_intraday({interval}) chunk error: {e}")
        if i + BATCH_CHUNK < len(tickers):
            time.sleep(RATE_LIMIT_PAUSE)
    log.info(f"  Intraday {interval} batch: {len(result)}/{len(tickers)} tickers fetched")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  ATR + R-RATIO
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df_daily: pd.DataFrame, period: int = 14) -> float:
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
    if np.isnan(atr) or atr <= 0:
        return None
    risk = trigger_close - stop_level if direction == "bullish" else stop_level - trigger_close
    return round(risk / atr, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  KEY LEVEL DISTANCES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_session_high_distance(df_daily: pd.DataFrame,
                                   trigger_close: float) -> Optional[float]:
    if df_daily is None or df_daily.empty:
        return None
    today_high = float(df_daily["high"].iloc[-1])
    if today_high <= 0:
        return None
    return round((trigger_close / today_high - 1) * 100, 2)


def compute_ema8w_distance(trigger_close: float,
                            ema8: Optional[float]) -> Optional[float]:
    if ema8 is None or ema8 <= 0:
        return None
    return round((trigger_close / ema8 - 1) * 100, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTRADAY VOLUME RVOL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_intraday_rvol(df_intraday: pd.DataFrame) -> float:
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
    score = 0
    score += {"HIGH": 4, "MED": 2}.get(conviction, 0)
    if streak_len >= 5:   score += 3
    elif streak_len >= 4: score += 2
    else:                  score += 1
    if r_ratio is not None:
        if r_ratio >= R_ELITE:     score += 3
        elif r_ratio >= R_GOOD:    score += 2
        elif r_ratio >= R_MINIMUM: score += 1
    if rvol >= 2.0:   score += 2
    elif rvol >= 1.5: score += 1
    score += 2 if "65" in timeframe else 1
    if dist_session_high is not None:
        if direction == "bullish" and dist_session_high >= -1.0: score += 1
        elif direction == "bearish" and dist_session_high <= -5.0: score += 1
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
    df = df[df["close"] > 0].copy()
    if len(df) < min_streak + 2:
        return None

    trigger_bar = df.iloc[-1]
    pivot_bar   = df.iloc[-2]

    def count_streak(start_idx: int, qualifier) -> int:
        count = 0
        for i in range(start_idx, min(start_idx + 10, len(df))):
            if qualifier(df.iloc[-(i)]):
                count += 1
            else:
                break
        return count

    if is_green(pivot_bar):
        streak = count_streak(3, is_red)
        if streak >= min_streak and float(trigger_bar["close"]) > float(pivot_bar["high"]):
            return {
                "direction":    "bullish",
                "streak_len":   streak,
                "pivot_bar":    pivot_bar,
                "trigger_bar":  trigger_bar,
                "pivot_time":   df.index[-2],
                "trigger_time": df.index[-1],
            }

    if is_red(pivot_bar):
        streak = count_streak(3, is_green)
        if streak >= min_streak and float(trigger_bar["close"]) < float(pivot_bar["low"]):
            return {
                "direction":    "bearish",
                "streak_len":   streak,
                "pivot_bar":    pivot_bar,
                "trigger_bar":  trigger_bar,
                "pivot_time":   df.index[-2],
                "trigger_time": df.index[-1],
            }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSIST TRIGGER
# ═══════════════════════════════════════════════════════════════════════════════

def persist_trigger(entry: dict, timeframe: str, result: dict,
                    df_daily, df_intraday):
    """Build full trigger record for a classic pivot and save it."""
    pb = result["pivot_bar"]
    tb = result["trigger_bar"]
    d  = result["direction"]

    stop_level    = float(pb["low"]) if d == "bullish" else float(pb["high"])
    trigger_close = float(tb["close"])

    atr     = compute_atr(df_daily)
    r_ratio = compute_r_ratio(trigger_close, stop_level, d, atr)
    rvol    = compute_intraday_rvol(df_intraday)
    dist_sh = compute_session_high_distance(df_daily, trigger_close)
    dist_e8 = compute_ema8w_distance(trigger_close, entry.get("ema8"))

    score = compute_trigger_score(
        conviction=entry["conviction"],
        streak_len=result["streak_len"],
        r_ratio=r_ratio,
        rvol=rvol,
        timeframe=timeframe,
        dist_session_high=dist_sh,
        direction=d,
    )

    trigger = {
        "ticker":              entry["ticker"],
        "timeframe":           timeframe,
        "direction":           d,
        "conviction":          entry["conviction"],
        "trigger_type":        "PIVOT",
        "theme":               entry.get("theme", ""),
        "industry":            entry.get("industry", ""),
        "industry_rank":       entry.get("industry_rank", 99),
        "theme_rank":          entry.get("theme_rank", 99),
        "weekly_stage":        entry.get("weekly_stage"),
        "daily_stage":         entry.get("daily_stage"),
        "trend_template":      entry.get("trend_template"),
        "weekly_bbuw":         entry.get("weekly_bbuw"),
        "daily_bbuw":          entry.get("daily_bbuw"),
        "ep_tier":             entry.get("ep_tier", "NONE"),
        "rs_rating":           entry.get("rs_rating"),
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
        "stop_level":          stop_level,
        "atr_14d":             round(float(atr), 4) if pd.notna(atr) else None,
        "r_ratio":             r_ratio,
        "rvol_trigger":        rvol,
        "dist_session_high_%": dist_sh,
        "dist_ema8w_%":        dist_e8,
        "trigger_score":       score,
        "entry_note":          "ITM/ATM weekly calls" if d == "bullish" else "ITM/ATM weekly puts",
    }

    save_trigger(trigger)
    r_str = f"  R={r_ratio:.1f}" if r_ratio is not None else ""
    log.info(f"  💾 {entry['ticker']} {timeframe} {d.upper()}"
             f"  score={score}{r_str}  rvol={rvol:.1f}×"
             f"  streak={result['streak_len']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN  (v4.3 — 30-min pivot only)
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting INTRADAY pivot scanner v4.3 (30-min pivot, batch downloads)...")

    daily_data    = get_latest_daily_watchlist()
    entries       = daily_data.get("entries", [])
    pivot_entries = [e for e in entries if e["conviction"] in ALERT_TIERS]

    if not pivot_entries:
        log.error("No entries in daily watchlist. Run daily_screener.py first.")
        return

    log.info(f"Watchlist: {len(pivot_entries)} tickers ({ALERT_TIERS})")

    tickers = list({e["ticker"] for e in pivot_entries})

    # ── 2 batch downloads ─────────────────────────────────────────────────
    log.info("Batch downloading daily bars...")
    daily_bars = batch_fetch_daily(tickers)
    time.sleep(RATE_LIMIT_PAUSE)

    log.info("Batch downloading 30m bars...")
    bars_30m = batch_fetch_intraday(tickers, "30m")

    # ── 30-min pivot scan ─────────────────────────────────────────────────
    fired        = set()
    trigger_count = 0

    log.info(f"─── PIVOT SCAN ({len(pivot_entries)} tickers) ───")
    for entry in pivot_entries:
        ticker = entry["ticker"]
        df_30  = bars_30m.get(ticker)
        if df_30 is None:
            continue
        r = detect_pivot(df_30)
        if r:
            fired.add(ticker)
            persist_trigger(entry, "30-MIN", r, daily_bars.get(ticker), df_30)
            trigger_count += 1

    log.info(f"\n  SCAN COMPLETE")
    log.info(f"    Triggers : {trigger_count}")
    log.info(f"    API calls: 2 batch downloads (daily + 30m)")


if __name__ == "__main__":
    main()
