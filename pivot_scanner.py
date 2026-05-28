"""
INTRADAY PIVOT SCANNER (v5.1) — Cross-day pivot detection + scan-window.

Signal types
  PIVOT       — Classic 30-min streak reversal (3+ bar contraction → breakout)
                Streaks may span session boundaries (cross-day pivots supported).
  RANGE_BREAK — First-hour range breakout (9:30–10:30 ET range broken at bar 3+)
  CONTINUATION— EOD grade on all triggers: STRONG / HOLDING / FAILED

v5.1 changes (cross-day pivot support)
  1. detect_pivot now scans a configurable PIVOT_SCAN_WINDOW (default 4) to
     find the pivot bar — previously hardcoded to iloc[-2] which forced the
     pivot to be the most recent bar before the trigger. Now the trigger bar
     is still always the most-recently-closed bar (iloc[-1]), but the pivot
     can sit 1–4 positions back, enabling morning triggers where today's
     opening bars haven't yet formed a clean reversal bar.
  2. Streak lookback cap raised from 10 → STREAK_LOOKBACK (default 20) bars.
     At 30m, 20 bars = ~10 trading hours (>1 full session), so the streak
     search can fully reach into yesterday's bars.
  3. detect_pivot result now includes `is_cross_day` (bool) and
     `streak_start_date` so the dashboard and logs can surface cross-day setups.

Speed improvements vs v4.3
  1. batch_fetch_daily REMOVED — ATR, prev_close, high_20d pre-computed in
     daily_screener.py and stored in watchlist JSON. Saves ~6-8s per run.
  2. Intraday period changed from "60d" → "5d" — ~12× faster download.
  3. RATE_LIMIT_PAUSE reduced from 2s → 0.5s (batch downloads, not serial).

Schedule (GitHub Actions)
  Intraday: M-F :01/:31 14:00-21:31 UTC — all signals
  Premarket: M-F 13:25 UTC — premarket_gap_scan.py (separate script)
"""

import os
import time
import logging
from datetime import datetime, timezone, date
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from data_layer import (
    save_trigger,
    save_trigger_continuations,
    get_today_triggers,
    get_latest_daily_watchlist,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MIN_STREAK         = 3        # bars of contraction before pivot fires
RATE_LIMIT_PAUSE   = 0.5      # seconds between batch chunks (was 2s)
ALERT_TIERS        = ["HIGH", "MED"]

# Pivot scan window — how many positions back to search for the pivot bar.
# 1 = old behaviour (pivot must be immediately before trigger at iloc[-2]).
# 4 = allows the pivot bar to sit up to 4 positions before the trigger,
#     so morning setups where today's early bars haven't yet formed a clean
#     reversal can still resolve off a cross-day pivot.
PIVOT_SCAN_WINDOW  = 4

# Max bars to look back when counting the streak (raised from 10 → 20).
# At 30m cadence, 20 bars ≈ 10 trading hours (~1.5 sessions), ensuring the
# streak search reaches fully into the previous day's bars.
STREAK_LOOKBACK    = 20

# R-ratio thresholds (matched to dashboard display)
R_ELITE   = 3.0
R_GOOD    = 2.0
R_MINIMUM = 1.0

# Range break RVOL minimum
RANGE_BREAK_MIN_RVOL = 1.3

# Intraday batch chunk size
BATCH_CHUNK = 200

# Time gates (UTC hours)
# Range break: fires after the first-hour (9:30-10:30 ET) closes
# = UTC 15:00 onwards (10:30 ET + 30min grace) until 16:30 UTC
RANGE_BREAK_UTC_START = 15
RANGE_BREAK_UTC_END   = 16

# Continuation grader: after market close = 21:00+ UTC
CONTINUATION_UTC_START = 21

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH DATA FETCHING  (Speed 2: period="5d" intraday)
# ═══════════════════════════════════════════════════════════════════════════════

def _slice_ticker(raw: pd.DataFrame, ticker: str, min_rows: int = 5) -> Optional[pd.DataFrame]:
    """Extract single ticker from yf.download() MultiIndex result."""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw.xs(ticker, axis=1, level=1).copy()
        else:
            df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(how="all")
        if len(df) < min_rows:
            return None
        return df
    except (KeyError, Exception):
        return None


def batch_fetch_intraday(tickers: list, interval: str = "30m") -> dict:
    """
    Download 5-day intraday bars — 12x faster than the previous 60d fetch.
    5d gives ~65 bars of 30m data — more than enough for pattern detection.
    """
    result = {}
    for i in range(0, len(tickers), BATCH_CHUNK):
        chunk = tickers[i: i + BATCH_CHUNK]
        try:
            raw = yf.download(
                chunk, period="5d", interval=interval,
                auto_adjust=True, progress=False, threads=True,
            )
            if raw.empty:
                continue
            if raw.index.tzinfo is None:
                raw.index = raw.index.tz_localize("UTC")
            else:
                raw.index = raw.index.tz_convert("UTC")
            raw.index = raw.index.tz_localize(None)
            for tk in chunk:
                df = _slice_ticker(raw, tk, min_rows=5)
                if df is not None:
                    result[tk] = df.sort_index().tail(100)
        except Exception as e:
            log.warning(f"batch_fetch_intraday({interval}) chunk error: {e}")
        if i + BATCH_CHUNK < len(tickers):
            time.sleep(RATE_LIMIT_PAUSE)
    log.info(f"  Intraday {interval} batch: {len(result)}/{len(tickers)} tickers  (5d period)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS  (Speed 1: ATR + session high now read from pre-computed entry dict)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_r_ratio(trigger_close: float, stop_level: float,
                    direction: str, atr) -> Optional[float]:
    if atr is None or (isinstance(atr, float) and np.isnan(atr)) or atr <= 0:
        return None
    risk = trigger_close - stop_level if direction == "bullish" else stop_level - trigger_close
    return round(risk / float(atr), 2)


def compute_ema8w_distance(trigger_close: float, ema8: Optional[float]) -> Optional[float]:
    if ema8 is None or ema8 <= 0:
        return None
    return round((trigger_close / ema8 - 1) * 100, 2)


def compute_intraday_rvol(df_intraday: pd.DataFrame) -> float:
    if df_intraday is None or len(df_intraday) < 5:
        return 1.0
    avg_vol = df_intraday["volume"].iloc[-21:-1].mean()
    trigger_vol = df_intraday["volume"].iloc[-1]
    if avg_vol <= 0:
        return 1.0
    return round(float(trigger_vol / avg_vol), 2)


def compute_trigger_score(conviction: str, streak_len: int,
                           r_ratio: Optional[float], rvol: float,
                           timeframe: str, dist_session_high: Optional[float],
                           direction: str, trigger_type: str = "PIVOT") -> int:
    score = 0
    score += {"HIGH": 4, "MED": 2}.get(conviction, 0)
    # Streak scoring (N/A for non-PIVOT types)
    if trigger_type == "PIVOT":
        if streak_len >= 5:   score += 3
        elif streak_len >= 4: score += 2
        else:                  score += 1
    else:
        score += 1  # flat bonus for non-streak signals
    if r_ratio is not None:
        if r_ratio >= R_ELITE:     score += 3
        elif r_ratio >= R_GOOD:    score += 2
        elif r_ratio >= R_MINIMUM: score += 1
    if rvol >= 2.0:   score += 2
    elif rvol >= 1.5: score += 1
    score += 1  # timeframe base (all now 30-MIN)
    if dist_session_high is not None:
        if direction == "bullish" and dist_session_high >= -1.0: score += 1
        elif direction == "bearish" and dist_session_high <= -5.0: score += 1
    if trigger_type == "RANGE_BREAK": score += 1  # range break quality bonus
    return score


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 1 — CLASSIC PIVOT (streak reversal)
# ═══════════════════════════════════════════════════════════════════════════════

def is_green(row): return row["close"] > row["open"]
def is_red(row):   return row["close"] <= row["open"]


def detect_pivot(df: pd.DataFrame,
                 min_streak: int = MIN_STREAK,
                 scan_window: int = PIVOT_SCAN_WINDOW) -> Optional[dict]:
    """
    Detect a classic 30-min streak-reversal pivot.

    Cross-day behaviour (v5.1)
    --------------------------
    * The trigger bar is always df.iloc[-1] (most recently closed bar).
    * The pivot bar is searched across the last `scan_window` positions
      (df.iloc[-2] … df.iloc[-(scan_window+1)]).  This means a pivot that
      formed at end-of-yesterday followed by today's opening bars is caught
      without requiring today's first bar itself to be the pivot.
    * The streak is counted backward from the bar immediately before the
      candidate pivot, capped at STREAK_LOOKBACK (20) bars rather than the
      old hard limit of 10.  At 30m cadence, 20 bars covers > 1 full session.

    The result dict includes `is_cross_day` (True when at least one streak
    bar or the pivot bar belongs to a different calendar date than the trigger)
    and `streak_start_date` for logging.
    """
    # Need at least: trigger(1) + pivot(1) + min_streak bars
    min_needed = min_streak + 2
    if len(df) < min_needed:
        return None
    df = df[df["close"] > 0].copy()
    if len(df) < min_needed:
        return None

    trigger_bar  = df.iloc[-1]
    trigger_date = df.index[-1].date() if hasattr(df.index[-1], "date") else None

    def count_streak(streak_start_pos: int, qualifier) -> int:
        """Count consecutive qualifying bars going backward from streak_start_pos."""
        count = 0
        # Upper bound is len(df)+1 so df.iloc[-len(df)] (the very first bar) is reachable.
        for i in range(streak_start_pos, min(streak_start_pos + STREAK_LOOKBACK, len(df) + 1)):
            if qualifier(df.iloc[-i]):
                count += 1
            else:
                break
        return count

    def oldest_streak_date(streak_start_pos: int, qualifier) -> Optional[date]:
        """Return the calendar date of the OLDEST (furthest-back) qualifying streak bar."""
        last_date = None
        for i in range(streak_start_pos, min(streak_start_pos + STREAK_LOOKBACK, len(df) + 1)):
            if qualifier(df.iloc[-i]):
                ts = df.index[-i]
                last_date = ts.date() if hasattr(ts, "date") else None
            else:
                break
        return last_date

    # ── Scan pivot positions from -2 back to -(scan_window+1) ────────────
    for pivot_offset in range(2, scan_window + 2):
        if pivot_offset >= len(df):
            break

        pivot_bar   = df.iloc[-pivot_offset]
        # streak bars start immediately after the pivot (going further back)
        streak_pos  = pivot_offset + 1

        if is_green(pivot_bar):
            streak = count_streak(streak_pos, is_red)
            if streak >= min_streak and float(trigger_bar["close"]) > float(pivot_bar["high"]):
                pivot_date  = df.index[-pivot_offset].date() if hasattr(df.index[-pivot_offset], "date") else None
                sd          = oldest_streak_date(streak_pos, is_red)
                cross_day   = bool(
                    trigger_date and pivot_date and (
                        pivot_date < trigger_date or (sd and sd < trigger_date)
                    )
                )
                return {
                    "direction":        "bullish",
                    "streak_len":       streak,
                    "pivot_bar":        pivot_bar,
                    "trigger_bar":      trigger_bar,
                    "pivot_time":       df.index[-pivot_offset],
                    "trigger_time":     df.index[-1],
                    "is_cross_day":     cross_day,
                    "streak_start_date": str(sd) if sd else None,
                    "pivot_offset":     pivot_offset,   # 2 = immediately before trigger
                }

        if is_red(pivot_bar):
            streak = count_streak(streak_pos, is_green)
            if streak >= min_streak and float(trigger_bar["close"]) < float(pivot_bar["low"]):
                pivot_date  = df.index[-pivot_offset].date() if hasattr(df.index[-pivot_offset], "date") else None
                sd          = oldest_streak_date(streak_pos, is_green)
                cross_day   = bool(
                    trigger_date and pivot_date and (
                        pivot_date < trigger_date or (sd and sd < trigger_date)
                    )
                )
                return {
                    "direction":        "bearish",
                    "streak_len":       streak,
                    "pivot_bar":        pivot_bar,
                    "trigger_bar":      trigger_bar,
                    "pivot_time":       df.index[-pivot_offset],
                    "trigger_time":     df.index[-1],
                    "is_cross_day":     cross_day,
                    "streak_start_date": str(sd) if sd else None,
                    "pivot_offset":     pivot_offset,
                }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 2 — FIRST-HOUR RANGE BREAK
# ═══════════════════════════════════════════════════════════════════════════════

def detect_first_hour_range_break(df_30m: pd.DataFrame,
                                   min_rvol: float = RANGE_BREAK_MIN_RVOL) -> Optional[dict]:
    """
    Detect a breakout above / breakdown below the first-hour trading range.
    First hour = bars with open time 13:30–14:30 UTC (9:30–10:30 ET).
    Fires when a LATER bar closes outside that range with RVOL ≥ min_rvol.

    Intended to run at 15:01+ UTC (first bar AFTER first hour closes).
    Fires once per session — main() guards against re-firing.
    """
    if df_30m is None or len(df_30m) < 3:
        return None

    today = date.today()

    # Split bars into first-hour (13:30–14:29 UTC) and post-first-hour
    first_hour, post_fh = [], []
    for ts, row in df_30m.iterrows():
        # Handle tz-naive index
        bar_date = ts.date() if hasattr(ts, "date") else ts
        if bar_date != today:
            continue
        bar_hour = ts.hour if hasattr(ts, "hour") else 0
        bar_min  = ts.minute if hasattr(ts, "minute") else 0
        utc_mins = bar_hour * 60 + bar_min
        if 13 * 60 + 30 <= utc_mins < 14 * 60 + 30:   # 13:30–14:29 UTC
            first_hour.append(row)
        elif utc_mins >= 14 * 60 + 30:                  # 14:30 UTC onward
            post_fh.append(row)

    if len(first_hour) < 1 or len(post_fh) < 1:
        return None

    fh_df   = pd.DataFrame(first_hour)
    or_high = float(fh_df["high"].max())
    or_low  = float(fh_df["low"].min())
    or_range = round(or_high - or_low, 4)

    # Check the most recent post-first-hour bar
    current = post_fh[-1]
    current_close = float(current["close"])

    rvol = compute_intraday_rvol(df_30m)
    if rvol < min_rvol:
        return None

    if current_close > or_high:
        return {
            "direction":    "bullish",
            "or_high":      or_high,
            "or_low":       or_low,
            "or_range":     or_range,
            "breakout_pct": round((current_close / or_high - 1) * 100, 2),
            "trigger_bar":  current,
            "trigger_time": df_30m.index[-1],
            "rvol":         rvol,
        }
    if current_close < or_low:
        return {
            "direction":    "bearish",
            "or_high":      or_high,
            "or_low":       or_low,
            "or_range":     or_range,
            "breakout_pct": round((1 - current_close / or_low) * 100, 2),
            "trigger_bar":  current,
            "trigger_time": df_30m.index[-1],
            "rvol":         rvol,
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 3 — END-OF-SESSION CONTINUATION GRADER
# ═══════════════════════════════════════════════════════════════════════════════

def run_continuation_grader(bars_30m: dict):
    """
    After market close (called when UTC hour >= 21), grade all of today's
    triggers: did price hold, extend, or fail from the trigger close?

    STRONG  — held trigger level AND closed within 2% of session high/low
    HOLDING — held trigger level, closed in the middle
    FAILED  — reversed back through the trigger close level

    Saves to data/trigger_continuations.json for dashboard enrichment.
    """
    today_triggers = get_today_triggers()
    if not today_triggers:
        log.info("  Continuation grader: no triggers today")
        return

    grades = {}
    graded = 0

    for t in today_triggers:
        ticker        = t.get("ticker")
        direction     = t.get("direction")
        trigger_close = t.get("trigger_close")
        if not ticker or not direction or trigger_close is None:
            continue

        df_30m = bars_30m.get(ticker)
        if df_30m is None or df_30m.empty:
            continue

        # Today's bars only
        today = date.today()
        today_bars = df_30m[[r.date() == today for r in df_30m.index]]
        if today_bars.empty:
            continue

        eod_close  = float(today_bars["close"].iloc[-1])
        session_hi = float(today_bars["high"].max())
        session_lo = float(today_bars["low"].min())
        pct_chg    = round((eod_close / trigger_close - 1) * 100, 2)

        if direction == "bullish":
            held       = eod_close >= trigger_close
            at_extreme = session_hi > 0 and eod_close >= session_hi * 0.98
        else:
            held       = eod_close <= trigger_close
            at_extreme = session_lo > 0 and eod_close <= session_lo * 1.02

        if held and at_extreme:
            grade = "STRONG"
            note  = f"Held + closed at extreme · {pct_chg:+.1f}%"
        elif held:
            grade = "HOLDING"
            note  = f"Above trigger level · {pct_chg:+.1f}%"
        else:
            grade = "FAILED"
            note  = f"Reversed through trigger · {pct_chg:+.1f}%"

        key = f"{ticker}_{direction}"
        grades[key] = {
            "ticker":        ticker,
            "direction":     direction,
            "trigger_type":  t.get("trigger_type", "PIVOT"),
            "trigger_close": trigger_close,
            "eod_close":     round(eod_close, 4),
            "session_change_pct": pct_chg,
            "held_trigger":  held,
            "grade":         grade,
            "grade_note":    note,
        }
        graded += 1

    save_trigger_continuations({
        "scan_time": datetime.now().isoformat(),
        "grades":    grades,
    })
    counts = {g: sum(1 for v in grades.values() if v["grade"] == g)
              for g in ("STRONG", "HOLDING", "FAILED")}
    log.info(f"  Continuation grades: {graded} triggers — "
             f"STRONG={counts['STRONG']} HOLDING={counts['HOLDING']} FAILED={counts['FAILED']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PERSIST TRIGGER  (Speed 1: df_daily removed — reads pre-computed entry fields)
# ═══════════════════════════════════════════════════════════════════════════════

def persist_trigger(entry: dict, result: dict, df_intraday: pd.DataFrame,
                    trigger_type: str = "PIVOT"):
    """
    Build a full trigger record and save it.
    Works for both PIVOT (streak reversal) and RANGE_BREAK signals.
    """
    d = result["direction"]
    tb = result["trigger_bar"]
    trigger_close = float(tb["close"])

    # ── Stop level ────────────────────────────────────────────────────────
    if trigger_type == "PIVOT":
        pb = result["pivot_bar"]
        stop_level = float(pb["low"]) if d == "bullish" else float(pb["high"])
    else:  # RANGE_BREAK — stop is the other side of the opening range
        stop_level = result["or_low"] if d == "bullish" else result["or_high"]

    # ── Metrics from pre-computed entry fields (Speed 1 — no daily fetch) ─
    atr     = entry.get("atr_14d")
    high_20d = entry.get("high_20d")
    dist_sh = round((trigger_close / high_20d - 1) * 100, 2) if high_20d else None
    dist_e8 = compute_ema8w_distance(trigger_close, entry.get("ema8"))
    r_ratio = compute_r_ratio(trigger_close, stop_level, d, atr)
    rvol    = result.get("rvol") or compute_intraday_rvol(df_intraday)

    score = compute_trigger_score(
        conviction=entry["conviction"],
        streak_len=result.get("streak_len", 0),
        r_ratio=r_ratio,
        rvol=rvol,
        timeframe="30-MIN",
        dist_session_high=dist_sh,
        direction=d,
        trigger_type=trigger_type,
    )

    # ── Pivot OHLC — pivot bar for PIVOT, OR levels for RANGE_BREAK ───────
    if trigger_type == "PIVOT":
        pb = result["pivot_bar"]
        p_open, p_high, p_low, p_close = (
            float(pb["open"]), float(pb["high"]),
            float(pb["low"]),  float(pb["close"]))
        p_time = str(result.get("pivot_time", ""))
    else:
        p_open  = result["or_low"]
        p_high  = result["or_high"]
        p_low   = result["or_low"]
        p_close = result["or_high"]
        p_time  = str(result.get("trigger_time", ""))

    trigger = {
        "ticker":               entry["ticker"],
        "timeframe":            "30-MIN",
        "direction":            d,
        "conviction":           entry["conviction"],
        "trigger_type":         trigger_type,
        "theme":                entry.get("theme", ""),
        "industry":             entry.get("industry", ""),
        "industry_rank":        entry.get("industry_rank", 99),
        "theme_rank":           entry.get("theme_rank", 99),
        "weekly_stage":         entry.get("weekly_stage"),
        "daily_stage":          entry.get("daily_stage"),
        "trend_template":       entry.get("trend_template"),
        "weekly_bbuw":          entry.get("weekly_bbuw"),
        "daily_bbuw":           entry.get("daily_bbuw"),
        "ep_tier":              entry.get("ep_tier", "NONE"),
        "rs_rating":            entry.get("rs_rating"),
        "streak_len":           result.get("streak_len", 0),
        "pivot_open":           p_open,
        "pivot_high":           p_high,
        "pivot_low":            p_low,
        "pivot_close":          p_close,
        "pivot_time":           p_time,
        "trigger_open":         float(tb["open"]),
        "trigger_high":         float(tb["high"]),
        "trigger_low":          float(tb["low"]),
        "trigger_close":        trigger_close,
        "trigger_time":         str(result.get("trigger_time", df_intraday.index[-1])),
        "stop_level":           round(stop_level, 4),
        "atr_14d":              round(float(atr), 4) if atr else None,
        "r_ratio":              r_ratio,
        "rvol_trigger":         rvol,
        "dist_session_high_%":  dist_sh,
        "dist_ema8w_%":         dist_e8,
        "trigger_score":        score,
        # Range break specific (None for PIVOT)
        "or_high":              result.get("or_high"),
        "or_low":               result.get("or_low"),
        "or_range":             result.get("or_range"),
        "breakout_pct":         result.get("breakout_pct"),
        "entry_note":           (
            f"First-hour range break · OR {result.get('or_low',0):.2f}–{result.get('or_high',0):.2f}"
            if trigger_type == "RANGE_BREAK"
            else ("ITM/ATM weekly calls" if d == "bullish" else "ITM/ATM weekly puts")
        ),
        # Cross-day metadata (PIVOT only; None for RANGE_BREAK)
        "is_cross_day":         result.get("is_cross_day"),
        "streak_start_date":    result.get("streak_start_date"),
        "pivot_offset":         result.get("pivot_offset"),
    }

    save_trigger(trigger)
    r_str       = f"  R={r_ratio:.1f}" if r_ratio is not None else ""
    xday_str    = "  🗓 CROSS-DAY" if result.get("is_cross_day") else ""
    offset_str  = (f"  offset={result.get('pivot_offset','?')}"
                   if result.get("pivot_offset", 2) != 2 else "")
    log.info(f"  💾 [{trigger_type}] {entry['ticker']} {d.upper()}"
             f"  score={score}{r_str}  rvol={rvol:.1f}×"
             f"  streak={result.get('streak_len','-')}{offset_str}{xday_str}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    now_utc = datetime.now(timezone.utc)
    is_range_break_window = (
        RANGE_BREAK_UTC_START <= now_utc.hour < RANGE_BREAK_UTC_END + 1
    )
    is_eod_window = (now_utc.hour >= CONTINUATION_UTC_START)

    log.info(
        f"Starting INTRADAY scanner v5.1  "
        f"[PIVOT=✓  RANGE_BREAK={'✓' if is_range_break_window else '—'}  "
        f"CONTINUATION={'✓' if is_eod_window else '—'}]"
    )

    daily_data    = get_latest_daily_watchlist()
    entries       = daily_data.get("entries", [])
    pivot_entries = [e for e in entries if e["conviction"] in ALERT_TIERS]

    if not pivot_entries:
        log.error("No entries in daily watchlist. Run daily_screener.py first.")
        return

    tickers = list({e["ticker"] for e in pivot_entries})
    log.info(f"Watchlist: {len(pivot_entries)} tickers ({ALERT_TIERS})")

    # ── SINGLE batch download (Speed 1+2: daily bars removed, 5d intraday) ─
    log.info("Batch downloading 30m bars (5d)...")
    bars_30m = batch_fetch_intraday(tickers, "30m")

    # ── Dedup guards — prevent re-firing same signal same session ─────────
    today_triggers = get_today_triggers()
    fired_pivots = {
        (t["ticker"], t.get("direction"))
        for t in today_triggers if t.get("trigger_type") == "PIVOT"
    }
    fired_range_breaks = {
        (t["ticker"], t.get("direction"))
        for t in today_triggers if t.get("trigger_type") == "RANGE_BREAK"
    }

    pivot_count = range_break_count = 0

    # ── Signal 1: PIVOT scan ──────────────────────────────────────────────
    log.info(f"─── PIVOT SCAN ({len(pivot_entries)} tickers) ───")
    for entry in pivot_entries:
        ticker = entry["ticker"]
        df_30  = bars_30m.get(ticker)
        if df_30 is None:
            continue
        r = detect_pivot(df_30)
        if r:
            key = (ticker, r["direction"])
            if key not in fired_pivots:
                persist_trigger(entry, r, df_30, trigger_type="PIVOT")
                fired_pivots.add(key)
                pivot_count += 1

    # ── Signal 2: RANGE_BREAK scan (time-gated) ───────────────────────────
    if is_range_break_window:
        log.info(f"─── RANGE BREAK SCAN ({len(pivot_entries)} tickers) ───")
        for entry in pivot_entries:
            ticker = entry["ticker"]
            df_30  = bars_30m.get(ticker)
            if df_30 is None:
                continue
            r = detect_first_hour_range_break(df_30)
            if r:
                key = (ticker, r["direction"])
                if key not in fired_range_breaks:
                    persist_trigger(entry, r, df_30, trigger_type="RANGE_BREAK")
                    fired_range_breaks.add(key)
                    range_break_count += 1
    else:
        log.info(f"  Range break window: not active (UTC {now_utc.hour}:{now_utc.minute:02d})")

    # ── Signal 3: CONTINUATION GRADER (EOD only) ─────────────────────────
    if is_eod_window:
        log.info("─── EOD CONTINUATION GRADER ───")
        run_continuation_grader(bars_30m)

    log.info(
        f"\n  SCAN COMPLETE — "
        f"PIVOT={pivot_count}  RANGE_BREAK={range_break_count}  "
        f"API calls: 1 batch (30m 5d)"
    )


if __name__ == "__main__":
    main()
