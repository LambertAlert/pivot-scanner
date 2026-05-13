"""
INTRADAY PIVOT SCANNER (v4.1) — R-ratio, key levels, RVOL, composite score, ORB.

Triggers:
  1. Classic 30/65-min pivot (3+ bar streak reversal)
  2. Opening Range Breakout (ORB) — HIGH conviction only
     First 30-min bar sets range; second bar breaks out with volume confirmation.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pytz

from data_layer import save_trigger, get_latest_daily_watchlist


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MIN_STREAK       = 3
RATE_LIMIT_PAUSE = 2
SCAN_30MIN       = True
SCAN_65MIN       = True
SCAN_ORB         = True          # Opening Range Breakout scan
ALERT_TIERS      = ["HIGH", "MED"]
ORB_TIERS        = ["HIGH", "MED"]  # ORB fires on HIGH + MED (OR logic with pivot)

# R-ratio thresholds for scoring
R_ELITE   = 3.0
R_GOOD    = 2.0
R_MINIMUM = 1.0

# ORB config
ORB_RVOL_MIN      = 1.0          # breakout bar must be ≥ 1.0× average volume (relaxed)
ORB_OPEN_HOUR_CT  = 9            # market open hour (CT)
ORB_OPEN_MIN_CT   = 30           # market open minute (CT)
CT_TZ             = pytz.timezone("America/Chicago")

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
    Composite score for ranking triggers. Max ~15 points.
    """
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
#  OPENING RANGE BREAKOUT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_orb(df_30m: pd.DataFrame) -> Optional[dict]:
    """
    Opening Range Breakout on the 30-minute timeframe.

    Definition:
      • Bar 1 (9:30–10:00 CT) = the opening range. High = OR high, Low = OR low.
      • Bar 2 (10:00–10:30 CT) = the trigger bar.
        - Bullish ORB: bar 2 closes ABOVE bar 1's high AND volume ≥ ORB_RVOL_MIN × 20-bar avg
        - Bearish ORB: bar 2 closes BELOW bar 1's low  AND volume ≥ ORB_RVOL_MIN × 20-bar avg

    Only fires if we're currently in bar 2 (i.e. last bar IS bar 2 of today).
    This prevents re-firing on subsequent bars after the ORB already triggered.

    Returns dict with direction, range, breakout bar, and volume confirmation.
    Returns None if no ORB or conditions not met.
    """
    if df_30m is None or len(df_30m) < 3:
        return None

    # Convert index to CT for open detection
    try:
        idx_ct = df_30m.index.tz_convert(CT_TZ)
    except Exception:
        idx_ct = df_30m.index

    today_ct = datetime.now(CT_TZ).date()

    # Find today's bars
    today_mask = pd.Series([i.date() == today_ct for i in idx_ct], index=df_30m.index)
    today_bars = df_30m[today_mask]

    if len(today_bars) < 2:
        return None  # Need at least 2 bars today

    # Bar 1 = first 30-min bar of the day (opening range)
    bar1 = today_bars.iloc[0]
    bar1_time = idx_ct[today_mask][0]

    # Loose timing check — bar1 should start between 9:00 and 10:00 CT
    # (allows for DST offsets and yfinance rounding)
    if bar1_time.hour < 9 or bar1_time.hour >= 10:
        return None  # First bar is not in opening hour

    # Bar 2 = second 30-min bar — the breakout confirmation bar
    # We check bar 2 regardless of how many bars exist today
    # (ORB is evaluated once: did bar 2 break the range?)
    bar2 = today_bars.iloc[1]

    or_high = float(bar1["high"])
    or_low  = float(bar1["low"])
    or_range = or_high - or_low

    if or_range <= 0:
        return None

    # Volume confirmation
    avg_vol_20 = df_30m["volume"].iloc[-22:-2].mean()
    bar2_rvol  = float(bar2["volume"] / avg_vol_20) if avg_vol_20 > 0 else 1.0

    if bar2_rvol < ORB_RVOL_MIN:
        log.info(f"  ORB: volume low ({bar2_rvol:.2f}× < {ORB_RVOL_MIN}×) — skipping")
        return None

    bar2_close = float(bar2["close"])

    # Bullish ORB
    if bar2_close > or_high:
        breakout_pct = (bar2_close - or_high) / or_high * 100
        return {
            "direction":     "bullish",
            "or_high":       round(or_high, 2),
            "or_low":        round(or_low, 2),
            "or_range":      round(or_range, 2),
            "breakout_pct":  round(breakout_pct, 2),
            "bar1":          bar1,
            "bar2":          bar2,
            "bar1_time":     str(df_30m.index[today_mask][0]),
            "bar2_time":     str(df_30m.index[today_mask][1]),
            "rvol":          round(bar2_rvol, 2),
            "stop_level":    round(or_low, 2),   # stop = bottom of opening range
        }

    # Bearish ORB
    if bar2_close < or_low:
        breakout_pct = (or_low - bar2_close) / or_low * 100
        return {
            "direction":     "bearish",
            "or_high":       round(or_high, 2),
            "or_low":        round(or_low, 2),
            "or_range":      round(or_range, 2),
            "breakout_pct":  round(breakout_pct, 2),
            "bar1":          bar1,
            "bar2":          bar2,
            "bar1_time":     str(df_30m.index[today_mask][0]),
            "bar2_time":     str(df_30m.index[today_mask][1]),
            "rvol":          round(bar2_rvol, 2),
            "stop_level":    round(or_high, 2),  # stop = top of opening range
        }

    return None  # Bar 2 inside the range — no ORB


# ═══════════════════════════════════════════════════════════════════════════════
#  PIVOT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def is_green(row): return row["close"] > row["open"]
def is_red(row):   return row["close"] <= row["open"]


def detect_pivot(df: pd.DataFrame,
                 min_streak: int = MIN_STREAK) -> Optional[dict]:
    """
    Detect a pivot reversal signal.
    Bullish: pivot bar is green, preceded by ≥ min_streak red bars,
             trigger bar closes above pivot bar's high.
    Bearish: mirror image.
    Streak counted conservatively — stops at first non-qualifying bar.
    """
    if len(df) < min_streak + 2:
        return None

    # Drop zero-close bars before evaluation
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

    # Bullish: pivot green, preceded by red bars, trigger breaks above pivot high
    if is_green(pivot_bar):
        streak = count_streak(3, is_red)
        if streak >= min_streak and float(trigger_bar["close"]) > float(pivot_bar["high"]):
            return {
                "direction":   "bullish",
                "streak_len":  streak,
                "pivot_bar":   pivot_bar,
                "trigger_bar": trigger_bar,
                "pivot_time":  df.index[-2],
                "trigger_time":df.index[-1],
            }

    # Bearish: pivot red, preceded by green bars, trigger breaks below pivot low
    if is_red(pivot_bar):
        streak = count_streak(3, is_green)
        if streak >= min_streak and float(trigger_bar["close"]) < float(pivot_bar["low"]):
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


def persist_orb_trigger(entry: dict, result: dict,
                         df_daily: Optional[pd.DataFrame]):
    """Build and save an ORB trigger record."""
    d  = result["direction"]
    b2 = result["bar2"]

    trigger_close = float(b2["close"])
    stop_level    = result["stop_level"]

    atr     = compute_atr(df_daily)
    r_ratio = compute_r_ratio(trigger_close, stop_level, d, atr)

    dist_sh = compute_session_high_distance(df_daily, trigger_close)
    dist_e8 = compute_ema8w_distance(trigger_close, entry.get("ema8"))

    # ORB score — starts higher because it's a cleaner, tighter setup
    # Base: HIGH conviction = 5 (bonus over pivot), RVOL premium, R premium
    orb_base = 5  # ORB on HIGH always starts higher than standard pivot
    score = orb_base
    if result["rvol"] >= 2.0:   score += 2
    elif result["rvol"] >= 1.5: score += 1
    if r_ratio is not None:
        if r_ratio >= R_ELITE:     score += 3
        elif r_ratio >= R_GOOD:    score += 2
        elif r_ratio >= R_MINIMUM: score += 1
    # Tight opening range = cleaner signal
    or_pct = result["or_range"] / result["or_high"] * 100
    if or_pct < 1.0: score += 2   # very tight range = strong conviction
    elif or_pct < 2.0: score += 1

    entry_note = (
        "ORB — ITM/ATM weekly calls (stop: OR low)"
        if d == "bullish"
        else "ORB — ITM/ATM weekly puts (stop: OR high)"
    )

    trigger = {
        "ticker":              entry["ticker"],
        "timeframe":           "ORB-30",
        "direction":           d,
        "conviction":          entry["conviction"],
        "trigger_type":        "ORB",          # distinguishes from pivot triggers
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
        # ORB-specific fields
        "streak_len":          0,              # N/A for ORB
        "or_high":             result["or_high"],
        "or_low":              result["or_low"],
        "or_range":            result["or_range"],
        "breakout_pct":        result["breakout_pct"],
        # Bar data
        "pivot_open":          float(result["bar1"]["open"]),
        "pivot_high":          result["or_high"],
        "pivot_low":           result["or_low"],
        "pivot_close":         float(result["bar1"]["close"]),
        "pivot_time":          result["bar1_time"],
        "trigger_open":        float(b2["open"]),
        "trigger_high":        float(b2["high"]),
        "trigger_low":         float(b2["low"]),
        "trigger_close":       trigger_close,
        "trigger_time":        result["bar2_time"],
        # Risk metrics
        "stop_level":          stop_level,
        "atr_14d":             round(atr, 4) if pd.notna(atr) else None,
        "r_ratio":             r_ratio,
        "rvol_trigger":        result["rvol"],
        # Key levels
        "dist_session_high_%": dist_sh,
        "dist_ema8w_%":        dist_e8,
        "trigger_score":       score,
        "entry_note":          entry_note,
    }

    save_trigger(trigger)
    r_str = f"  R={r_ratio:.1f}" if r_ratio is not None else ""
    log.info(f"  🎯 ORB {d.upper()}: {entry['ticker']}"
             f"  OR={result['or_high']:.2f}/{result['or_low']:.2f}"
             f"  rvol={result['rvol']:.1f}×  +{result['breakout_pct']:.1f}%{r_str}"
             f"  score={score}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting INTRADAY pivot scanner v4.1 (pivot + ORB)...")

    daily_data = get_latest_daily_watchlist()
    entries    = daily_data.get("entries", [])

    orb_entries    = [e for e in entries if e["conviction"] in ORB_TIERS]
    pivot_entries  = [e for e in entries if e["conviction"] in ALERT_TIERS]

    if not pivot_entries:
        log.error("No entries in daily watchlist. Run daily_screener.py first.")
        return

    now_ct = datetime.now(CT_TZ)
    log.info(f"Scanning {len(pivot_entries)} tickers for pivots ({ALERT_TIERS})")
    log.info(f"Scanning {len(orb_entries)} tickers for ORB ({ORB_TIERS})")
    log.info(f"Current time: {now_ct.strftime('%H:%M CT')} — market hours 9:30–16:00 CT")
    log.info(f"ORB window: 10:00–10:30 CT (fires when 2nd 30-min bar is most recent)")

    fired_tickers_30min = set()
    fired_tickers_65min = set()
    fired_orb           = set()
    trigger_count       = 0
    orb_count           = 0

    # ── ORB scan (HIGH conviction only) ───────────────────────────────────────
    # Run first so ORB triggers appear at top of today's list
    if SCAN_ORB:
        log.info("─── ORB SCAN ───")
        for i, entry in enumerate(orb_entries, 1):
            ticker = entry["ticker"]
            log.info(f"  ORB [{i}/{len(orb_entries)}] {ticker}")

            df_30 = fetch_intraday_bars(ticker, "30m")
            if df_30 is None:
                continue

            orb = detect_orb(df_30)
            if orb:
                df_daily = fetch_daily_bars(ticker)
                persist_orb_trigger(entry, orb, df_daily)
                fired_orb.add(ticker)
                orb_count += 1

            if i < len(orb_entries):
                time.sleep(RATE_LIMIT_PAUSE)

    # ── Pivot scan (HIGH + MED conviction) ───────────────────────────────────
    log.info("─── PIVOT SCAN ───")
    for i, entry in enumerate(pivot_entries, 1):
        ticker = entry["ticker"]
        log.info(f"  [{i}/{len(pivot_entries)}] {ticker} ({entry['conviction']})")

        df_daily = fetch_daily_bars(ticker)

        if SCAN_30MIN:
            df_30 = fetch_intraday_bars(ticker, "30m")
            if df_30 is not None:
                r = detect_pivot(df_30)
                if r:
                    fired_tickers_30min.add(ticker)
                    persist_trigger(entry, "30-MIN", r, df_daily, df_30)
                    trigger_count += 1

        if SCAN_65MIN:
            df_60 = fetch_intraday_bars(ticker, "60m")
            if df_60 is not None:
                r = detect_pivot(df_60)
                if r:
                    fired_tickers_65min.add(ticker)
                    persist_trigger(entry, "65-MIN", r, df_daily, df_60)
                    trigger_count += 1

        if i < len(pivot_entries):
            time.sleep(RATE_LIMIT_PAUSE)

    dupes = fired_tickers_30min & fired_tickers_65min
    if dupes:
        log.info(f"  Dupes (both TF): {', '.join(sorted(dupes))}")

    log.info(f"\n  SCAN COMPLETE")
    log.info(f"    ORB triggers   : {orb_count}")
    log.info(f"    Pivot triggers : {trigger_count}")
    log.info(f"    Total          : {orb_count + trigger_count}")
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
