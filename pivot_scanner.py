"""
INTRADAY PIVOT SCANNER (v4.2) — batch downloads, ORB restored, rate-limit fix.

Changes over v4.1:
  - Replaced serial per-ticker yfinance calls with batch yf.download() for
    daily, 30m, and 60m intervals.  189 API calls → 3.  No more rate-limit wall.
  - Removed duplicate main() that was silently overriding v4.1 ORB logic.
  - Single clean main() with ORB scan fully operational.

Triggers:
  1. Classic 30/65-min pivot (3+ bar streak reversal)
  2. Opening Range Breakout (ORB) — HIGH + MED conviction
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
RATE_LIMIT_PAUSE = 2          # seconds between batch downloads (only 3 calls total now)
SCAN_30MIN       = True
SCAN_65MIN       = True
SCAN_ORB         = True
ALERT_TIERS      = ["HIGH", "MED"]
ORB_TIERS        = ["HIGH", "MED"]

# R-ratio thresholds
R_ELITE   = 3.0
R_GOOD    = 2.0
R_MINIMUM = 1.0

# ORB config
ORB_RVOL_MIN      = 1.0
ORB_OPEN_HOUR_CT  = 9
ORB_OPEN_MIN_CT   = 30
CT_TZ             = pytz.timezone("America/Chicago")

# Batch download chunk size — yfinance handles ~200 tickers fine; split if larger
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
#  OPENING RANGE BREAKOUT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_orb(df_30m: pd.DataFrame) -> Optional[dict]:
    """
    Opening Range Breakout on the 30-minute timeframe.
    Bar 1 (9:30–10:00 CT) = opening range.
    Bar 2 closes above OR high (bullish) or below OR low (bearish) with RVOL confirmation.
    """
    if df_30m is None or len(df_30m) < 3:
        return None

    try:
        idx_ct = df_30m.index.tz_convert(CT_TZ)
    except Exception:
        idx_ct = df_30m.index

    today_ct = datetime.now(CT_TZ).date()
    today_mask = pd.Series([i.date() == today_ct for i in idx_ct], index=df_30m.index)
    today_bars = df_30m[today_mask]

    if len(today_bars) < 2:
        return None

    bar1 = today_bars.iloc[0]
    bar1_time = idx_ct[today_mask][0]

    if bar1_time.hour < 9 or bar1_time.hour >= 10:
        return None

    bar2 = today_bars.iloc[1]
    or_high = float(bar1["high"])
    or_low  = float(bar1["low"])
    or_range = or_high - or_low

    if or_range <= 0:
        return None

    avg_vol_20 = df_30m["volume"].iloc[-22:-2].mean()
    bar2_rvol  = float(bar2["volume"] / avg_vol_20) if avg_vol_20 > 0 else 1.0

    if bar2_rvol < ORB_RVOL_MIN:
        log.info(f"  ORB: volume low ({bar2_rvol:.2f}× < {ORB_RVOL_MIN}×) — skipping")
        return None

    bar2_close = float(bar2["close"])

    if bar2_close > or_high:
        return {
            "direction":     "bullish",
            "or_high":       round(or_high, 2),
            "or_low":        round(or_low, 2),
            "or_range":      round(or_range, 2),
            "breakout_pct":  round((bar2_close - or_high) / or_high * 100, 2),
            "bar1":          bar1,
            "bar2":          bar2,
            "bar1_time":     str(df_30m.index[today_mask][0]),
            "bar2_time":     str(df_30m.index[today_mask][1]),
            "rvol":          round(bar2_rvol, 2),
            "stop_level":    round(or_low, 2),
        }

    if bar2_close < or_low:
        return {
            "direction":     "bearish",
            "or_high":       round(or_high, 2),
            "or_low":        round(or_low, 2),
            "or_range":      round(or_range, 2),
            "breakout_pct":  round((or_low - bar2_close) / or_low * 100, 2),
            "bar1":          bar1,
            "bar2":          bar2,
            "bar1_time":     str(df_30m.index[today_mask][0]),
            "bar2_time":     str(df_30m.index[today_mask][1]),
            "rvol":          round(bar2_rvol, 2),
            "stop_level":    round(or_high, 2),
        }

    return None


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

    orb_base = 5
    score = orb_base
    if result["rvol"] >= 2.0:   score += 2
    elif result["rvol"] >= 1.5: score += 1
    if r_ratio is not None:
        if r_ratio >= R_ELITE:     score += 3
        elif r_ratio >= R_GOOD:    score += 2
        elif r_ratio >= R_MINIMUM: score += 1
    or_pct = result["or_range"] / result["or_high"] * 100
    if or_pct < 1.0:   score += 2
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
        "trigger_type":        "ORB",
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
        "streak_len":          0,
        "or_high":             result["or_high"],
        "or_low":              result["or_low"],
        "or_range":            result["or_range"],
        "breakout_pct":        result["breakout_pct"],
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
        "stop_level":          stop_level,
        "atr_14d":             round(atr, 4) if pd.notna(atr) else None,
        "r_ratio":             r_ratio,
        "rvol_trigger":        result["rvol"],
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
#  MAIN  (single definition — v4.2)
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting INTRADAY pivot scanner v4.2 (batch downloads + ORB)...")

    daily_data    = get_latest_daily_watchlist()
    entries       = daily_data.get("entries", [])
    orb_entries   = [e for e in entries if e["conviction"] in ORB_TIERS]
    pivot_entries = [e for e in entries if e["conviction"] in ALERT_TIERS]

    if not pivot_entries:
        log.error("No entries in daily watchlist. Run daily_screener.py first.")
        return

    now_ct = datetime.now(CT_TZ)
    log.info(f"Watchlist: {len(pivot_entries)} pivot tickers, {len(orb_entries)} ORB tickers")
    log.info(f"Current time: {now_ct.strftime('%H:%M CT')}")

    tickers = list({e["ticker"] for e in pivot_entries})
    entry_map = {e["ticker"]: e for e in pivot_entries}

    # ── 3 batch downloads replace 189 serial calls ────────────────────────
    log.info("Batch downloading daily bars...")
    daily_bars = batch_fetch_daily(tickers)
    time.sleep(RATE_LIMIT_PAUSE)

    log.info("Batch downloading 30m bars...")
    bars_30m = batch_fetch_intraday(tickers, "30m") if SCAN_30MIN or SCAN_ORB else {}
    time.sleep(RATE_LIMIT_PAUSE)

    log.info("Batch downloading 60m bars...")
    bars_60m = batch_fetch_intraday(tickers, "60m") if SCAN_65MIN else {}

    # ── ORB scan ──────────────────────────────────────────────────────────
    fired_orb   = set()
    orb_count   = 0

    if SCAN_ORB:
        log.info(f"─── ORB SCAN ({len(orb_entries)} tickers) ───")
        for entry in orb_entries:
            ticker = entry["ticker"]
            df_30  = bars_30m.get(ticker)
            if df_30 is None:
                log.info(f"  ORB [{ticker}] no 30m data — skip")
                continue
            orb = detect_orb(df_30)
            if orb:
                persist_orb_trigger(entry, orb, daily_bars.get(ticker))
                fired_orb.add(ticker)
                orb_count += 1

    # ── Pivot scan ────────────────────────────────────────────────────────
    fired_30 = set()
    fired_65 = set()
    trigger_count = 0

    log.info(f"─── PIVOT SCAN ({len(pivot_entries)} tickers) ───")
    for entry in pivot_entries:
        ticker   = entry["ticker"]
        df_daily = daily_bars.get(ticker)

        if SCAN_30MIN:
            df_30 = bars_30m.get(ticker)
            if df_30 is not None:
                r = detect_pivot(df_30)
                if r:
                    fired_30.add(ticker)
                    persist_trigger(entry, "30-MIN", r, df_daily, df_30)
                    trigger_count += 1

        if SCAN_65MIN:
            df_60 = bars_60m.get(ticker)
            if df_60 is not None:
                r = detect_pivot(df_60)
                if r:
                    fired_65.add(ticker)
                    persist_trigger(entry, "65-MIN", r, df_daily, df_60)
                    trigger_count += 1

    dupes = fired_30 & fired_65
    if dupes:
        log.info(f"  Both TF fired: {', '.join(sorted(dupes))}")

    log.info(f"\n  SCAN COMPLETE")
    log.info(f"    ORB triggers   : {orb_count}")
    log.info(f"    Pivot triggers : {trigger_count}  (30m={len(fired_30)}  65m={len(fired_65)})")
    log.info(f"    Total          : {orb_count + trigger_count}")
    log.info(f"    API calls made : ~3 batch downloads (was ~{len(tickers) * 3} serial)")


if __name__ == "__main__":
    main()
