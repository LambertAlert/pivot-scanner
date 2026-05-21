"""
DAILY SCREENER (v3) — batch yf.download replaces serial per-ticker calls.

Changes over v2:
  - batch_fetch_daily() downloads all tickers + SPY in one yf.download() call
    instead of 439 individual requests.  ~440 serial calls → 1 batch call.
  - RATE_LIMIT_PAUSE removed from the per-ticker loop (no longer needed).
  - All stage/BBUW/conviction logic unchanged.
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

from data_layer import save_daily_watchlist, get_latest_weekly_watchlist, get_industry_ranks


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

THEMES_CSV              = "sector_themes.csv"
LOG_LEVEL               = logging.INFO
MIN_DAILY_BBUW_SCORE    = 40

# Batch download config
BATCH_CHUNK = 200   # yfinance handles ~200 tickers cleanly; splits if larger


def compute_ibd_rs_factor(close_series: pd.Series) -> float:
    """
    Compute the IBD-style RS raw factor.
    Formula: 0.4*(C/C63) + 0.2*(C/C126) + 0.2*(C/C189) + 0.2*(C/C252)
    Returns nan if insufficient data.
    """
    s = close_series.dropna()
    n = len(s)
    if n < 63:
        return float("nan")
    c = float(s.iloc[-1])
    def _r(p):
        if n <= p:
            return float("nan")
        past = float(s.iloc[-(p + 1)])
        return c / past if past > 0 else float("nan")
    weights = [(_r(63), 0.40), (_r(126), 0.20), (_r(189), 0.20), (_r(252), 0.20)]
    valid = [(v, w) for v, w in weights if not (v != v)]  # filter NaN
    if not valid:
        return float("nan")
    total_w = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / total_w


def compute_daily_quick_metrics(df) -> dict:
    """
    Pre-compute intraday-scanner metrics from daily bars fetched by daily_screener.
    Storing these in the watchlist JSON eliminates the batch_fetch_daily call
    that pivot_scanner.py previously made on every intraday scan run.

    Returns: atr_14d, prev_close, high_20d, low_20d
    """
    result = {"atr_14d": None, "prev_close": None, "high_20d": None, "low_20d": None}
    try:
        if df is None or len(df) < 15:
            return result
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        prev_c = close.shift(1)
        tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
        atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
        result["atr_14d"]    = round(atr, 4) if pd.notna(atr) else None
        result["prev_close"] = round(float(close.iloc[-1]), 4)
        if len(df) >= 20:
            result["high_20d"] = round(float(high.iloc[-20:].max()), 4)
            result["low_20d"]  = round(float(low.iloc[-20:].min()),  4)
    except Exception:
        pass
    return result


def compute_universe_rs_ratings(bars: dict) -> dict:
    """
    Compute IBD RS ratings (1-99 percentile) for all tickers in bars dict.
    Ranked against the full fetched universe so percentiles are meaningful.

    Returns {ticker: {"rs_rating": int, "rs_factor": float}}
    """
    factors = {}
    for tk, df in bars.items():
        if tk == "SPY" or df is None or df.empty:
            continue
        if "close" not in df.columns:
            continue
        f = compute_ibd_rs_factor(df["close"])
        if f == f:  # not NaN
            factors[tk] = f

    if not factors:
        return {}

    # Rank within universe → percentile 1-99
    tickers = list(factors.keys())
    values  = [factors[t] for t in tickers]
    ranks   = pd.Series(values, index=tickers).rank(ascending=True, method="average")
    n       = len(ranks)
    result  = {}
    for tk in tickers:
        pct = int(round(ranks[tk] / n * 99))
        result[tk] = {
            "rs_rating": max(1, min(99, pct)),
            "rs_factor": round(factors[tk], 5),
        }
    return result



logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH DATA FETCHING  (replaces serial fetch_daily_bars)
# ═══════════════════════════════════════════════════════════════════════════════

def batch_fetch_daily(tickers: list, period: str = "1y") -> dict:
    """
    Download daily bars for all tickers in one yf.download() call.
    Returns {ticker: DataFrame} — tickers with < 60 usable rows are absent.
    SPY should be included in the tickers list to avoid a separate call.
    """
    result = {}
    tickers = list(set(tickers))  # deduplicate

    for i in range(0, len(tickers), BATCH_CHUNK):
        chunk = tickers[i: i + BATCH_CHUNK]
        try:
            raw = yf.download(
                chunk,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                log.warning(f"Batch chunk {i//BATCH_CHUNK + 1}: empty result")
                continue

            for tk in chunk:
                try:
                    # MultiIndex when >1 ticker; flat when exactly 1
                    if isinstance(raw.columns, pd.MultiIndex):
                        df = raw.xs(tk, axis=1, level=1).copy()
                    else:
                        df = raw.copy()

                    df.columns = [c.lower() for c in df.columns]
                    needed = [c for c in ["open", "high", "low", "close", "volume"]
                              if c in df.columns]
                    df = df[needed].dropna(how="all")
                    df = df[df["close"] > 0]

                    if len(df) >= 60:
                        result[tk] = df
                except Exception:
                    pass  # ticker absent from batch result — skip silently

        except Exception as e:
            log.warning(f"Batch fetch chunk error: {e}")

        # Brief pause between chunks (only fires if > BATCH_CHUNK tickers)
        if i + BATCH_CHUNK < len(tickers):
            time.sleep(2)

    log.info(f"Batch daily fetch: {len(result)}/{len(tickers)} tickers with ≥60 rows")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE & BBUW (DAILY) — unchanged
# ═══════════════════════════════════════════════════════════════════════════════

def calc_bbuw_daily(df: pd.DataFrame, df_spy: pd.DataFrame = None) -> dict:
    if len(df) < 60:
        return {"score": 0, "components": {}}
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    daily_range = high - low
    recent_range = daily_range.iloc[-5:].mean()
    baseline_range = daily_range.iloc[-25:-5].mean()
    range_ratio = recent_range / baseline_range if baseline_range > 0 else 1.0
    range_score = max(0, min(100, (1.0 - range_ratio) * 200))
    recent_vol = vol.iloc[-5:].mean()
    baseline_vol = vol.iloc[-25:-5].mean()
    vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1.0
    vol_score = max(0, min(100, (1.0 - vol_ratio) * 200))
    recent_lows = low.iloc[-5:].values
    hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] >= recent_lows[i-1])
    hl_score = (hl_count / 4) * 100
    avg_range_20 = daily_range.iloc[-20:].mean()
    coil_days = sum(1 for r in daily_range.iloc[-10:] if r < avg_range_20 * 0.6)
    coil_score = min(100, coil_days * 12)
    rs_score = 50
    if df_spy is not None and len(df_spy) >= 60 and len(df) >= 60:
        try:
            stock_perf = (close.iloc[-1] / close.iloc[-60]) - 1
            spy_perf = (df_spy["close"].iloc[-1] / df_spy["close"].iloc[-60]) - 1
            rs_score = max(0, min(100, 50 + (stock_perf - spy_perf) * 200))
        except Exception:
            pass
    high_30d = high.iloc[-30:].max()
    pullback_pct = ((close.iloc[-1] / high_30d) - 1) * 100
    if -10 <= pullback_pct <= -3:
        pb_score = 100
    elif -20 <= pullback_pct < -10:
        pb_score = 70
    elif -3 < pullback_pct <= 0:
        pb_score = 60
    else:
        pb_score = 30
    ema_21 = close.ewm(span=21).mean().iloc[-1]
    ema_proximity_pct = abs((close.iloc[-1] - ema_21) / ema_21) * 100
    if ema_proximity_pct < 2:
        ema_score = 100
    elif ema_proximity_pct < 5:
        ema_score = 60
    else:
        ema_score = 30
    weights = {
        "range_contraction": 0.20, "volume_contraction": 0.15,
        "higher_lows": 0.15,       "coil_duration": 0.10,
        "rs_vs_spy": 0.20,         "pullback_depth": 0.10,
        "ema_proximity": 0.10,
    }
    components = {
        "range_contraction": round(range_score, 1),
        "volume_contraction": round(vol_score, 1),
        "higher_lows":        round(hl_score, 1),
        "coil_duration":      round(coil_score, 1),
        "rs_vs_spy":          round(rs_score, 1),
        "pullback_depth":     round(pb_score, 1),
        "ema_proximity":      round(ema_score, 1),
    }
    composite = sum(components[k] * weights[k] for k in weights)
    return {"score": round(composite, 1), "components": components}


# ═══════════════════════════════════════════════════════════════════════════════
#  THEMES — unchanged
# ═══════════════════════════════════════════════════════════════════════════════

def load_sector_themes(csv_path: str) -> dict:
    """
    Load sector theme map from sector_themes.csv.
    Supports both old format (Symbol, Theme, ThemeRank)
    and new format (Symbol, Theme, ThemeRank, Industry).
    Returns: {ticker: {"theme": str, "rank": int, "industry": str}}
    """
    themes = {}
    if not os.path.exists(csv_path):
        log.warning(f"Sector themes file not found: {csv_path}")
        return themes
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row.get("Symbol", "").strip().upper()
            theme = row.get("Theme", "").strip()
            industry = row.get("Industry", "").strip()
            try:
                rank = int(row.get("ThemeRank", 99))
            except ValueError:
                rank = 99
            if sym:
                themes[sym] = {"theme": theme, "rank": rank, "industry": industry}
    log.info(f"Loaded {len(themes)} sector theme mappings.")
    return themes


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVICTION — unchanged
# ═══════════════════════════════════════════════════════════════════════════════

def assign_conviction(weekly_bbuw, daily_bbuw, trend_template, industry_rank: int,
                       pivot_8w_tier: str = "NONE",
                       ep_tier: str = "NONE") -> str:
    """
    Tiered conviction scoring (max 10 points base + EP bonus).

      Weekly BBUW  ≥ 60 → +1, ≥ 75 → +1              (max 2)
      Daily  BBUW  ≥ 60 → +1, ≥ 75 → +1              (max 2)
      Trend Template ≥ 6 → +1, = 8 → +1              (max 2)
      Industry Rank ≤ 10 → +1, ≤ 3 → +1              (max 2)
      8W Pivot STANDARD → +1, STRONG → +2             (max 2)
      EP WATCH → +1, STANDARD → +2, STRONG → +3      (bonus)

    HIGH ≥ 7 | MED ≥ 4 | LOW < 4
    """
    points = 0
    if weekly_bbuw    >= 60: points += 1
    if weekly_bbuw    >= 75: points += 1
    if daily_bbuw     >= 60: points += 1
    if daily_bbuw     >= 75: points += 1
    if trend_template >= 6:  points += 1
    if trend_template == 8:  points += 1
    if industry_rank  <= 10: points += 1
    if industry_rank  <= 3:  points += 1

    if pivot_8w_tier == "STANDARD": points += 1
    elif pivot_8w_tier == "STRONG": points += 2

    if ep_tier == "WATCH":      points += 1
    elif ep_tier == "STANDARD": points += 2
    elif ep_tier == "STRONG":   points += 3

    if points >= 7: return "HIGH"
    if points >= 4: return "MED"
    return "LOW"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting DAILY screener (v4 — batch downloads, weekly stage trusted)...")

    weekly_data    = get_latest_weekly_watchlist()
    weekly_entries = weekly_data.get("entries", [])
    if not weekly_entries:
        log.error("No weekly watchlist found. Run weekly_screener.py first.")
        return

    log.info(f"Loaded {len(weekly_entries)} entries from weekly screen.")
    themes               = load_sector_themes(THEMES_CSV)
    industry_rank_lookup = get_industry_ranks()
    log.info(f"Loaded {len(industry_rank_lookup)} industry ranks.")

    # ── Batch download: all weekly tickers + SPY in one call ─────────────
    all_tickers = [e["ticker"] for e in weekly_entries] + ["SPY"]
    log.info(f"Batch downloading daily bars for {len(all_tickers)} tickers...")
    bars = batch_fetch_daily(all_tickers)

    df_spy = bars.get("SPY")

    # ── IBD RS ratings across full fetched universe ───────────────────────
    # Computed BEFORE filtering so percentiles reflect the full 435-ticker pool.
    # Formula: 0.4*(C/C63)+0.2*(C/C126)+0.2*(C/C189)+0.2*(C/C252)
    log.info("Computing IBD RS ratings across full universe...")
    universe_rs = compute_universe_rs_ratings(bars)
    log.info(f"  RS ratings computed for {len(universe_rs)} tickers")
    if df_spy is None:
        log.warning("SPY fetch failed — RS vs SPY component will default to 50")

    # ── Per-ticker evaluation (no API calls inside loop) ──────────────────
    qualified       = []
    skipped_no_data = 0
    skipped_bbuw    = 0

    for i, entry in enumerate(weekly_entries, 1):
        ticker = entry["ticker"]

        df_daily = bars.get(ticker)
        if df_daily is None:
            skipped_no_data += 1
            log.debug(f"[{i}/{len(weekly_entries)}] {ticker} — no data")
            continue

        daily_bbuw = calc_bbuw_daily(df_daily, df_spy)
        if daily_bbuw["score"] < MIN_DAILY_BBUW_SCORE:
            skipped_bbuw += 1
            continue

        # Pre-compute daily metrics for intraday scanner — avoids re-fetching daily bars
        daily_metrics = compute_daily_quick_metrics(df_daily)

        theme_info      = themes.get(ticker, {"theme": "Unclassified", "rank": 99, "industry": ""})
        pivot_8w_tier   = entry.get("pivot_8w_tier", "NONE")
        ticker_industry = theme_info.get("industry", "")
        industry_rank   = industry_rank_lookup.get(ticker_industry, 99)

        conviction = assign_conviction(
            entry.get("bbuw_score", 0),
            daily_bbuw["score"],
            entry.get("trend_template_score", 0),
            industry_rank,
            pivot_8w_tier=pivot_8w_tier,
            ep_tier=entry.get("ep_tier", "NONE"),
        )

        tier_emoji = {
            "STRONG": "🔥", "STANDARD": "✅", "WEAK": "⚠️",
            "PROXIMITY": "👀", "NONE": "  ",
        }.get(pivot_8w_tier, "  ")
        log.info(f"  ✅ [{i}/{len(weekly_entries)}] {ticker}"
                 f" — {conviction} | DailyBBUW={daily_bbuw['score']:.0f}"
                 f" | Ind.Rank=#{industry_rank}"
                 f" | {tier_emoji}{pivot_8w_tier}")

        qualified.append({
            "ticker":                ticker,
            "conviction":            conviction,
            "weekly_stage":          entry.get("stage"),
            "trend_template":        entry.get("trend_template_score"),
            "weekly_bbuw":           entry.get("bbuw_score"),
            "daily_bbuw":            daily_bbuw["score"],
            "daily_bbuw_components": daily_bbuw["components"],
            "theme":                 theme_info["theme"],
            "theme_rank":            theme_info["rank"],
            "industry":              theme_info.get("industry", ""),
            "industry_rank":         industry_rank,
            "ep_tier":               entry.get("ep_tier", "NONE"),
            "ep_score":              entry.get("ep_score", 0),
            "ep_week_pct":           entry.get("ep_week_pct"),
            "pivot_8w_fired":        entry.get("pivot_8w_fired", False),
            "pivot_8w_tier":         pivot_8w_tier,
            "ema8":                  entry.get("ema8"),
            "pct_from_ema8":         entry.get("pct_from_ema8"),
            "ema8_rising":           entry.get("ema8_rising", False),
            "pivot_8w_volume_spike": entry.get("pivot_8w_volume_spike", False),
            # IBD RS Rating (1-99 percentile, ranked vs full 435-ticker universe)
            "rs_rating":             universe_rs.get(ticker, {}).get("rs_rating"),
            "rs_factor":             universe_rs.get(ticker, {}).get("rs_factor"),
            # Pre-computed daily metrics for intraday scanner (Speed 1 — no re-fetch)
            "atr_14d":               daily_metrics.get("atr_14d"),
            "prev_close":            daily_metrics.get("prev_close"),
            "high_20d":              daily_metrics.get("high_20d"),
            "low_20d":               daily_metrics.get("low_20d"),
        })

    conviction_order = {"HIGH": 0, "MED": 1, "LOW": 2}
    qualified.sort(key=lambda x: (conviction_order[x["conviction"]], -x["daily_bbuw"]))
    save_daily_watchlist(qualified)

    high_count   = sum(1 for r in qualified if r["conviction"] == "HIGH")
    med_count    = sum(1 for r in qualified if r["conviction"] == "MED")
    pivot_strong = sum(1 for r in qualified if r.get("pivot_8w_tier") == "STRONG")
    pivot_std    = sum(1 for r in qualified if r.get("pivot_8w_tier") == "STANDARD")
    pivot_prox   = sum(1 for r in qualified if r.get("pivot_8w_tier") == "PROXIMITY")

    log.info(f"\n  DAILY COMPLETE — {len(qualified)} qualified"
             f" ({high_count} HIGH, {med_count} MED)")
    log.info(f"    Skipped: {skipped_no_data} no-data | {skipped_bbuw} low-BBUW")
    log.info(f"    8W Pivots: 🔥 STRONG={pivot_strong}"
             f"  ✅ STANDARD={pivot_std}  👀 PROXIMITY={pivot_prox}")


if __name__ == "__main__":
    main()
