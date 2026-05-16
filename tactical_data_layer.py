"""
data_layer.py — Single source of truth for all data fetching and metric computation.

Pulls daily OHLCV via yfinance with hard verification (no silent failures).
Computes all metrics for both Macro View and Speculative Themes tabs.
Cached aggressively to avoid hammering yfinance.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

from themes import THEMES, get_all_unique_tickers, get_themes_for_ticker

# =============================================================================
# UNIVERSE DEFINITIONS
# =============================================================================

# Indices and breadth proxies
MACRO_INDICES = ["SPY", "RSP", "QQQ", "QQQE", "IWM", "MDY"]

# Style / factor box (large, mid, small × growth, value)
MACRO_STYLE = ["IJS", "IJT", "IJJ", "IJK", "IVE", "IVW"]

# Canonical 11 SPDR sectors for breadth math
MACRO_SECTORS = ["XLK", "XLF", "XLY", "XLC", "XLP", "XLE", "XLB", "XLI", "XLU", "XLV", "XLRE"]

# Leadership pair components (some overlap with sectors above is fine)
MACRO_PAIRS = ["IGV", "SOXX", "SMH", "CIBR", "KRE", "XBI", "IBB", "XRT",
               "ITB", "XHB", "XOP", "OIH", "GDX", "ITA", "IYT"]

# Stress tape
MACRO_STRESS = ["^VIX", "^VIX3M", "HYG", "LQD", "TLT", "GLD"]

# Cross-asset narrative tracker
MACRO_NARRATIVE = ["^GSPC", "DX-Y.NYB", "^TNX"]


def all_macro_tickers():
    """De-dup union of every ticker the macro tab needs."""
    return sorted(set(MACRO_INDICES + MACRO_STYLE + MACRO_SECTORS +
                      MACRO_PAIRS + MACRO_STRESS + MACRO_NARRATIVE))


# =============================================================================
# LEADERSHIP PAIR DEFINITIONS
# (numerator, denominator, label, group)
# =============================================================================

LEADERSHIP_PAIRS = [
    # ── Speculative Risk-On (5 pairs) ──────────────────────────────────────
    # SOXX/SPY + IGV/SPY capture the full tech complex vs market directly —
    # semis outperforming = hardware risk-on; software outperforming = growth risk-on.
    # IBB/XLV replaces XBI/IBB — biotech vs total healthcare is a cleaner
    # risk-appetite signal than small vs large biotech (too noisy internally).
    ("SOXX", "SPY",  "Semis vs Market",            "Speculative_Risk_On"),
    ("IGV",  "SPY",  "Software vs Market",         "Speculative_Risk_On"),
    ("KRE",  "XLF",  "Regional vs Total Banks",    "Speculative_Risk_On"),
    ("IBB",  "XLV",  "Biotech vs Healthcare",      "Speculative_Risk_On"),
    ("XRT",  "XLY",  "Retail vs Total Disc",       "Speculative_Risk_On"),
    ("CIBR", "XLK",  "Cyber vs Tech",              "Idiosyncratic"),
    ("ITB",  "XHB",  "Pure Builders vs Construction", "Cyclical_Expansion"),
    ("XOP",  "XLE",  "E&P vs Integrated Oil",      "Commodity_Confirmation"),
    ("OIH",  "XLE",  "Services vs Integrated",     "Cyclical_Expansion"),
    ("GDX",  "GLD",  "Miners vs Gold",             "Commodity_Confirmation"),
    ("ITA",  "XLI",  "Defense vs Industrials",     "Idiosyncratic"),
    ("IYT",  "SPY",  "Transports vs Market",       "Cyclical_Expansion"),
]

# Pair confirmation threshold (return differential)
PAIR_THRESHOLD = 0.005  # 0.5%


# =============================================================================
# CROSS-ASSET NARRATIVE STATE MAP
# (spx_dir, dxy_dir, rates_dir) → (state_id, name, sector_tilt)
# 1 = up, -1 = down (close-to-close)
# =============================================================================

NARRATIVE_STATES = {
    ( 1, -1, -1): (1, "Goldilocks",         "Growth, semis, software, biotech — ideal backdrop"),
    ( 1, -1,  1): (2, "Broad Risk-On",      "Energy, materials, commodities, industrials — risk + growth bid together"),
    ( 1,  1,  1): (3, "US Exceptionalism",  "Large-cap quality, mega-caps, USD-earners — US is the destination"),
    ( 1,  1, -1): (4, "Soft Landing Print", "Broad equity, consumer disc — Goldilocks with stronger dollar"),
    (-1,  1, -1): (5, "Flight to Safety",   "Defensive — Treasuries, utilities, staples, gold"),
    (-1,  1,  1): (6, "Hawkish Squeeze",    "Cash; no longs — Fed tightening pain, stocks AND bonds selling"),
    (-1, -1, -1): (7, "Growth Scare",       "Gold, duration, defensive value — market pricing in weakness"),
    (-1, -1,  1): (8, "Supply Shock",       "Real assets, gold miners, commodities — dollar weak but rates rising"),
}


# =============================================================================
# DATA FETCHING (cached, batched, verified)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_universe(tickers_tuple, lookback_days=750):
    """
    Batch-fetch daily OHLCV for a tuple of tickers via yfinance.

    Returns:
        prices: DataFrame indexed by date, columns are tickers (Close prices, auto-adjusted)
        failed: list of tickers that returned no data

    Uses tuple input so it's hashable for st.cache_data.
    """
    tickers = list(tickers_tuple)
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    # yfinance can be flaky with one giant batch; chunk it
    chunk_size = 30
    all_data = {}
    failed = []

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            df = yf.download(
                tickers=chunk,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as e:
            for t in chunk:
                failed.append(t)
            continue

        for t in chunk:
            try:
                if len(chunk) == 1:
                    series = df["Close"]
                else:
                    series = df[t]["Close"]

                series = series.dropna()
                if len(series) < 30:
                    failed.append(t)
                else:
                    all_data[t] = series
            except (KeyError, TypeError):
                failed.append(t)

    if not all_data:
        return pd.DataFrame(), failed

    prices = pd.DataFrame(all_data)
    prices.index = pd.to_datetime(prices.index)
    return prices, failed


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def safe_ret(series, periods):
    """% return over `periods` trading days; NaN if not enough history."""
    if len(series) < periods + 1:
        return np.nan
    return (series.iloc[-1] / series.iloc[-periods - 1] - 1) * 100.0


def above_ma(series, periods):
    """Boolean: latest close above its `periods`-day SMA."""
    if len(series) < periods:
        return False
    sma = series.rolling(periods).mean().iloc[-1]
    return float(series.iloc[-1]) > float(sma)


def realized_vol(series, days=30):
    """Annualized realized volatility from daily returns."""
    if len(series) < days + 1:
        return np.nan
    returns = series.pct_change().dropna().iloc[-days:]
    return returns.std() * np.sqrt(252) * 100.0


def percentile_rank(series, lookback=252):
    """Latest value's percentile rank vs `lookback` history (0-100)."""
    if len(series) < lookback:
        lookback = len(series)
    window = series.iloc[-lookback:]
    return (window.rank(pct=True).iloc[-1]) * 100.0


# =============================================================================
# LEADERSHIP PAIR LOGIC
# =============================================================================

def pair_differentials(prices, num, den, windows=(5, 10, 20)):
    """
    Returns dict of {window_days: (num_return - den_return)} as decimals.
    """
    out = {}
    for w in windows:
        n_ret = safe_ret(prices[num], w) / 100.0 if num in prices else np.nan
        d_ret = safe_ret(prices[den], w) / 100.0 if den in prices else np.nan
        if np.isnan(n_ret) or np.isnan(d_ret):
            out[w] = np.nan
        else:
            out[w] = n_ret - d_ret
    return out


def classify_pair_state(diff_5, diff_10, diff_20, threshold=PAIR_THRESHOLD):
    """
    Returns one of: Confirmed, Fresh, Fading, False Start, Denied, Unknown.
    """
    if any(np.isnan(x) for x in (diff_5, diff_10, diff_20)):
        return "Unknown"

    pos_5 = diff_5 > threshold
    pos_10 = diff_10 > threshold
    pos_20 = diff_20 > threshold

    if pos_5 and pos_10 and pos_20:
        return "Confirmed"
    if pos_5 and pos_10 and not pos_20:
        return "Fresh"
    if not pos_5 and pos_10 and pos_20:
        return "Fading"
    if pos_5 and not pos_10 and not pos_20:
        return "False Start"
    if not pos_5 and not pos_10 and not pos_20:
        return "Denied"
    return "Mixed"


# =============================================================================
# CROSS-ASSET NARRATIVE
# =============================================================================

def compute_narrative(prices):
    """
    Compute today's narrative state and 10-day rolling frequencies.
    """
    spx = prices.get("^GSPC")
    dxy = prices.get("DX-Y.NYB")
    rates = prices.get("^TNX")

    if spx is None or dxy is None or rates is None:
        return None

    aligned = pd.concat([spx, dxy, rates], axis=1, keys=["SPX", "DXY", "RATES"]).dropna()
    if len(aligned) < 30:
        return None

    # Daily directions, close-to-close (1 = up, -1 = down)
    dirs = np.sign(aligned.diff()).fillna(0).astype(int)
    dirs = dirs.replace(0, 1)  # treat flat as up to avoid undefined state

    # Today's state
    today = tuple(dirs.iloc[-1].tolist())
    today_state = NARRATIVE_STATES.get(today, (0, "Unknown", ""))

    # 10-day rolling frequency of each state
    last10 = dirs.tail(10)
    state_counts = {sid: 0 for sid in range(1, 9)}
    for _, row in last10.iterrows():
        key = tuple(row.tolist())
        if key in NARRATIVE_STATES:
            sid = NARRATIVE_STATES[key][0]
            state_counts[sid] += 1

    dominant_id = max(state_counts, key=state_counts.get)
    dominant_count = state_counts[dominant_id]
    dominant_meta = next(
        (meta for k, meta in NARRATIVE_STATES.items() if meta[0] == dominant_id),
        (0, "Unknown", "")
    )

    # 60-day history of state IDs for the heatmap
    last60 = dirs.tail(60)
    state_history = []
    for date, row in last60.iterrows():
        key = tuple(row.tolist())
        sid = NARRATIVE_STATES.get(key, (0, "Unknown", ""))[0]
        state_history.append({"date": date, "state_id": sid})

    return {
        "today_state_id": today_state[0],
        "today_state_name": today_state[1],
        "today_sector_tilt": today_state[2],
        "spx_dir": dirs["SPX"].iloc[-1],
        "dxy_dir": dirs["DXY"].iloc[-1],
        "rates_dir": dirs["RATES"].iloc[-1],
        "rolling_10_counts": state_counts,
        "dominant_id": dominant_id,
        "dominant_name": dominant_meta[1],
        "dominant_tilt": dominant_meta[2],
        "dominant_pct": dominant_count * 10,  # out of 10 days = pct
        "history_60d": state_history,
    }


# =============================================================================
# MASTER MACRO METRIC COMPUTATION
# =============================================================================

def compute_macro_metrics(prices):
    """
    Single function that computes everything the Macro View tab needs.
    Returns a nested dict keyed by section.
    """
    metrics = {
        "as_of": prices.index[-1] if len(prices) else None,
        "headline": {},
        "participation": {},
        "style": {},
        "sector_rotation": {},
        "leadership_pairs": [],
        "group_scores": {},
        "stress": {},
        "regime_history": [],
        "narrative": None,
    }

    # ── Headline Breadth ────────────────────────────────────────────────
    if "SPY" in prices:
        spy = prices["SPY"]
        metrics["headline"]["spy_close"] = float(spy.iloc[-1])
        metrics["headline"]["spy_above_50"] = above_ma(spy, 50)
        metrics["headline"]["spy_above_200"] = above_ma(spy, 200)
        metrics["headline"]["spy_5d"] = safe_ret(spy, 5)
        metrics["headline"]["spy_20d"] = safe_ret(spy, 20)

    if "^VIX" in prices:
        vix = prices["^VIX"]
        metrics["headline"]["vix_level"] = float(vix.iloc[-1])

    if "^VIX3M" in prices and "^VIX" in prices:
        try:
            ratio = float(prices["^VIX"].iloc[-1] / prices["^VIX3M"].iloc[-1])
            metrics["headline"]["vix_term_ratio"] = ratio
            metrics["headline"]["vix_contango"] = ratio < 1.0
        except Exception:
            metrics["headline"]["vix_term_ratio"] = np.nan
            metrics["headline"]["vix_contango"] = False

    # MMTW proxy: % of SPDR sectors above 20DMA (since real MMTW is not on yfinance)
    sector_above_20 = sum(above_ma(prices[s], 20) for s in MACRO_SECTORS if s in prices)
    sector_above_50 = sum(above_ma(prices[s], 50) for s in MACRO_SECTORS if s in prices)
    n_sectors_present = sum(1 for s in MACRO_SECTORS if s in prices)
    if n_sectors_present > 0:
        metrics["headline"]["pct_sectors_above_20"] = (sector_above_20 / n_sectors_present) * 100
        metrics["headline"]["pct_sectors_above_50"] = (sector_above_50 / n_sectors_present) * 100
    else:
        metrics["headline"]["pct_sectors_above_20"] = np.nan
        metrics["headline"]["pct_sectors_above_50"] = np.nan

    # ── Participation Ratios ────────────────────────────────────────────
    def make_ratio(num, den):
        if num not in prices or den not in prices:
            return None
        n_aligned, d_aligned = prices[num].align(prices[den], join="inner")
        ratio = n_aligned / d_aligned
        return ratio.dropna()

    for name, num, den in [
        ("rsp_spy", "RSP", "SPY"),
        ("qqqe_qqq", "QQQE", "QQQ"),
        ("iwm_spy", "IWM", "SPY"),
        ("mdy_spy", "MDY", "SPY"),
    ]:
        r = make_ratio(num, den)
        if r is not None and len(r) > 50:
            metrics["participation"][name] = {
                "current": float(r.iloc[-1]),
                "sma_50": float(r.rolling(50).mean().iloc[-1]),
                "above_sma": float(r.iloc[-1]) > float(r.rolling(50).mean().iloc[-1]),
                "5d_change_pct": (float(r.iloc[-1]) / float(r.iloc[-6]) - 1) * 100 if len(r) > 5 else np.nan,
                "series_60d": r.tail(60).reset_index().to_dict("records"),
            }

    # ── Style / Factor Box ──────────────────────────────────────────────
    for name, num, den in [
        ("large_gv", "IVW", "IVE"),
        ("mid_gv", "IJK", "IJJ"),
        ("small_gv", "IJT", "IJS"),
    ]:
        r = make_ratio(num, den)
        if r is not None and len(r) > 50:
            metrics["style"][name] = {
                "current": float(r.iloc[-1]),
                "5d_change_pct": (float(r.iloc[-1]) / float(r.iloc[-6]) - 1) * 100 if len(r) > 5 else np.nan,
                "series_60d": r.tail(60).reset_index().to_dict("records"),
            }
    # Growth premium composite
    if all(k in metrics["style"] for k in ("large_gv", "mid_gv", "small_gv")):
        composite = np.mean([metrics["style"][k]["current"] for k in ("large_gv", "mid_gv", "small_gv")])
        metrics["style"]["growth_premium"] = float(composite)

    # ── Sector Rotation ─────────────────────────────────────────────────
    def avg_return(tickers, days):
        rets = []
        for t in tickers:
            if t in prices:
                r = safe_ret(prices[t], days)
                if not np.isnan(r):
                    rets.append(r)
        return np.mean(rets) if rets else np.nan

    defensive = ["XLP", "XLU", "XLV"]
    offensive = ["XLK", "XLY", "XLF", "XLC", "XLI"]
    metrics["sector_rotation"]["defensive_5d"] = avg_return(defensive, 5)
    metrics["sector_rotation"]["offensive_5d"] = avg_return(offensive, 5)
    metrics["sector_rotation"]["defensive_20d"] = avg_return(defensive, 20)
    metrics["sector_rotation"]["offensive_20d"] = avg_return(offensive, 20)

    if not np.isnan(metrics["sector_rotation"]["defensive_5d"]) and \
       not np.isnan(metrics["sector_rotation"]["offensive_5d"]):
        metrics["sector_rotation"]["def_off_diff_5d"] = (
            metrics["sector_rotation"]["defensive_5d"] - metrics["sector_rotation"]["offensive_5d"]
        )

    # All sectors top/bottom 5d
    sector_5d = []
    for s in MACRO_SECTORS:
        if s in prices:
            r = safe_ret(prices[s], 5)
            if not np.isnan(r):
                sector_5d.append((s, r))
    sector_5d.sort(key=lambda x: x[1], reverse=True)
    metrics["sector_rotation"]["top_3_5d"] = sector_5d[:3]
    metrics["sector_rotation"]["bottom_3_5d"] = sector_5d[-3:][::-1]
    metrics["sector_rotation"]["all_sectors_5d"] = sector_5d

    # ── Leadership Pairs ────────────────────────────────────────────────
    group_results = {
        "Speculative_Risk_On": [],
        "Cyclical_Expansion": [],
        "Commodity_Confirmation": [],
        "Idiosyncratic": [],
    }

    for num, den, label, group in LEADERSHIP_PAIRS:
        diffs = pair_differentials(prices, num, den)
        state = classify_pair_state(diffs.get(5), diffs.get(10), diffs.get(20))

        # Build 60-day ratio history for spark line
        ratio_series = None
        if num in prices and den in prices:
            n_a, d_a = prices[num].align(prices[den], join="inner")
            r = (n_a / d_a).dropna()
            if len(r) > 0:
                ratio_series = r.tail(60).reset_index()
                ratio_series.columns = ["date", "ratio"]
                ratio_series = ratio_series.to_dict("records")

        pair_record = {
            "num": num,
            "den": den,
            "label": label,
            "group": group,
            "diff_5d": diffs.get(5),
            "diff_10d": diffs.get(10),
            "diff_20d": diffs.get(20),
            "state": state,
            "ratio_series_60d": ratio_series,
        }
        metrics["leadership_pairs"].append(pair_record)
        group_results[group].append(pair_record)

    # Group scores: count of "Confirmed" or "Fresh" pairs in each group
    for grp, pairs in group_results.items():
        confirmed = sum(1 for p in pairs if p["state"] in ("Confirmed", "Fresh"))
        total = len(pairs)
        metrics["group_scores"][grp] = {
            "confirmed": confirmed,
            "total": total,
            "pct": (confirmed / total * 100) if total else 0,
            "pairs": pairs,
        }

    # ── Composite Regime Label ──────────────────────────────────────────
    metrics["regime_label"] = determine_rotation_regime(metrics)

    # ── Stress Tape ─────────────────────────────────────────────────────
    if "HYG" in prices and "LQD" in prices:
        hyg, lqd = prices["HYG"].align(prices["LQD"], join="inner")
        ratio = (hyg / lqd).dropna()
        if len(ratio) > 50:
            metrics["stress"]["hyg_lqd_ratio"] = float(ratio.iloc[-1])
            metrics["stress"]["hyg_lqd_above_50"] = float(ratio.iloc[-1]) > float(ratio.rolling(50).mean().iloc[-1])
            metrics["stress"]["hyg_lqd_5d_pct"] = (float(ratio.iloc[-1]) / float(ratio.iloc[-6]) - 1) * 100 if len(ratio) > 5 else np.nan
            metrics["stress"]["hyg_lqd_series_60d"] = ratio.tail(60).reset_index().rename(columns={ratio.name: "ratio"}).to_dict("records") if hasattr(ratio, "name") else None

    if "HYG" in prices:
        metrics["stress"]["hyg_above_50"] = above_ma(prices["HYG"], 50)
        metrics["stress"]["hyg_5d"] = safe_ret(prices["HYG"], 5)

    if "TLT" in prices:
        metrics["stress"]["tlt_realized_vol_30d"] = realized_vol(prices["TLT"], 30)
        metrics["stress"]["tlt_close"] = float(prices["TLT"].iloc[-1])
        metrics["stress"]["tlt_5d"] = safe_ret(prices["TLT"], 5)

    if "GLD" in prices:
        metrics["stress"]["gld_above_50"] = above_ma(prices["GLD"], 50)
        metrics["stress"]["gld_5d"] = safe_ret(prices["GLD"], 5)
        metrics["stress"]["gld_close"] = float(prices["GLD"].iloc[-1])

    if "^VIX" in prices:
        metrics["stress"]["vix_series_60d"] = prices["^VIX"].tail(60).reset_index().rename(columns={"^VIX": "vix"}).to_dict("records")

    # ── Cross-Asset Narrative ───────────────────────────────────────────
    metrics["narrative"] = compute_narrative(prices)

    return metrics


def determine_rotation_regime(metrics):
    """
    Classifies the overall rotation regime using all signals.
    Returns a string label.
    """
    g_spec = metrics["group_scores"].get("Speculative_Risk_On", {}).get("confirmed", 0)
    g_cyc = metrics["group_scores"].get("Cyclical_Expansion", {}).get("confirmed", 0)
    g_com = metrics["group_scores"].get("Commodity_Confirmation", {}).get("confirmed", 0)

    pct_above_20 = metrics["headline"].get("pct_sectors_above_20", np.nan)

    spy_above_50 = metrics["headline"].get("spy_above_50", False)
    rsp_5d = (metrics.get("participation", {})
                       .get("rsp_spy", {})
                       .get("5d_change_pct", np.nan))

    def_off_diff = metrics["sector_rotation"].get("def_off_diff_5d", np.nan)

    hyg_above_50 = metrics["stress"].get("hyg_above_50", False)

    # Capitulation
    if not np.isnan(pct_above_20) and pct_above_20 < 20 and \
       not np.isnan(def_off_diff) and def_off_diff > 1.0:
        return "CAPITULATION SETUP"

    # Overbought distribution
    if not np.isnan(pct_above_20) and pct_above_20 > 85 and \
       not np.isnan(rsp_5d) and rsp_5d < -0.5:
        return "OVERBOUGHT DISTRIBUTION"

    # Mega-cap mirage
    if spy_above_50 and not np.isnan(rsp_5d) and rsp_5d < -1.0:
        return "MEGA-CAP MIRAGE"

    # Stealth risk-off
    if not np.isnan(def_off_diff) and def_off_diff > 1.5 and not hyg_above_50:
        return "STEALTH RISK-OFF"

    # Healthy
    if g_spec >= 3 and g_cyc >= 2:
        return "HEALTHY RISK-ON"

    if g_cyc >= 2 and g_com <= 1:
        return "CYCLICAL EXPANSION"

    if g_com >= 2 and g_cyc >= 1:
        return "INFLATION REFLATION"

    if g_spec >= 2:
        return "GROWTH-LED RISK-ON"

    return "MIXED / CHOPPY"


# =============================================================================
# THEMATIC METRICS (for Speculative Themes tab)
# =============================================================================

def compute_theme_metrics(prices, themes_dict=None):
    """
    Compute per-theme performance metrics:
      - Avg 1M / 3M / 6M / YTD return
      - % of tickers up
      - Top mover
      - Per-ticker detail (for drill-down)

    Returns:
      theme_records: list of dicts
      ticker_metrics: dict[ticker -> dict] for fast drill-downs
    """
    if themes_dict is None:
        themes_dict = THEMES

    today = datetime.now()
    ytd_start = pd.Timestamp(year=today.year, month=1, day=1)

    # Pre-compute ticker metrics once for everything
    ticker_metrics = {}
    for t in prices.columns:
        s = prices[t].dropna()
        if len(s) < 30:
            continue

        ret_1m = safe_ret(s, 21)
        ret_3m = safe_ret(s, 63)
        ret_6m = safe_ret(s, 126)

        # YTD return
        try:
            ytd_anchor = s.loc[s.index >= ytd_start]
            if len(ytd_anchor) > 1:
                ret_ytd = (s.iloc[-1] / ytd_anchor.iloc[0] - 1) * 100
            else:
                ret_ytd = np.nan
        except Exception:
            ret_ytd = np.nan

        # 52w high distance
        try:
            high_252 = s.iloc[-252:].max() if len(s) >= 50 else s.max()
            dist_52w = (s.iloc[-1] / high_252 - 1) * 100
        except Exception:
            dist_52w = np.nan

        ticker_metrics[t] = {
            "ticker": t,
            "last": float(s.iloc[-1]),
            "ret_1m": float(ret_1m) if not np.isnan(ret_1m) else None,
            "ret_3m": float(ret_3m) if not np.isnan(ret_3m) else None,
            "ret_6m": float(ret_6m) if not np.isnan(ret_6m) else None,
            "ret_ytd": float(ret_ytd) if not np.isnan(ret_ytd) else None,
            "dist_52w_high": float(dist_52w) if not np.isnan(dist_52w) else None,
            "above_50dma": above_ma(s, 50),
            "above_200dma": above_ma(s, 200),
        }

    theme_records = []
    for theme_name, theme_tickers in themes_dict.items():
        rets_1m, rets_3m, rets_6m, rets_ytd = [], [], [], []
        ticker_details = []

        for t in theme_tickers:
            if t not in ticker_metrics:
                continue
            tm = ticker_metrics[t]
            ticker_details.append(tm)
            for collector, key in [(rets_1m, "ret_1m"), (rets_3m, "ret_3m"),
                                    (rets_6m, "ret_6m"), (rets_ytd, "ret_ytd")]:
                if tm[key] is not None:
                    collector.append(tm[key])

        if not rets_1m:
            theme_records.append({
                "theme": theme_name,
                "n_tickers": len(theme_tickers),
                "n_with_data": 0,
                "avg_1m": None, "avg_3m": None, "avg_6m": None, "avg_ytd": None,
                "median_1m": None,
                "pct_up_1m": None, "top_mover_1m": None, "top_mover_1m_ret": None,
                "ticker_details": [],
            })
            continue

        top_idx = int(np.argmax(rets_1m))
        # Find which ticker that corresponds to (reconstructing from ticker_details with valid ret_1m)
        with_1m = [t for t in ticker_details if t["ret_1m"] is not None]
        top_mover = with_1m[top_idx] if top_idx < len(with_1m) else None

        theme_records.append({
            "theme": theme_name,
            "n_tickers": len(theme_tickers),
            "n_with_data": len(with_1m),
            "avg_1m": float(np.mean(rets_1m)),
            "avg_3m": float(np.mean(rets_3m)) if rets_3m else None,
            "avg_6m": float(np.mean(rets_6m)) if rets_6m else None,
            "avg_ytd": float(np.mean(rets_ytd)) if rets_ytd else None,
            "median_1m": float(np.median(rets_1m)),
            "pct_up_1m": float(sum(1 for r in rets_1m if r > 0) / len(rets_1m) * 100),
            "top_mover_1m": top_mover["ticker"] if top_mover else None,
            "top_mover_1m_ret": top_mover["ret_1m"] if top_mover else None,
            "ticker_details": sorted(ticker_details,
                                     key=lambda x: x["ret_1m"] if x["ret_1m"] is not None else -999,
                                     reverse=True),
        })

    return theme_records, ticker_metrics


def compute_cross_theme_movers(themes_dict=None, ticker_metrics=None):
    """
    Tickers ranked by # of themes they appear in (cross-theme exposure).
    Higher count = more narrative tailwinds.
    """
    if themes_dict is None:
        themes_dict = THEMES

    counts = {}
    for theme_name, tickers in themes_dict.items():
        for t in tickers:
            counts.setdefault(t, []).append(theme_name)

    rows = []
    for ticker, theme_list in counts.items():
        row = {
            "ticker": ticker,
            "n_themes": len(theme_list),
            "themes": theme_list,
        }
        if ticker_metrics and ticker in ticker_metrics:
            row.update({
                "ret_1m": ticker_metrics[ticker].get("ret_1m"),
                "ret_3m": ticker_metrics[ticker].get("ret_3m"),
                "last": ticker_metrics[ticker].get("last"),
            })
        rows.append(row)

    rows.sort(key=lambda x: (x["n_themes"], x.get("ret_1m") or -999), reverse=True)
    return rows
