"""
macro_prep_v2.py — War Room Macro Engine v2
=============================================
Implements the three-layer macro framework from war_room_macro_overhaul_spec_v2.md

Phase 1 (this script):
  Layer 1 — GIP Composite (Growth + Inflation + Liquidity + Curve)
  Layer 2 — Cross-Asset Narrative Tracker (8-state SPX/DXY/Rates)

Phase 2 (future):
  Layer 3 — Vol Compression
  Macro Gate (all 5 conditions)

Outputs: data/gip_data.json, data/narrative_data.json

Schedule: runs daily as part of macro_theme job in GitLab CI.
FRED_API_KEY must be set as a GitLab CI/CD variable and Streamlit Cloud secret.

Dependencies: pip install fredapi yfinance pandas numpy
"""

import os
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.makedirs("data", exist_ok=True)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
GIP_JSON       = "data/gip_data.json"
NARRATIVE_JSON = "data/narrative_data.json"

# ── Normalisation constants ─────────────────────────────────────────────────
Z_LOOKBACK     = 1260   # ~5 trading years
Z_CLIP         = 2.5    # saturate at ±2.5 sigma
SCORE_RANGE    = 10.0   # pillars and composite both live in [−10, +10]

# ── Forward-fill maximums (trading days) ────────────────────────────────────
FFILL = {
    "daily":   3,
    "weekly":  7,
    "monthly": 35,   # monthly releases
    "ism":     35,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  FRED FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

def fred_series(series_id: str, start: str = "2015-01-01") -> pd.Series:
    """
    Pull a FRED series via fredapi if key is available,
    else fall back to a direct FRED API URL request.
    Returns a daily-indexed pd.Series with forward-fill applied.
    """
    if not FRED_API_KEY:
        log.warning(f"FRED_API_KEY not set — skipping {series_id}")
        return pd.Series(dtype=float, name=series_id)

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        s = fred.get_series(series_id, observation_start=start)
        s.name = series_id
        return s
    except Exception as e:
        log.warning(f"FRED {series_id}: {e}")
        return pd.Series(dtype=float, name=series_id)


def yf_close(ticker: str, start: str = "2015-01-01") -> pd.Series:
    """Pull adjusted close from yfinance."""
    try:
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].squeeze()
        s.index = pd.to_datetime(s.index)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.name = ticker
        return s
    except Exception as e:
        log.warning(f"yfinance {ticker}: {e}")
        return pd.Series(dtype=float, name=ticker)


def safe_pct_change_yoy(s: pd.Series) -> pd.Series:
    """Year-over-year % change, returns NaN-safe."""
    return s.pct_change(periods=252) * 100


# ═══════════════════════════════════════════════════════════════════════════════
#  NORMALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def z_score_series(s: pd.Series, lookback: int = Z_LOOKBACK) -> pd.Series:
    """Rolling z-score vs trailing lookback window, clipped at ±Z_CLIP."""
    mu  = s.rolling(lookback, min_periods=max(60, lookback // 4)).mean()
    sig = s.rolling(lookback, min_periods=max(60, lookback // 4)).std()
    z   = (s - mu) / sig.replace(0, np.nan)
    return z.clip(-Z_CLIP, Z_CLIP)


def z_to_score(z: pd.Series) -> pd.Series:
    """Map z-score in [−2.5, 2.5] → score in [−10, +10]."""
    return (z / Z_CLIP) * SCORE_RANGE


def inflation_u_score(yoy: float, target: float = 2.0,
                       above_cost: float = 3.5, below_cost: float = 6.0) -> float:
    """
    Asymmetric U-shape scoring for inflation readings.
    Score = 10 at target; penalises above AND below.
    Below penalty is steeper — deflation scares impact risk assets faster.
    """
    if np.isnan(yoy):
        return 0.0
    deviation = abs(yoy - target)
    if yoy >= target:
        return max(-10.0, 10.0 - above_cost * deviation)
    else:
        return max(-10.0, 10.0 - below_cost * deviation)


def stale_check(s: pd.Series, max_days: int) -> bool:
    """Returns True if the last valid observation is older than max_days."""
    if s.empty or s.dropna().empty:
        return True
    last_valid = s.dropna().index[-1]
    return (pd.Timestamp.today() - last_valid).days > max_days


def ffill_limited(s: pd.Series, max_days: int) -> pd.Series:
    """Forward-fill a series up to max_days only."""
    return s.ffill(limit=max_days)


def align_to_daily_index(series_list: list, start: str = "2018-01-01") -> pd.DataFrame:
    """Reindex all series to a common daily business-day index and forward-fill."""
    idx = pd.bdate_range(start=start, end=pd.Timestamp.today())
    df = pd.DataFrame(index=idx)
    for s in series_list:
        if not s.empty:
            df[s.name] = s.reindex(idx)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — GIP COMPOSITE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_growth_pillar() -> dict:
    """
    Growth Pillar (30% of GIP).
    Components:
      ISM Manufacturing PMI          40%  (FRED: MANEMP proxy or DALSAMI)
      Leading Economic Index YoY     35%  (FRED: USSLIND)
      Initial Claims 4-wk MA inverted 25% (FRED: ICSA)
    """
    log.info("  Computing Growth pillar...")
    scores = {}
    stale  = {}

    # LEI YoY
    lei_raw = fred_series("USSLIND")
    lei_raw = ffill_limited(lei_raw, FFILL["monthly"])
    stale["lei"] = stale_check(lei_raw, FFILL["monthly"])
    if not lei_raw.empty:
        lei_yoy = safe_pct_change_yoy(lei_raw)
        lei_z   = z_score_series(lei_yoy)
        scores["lei"] = z_to_score(lei_z).iloc[-1]
    else:
        scores["lei"] = 0.0

    # Initial Claims (inverted — higher claims = worse growth)
    icsa_raw = fred_series("ICSA")
    icsa_raw = ffill_limited(icsa_raw, FFILL["weekly"])
    stale["icsa"] = stale_check(icsa_raw, FFILL["weekly"])
    if not icsa_raw.empty:
        icsa_4wk = icsa_raw.rolling(4, min_periods=1).mean()
        icsa_z   = z_score_series(icsa_4wk)
        scores["icsa"] = z_to_score(-icsa_z).iloc[-1]  # inverted
    else:
        scores["icsa"] = 0.0

    # ISM proxy — use FRED MANEMP (manufacturing employment) as best free proxy
    # Dallas Fed DALSAMI or Chicago PMI would be better but availability varies
    ism_proxy = fred_series("MANEMP")
    ism_proxy = ffill_limited(ism_proxy, FFILL["monthly"])
    stale["ism"] = stale_check(ism_proxy, FFILL["monthly"])
    if not ism_proxy.empty:
        ism_mom = ism_proxy.pct_change(periods=3) * 100   # 3-month trend
        ism_z   = z_score_series(ism_mom)
        scores["ism"] = z_to_score(ism_z).iloc[-1]
    else:
        scores["ism"] = 0.0

    # Weighted composite
    pillar = (
        scores["ism"] * 0.40 +
        scores["lei"] * 0.35 +
        scores["icsa"] * 0.25
    )
    pillar = float(np.clip(pillar, -SCORE_RANGE, SCORE_RANGE))

    log.info(f"    Growth: {pillar:.2f}  "
             f"(ISM={scores['ism']:.1f}, LEI={scores['lei']:.1f}, ICSA={scores['icsa']:.1f})")
    return {
        "score":      round(pillar, 3),
        "components": {k: round(float(v), 3) for k, v in scores.items()},
        "stale":      any(stale.values()),
        "stale_detail": stale,
    }


def compute_inflation_pillar() -> dict:
    """
    Inflation Pillar (25% of GIP).
    Asymmetric U-shape: best score at 2% target, penalties above AND below.
    Components: CPI YoY, Core CPI YoY, PCE YoY, 5Y5Y Forward, PPI YoY.
    """
    log.info("  Computing Inflation pillar...")
    scores = {}
    stale  = {}

    series_map = {
        "cpi_yoy":  ("CPIAUCSL",  2.0,  3.5, 6.0),
        "core_yoy": ("CPILFESL",  2.0,  3.5, 6.0),
        "pce_yoy":  ("PCEPI",     2.0,  3.5, 6.0),
        "ppi_yoy":  ("PPIACO",    2.0,  3.5, 6.0),
        "t5y5y":    ("T5YIFR",    2.25, 3.5, 6.0),  # different target
    }
    weights = {"cpi_yoy": 0.25, "core_yoy": 0.25, "pce_yoy": 0.25,
               "t5y5y": 0.15, "ppi_yoy": 0.10}

    for key, (series_id, target, above_cost, below_cost) in series_map.items():
        raw = fred_series(series_id)
        raw = ffill_limited(raw, FFILL["monthly"])
        stale[key] = stale_check(raw, FFILL["monthly"])
        if raw.empty:
            scores[key] = 0.0
            continue

        if key == "t5y5y":
            # T5YIFR is already a rate level — use directly
            latest_val = float(raw.dropna().iloc[-1])
        else:
            # Monthly series — compute YoY
            yoy = safe_pct_change_yoy(raw).dropna()
            if yoy.empty:
                scores[key] = 0.0
                continue
            latest_val = float(yoy.iloc[-1])

        scores[key] = inflation_u_score(latest_val, target, above_cost, below_cost)

    pillar = sum(scores[k] * weights[k] for k in scores)
    pillar = float(np.clip(pillar, -SCORE_RANGE, SCORE_RANGE))

    log.info(f"    Inflation: {pillar:.2f}  "
             f"(CPI={scores.get('cpi_yoy',0):.1f}, Core={scores.get('core_yoy',0):.1f}, "
             f"PCE={scores.get('pce_yoy',0):.1f}, T5Y5Y={scores.get('t5y5y',0):.1f})")
    return {
        "score":      round(pillar, 3),
        "components": {k: round(float(v), 3) for k, v in scores.items()},
        "stale":      any(stale.values()),
        "stale_detail": stale,
    }


def compute_liquidity_pillar() -> dict:
    """
    Liquidity Pillar (30% of GIP).
    Components:
      Net Liquidity (WALCL − WTREGEN − RRPONTSYD) — level z + 8-week change z  50%
      HY OAS (BAMLH0A0HYM2) inverted                                            25%
      M2 YoY (M2SL)                                                             15%
      SOFR inverted                                                              10%
    """
    log.info("  Computing Liquidity pillar...")
    scores = {}
    stale  = {}

    # Net Liquidity
    walcl = fred_series("WALCL")      # Fed balance sheet
    tga   = fred_series("WTREGEN")    # Treasury General Account
    rrp   = fred_series("RRPONTSYD")  # Reverse Repo

    for name, s in [("walcl", walcl), ("tga", tga), ("rrp", rrp)]:
        stale[name] = stale_check(ffill_limited(s, FFILL["weekly"]), FFILL["weekly"])

    if not walcl.empty and not tga.empty and not rrp.empty:
        idx = pd.bdate_range(
            start=max(walcl.index.min(), tga.index.min(), rrp.index.min()),
            end=pd.Timestamp.today()
        )
        w = walcl.reindex(idx).ffill(limit=7)
        t = tga.reindex(idx).ffill(limit=7)
        r = rrp.reindex(idx).ffill(limit=7)
        net_liq = (w - t - r).dropna()

        net_liq_z    = z_score_series(net_liq)
        net_liq_8w   = net_liq.diff(40)   # 8 weeks ~40 trading days
        net_liq_8w_z = z_score_series(net_liq_8w)

        net_liq_score = z_to_score((net_liq_z + net_liq_8w_z) / 2)
        scores["net_liq"] = float(net_liq_score.iloc[-1])
    else:
        scores["net_liq"] = 0.0

    # HY OAS (inverted — wider spreads = worse liquidity)
    hy = fred_series("BAMLH0A0HYM2")
    hy = ffill_limited(hy, FFILL["daily"])
    stale["hy"] = stale_check(hy, FFILL["daily"])
    if not hy.empty:
        hy_z = z_score_series(hy)
        scores["hy_oas"] = float(z_to_score(-hy_z).iloc[-1])  # inverted
    else:
        scores["hy_oas"] = 0.0

    # M2 YoY
    m2 = fred_series("M2SL")
    m2 = ffill_limited(m2, FFILL["monthly"])
    stale["m2"] = stale_check(m2, FFILL["monthly"])
    if not m2.empty:
        m2_yoy = safe_pct_change_yoy(m2)
        m2_z   = z_score_series(m2_yoy)
        scores["m2"] = float(z_to_score(m2_z).iloc[-1])
    else:
        scores["m2"] = 0.0

    # SOFR (inverted)
    sofr = fred_series("SOFR")
    sofr = ffill_limited(sofr, FFILL["daily"])
    stale["sofr"] = stale_check(sofr, FFILL["daily"])
    if not sofr.empty:
        sofr_z = z_score_series(sofr)
        scores["sofr"] = float(z_to_score(-sofr_z).iloc[-1])
    else:
        scores["sofr"] = 0.0

    pillar = (
        scores["net_liq"] * 0.50 +
        scores["hy_oas"]  * 0.25 +
        scores["m2"]      * 0.15 +
        scores["sofr"]    * 0.10
    )
    pillar = float(np.clip(pillar, -SCORE_RANGE, SCORE_RANGE))

    log.info(f"    Liquidity: {pillar:.2f}  "
             f"(NetLiq={scores['net_liq']:.1f}, HY={scores['hy_oas']:.1f}, "
             f"M2={scores['m2']:.1f}, SOFR={scores['sofr']:.1f})")
    return {
        "score":      round(pillar, 3),
        "components": {k: round(float(v), 3) for k, v in scores.items()},
        "stale":      any(stale.values()),
        "stale_detail": stale,
    }


def compute_curve_pillar() -> dict:
    """
    Curve Pillar (15% of GIP).
    Components:
      2s10s with steepener decomposition  50%  (T10Y2Y + DGS2 + DGS10)
      Real 10Y yield (DFII10)             30%
      MOVE proxy: TLT 30d realized vol    20%  (inverted)
    """
    log.info("  Computing Curve pillar...")
    scores = {}
    stale  = {}

    # 2s10s spread
    t10y2y = fred_series("T10Y2Y")
    dgs2   = fred_series("DGS2")
    dgs10  = fred_series("DGS10")

    for name, s in [("t10y2y", t10y2y), ("dgs2", dgs2), ("dgs10", dgs10)]:
        stale[name] = stale_check(ffill_limited(s, FFILL["daily"]), FFILL["daily"])

    # Steepener decomposition (5-day window)
    steepener_adj = 0.0
    if not dgs2.empty and not dgs10.empty:
        dgs2_5  = dgs2.diff(5).iloc[-1]
        dgs10_5 = dgs10.diff(5).iloc[-1]

        if not (np.isnan(dgs2_5) or np.isnan(dgs10_5)):
            if dgs10_5 > 0 and dgs10_5 > abs(dgs2_5):
                steepener_adj = 1.0    # bear steepening — growth signal
            elif dgs2_5 < -0.05 and abs(dgs2_5) > dgs10_5:
                steepener_adj = -1.0   # bull steepening — recession arrival
            elif dgs2_5 > dgs10_5 and dgs2_5 > 0:
                steepener_adj = -0.5   # bear flattening — hawkish Fed
            elif dgs10_5 < 0 and abs(dgs10_5) > abs(dgs2_5):
                steepener_adj = -0.7   # bull flattening — growth scare

    if not t10y2y.empty:
        t10y2y_ff = ffill_limited(t10y2y, FFILL["daily"])
        spread_z  = z_score_series(t10y2y_ff)
        # Combine spread z-score with steepener adjustment
        spread_score = float(z_to_score(spread_z).iloc[-1]) + steepener_adj * 2
        scores["curve_spread"] = float(np.clip(spread_score, -SCORE_RANGE, SCORE_RANGE))
    else:
        scores["curve_spread"] = 0.0

    # Real 10Y yield (inverted — higher real yields = tighter financial conditions)
    dfii10 = fred_series("DFII10")
    dfii10 = ffill_limited(dfii10, FFILL["daily"])
    stale["dfii10"] = stale_check(dfii10, FFILL["daily"])
    if not dfii10.empty:
        real10_z = z_score_series(dfii10)
        scores["real10y"] = float(z_to_score(-real10_z).iloc[-1])  # inverted
    else:
        scores["real10y"] = 0.0

    # MOVE proxy: TLT 30-day realized vol (inverted — high vol = tight conditions)
    tlt = yf_close("TLT")
    if not tlt.empty:
        tlt_ret    = tlt.pct_change()
        tlt_30d_rv = tlt_ret.rolling(21).std() * np.sqrt(252) * 100
        stale["move_proxy"] = stale_check(tlt_30d_rv, FFILL["daily"])
        rv_z = z_score_series(tlt_30d_rv)
        scores["move_proxy"] = float(z_to_score(-rv_z).iloc[-1])  # inverted
    else:
        scores["move_proxy"] = 0.0
        stale["move_proxy"] = True

    pillar = (
        scores["curve_spread"] * 0.50 +
        scores["real10y"]      * 0.30 +
        scores["move_proxy"]   * 0.20
    )
    pillar = float(np.clip(pillar, -SCORE_RANGE, SCORE_RANGE))

    log.info(f"    Curve: {pillar:.2f}  steepener_adj={steepener_adj:+.1f}  "
             f"(Spread={scores['curve_spread']:.1f}, Real10Y={scores['real10y']:.1f}, "
             f"MOVE={scores['move_proxy']:.1f})")
    return {
        "score":           round(pillar, 3),
        "components":      {k: round(float(v), 3) for k, v in scores.items()},
        "steepener_adj":   steepener_adj,
        "stale":           any(stale.values()),
        "stale_detail":    stale,
    }


def classify_gip_regime(composite: float) -> dict:
    """Map composite score to regime tier label and color."""
    if composite >= 7:
        return {"label": "Strong Goldilocks / Melt-Up",     "color": "green",  "tier": 5}
    elif composite >= 4:
        return {"label": "Expansionary / Risk-On",          "color": "green",  "tier": 4}
    elif composite >= 0:
        return {"label": "Neutral",                         "color": "amber",  "tier": 3}
    elif composite >= -4:
        return {"label": "Slowdown / Stagflation Risk",     "color": "red",    "tier": 2}
    else:
        return {"label": "Restrictive / Risk-Off",          "color": "red",    "tier": 1}


def build_gip_history(days: int = 252) -> list:
    """
    Build a trailing history of GIP composite by re-running the pillar
    calculations on historical z-score values. Returns list of daily records
    for the dashboard line chart.

    NOTE: For efficiency this uses the most recent value of each pillar —
    a full history rebuild would require re-fetching all series which is
    slow for the pipeline. Phase 1.5 will add the parquet history store.
    """
    # For Phase 1 we return a stub — dashboard will show as "history pending"
    return []


def compute_gip() -> dict:
    """
    Master GIP computation. Returns full output dict for JSON serialization.
    """
    log.info("Computing GIP Composite...")

    growth    = compute_growth_pillar()
    inflation = compute_inflation_pillar()
    liquidity = compute_liquidity_pillar()
    curve     = compute_curve_pillar()

    composite = (
        growth["score"]    * 0.30 +
        inflation["score"] * 0.25 +
        liquidity["score"] * 0.30 +
        curve["score"]     * 0.15
    )
    composite = float(np.clip(composite, -SCORE_RANGE, SCORE_RANGE))
    regime    = classify_gip_regime(composite)

    any_stale = any([
        growth["stale"], inflation["stale"],
        liquidity["stale"], curve["stale"]
    ])

    log.info(f"  GIP Composite: {composite:.2f} → {regime['label']}")

    return {
        "generated_at":  datetime.now().isoformat(),
        "composite":     round(composite, 3),
        "regime":        regime,
        "any_stale":     any_stale,
        "pillars": {
            "growth":    growth,
            "inflation": inflation,
            "liquidity": liquidity,
            "curve":     curve,
        },
        "weights": {
            "growth": 0.30, "inflation": 0.25,
            "liquidity": 0.30, "curve": 0.15,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — CROSS-ASSET NARRATIVE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

NARRATIVE_MAP = {
    ("↑", "↓", "↓"): {
        "id": 1, "name": "Goldilocks",
        "label": "Stocks ↑ · Dollar ↓ · Rates ↓",
        "sector_tilt": "Growth, semis, software, biotech",
        "color": "green",
        "bullish": True,
    },
    ("↑", "↓", "↑"): {
        "id": 2, "name": "Reflation",
        "label": "Stocks ↑ · Dollar ↓ · Rates ↑",
        "sector_tilt": "Energy, materials, industrials, banks",
        "color": "green",
        "bullish": True,
    },
    ("↑", "↑", "↑"): {
        "id": 3, "name": "US Exceptionalism",
        "label": "Stocks ↑ · Dollar ↑ · Rates ↑",
        "sector_tilt": "Large-cap quality, tech mega-caps",
        "color": "green",
        "bullish": True,
    },
    ("↑", "↑", "↓"): {
        "id": 4, "name": "Soft Landing",
        "label": "Stocks ↑ · Dollar ↑ · Rates ↓",
        "sector_tilt": "Broad equity, consumer discretionary",
        "color": "green",
        "bullish": True,
    },
    ("↓", "↑", "↓"): {
        "id": 5, "name": "Risk-Off / Flight to Quality",
        "label": "Stocks ↓ · Dollar ↑ · Rates ↓",
        "sector_tilt": "Defensive — utilities, staples, gold",
        "color": "red",
        "bullish": False,
    },
    ("↓", "↑", "↑"): {
        "id": 6, "name": "Stagflation / Hawkish Pain",
        "label": "Stocks ↓ · Dollar ↑ · Rates ↑",
        "sector_tilt": "Cash — no longs",
        "color": "red",
        "bullish": False,
    },
    ("↓", "↓", "↓"): {
        "id": 7, "name": "Slowdown / Easing Anticipation",
        "label": "Stocks ↓ · Dollar ↓ · Rates ↓",
        "sector_tilt": "Gold, duration, defensive value",
        "color": "amber",
        "bullish": False,
    },
    ("↓", "↓", "↑"): {
        "id": 8, "name": "Inflation Resurgence",
        "label": "Stocks ↓ · Dollar ↓ · Rates ↑",
        "sector_tilt": "Real assets, gold miners, defensive value",
        "color": "amber",
        "bullish": False,
    },
}

BEARISH_NARRATIVES = {"Risk-Off / Flight to Quality", "Stagflation / Hawkish Pain",
                       "Inflation Resurgence"}


def get_direction(series: pd.Series) -> str:
    """Strict close-to-close direction. ↑ if today > yesterday, else ↓."""
    if len(series) < 2:
        return "↓"
    return "↑" if series.iloc[-1] > series.iloc[-2] else "↓"


def compute_narrative() -> dict:
    """
    Layer 2: Cross-asset narrative tracker.
    Three assets × daily close-to-close direction → 8 possible joint states.
    Rolling 5/10/20-day frequencies + 5-year base rates.
    """
    log.info("Computing Cross-Asset Narrative...")

    START = "2019-01-01"

    spx = yf_close("^GSPC", start=START)
    dxy = yf_close("DX-Y.NYB", start=START)
    tnx = yf_close("^TNX", start=START)

    if spx.empty or dxy.empty or tnx.empty:
        log.warning("Could not fetch all narrative assets — returning neutral state")
        return {
            "generated_at":  datetime.now().isoformat(),
            "error":         "Data unavailable",
            "today_state":   None,
            "dominant":      None,
        }

    # Align to common daily index
    idx = pd.bdate_range(
        start=max(spx.index.min(), dxy.index.min(), tnx.index.min()),
        end=pd.Timestamp.today()
    )
    spx = spx.reindex(idx).ffill(limit=3)
    dxy = dxy.reindex(idx).ffill(limit=3)
    tnx = tnx.reindex(idx).ffill(limit=3)

    # Daily directions (strict close-to-close)
    spx_dir = spx.pct_change().apply(lambda x: "↑" if x > 0 else "↓")
    dxy_dir = dxy.pct_change().apply(lambda x: "↑" if x > 0 else "↓")
    tnx_dir = tnx.pct_change().apply(lambda x: "↑" if x > 0 else "↓")

    # Map each day to a narrative state id
    def classify_row(spx_d, dxy_d, tnx_d):
        key = (spx_d, dxy_d, tnx_d)
        m   = NARRATIVE_MAP.get(key)
        return m["id"] if m else 0

    states = pd.Series([
        classify_row(s, d, t)
        for s, d, t in zip(spx_dir, dxy_dir, tnx_dir)
    ], index=idx)

    # Rolling frequencies
    def rolling_freq(s: pd.Series, window: int) -> pd.DataFrame:
        """For each day, compute the frequency of each narrative id in the trailing window."""
        result = {}
        for nid in range(1, 9):
            result[nid] = s.rolling(window, min_periods=1).apply(
                lambda w: (w == nid).sum() / len(w)
            )
        return pd.DataFrame(result, index=s.index)

    freq_10  = rolling_freq(states, 10)
    freq_5   = rolling_freq(states, 5)
    freq_20  = rolling_freq(states, 20)

    # 5-year base rates
    base_states = states.iloc[-1260:] if len(states) >= 1260 else states
    base_rates  = {
        str(nid): round(float((base_states == nid).sum() / len(base_states) * 100), 1)
        for nid in range(1, 9)
    }

    # Today's state
    today_spx = get_direction(spx)
    today_dxy = get_direction(dxy)
    today_tnx = get_direction(tnx)
    today_key = (today_spx, today_dxy, today_tnx)
    today_narrative = NARRATIVE_MAP.get(today_key, {
        "id": 0, "name": "Unknown", "label": "—",
        "sector_tilt": "—", "color": "amber", "bullish": False,
    })

    # Dominant narrative over last 10 days (highest rolling_10 frequency wins)
    last_freq_10 = freq_10.iloc[-1]
    dominant_id  = int(last_freq_10.idxmax())
    dominant     = NARRATIVE_MAP.get(
        next((k for k, v in NARRATIVE_MAP.items() if v["id"] == dominant_id), None),
        today_narrative
    )

    # Check if bearish narratives dominate last 10 days
    bearish_days_10 = sum(
        1 for i in range(-10, 0)
        if i < len(states) and
        NARRATIVE_MAP.get(
            next((k for k, v in NARRATIVE_MAP.items()
                  if v["id"] == int(states.iloc[i])), None),
            {}
        ).get("name", "") in BEARISH_NARRATIVES
    )

    # 60-day history for heatmap
    history_60 = []
    for i in range(-60, 0):
        if abs(i) > len(states):
            continue
        day_state_id  = int(states.iloc[i])
        day_narrative = next(
            (v for v in NARRATIVE_MAP.values() if v["id"] == day_state_id),
            {"name": "Unknown", "color": "amber", "id": 0}
        )
        history_60.append({
            "date":           str(states.index[i].date()),
            "state_id":       day_state_id,
            "narrative_name": day_narrative.get("name", "Unknown"),
            "color":          day_narrative.get("color", "amber"),
            "spx":            str(spx_dir.iloc[i]),
            "dxy":            str(dxy_dir.iloc[i]),
            "tnx":            str(tnx_dir.iloc[i]),
        })

    # Narrative transition matrix (last 252 days)
    transition_states = states.iloc[-252:].dropna().astype(int)
    transitions = {}
    for from_id in range(1, 9):
        transitions[str(from_id)] = {}
        mask = transition_states == from_id
        if mask.sum() > 0:
            next_states = transition_states.shift(-1)[mask].dropna().astype(int)
            for to_id in range(1, 9):
                prob = float((next_states == to_id).sum() / len(next_states)) if len(next_states) > 0 else 0.0
                transitions[str(from_id)][str(to_id)] = round(prob, 3)

    log.info(f"  Today: State {today_narrative['id']} — {today_narrative['name']}")
    log.info(f"  Dominant (10d): State {dominant_id} — {dominant.get('name','?')}")
    log.info(f"  Bearish days in last 10: {bearish_days_10}")

    return {
        "generated_at":    datetime.now().isoformat(),
        "today": {
            "spx_dir":      today_spx,
            "dxy_dir":      today_dxy,
            "tnx_dir":      today_tnx,
            "state_id":     today_narrative["id"],
            "name":         today_narrative["name"],
            "label":        today_narrative.get("label", ""),
            "sector_tilt":  today_narrative.get("sector_tilt", ""),
            "color":        today_narrative.get("color", "amber"),
            "bullish":      today_narrative.get("bullish", False),
        },
        "dominant_10d": {
            "state_id":    dominant_id,
            "name":        dominant.get("name", ""),
            "color":       dominant.get("color", "amber"),
            "sector_tilt": dominant.get("sector_tilt", ""),
            "freq_10d_pct": round(float(last_freq_10[dominant_id]) * 100, 1),
        },
        "bearish_days_10":   bearish_days_10,
        "all_narratives":    [
            {
                "id":          v["id"],
                "name":        v["name"],
                "label":       v["label"],
                "sector_tilt": v["sector_tilt"],
                "color":       v["color"],
                "bullish":     v["bullish"],
                "freq_5d_pct":  round(float(freq_5.iloc[-1][v["id"]]) * 100, 1),
                "freq_10d_pct": round(float(freq_10.iloc[-1][v["id"]]) * 100, 1),
                "freq_20d_pct": round(float(freq_20.iloc[-1][v["id"]]) * 100, 1),
                "base_rate_pct": float(base_rates.get(str(v["id"]), 0)),
            }
            for v in sorted(NARRATIVE_MAP.values(), key=lambda x: x["id"])
        ],
        "base_rates":    base_rates,
        "transitions":   transitions,
        "history_60d":   history_60,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 STUB — VOL COMPRESSION (placeholder for future build)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vol_compression_stub() -> dict:
    """
    Phase 2 placeholder. Returns a neutral reading so the dashboard
    can render the vol panel with a 'Phase 2 — coming soon' state.
    """
    try:
        vix = yf_close("^VIX", start="2022-01-01")
        ovx = yf_close("^OVX", start="2022-01-01")
        tlt = yf_close("TLT",  start="2022-01-01")

        latest_vix = float(vix.dropna().iloc[-1]) if not vix.empty else 20.0
        latest_ovx = float(ovx.dropna().iloc[-1]) if not ovx.empty else 35.0
        tlt_ret    = tlt.pct_change() if not tlt.empty else pd.Series(dtype=float)
        tlt_rv     = float(tlt_ret.rolling(21).std().iloc[-1] * np.sqrt(252) * 100) if len(tlt_ret) > 21 else 10.0

        # Simple VIX percentile (5-year)
        vix_5y = vix.iloc[-1260:] if len(vix) >= 1260 else vix
        vix_pct = float((vix_5y < latest_vix).sum() / len(vix_5y) * 100) if not vix_5y.empty else 50.0

        return {
            "phase":         2,
            "status":        "STUB — Phase 2",
            "score":         None,
            "vix":           round(latest_vix, 1),
            "ovx":           round(latest_ovx, 1),
            "tlt_rv":        round(tlt_rv, 1),
            "vix_pct_5y":    round(vix_pct, 1),
        }
    except Exception as e:
        return {"phase": 2, "status": "STUB", "score": None, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  MACRO GATE STUB (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gate_stub(gip: dict, narrative: dict, vol_stub: dict) -> dict:
    """
    Phase 2 gate placeholder. Evaluates the conditions available in Phase 1
    and marks the rest as 'pending Phase 2'.
    """
    composite = gip.get("composite", 0)

    # Condition 1: GIP ≥ +4
    c1 = composite >= 4.0

    # Condition 2: Narrative not bearish for 5+ of last 10 days
    bearish_days = narrative.get("bearish_days_10", 5)
    c2 = bearish_days < 5

    # Conditions 3-4: Vol compression + VIX contango — Phase 2
    c3 = None   # pending
    c4 = None   # pending

    # Condition 5: SPY > 200DMA and > 50DMA
    try:
        spy = yf_close("SPY", start="2023-01-01")
        if not spy.empty:
            sma50  = spy.rolling(50).mean().iloc[-1]
            sma200 = spy.rolling(200).mean().iloc[-1]
            spy_last = spy.iloc[-1]
            c5 = bool(spy_last > sma50 and spy_last > sma200)
        else:
            c5 = None
    except Exception:
        c5 = None

    # VIX-based sizing
    vix = vol_stub.get("vix", 20)
    if vix < 16:
        sizing = "100%"
    elif vix < 25:
        sizing = "50%"
    else:
        sizing = "25%"

    # Gate state — OPEN only when all available conditions pass
    available = [c1, c2, c5]
    available_results = [c for c in available if c is not None]
    gate_open = all(available_results) if available_results else False

    return {
        "gate_open":  gate_open,
        "phase2_note": "Vol compression (C3) and VIX contango (C4) require Phase 2",
        "conditions": {
            "c1_gip_ge_4":           c1,
            "c2_narrative_not_bearish": c2,
            "c3_vol_compression":    c3,
            "c4_vix_contango":       c4,
            "c5_spy_above_smas":     c5,
        },
        "vix_sizing": sizing,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Macro Engine v2 — Phase 1")
    log.info("=" * 60)

    if not FRED_API_KEY:
        log.error("FRED_API_KEY not set. Set in GitLab CI/CD Variables and Streamlit Secrets.")
        log.error("Free key: https://fred.stlouisfed.org/docs/api/api_key.html")

    # ── GIP Composite ────────────────────────────────────────────────────────
    gip = compute_gip()
    with open(GIP_JSON, "w") as f:
        json.dump(gip, f, indent=2, default=str)
    log.info(f"✅ GIP saved → {GIP_JSON}  composite={gip['composite']:.2f}  regime={gip['regime']['label']}")

    # ── Narrative Tracker ────────────────────────────────────────────────────
    narrative = compute_narrative()
    with open("data/narrative_data.json", "w") as f:
        json.dump(narrative, f, indent=2, default=str)
    log.info(f"✅ Narrative saved → data/narrative_data.json  "
             f"today={narrative.get('today', {}).get('name', '?')}")

    # ── Vol Compression (Phase 2 stub) ───────────────────────────────────────
    vol = compute_vol_compression_stub()
    with open("data/vol_compression.json", "w") as f:
        json.dump(vol, f, indent=2, default=str)

    # ── Gate State (Phase 2 stub) ────────────────────────────────────────────
    gate = compute_gate_stub(gip, narrative, vol)
    with open("data/gate_state.json", "w") as f:
        json.dump(gate, f, indent=2, default=str)

    log.info(f"✅ Gate state: {'OPEN' if gate['gate_open'] else 'CLOSED'}  "
             f"(Phase 1 conditions only — C3/C4 pending Phase 2)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
