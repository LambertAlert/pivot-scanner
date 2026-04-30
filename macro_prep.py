"""
macro_prep.py — Runs daily after close (or on-demand).
Fetches SPY/QQQ/VIX, macro indicators, sector ETFs.
Computes regimes, transition matrices, regime-sector stats.
Saves: data/macro_data.json (current state) + SQLite history.

Source: Indices_Analysis.ipynb (adapted for GitLab pipeline)
Dependencies: pip install yfinance pandas numpy polygon-api-client
"""

import os
import json
import warnings
import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
POLYGON_API_KEY  = os.getenv("POLYGON_API_KEY", "")
LOOKBACK_YEARS   = 4          # keep manageable for free tier
RATE_LIMIT_SLEEP = 13         # polygon free: 5 req/min
OUTPUT_JSON      = "data/macro_data.json"
os.makedirs("data", exist_ok=True)

SECTOR_TICKERS = {
    "XLK": "Technology",    "XLF": "Financials",   "XLE": "Energy",
    "XLV": "Health Care",   "XLI": "Industrials",  "XLY": "Cons Disc",
    "XLP": "Cons Staples",  "XLB": "Materials",    "XLU": "Utilities",
    "XLC": "Comm Svcs",     "XLRE": "Real Estate",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_yf(ticker: str, period: str = "5y") -> pd.Series:
    """Fetch adjusted close via yfinance. Returns named Series indexed by date."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float, name=ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].squeeze()
        s.index = pd.to_datetime(s.index)
        s.name = ticker
        return s
    except Exception as e:
        log.warning(f"yfinance {ticker}: {e}")
        return pd.Series(dtype=float, name=ticker)


def fetch_polygon_daily(ticker: str, label: str) -> pd.Series:
    """Fetch via Polygon if API key set, else fall back to yfinance."""
    if not POLYGON_API_KEY:
        return fetch_yf(ticker)
    try:
        from polygon import RESTClient
        client = RESTClient(POLYGON_API_KEY)
        start = (datetime.now() - timedelta(days=365 * LOOKBACK_YEARS)).strftime("%Y-%m-%d")
        end   = datetime.now().strftime("%Y-%m-%d")
        aggs  = list(client.get_aggs(ticker, 1, "day", start, end, limit=50000))
        if not aggs:
            return fetch_yf(ticker)
        records = {pd.Timestamp(a.timestamp, unit="ms").normalize(): a.close for a in aggs}
        s = pd.Series(records, name=label)
        s.index = pd.to_datetime(s.index)
        time.sleep(RATE_LIMIT_SLEEP)
        return s
    except Exception as e:
        log.warning(f"Polygon {ticker}: {e} — falling back to yfinance")
        time.sleep(1)
        return fetch_yf(ticker)


# ── Core build ────────────────────────────────────────────────────────────────

def build_macro_df() -> dict:
    log.info("Fetching SPY, QQQ, VIX...")
    spy = fetch_polygon_daily("SPY", "SPY")
    qqq = fetch_polygon_daily("QQQ", "QQQ")
    vix = fetch_yf("^VIX", period="5y").rename("VIX")

    df = pd.concat([spy, qqq, vix], axis=1).sort_index().dropna(subset=["SPY", "QQQ"])
    df.columns = ["SPY", "QQQ", "VIX"]
    df.index = pd.to_datetime(df.index)

    # Returns + momentum
    for col in ["SPY", "QQQ"]:
        df[f"{col}_ret"] = df[col].pct_change()
        for p in [1, 5, 10, 20]:
            df[f"{col}_{p}d"] = df[col].pct_change(p)

    # Rolling vol
    for w in [21, 63, 252]:
        df[f"SPY_vol_{w}"] = df["SPY_ret"].rolling(w).std() * np.sqrt(252) * 100
        df[f"QQQ_vol_{w}"] = df["QQQ_ret"].rolling(w).std() * np.sqrt(252) * 100

    # Macro — use yfinance for free
    log.info("Fetching macro indicators...")
    macro_map = {"^TNX": "US10Y", "DX-Y.NYB": "DXY", "GC=F": "GOLD", "CL=F": "OIL"}
    for yf_sym, label in macro_map.items():
        s = fetch_yf(yf_sym).rename(label)
        df = df.join(s, how="left")
    macro_cols = list(macro_map.values())
    df[macro_cols] = df[macro_cols].ffill()

    if "US10Y" in df.columns:
        df["SPY_vs_10Y_corr_60"] = df["SPY_ret"].rolling(60).corr(df["US10Y"].pct_change())

    # Yield curve
    try:
        us2y = fetch_yf("^IRX").rename("US2Y") / 100
        df = df.join(us2y, how="left")
        df["US2Y"] = df["US2Y"].ffill()
        df["Yield_Curve"] = df["US10Y"] - df["US2Y"]
    except Exception:
        pass

    # Sector ETFs
    log.info("Fetching sector ETFs...")
    for sym in SECTOR_TICKERS:
        s = fetch_polygon_daily(sym, sym)
        df = df.join(s, how="left")
    sector_cols = [s for s in SECTOR_TICKERS if s in df.columns]
    df[sector_cols] = df[sector_cols].ffill()
    for s in sector_cols:
        df[f"{s}_ret"]        = df[s].pct_change()
        df[f"{s}_vs_SPY_20d"] = (df[s] / df["SPY"]).pct_change(20) * 100
        df[f"{s}_mom_60d"]    = df[s].pct_change(60) * 100

    df = df.dropna(subset=["SPY", "QQQ"])

    # ── Regime classification ─────────────────────────────────────────────────
    log.info("Computing market regimes...")
    df["vix_regime"] = pd.cut(
        df["VIX"].fillna(20),
        bins=[0, 19, 30, np.inf],
        labels=["Low Vol (Investable)", "Moderate (Chop)", "High Vol (Fear)"],
    ).astype(str)

    rolling_highs = df["SPY"].rolling(10).max()
    rolling_lows  = df["SPY"].rolling(10).min()
    df["_lower_highs"] = rolling_highs < rolling_highs.shift(10) * 0.99
    df["_higher_lows"] = rolling_lows  > rolling_lows.shift(10) * 1.01
    df["_recent_ret"]  = df["SPY"].pct_change(20) * 100
    df["_corr_60"]     = df["SPY_ret"].rolling(60).corr(df["QQQ_ret"])

    def classify_regime(row):
        if row["_lower_highs"] and row["_recent_ret"] < -1 and row.get("SPY_vol_21", 0) > 20:
            return "Topping"
        if row["_higher_lows"] and row["_recent_ret"] > 1 and row.get("SPY_vol_21", 0) > 20:
            return "Bottoming"
        if abs(row["_corr_60"]) > 0.65 and row.get("SPY_vol_21", 99) < 18:
            return "Strong Trending"
        if row["_corr_60"] < 0.45 and row.get("SPY_vol_21", 99) < 16:
            return "Mean Reverting"
        return "Transition / Other"

    df["price_regime"]    = df.apply(classify_regime, axis=1)
    df["combined_regime"] = df["price_regime"] + " + " + df["vix_regime"]
    df.drop(columns=["_lower_highs", "_higher_lows", "_recent_ret", "_corr_60"], inplace=True)

    # ── Transition matrices ───────────────────────────────────────────────────
    log.info("Building transition matrices...")
    unique_regimes = sorted(df["combined_regime"].dropna().unique())

    def build_transition(days_ahead: int) -> dict:
        s = df["combined_regime"].dropna()
        from_r = s.iloc[:-days_ahead]
        to_r   = s.shift(-days_ahead).loc[from_r.index].dropna()
        from_r = from_r.loc[to_r.index]
        counts = pd.crosstab(
            pd.Categorical(from_r, categories=unique_regimes),
            pd.Categorical(to_r,   categories=unique_regimes),
        )
        row_sums = counts.sum(axis=1)
        probs = counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0)
        return probs.to_dict()

    transitions = {
        "1d":  build_transition(1),
        "5d":  build_transition(5),
        "10d": build_transition(10),
    }

    # ── Regime-sector forward stats ───────────────────────────────────────────
    log.info("Computing regime-sector stats...")
    all_assets = ["SPY", "QQQ"] + sector_cols
    for asset in all_assets:
        c = df[asset]
        df[f"{asset}_fwd_5d"]  = c.pct_change(5).shift(-5)  * 100
        df[f"{asset}_fwd_20d"] = c.pct_change(20).shift(-20) * 100

    regime_stats = {}
    for regime_name in df["combined_regime"].dropna().unique():
        rdata = df[df["combined_regime"] == regime_name]
        if len(rdata) < 20:
            continue
        stats = {}
        for asset in all_assets:
            if f"{asset}_ret" not in df.columns:
                continue
            ret = rdata[f"{asset}_ret"]
            std = ret.std()
            stats[asset] = {
                "Count":             int(len(rdata)),
                "Avg_5d_Return":     round(float(rdata[f"{asset}_fwd_5d"].mean()), 2),
                "Avg_20d_Return":    round(float(rdata[f"{asset}_fwd_20d"].mean()), 2),
                "Win_Rate_5d":       round(float((rdata[f"{asset}_fwd_5d"] > 0).mean() * 100), 1),
                "Volatility_Ann":    round(float(std * np.sqrt(252) * 100), 2),
                "Sharpe":            round(float(ret.mean() / std * np.sqrt(252)) if std > 0 else 0, 2),
            }
        regime_stats[regime_name] = stats

    # ── Current snapshot ──────────────────────────────────────────────────────
    last = df.iloc[-1]
    current_snapshot = {
        "date":             str(df.index[-1].date()),
        "SPY":              round(float(last["SPY"]), 2),
        "QQQ":              round(float(last["QQQ"]), 2),
        "VIX":              round(float(last["VIX"]), 1) if pd.notna(last["VIX"]) else None,
        "SPY_ret_1d":       round(float(last["SPY_ret"] * 100), 2) if pd.notna(last.get("SPY_ret")) else None,
        "QQQ_ret_1d":       round(float(last["QQQ_ret"] * 100), 2) if pd.notna(last.get("QQQ_ret")) else None,
        "SPY_20d":          round(float(last.get("SPY_20d", 0) * 100), 2),
        "QQQ_20d":          round(float(last.get("QQQ_20d", 0) * 100), 2),
        "SPY_vol_21":       round(float(last.get("SPY_vol_21", 0)), 1),
        "price_regime":     str(last["price_regime"]),
        "vix_regime":       str(last["vix_regime"]),
        "combined_regime":  str(last["combined_regime"]),
        "US10Y":            round(float(last["US10Y"]), 3) if "US10Y" in last and pd.notna(last["US10Y"]) else None,
        "DXY":              round(float(last["DXY"]), 2)   if "DXY"   in last and pd.notna(last["DXY"])   else None,
        "GOLD":             round(float(last["GOLD"]), 2)  if "GOLD"  in last and pd.notna(last["GOLD"])  else None,
        "OIL":              round(float(last["OIL"]), 2)   if "OIL"   in last and pd.notna(last["OIL"])   else None,
        "Yield_Curve":      round(float(last["Yield_Curve"]), 3) if "Yield_Curve" in last and pd.notna(last["Yield_Curve"]) else None,
    }

    # Sector performance snapshot
    sector_snapshot = []
    for sym, name in SECTOR_TICKERS.items():
        if sym not in df.columns:
            continue
        row = {
            "ticker": sym,
            "name":   name,
            "price":  round(float(last[sym]), 2) if pd.notna(last[sym]) else None,
            "vs_SPY_20d": round(float(last.get(f"{sym}_vs_SPY_20d", 0)), 2),
            "mom_60d":    round(float(last.get(f"{sym}_mom_60d", 0)), 2),
        }
        sector_snapshot.append(row)

    # Recent regime history (last 60 rows)
    regime_history_df = df[["combined_regime", "SPY", "QQQ", "VIX"]].tail(60).copy()
    # Convert the date index to a string column regardless of its name
    regime_history_df = regime_history_df.reset_index()
    regime_history_df.columns = ["date"] + list(regime_history_df.columns[1:])
    regime_history_df["date"] = regime_history_df["date"].astype(str)
    regime_history = regime_history_df.fillna(0).to_dict(orient="records")

    output = {
        "generated_at":    datetime.now().isoformat(),
        "current":         current_snapshot,
        "sector_snapshot": sector_snapshot,
        "unique_regimes":  unique_regimes,
        "regime_stats":    regime_stats,
        "transitions":     transitions,
        "regime_history":  regime_history,
    }

    return output


def main():
    log.info("Starting macro_prep.py...")
    data = build_macro_df()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2, default=str)

    log.info(f"✅ Saved to {OUTPUT_JSON}")
    log.info(f"   Current regime: {data['current']['combined_regime']}")
    log.info(f"   VIX: {data['current']['VIX']}")


if __name__ == "__main__":
    main()
