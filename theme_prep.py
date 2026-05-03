"""
theme_prep.py — Runs daily after close.
Replicates Sector_Theme.ipynb momentum engine v3.0.
Fetches all thematic ETFs, computes RS rankings + momentum scores.
Saves: data/theme_data.json

Source: Sector_Theme.ipynb (adapted for GitLab pipeline)
"""

import os
import json
import logging
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.makedirs("data", exist_ok=True)
OUTPUT_JSON = "data/theme_data.json"

# ── Full theme universe (from Sector_Theme.ipynb) ────────────────────────────
THEMES = {
    # ── Technology & Innovation ───────────────────────────────────────────────
    "Semiconductors": [
        "SOXX", "SMH", "PSI", "SOXQ", "XSD", "FTXL",
    ],
    "AI & Future Tech": [
        "ARTY", "AIQ", "CHAT", "THNQ",
        "ARKK",       # active innovation — includes AI leaders
    ],
    "Robotics & Automation": [
        "BOTZ", "ROBO", "IRBO", "ARKQ",
    ],
    "Cybersecurity": [
        "CIBR", "HACK", "BUG", "IHAK",
    ],
    "Cloud Computing": [
        "SKYY", "WCLD", "CLOU",
    ],
    "Expanded Tech / Software": [
        "IGV", "ARKW",
    ],
    "Data Centers / Digital Infra": [
        "DTCR", "SRVR",
    ],
    "Quantum Computing": [
        "QTUM",
    ],
    "Humanoid Robotics": [
        "KOID",
    ],

    # ── Defense, Space & Drones ───────────────────────────────────────────────
    "Defense & Aerospace": [
        "ITA", "SHLD", "PPA", "XAR",
    ],
    "Drones / UAV": [
        "DRNZ", "HELO",
    ],
    "Space Tech": [
        "UFO", "ARKX",
    ],

    # ── Energy & Resources ────────────────────────────────────────────────────
    "Nuclear / Uranium": [
        "URA", "NLR", "URNM", "NUKZ",
    ],
    "Solar Energy": [
        "TAN", "RAYS",
    ],
    "Clean Energy / Power": [
        "ICLN", "LIT", "HYDR", "CNRG", "QCLN",
    ],
    "Oil & Gas / Upstream": [
        "XOP", "OIH", "GUSH",
    ],
    "Rare Earth / Strategic Metals": [
        "REMX", "COPX", "PICK",
    ],
    "Gold & Precious Metals": [
        "GDX", "GDXJ", "SIL", "SGDM",
    ],
    "Commodities / Inflation": [
        "GSG", "PDBC", "DBA",
    ],

    # ── Health & Life Sciences ────────────────────────────────────────────────
    "Biotechnology": [
        "IBB", "XBI", "ARKG", "BBH", "PBE",
    ],
    "Healthcare Innovation": [
        "IDNA", "GNOM", "EDOC",
    ],
    "Aging Population": [
        "AGNG",
    ],

    # ── Consumer & Real Estate ────────────────────────────────────────────────
    "Home Construction": [
        "ITB", "XHB",
    ],
    "Consumer Discretionary": [
        "XLY", "IBUY", "FDIS",
    ],
    "Gaming / Esports": [
        "HERO", "NERD", "GAMR",
    ],

    # ── Finance & Digital Assets ──────────────────────────────────────────────
    "Fintech": [
        "FINX", "ARKF", "IPAY",
    ],
    "Crypto / Blockchain": [
        "BKCH", "IBIT", "FBTC", "BLOK", "BITQ",
    ],

    # ── EV & Mobility ─────────────────────────────────────────────────────────
    "EV & Self-Driving Tech": [
        "IDRV", "DRIV", "KARS",
    ],

    # ── Infrastructure & Utilities ────────────────────────────────────────────
    "Clean Water": [
        "AQWA", "PHO", "FIW",
    ],

    # ── Global / Geopolitical Themes ──────────────────────────────────────────
    "India": [
        "INDA", "SMIN", "INCO",
    ],
    "Japan": [
        "EWJ", "DXJ", "JPXN",
    ],
    "Emerging Markets": [
        "EEM", "VWO", "FM",
    ],
    "Latin America": [
        "ILF", "EWZ",
    ],

    # ── Size & Style Factors ──────────────────────────────────────────────────
    "Small Cap": [
        "IWM", "IJR", "VBK",
    ],
    "Dividend / Value": [
        "VYM", "SCHD", "DVY", "VTV",
    ],

    # ── Fixed Income / Macro ──────────────────────────────────────────────────
    "Bonds / Duration": [
        "TLT", "IEF", "HYG", "LQD",
    ],

    # ── Broad Sector SPDRs ────────────────────────────────────────────────────
    "Broad Sectors": [
        "XLK", "XLV", "XLF", "XLE", "XLI",
        "XLY", "XLP", "XLB", "XLU", "XLC", "XLRE",
    ],
}

PERIODS = [5, 10, 20, 40, 50]
WEIGHTS = {5: 0.40, 10: 0.20, 20: 0.20, 40: 0.10, 50: 0.10}


def build_theme_data() -> dict:
    # De-duplicate themes
    seen = set()
    deduped = {}
    for theme, tickers in THEMES.items():
        clean = [t for t in tickers if t not in seen]
        seen.update(clean)
        deduped[theme] = clean

    all_tickers = list(seen) + ["SPY"]
    log.info(f"Downloading {len(all_tickers)} ETFs...")

    data = yf.download(all_tickers, period="70d", auto_adjust=True, progress=False)
    close  = data["Close"].copy().ffill() if isinstance(data, pd.DataFrame) else data["Close"].copy().ffill()
    high   = data["High"].copy().ffill()
    volume = data["Volume"].copy().ffill()

    # Flatten MultiIndex if present
    if isinstance(close.columns, pd.MultiIndex):
        close.columns  = close.columns.get_level_values(0)
        high.columns   = high.columns.get_level_values(0)
        volume.columns = volume.columns.get_level_values(0)

    log.info(f"Data through {close.index[-1].date()}")

    # Drop tickers with insufficient history
    min_rows = 52
    valid_tickers = [t for t in close.columns if close[t].count() >= min_rows]
    close  = close[[t for t in valid_tickers]]
    high   = high[[t  for t in valid_tickers if t in high.columns]]
    volume = volume[[t for t in valid_tickers if t in volume.columns]]

    spy_close = close["SPY"]

    # ── RS Rankings ─────────────────────────────────────────────────────────
    results = []
    for p in PERIODS:
        etf_ret  = close.pct_change(periods=p).iloc[-1]
        spy_ret  = spy_close.pct_change(periods=p).iloc[-1]
        rs_ratio = (1 + etf_ret) / (1 + spy_ret)
        temp = pd.DataFrame({
            "Ticker":          rs_ratio.index,
            f"RS_{p}d":        rs_ratio.values,
            f"Return_{p}d_%":  (etf_ret.values * 100).round(2),
        }).set_index("Ticker")
        results.append(temp)

    rs_df = pd.concat(results, axis=1).reset_index()
    rs_df = rs_df[rs_df["Ticker"] != "SPY"].copy()

    ticker_to_theme = {t: theme for theme, tickers in deduped.items() for t in tickers}
    rs_df["Theme"] = rs_df["Ticker"].map(ticker_to_theme).fillna("Unknown")

    rs_df["Weighted_RS"] = sum(
        rs_df[f"RS_{p}d"].fillna(0) * w for p, w in WEIGHTS.items()
    )

    # 52-week distance
    high_52w   = high[[t for t in high.columns if t != "SPY"]].rolling(252).max().iloc[-1]
    last_close = close.iloc[-1]
    rs_df["Dist_52wHigh_%"] = rs_df["Ticker"].map(
        lambda t: round((last_close.get(t, np.nan) / high_52w.get(t, np.nan) - 1) * 100, 1)
        if t in high_52w.index else np.nan
    )

    # ── Momentum Engine v3.0 ─────────────────────────────────────────────────
    log.info("Computing momentum scores (v3.0)...")
    spy_full        = spy_close.reindex(close.index).ffill()
    rs_ratio_matrix = close.div(spy_full, axis=0)
    rs_ema12_matrix = rs_ratio_matrix.ewm(span=12, adjust=False).mean()
    rs_ema30_matrix = rs_ratio_matrix.ewm(span=30, adjust=False).mean()
    rs_macd_line    = rs_ema12_matrix - rs_ratio_matrix.ewm(span=26, adjust=False).mean()
    rs_macd_signal  = rs_macd_line.ewm(span=9, adjust=False).mean()
    rs_macd_hist    = rs_macd_line - rs_macd_signal

    price_ema9  = close.ewm(span=9,  adjust=False).mean()
    price_ema21 = close.ewm(span=21, adjust=False).mean()
    price_ema50 = close.ewm(span=50, adjust=False).mean()

    mom_records = []
    for tk in rs_df["Ticker"]:
        if tk not in close.columns or close[tk].dropna().shape[0] < 52:
            mom_records.append({"Ticker": tk, "Momentum_Score": 5, "Momentum_Label": "➡ Flat",
                                "Score_A": 0, "Score_B": 0, "Score_C": 0, "Score_D": 0, "Score_E": 0,
                                "Full_Bull_Stack": False})
            continue

        score = 0

        # A: RS EMA Crossover
        e12 = rs_ema12_matrix[tk].iloc[-1]
        e30 = rs_ema30_matrix[tk].iloc[-1]
        emom = e12 - e30
        a1, a2 = int(e12 > e30), int(emom > 0.005)
        score += a1 + a2

        # B: RS Regression Slope
        recent = rs_ema12_matrix[tk].dropna().iloc[-12:]
        slope = np.polyfit(np.arange(len(recent)), recent.values, 1)[0] if len(recent) >= 8 else 0.0
        b1, b2 = int(slope > 0), int(slope > 0.00015)
        score += b1 + b2

        # C: RS MACD Histogram
        hist_series = rs_macd_hist[tk].dropna()
        hist_now  = hist_series.iloc[-1]
        hist_prev = hist_series.iloc[-2] if len(hist_series) > 1 else hist_now
        c1, c2 = int(hist_now > 0), int(hist_now > hist_prev)
        score += c1 + c2

        # D: Volume Quality
        if tk in volume.columns:
            vol        = volume[tk].reindex(close.index).ffill().dropna()
            returns_   = close[tk].reindex(vol.index).pct_change()
            up_mask    = returns_ > 0
            avg_vol_20 = vol.iloc[-20:].mean()
            avg_vol_10 = vol.iloc[-10:].mean()
            up_vol_10  = vol[up_mask].iloc[-10:].mean() if up_mask.sum() >= 3 else np.nan
            rvol = (up_vol_10 / avg_vol_20) if pd.notna(up_vol_10) and avg_vol_20 > 0 else np.nan
            d1 = int(pd.notna(rvol) and rvol >= 1.5)
            d2 = int(avg_vol_20 > 0 and avg_vol_10 > avg_vol_20)
        else:
            d1, d2 = 0, 1
        score += d1 + d2

        # E: EMA Stack
        e9  = price_ema9[tk].iloc[-1]
        e21 = price_ema21[tk].iloc[-1]
        e50 = price_ema50[tk].iloc[-1]
        e50s = price_ema50[tk].dropna()
        slope_50 = bool(e50s.iloc[-1] > e50s.iloc[-6]) if len(e50s) >= 6 else False
        full_bull = bool(e9 > e21 and e21 > e50)
        e1, e2 = int(slope_50), int(full_bull)
        score += e1 + e2

        label = ("🔥 Leading" if score >= 8 else "✅ Improving" if score >= 6 else
                 "➡ Flat" if score >= 4 else "⚠ Fading" if score >= 2 else "🔴 Broken")

        mom_records.append({
            "Ticker": tk, "Momentum_Score": score, "Momentum_Label": label,
            "Score_A": a1+a2, "Score_B": b1+b2, "Score_C": c1+c2,
            "Score_D": d1+d2, "Score_E": e1+e2,
            "Full_Bull_Stack": full_bull,
        })

    mom_df = pd.DataFrame(mom_records)
    rs_df  = rs_df.merge(mom_df, on="Ticker", how="left")

    # ── Rankings ──────────────────────────────────────────────────────────────
    rank_cols = []
    for p in PERIODS:
        rcol = f"Rank_{p}d"
        rs_df[rcol] = rs_df[f"RS_{p}d"].rank(ascending=False, method="min", na_option="bottom").astype(int)
        rank_cols.append(rcol)

    rs_df["Avg_RS_Rank"]      = rs_df[rank_cols].mean(axis=1).round(1)
    rs_df["Weighted_RS_Rank"] = rs_df["Weighted_RS"].rank(ascending=False, method="min", na_option="bottom").astype(int)
    rs_df = rs_df.sort_values("Weighted_RS_Rank")

    # ── Theme aggregation ─────────────────────────────────────────────────────
    theme_agg = rs_df.groupby("Theme").agg(
        ETF_Count       = ("Ticker",          "count"),
        Avg_Mom_Score   = ("Momentum_Score",   "mean"),
        Leading_Count   = ("Momentum_Label",   lambda x: (x == "🔥 Leading").sum()),
        Improving_Count = ("Momentum_Label",   lambda x: (x == "✅ Improving").sum()),
        Bull_Stack_Pct  = ("Full_Bull_Stack",  lambda x: round(x.sum() / max(len(x), 1) * 100, 1)),
        MACD_Bull_Pct   = ("Score_C",          lambda x: round((x == 2).mean() * 100, 1)),
        **{f"RS_{p}d": (f"Return_{p}d_%", "mean") for p in PERIODS},
    ).reset_index()

    for p in PERIODS:
        theme_agg[f"RS_{p}d_rank"] = theme_agg[f"RS_{p}d"].rank(ascending=False, method="min")
    theme_agg["Avg_Theme_Rank"] = theme_agg[[f"RS_{p}d_rank" for p in PERIODS]].mean(axis=1).round(1)
    theme_agg = theme_agg.drop(columns=[f"RS_{p}d_rank" for p in PERIODS])
    theme_agg = theme_agg.sort_values("Avg_Theme_Rank")

    # ── Serialize ─────────────────────────────────────────────────────────────
    def safe_records(df_):
        return df_.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")

    output = {
        "generated_at":   datetime.now().isoformat(),
        "etf_rankings":   safe_records(rs_df),
        "theme_rankings": safe_records(theme_agg),
        "metadata": {
            "as_of":       datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_etfs":  len(rs_df),
            "total_themes": len(deduped),
            "periods":     PERIODS,
            "weights":     WEIGHTS,
        },
    }

    return output


def main():
    log.info("Starting theme_prep.py...")
    data = build_theme_data()
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"✅ Saved {OUTPUT_JSON}  ({data['metadata']['total_etfs']} ETFs, {data['metadata']['total_themes']} themes)")


if __name__ == "__main__":
    main()
