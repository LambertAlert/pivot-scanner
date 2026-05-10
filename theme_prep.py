"""
theme_prep.py — Theme Tracker Engine v4.0
==========================================
Replaces momentum-based scoring with IBD-style Relative Strength vs SPY.
"""
import os, json, logging, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
os.makedirs("data", exist_ok=True)
OUTPUT_JSON = "data/theme_data.json"

THEMES = {
    "Broad Market":         ["QQQ","IWM","MDY","DIA"],
    "Semiconductors":       ["SOXX","SMH","PSI","SOXQ","XSD","FTXL"],
    "Software & Cloud":     ["IGV","WCLD","CLOU","SKYY"],
    "AI & Innovation":      ["THNQ","AIQ","CHAT","ARKK","ARKW"],
    "Robotics & Automation":["BOTZ","ROBO","IRBO","ARKQ"],
    "Cybersecurity":        ["CIBR","HACK","BUG","IHAK"],
    "Data Centers":         ["DTCR","SRVR","GRID"],
    "Fintech":              ["FINX","IPAY","ARKF"],
    "Crypto & Blockchain":  ["BKCH","IBIT","FBTC","BLOK","BITQ"],
    "Speculative Basket":   ["QQQE","IWO","XSMO","MTUM"],
    "Defense & Aerospace":  ["ITA","PPA","XAR","SHLD"],
    "Space & Drones":       ["UFO","ARKX","DRNZ","HELO"],
    "Industrials":          ["XLI","IYT","PAVE","CARZ"],
    "Home Construction":    ["ITB","XHB"],
    "EV & Mobility":        ["IDRV","DRIV","KARS"],
    "Clean Water":          ["PHO","FIW","AQWA"],
    "Oil & Gas":            ["XOP","OIH","XLE"],
    "Nuclear & Uranium":    ["URA","URNM","NLR","NUKZ"],
    "Clean Energy":         ["ICLN","TAN","QCLN","CNRG"],
    "Lithium & Batteries":  ["LIT","BATT"],
    "Rare Earth & Metals":  ["REMX","COPX","PICK","GUNR"],
    "Gold & Precious":      ["GDX","GDXJ","SIL","SGDM"],
    "Commodities":          ["GSG","PDBC","DBA"],
    "Biotechnology":        ["IBB","XBI","ARKG","BBH","PBE"],
    "Healthcare":           ["XLV","IDNA","GNOM","EDOC"],
    "Quantum Computing":    ["QTUM"],
    "Technology Sector":    ["XLK"],
    "Financials Sector":    ["XLF","KRE","IAT"],
    "Consumer Disc":        ["XLY","IBUY","FDIS"],
    "Consumer Staples":     ["XLP","VDC"],
    "Materials":            ["XLB","VAW"],
    "Utilities":            ["XLU","IDU"],
    "Real Estate":          ["XLRE","VNQ","REZ"],
    "Communication Svcs":   ["XLC","VOX"],
    "Small Cap":            ["IJR","VBK","IWN"],
    "Dividend & Value":     ["VYM","SCHD","DVY","VTV"],
    "India":                ["INDA","SMIN","INCO"],
    "Japan":                ["EWJ","DXJ","JPXN"],
    "Emerging Markets":     ["EEM","VWO","FM"],
    "China":                ["KWEB","FXI","MCHI"],
    "Latin America":        ["ILF","EWZ"],
    "Bonds & Duration":     ["TLT","IEF","LQD","HYG"],
    "Gaming / Esports":     ["HERO","NERD","GAMR"],
}


def compute_raw_rs_factor(s: pd.Series) -> float:
    """IBD RS raw factor: 0.4*(C/C63)+0.2*(C/C126)+0.2*(C/C189)+0.2*(C/C252)"""
    s = s.dropna()
    n = len(s)
    if n < 63:
        return np.nan
    c = float(s.iloc[-1])
    def r(p):
        if n <= p: return np.nan
        past = float(s.iloc[-(p+1)])
        return c/past if past > 0 else np.nan
    components = [(r(63),0.40),(r(126),0.20),(r(189),0.20),(r(252),0.20)]
    valid = [(v,w) for v,w in components if pd.notna(v)]
    if not valid: return np.nan
    tw = sum(w for _,w in valid)
    return sum(v*w for v,w in valid)/tw


def build_theme_data() -> dict:
    # Deduplicate
    seen, deduped = set(), {}
    for theme, tickers in THEMES.items():
        clean = [t for t in tickers if t not in seen]
        seen.update(clean)
        if clean: deduped[theme] = clean

    all_tickers = sorted(seen) + ["SPY"]
    log.info(f"Fetching {len(all_tickers)} ETFs...")

    raw = yf.download(all_tickers, period="400d", auto_adjust=True, progress=False)
    close = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw).ffill()
    high  = (raw["High"]  if isinstance(raw.columns, pd.MultiIndex) else raw).ffill()

    valid = [t for t in close.columns if close[t].dropna().shape[0] >= 63]
    close = close[valid]
    high  = high[[t for t in valid if t in high.columns]]
    log.info(f"Valid: {len(valid)} tickers through {close.index[-1].date()}")

    # RS factors
    rs_factors = {tk: compute_raw_rs_factor(close[tk]) for tk in close.columns}
    spy_rs = rs_factors.get("SPY", np.nan)
    if np.isnan(spy_rs): spy_rs = 1.0

    etf_tickers = [t for t in valid if t != "SPY"]
    rs_vs_spy = {t: rs_factors[t]/spy_rs for t in etf_tickers if pd.notna(rs_factors.get(t))}

    # Percentile rank 1-99 across all theme ETFs
    values = sorted([(t,v) for t,v in rs_vs_spy.items() if pd.notna(v)], key=lambda x: x[1])
    n = len(values)
    rs_pct = {tk: int(min(99, max(1, round((i/(max(n-1,1)))*98+1))))
              for i,(tk,_) in enumerate(values)}

    def ret(tk, p):
        if tk not in close.columns: return None
        s = close[tk].dropna()
        if len(s) <= p: return None
        return round((float(s.iloc[-1])/float(s.iloc[-(p+1)])-1)*100, 2)

    last = close.iloc[-1]
    high_52w = high.rolling(252).max().iloc[-1]
    def d52(tk):
        if tk not in high.columns: return None
        h,c = high_52w.get(tk,np.nan), last.get(tk,np.nan)
        if pd.isna(h) or pd.isna(c) or h<=0: return None
        return round((c/h-1)*100,1)

    ticker_to_theme = {t:th for th,tks in deduped.items() for t in tks}

    etf_records = []
    for tk in etf_tickers:
        if tk not in close.columns: continue
        etf_records.append({
            "Ticker":         tk,
            "Theme":          ticker_to_theme.get(tk,"Unknown"),
            "RS_Pct":         rs_pct.get(tk),
            "RS_vs_SPY":      round(rs_vs_spy.get(tk,np.nan),4) if pd.notna(rs_vs_spy.get(tk,np.nan)) else None,
            "Ret_1M_%":       ret(tk,21),
            "Ret_3M_%":       ret(tk,63),
            "Ret_6M_%":       ret(tk,126),
            "Ret_12M_%":      ret(tk,252),
            "Dist_52wHigh_%": d52(tk),
        })

    etf_df = pd.DataFrame(etf_records).sort_values("RS_Pct",ascending=False,na_position="last")

    theme_records = []
    for theme, tickers in deduped.items():
        sub = etf_df[etf_df["Ticker"].isin(tickers)]
        rs_vals = sub["RS_Pct"].dropna()
        if rs_vals.empty: continue
        top_row = sub.nlargest(1,"RS_Pct")
        theme_records.append({
            "Theme":       theme,
            "ETF_Count":   len(sub),
            "Avg_RS_Pct":  round(rs_vals.mean(),1),
            "Median_RS":   round(rs_vals.median(),1),
            "Top_ETF":     top_row["Ticker"].iloc[0] if not top_row.empty else "—",
            "Top_ETF_RS":  int(top_row["RS_Pct"].iloc[0]) if not top_row.empty else None,
            "N_Above_80":  int((rs_vals>=80).sum()),
            "N_Above_70":  int((rs_vals>=70).sum()),
            "Ret_1M_%":    round(sub["Ret_1M_%"].mean(),2) if "Ret_1M_%" in sub else None,
            "Ret_3M_%":    round(sub["Ret_3M_%"].mean(),2) if "Ret_3M_%" in sub else None,
            "Ret_12M_%":   round(sub["Ret_12M_%"].mean(),2) if "Ret_12M_%" in sub else None,
            "ETFs":        ", ".join(sub["Ticker"].tolist()),
        })

    theme_df = pd.DataFrame(theme_records).sort_values("Avg_RS_Pct",ascending=False,na_position="last")

    log.info(f"Top 5 themes: " + " | ".join(f"{r['Theme']} RS={r['Avg_RS_Pct']:.0f}" for _,r in theme_df.head(5).iterrows()))

    def safe(df):
        return df.replace({np.nan:None,np.inf:None,-np.inf:None}).to_dict(orient="records")

    return {
        "generated_at":   datetime.now().isoformat(),
        "etf_rankings":   safe(etf_df),
        "theme_rankings": safe(theme_df),
        "metadata": {
            "as_of":        datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_etfs":   len(etf_df),
            "total_themes": len(theme_df),
            "version":      "4.0 — IBD RS vs SPY",
        },
    }


def main():
    log.info("theme_prep.py v4.0 — IBD RS vs SPY")
    data = build_theme_data()
    with open(OUTPUT_JSON,"w") as f:
        json.dump(data,f,indent=2,default=str)
    m = data["metadata"]
    log.info(f"✅ {OUTPUT_JSON}: {m['total_etfs']} ETFs, {m['total_themes']} themes")

if __name__ == "__main__":
    main()
