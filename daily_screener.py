"""
DAILY SCREENER (v2) — reads weekly_watchlist.json, writes via data_layer.
"""

import os
import csv
import time
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from data_layer import save_daily_watchlist, get_latest_weekly_watchlist, get_industry_ranks


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

THEMES_CSV              = "sector_themes.csv"
RATE_LIMIT_PAUSE        = 1.0
LOG_LEVEL               = logging.INFO
MIN_DAILY_BBUW_SCORE    = 50
QUALIFYING_DAILY_STAGES = [1, 2]

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_daily_bars(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        log.error(f"[{ticker}] Daily fetch error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE & BBUW (DAILY)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_daily_stage(df: pd.DataFrame) -> int:
    close = df["close"]
    ma_150 = close.rolling(150).mean()
    if len(df) < 150 or pd.isna(ma_150.iloc[-1]):
        return 0
    ma_now = ma_150.iloc[-1]
    ma_50d_ago = ma_150.iloc[-50] if len(ma_150) >= 50 else ma_150.iloc[0]
    ma_slope_pct = ((ma_now - ma_50d_ago) / ma_50d_ago) * 100
    price = close.iloc[-1]
    above_ma = price > ma_now
    pct_from_ma = ((price - ma_now) / ma_now) * 100
    if above_ma and ma_slope_pct > 2.0:
        return 2
    if not above_ma and ma_slope_pct < -2.0:
        return 4
    if abs(pct_from_ma) < 7 and abs(ma_slope_pct) < 2.0:
        return 1
    if above_ma and ma_slope_pct < 2.0:
        return 3
    return 1


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
    weights = {"range_contraction": 0.20, "volume_contraction": 0.15, "higher_lows": 0.15,
               "coil_duration": 0.10, "rs_vs_spy": 0.20, "pullback_depth": 0.10, "ema_proximity": 0.10}
    components = {
        "range_contraction": round(range_score, 1),
        "volume_contraction": round(vol_score, 1),
        "higher_lows": round(hl_score, 1),
        "coil_duration": round(coil_score, 1),
        "rs_vs_spy": round(rs_score, 1),
        "pullback_depth": round(pb_score, 1),
        "ema_proximity": round(ema_score, 1),
    }
    composite = sum(components[k] * weights[k] for k in weights)
    return {"score": round(composite, 1), "components": components}


# ═══════════════════════════════════════════════════════════════════════════════
#  THEMES
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
            industry = row.get("Industry", "").strip()   # new column — empty in old format
            try:
                rank = int(row.get("ThemeRank", 99))
            except ValueError:
                rank = 99
            if sym:
                themes[sym] = {"theme": theme, "rank": rank, "industry": industry}
    log.info(f"Loaded {len(themes)} sector theme mappings.")
    return themes


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVICTION
# ═══════════════════════════════════════════════════════════════════════════════

def assign_conviction(weekly_bbuw, daily_bbuw, trend_template, industry_rank: int,
                       pivot_8w_tier: str = "NONE",
                       ep_tier: str = "NONE") -> str:
    """
    Tiered conviction scoring (max 10 points).

    Points awarded:
      • Weekly BBUW    ≥ 60 → +1, ≥ 75 → +1              (max 2)
      • Daily  BBUW    ≥ 60 → +1, ≥ 75 → +1              (max 2)
      • Trend Template ≥ 6  → +1, = 8  → +1              (max 2)
      • Industry Rank  ≤ 10 → +1, ≤ 3  → +1              (max 2)
      • 8W Pivot STANDARD → +1, STRONG → +2              (max 2)
      • EP Tier  WATCH → +1, STANDARD → +2, STRONG → +3  (bonus — can push above 10)

    Tiers:
      HIGH ≥ 7
      MED  ≥ 4
      LOW  < 4
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

    # 8-Week Pivot bonus
    if pivot_8w_tier == "STANDARD": points += 1
    elif pivot_8w_tier == "STRONG": points += 2

    # Episodic Pivot bonus — EPs are high-conviction catalyst events
    if ep_tier == "WATCH":    points += 1
    elif ep_tier == "STANDARD": points += 2
    elif ep_tier == "STRONG":   points += 3

    if points >= 7: return "HIGH"
    if points >= 4: return "MED"
    return "LOW"


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting DAILY screener (v2)...")
    weekly_data = get_latest_weekly_watchlist()
    weekly_entries = weekly_data.get("entries", [])
    if not weekly_entries:
        log.error("No weekly watchlist found. Run weekly_screener.py first.")
        return

    log.info(f"Loaded {len(weekly_entries)} entries from weekly screen.")
    themes = load_sector_themes(THEMES_CSV)
    log.info(f"Loaded {len(themes)} sector theme mappings.")

    # Load dynamic industry ranks (computed Sunday by weekly_screener.py)
    industry_rank_lookup = get_industry_ranks()   # {industry_name: rank_int}
    log.info(f"Loaded {len(industry_rank_lookup)} industry ranks.")

    df_spy = fetch_daily_bars("SPY")

    qualified = []
    for i, entry in enumerate(weekly_entries, 1):
        ticker = entry["ticker"]
        log.info(f"[{i}/{len(weekly_entries)}] {ticker}")
        df_daily = fetch_daily_bars(ticker)
        if df_daily is None:
            continue

        daily_stage = classify_daily_stage(df_daily)
        daily_bbuw = calc_bbuw_daily(df_daily, df_spy)

        if daily_stage not in QUALIFYING_DAILY_STAGES:
            continue
        if daily_bbuw["score"] < MIN_DAILY_BBUW_SCORE:
            continue

        theme_info = themes.get(ticker, {"theme": "Unclassified", "rank": 99, "industry": ""})
        pivot_8w_tier = entry.get("pivot_8w_tier", "NONE")

        # Look up the dynamic industry rank (default 99 = unranked)
        ticker_industry = theme_info.get("industry", "")
        industry_rank = industry_rank_lookup.get(ticker_industry, 99)

        conviction = assign_conviction(
            entry.get("bbuw_score", 0),
            daily_bbuw["score"],
            entry.get("trend_template_score", 0),
            industry_rank,
            pivot_8w_tier=pivot_8w_tier,
            ep_tier=entry.get("ep_tier", "NONE"),
        )

        qualified.append({
            "ticker": ticker,
            "conviction": conviction,
            "weekly_stage": entry.get("stage"),
            "daily_stage": daily_stage,
            "trend_template": entry.get("trend_template_score"),
            "weekly_bbuw": entry.get("bbuw_score"),
            "daily_bbuw": daily_bbuw["score"],
            "daily_bbuw_components": daily_bbuw["components"],
            "theme": theme_info["theme"],
            "theme_rank": theme_info["rank"],
            "industry": theme_info.get("industry", ""),
            "industry_rank": industry_rank,
            # EP fields passthrough
            "ep_tier":     entry.get("ep_tier", "NONE"),
            "ep_score":    entry.get("ep_score", 0),
            "ep_week_pct": entry.get("ep_week_pct"),
            # 8-Week Pivot passthrough from weekly screener
            "pivot_8w_fired":        entry.get("pivot_8w_fired", False),
            "pivot_8w_tier":         pivot_8w_tier,
            "ema8":                  entry.get("ema8"),
            "pct_from_ema8":         entry.get("pct_from_ema8"),
            "ema8_rising":           entry.get("ema8_rising", False),
            "pivot_8w_volume_spike": entry.get("pivot_8w_volume_spike", False),
        })
        tier_emoji = {
            "STRONG": "🔥", "STANDARD": "✅", "WEAK": "⚠️",
            "PROXIMITY": "👀", "NONE": "  ",
        }.get(pivot_8w_tier, "  ")
        log.info(f"  ✅ {ticker} — {conviction} | Ind.Rank #{industry_rank} | {tier_emoji}{pivot_8w_tier}")
        time.sleep(RATE_LIMIT_PAUSE)

    conviction_order = {"HIGH": 0, "MED": 1, "LOW": 2}
    qualified.sort(key=lambda x: (conviction_order[x["conviction"]], -x["daily_bbuw"]))
    save_daily_watchlist(qualified)

    high_count = sum(1 for r in qualified if r["conviction"] == "HIGH")
    med_count  = sum(1 for r in qualified if r["conviction"] == "MED")
    pivot_strong = sum(1 for r in qualified if r.get("pivot_8w_tier") == "STRONG")
    pivot_std    = sum(1 for r in qualified if r.get("pivot_8w_tier") == "STANDARD")
    pivot_prox   = sum(1 for r in qualified if r.get("pivot_8w_tier") == "PROXIMITY")
    log.info(f"\n  DAILY COMPLETE — {len(qualified)} qualified ({high_count} HIGH, {med_count} MED)")
    log.info(f"    8W Pivots: 🔥 STRONG={pivot_strong}  ✅ STANDARD={pivot_std}  👀 PROXIMITY={pivot_prox}")


if __name__ == "__main__":
    main()
