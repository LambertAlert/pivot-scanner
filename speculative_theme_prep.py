"""
speculative_theme_prep.py — Speculative Themes Data Pipeline
=============================================================
Fetches the full speculative theme universe (~332 tickers) via yfinance,
computes all theme and ticker metrics, and writes parquets to data/.

Outputs:
  data/theme_state.parquet          — one row per theme (aggregated metrics)
  data/theme_ticker_metrics.parquet — one row per ticker (per-ticker metrics)

Schedule: runs daily alongside other daily prep scripts (JOB_TYPE=daily).
This is the slowest prep step (~300 tickers in batches of 30 = ~60s).

Usage:
  python speculative_theme_prep.py
"""

import os
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.makedirs("data", exist_ok=True)

# Stub streamlit before importing tactical_data_layer
import types, sys
fake_st = types.ModuleType("streamlit")
fake_st.cache_data = lambda *a, **kw: (lambda f: f)
fake_st.warning = log.warning
fake_st.error = log.error
sys.modules.setdefault("streamlit", fake_st)

from tactical_data_layer import (
    fetch_universe,
    compute_theme_metrics,
    compute_cross_theme_movers,
)
from themes import (
    THEMES,
    THEME_TO_MACRO_GROUPS,
    get_all_unique_tickers,
)


def main():
    log.info("Starting speculative_theme_prep.py...")

    tickers = tuple(get_all_unique_tickers())
    log.info(f"Fetching {len(tickers)} theme universe tickers...")
    prices, failed = fetch_universe(tickers)

    if prices.empty:
        log.error("Theme universe fetch returned no data — aborting")
        return

    if failed:
        log.warning(f"Failed tickers ({len(failed)}): {', '.join(failed[:20])}"
                    f"{'...' if len(failed) > 20 else ''}")

    log.info("Computing theme metrics...")
    theme_records, ticker_metrics = compute_theme_metrics(prices, THEMES)

    # ── D4: Per-ticker 5d vs 21d acceleration ────────────────────────────
    # More responsive than 1M vs 3M. Captures week-vs-month momentum shifts.
    # Positive = accelerating short-term; Negative = momentum fading.
    from tactical_data_layer import fetch_universe as _fu
    ticker_accel = {}
    for tk in prices.columns:
        s = prices[tk].dropna()
        if len(s) < 22:
            continue
        r5d  = (float(s.iloc[-1]) / float(s.iloc[-6])  - 1) * 100
        r21d = (float(s.iloc[-1]) / float(s.iloc[-22]) - 1) * 100
        ticker_accel[tk] = r5d - r21d   # positive = recently accelerating

    # ── Cross-reference EP events for D2 ─────────────────────────────────
    ep_tickers = set()
    try:
        from data_layer import get_ep_events
        ep_data = get_ep_events()
        ep_tickers = {
            e.get("ticker", "")
            for e in ep_data.get("events", [])
            if e.get("ep_tier") in ("STRONG", "STANDARD", "WATCH")
        }
        log.info(f"  EP events: {len(ep_tickers)} active tickers")
    except Exception as e:
        log.warning(f"EP events load failed: {e}")

    # ── Cross-reference volume surges for D3 ─────────────────────────────
    surge_tickers = set()
    try:
        from data_layer import get_volume_surges
        vol_data = get_volume_surges()
        surge_tickers = {
            e.get("ticker", "")
            for e in vol_data.get("events", [])
            if e.get("is_fresh")
        }
        log.info(f"  Volume surges: {len(surge_tickers)} fresh tickers")
    except Exception as e:
        log.warning(f"Volume surges load failed: {e}")

    # ── Save 1: Per-theme aggregated state ───────────────────────────────
    theme_rows = []
    for r in theme_records:
        macro_groups = THEME_TO_MACRO_GROUPS.get(r["theme"], [])
        theme_tickers_list = list(THEMES.get(r["theme"], []))

        # D4: Theme-level acceleration = avg of (5d - 21d) per ticker in theme
        accel_vals = [ticker_accel[t] for t in theme_tickers_list if t in ticker_accel]
        acceleration = float(np.nanmean(accel_vals)) if accel_vals else None

        # D2: Count of active EP events in this theme
        ep_active_count = sum(1 for t in theme_tickers_list if t in ep_tickers)

        # D3: Count of fresh volume surges in this theme
        vol_surge_count = sum(1 for t in theme_tickers_list if t in surge_tickers)

        row = {
            "theme":             r["theme"],
            "n_tickers":         r["n_tickers"],
            "n_with_data":       r["n_with_data"],
            "avg_1m":            r["avg_1m"],
            "avg_3m":            r["avg_3m"],
            "avg_6m":            r["avg_6m"],
            "avg_ytd":           r["avg_ytd"],
            "median_1m":         r["median_1m"],
            "pct_up_1m":         r["pct_up_1m"],
            "top_mover_1m":      r["top_mover_1m"],
            "top_mover_1m_ret":  r["top_mover_1m_ret"],
            "macro_groups":      json.dumps(macro_groups),
            "generated_at":      datetime.now().isoformat(),
            "acceleration":      acceleration,          # D4: refined 5d vs 21d
            "ep_active_count":   ep_active_count,       # D2
            "vol_surge_count":   vol_surge_count,       # D3
            "composite_score":   None,                  # D5: filled below
        }
        theme_rows.append(row)

    # ── D5: Composite momentum score (0-100) ─────────────────────────────
    # Percentile-rank each component then blend. Robust to outliers vs z-score.
    # Components: level (1M return), breadth (% up), velocity (acceleration), activity (EP+vol)
    if len(theme_rows) >= 3:
        df_sc = pd.DataFrame(theme_rows)

        def _pct_rank(series):
            return series.rank(pct=True, na_option="keep").fillna(0.5) * 100

        df_sc["_lvl"] = _pct_rank(df_sc["avg_1m"])
        df_sc["_brd"] = _pct_rank(df_sc["pct_up_1m"])
        df_sc["_vel"] = _pct_rank(df_sc["acceleration"])
        df_sc["_act"] = _pct_rank(df_sc["ep_active_count"] + df_sc["vol_surge_count"])

        df_sc["composite_score"] = (
            0.35 * df_sc["_lvl"]
            + 0.25 * df_sc["_brd"]
            + 0.25 * df_sc["_vel"]
            + 0.15 * df_sc["_act"]
        ).round(1)

        score_map = dict(zip(df_sc["theme"], df_sc["composite_score"]))
        for row in theme_rows:
            row["composite_score"] = float(score_map.get(row["theme"], 50.0))

    theme_path = "data/theme_state.parquet"
    pd.DataFrame(theme_rows).to_parquet(theme_path, index=False)
    log.info(f"✅ {theme_path}  ({len(theme_rows)} themes)")

    # ── Save 2: Per-ticker metrics ────────────────────────────────────────
    ticker_rows = []
    for ticker, tm in ticker_metrics.items():
        theme_memberships = [
            name for name, ticks in THEMES.items() if ticker in ticks
        ]
        row = {
            "ticker":            ticker,
            "last":              tm.get("last"),
            "ret_1m":            tm.get("ret_1m"),
            "ret_3m":            tm.get("ret_3m"),
            "ret_6m":            tm.get("ret_6m"),
            "ret_ytd":           tm.get("ret_ytd"),
            "dist_52w_high":     tm.get("dist_52w_high"),
            "above_50dma":       tm.get("above_50dma", False),
            "above_200dma":      tm.get("above_200dma", False),
            "n_themes":          len(theme_memberships),
            "theme_memberships": json.dumps(theme_memberships),
            "has_ep":            ticker in ep_tickers,
            "has_vol_surge":     ticker in surge_tickers,
            "accel_5d_21d":      ticker_accel.get(ticker),
            "generated_at":      datetime.now().isoformat(),
        }
        ticker_rows.append(row)

    ticker_path = "data/theme_ticker_metrics.parquet"
    pd.DataFrame(ticker_rows).to_parquet(ticker_path, index=False)
    log.info(f"✅ {ticker_path}  ({len(ticker_rows)} tickers)")

    # Summary
    top5 = sorted(
        [r for r in theme_rows if r["composite_score"] is not None],
        key=lambda x: x["composite_score"], reverse=True
    )[:5]
    log.info("\n  TOP 5 THEMES (composite score):")
    for r in top5:
        ep_str  = f"  EP:{r['ep_active_count']}" if r['ep_active_count'] else ""
        vol_str = f"  VOL:{r['vol_surge_count']}" if r['vol_surge_count'] else ""
        log.info(f"    {r['theme']:<40} score={r['composite_score']:.0f}"
                 f"  1M={r['avg_1m']:+.1f}%  accel={r['acceleration']:+.2f}{ep_str}{vol_str}"
                 if r['avg_1m'] is not None and r['acceleration'] is not None
                 else f"    {r['theme']:<40} score={r['composite_score']:.0f}")
    log.info(f"\n  THEME PREP COMPLETE — {len(theme_rows)} themes, {len(ticker_rows)} tickers")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
