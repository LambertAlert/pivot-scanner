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

    # ── Save 1: Per-theme aggregated state ───────────────────────────────
    theme_rows = []
    for r in theme_records:
        macro_groups = THEME_TO_MACRO_GROUPS.get(r["theme"], [])
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
            "macro_groups":      json.dumps(macro_groups),  # stored as JSON string
            "generated_at":      datetime.now().isoformat(),
            # Acceleration for rotation detection
            "acceleration":      (r["avg_1m"] - r["avg_3m"] / 3.0)
                                 if r["avg_1m"] is not None and r["avg_3m"] is not None
                                 else None,
        }
        theme_rows.append(row)

    theme_path = "data/theme_state.parquet"
    pd.DataFrame(theme_rows).to_parquet(theme_path, index=False)
    log.info(f"✅ {theme_path}  ({len(theme_rows)} themes)")

    # ── Save 2: Per-ticker metrics ────────────────────────────────────────
    ticker_rows = []
    for ticker, tm in ticker_metrics.items():
        # Themes this ticker belongs to
        theme_memberships = [
            name for name, ticks in THEMES.items() if ticker in ticks
        ]
        row = {
            "ticker":           ticker,
            "last":             tm.get("last"),
            "ret_1m":           tm.get("ret_1m"),
            "ret_3m":           tm.get("ret_3m"),
            "ret_6m":           tm.get("ret_6m"),
            "ret_ytd":          tm.get("ret_ytd"),
            "dist_52w_high":    tm.get("dist_52w_high"),
            "above_50dma":      tm.get("above_50dma", False),
            "above_200dma":     tm.get("above_200dma", False),
            "n_themes":         len(theme_memberships),
            "theme_memberships": json.dumps(theme_memberships),
            "generated_at":     datetime.now().isoformat(),
        }
        ticker_rows.append(row)

    ticker_path = "data/theme_ticker_metrics.parquet"
    pd.DataFrame(ticker_rows).to_parquet(ticker_path, index=False)
    log.info(f"✅ {ticker_path}  ({len(ticker_rows)} tickers)")

    # Summary
    top5 = sorted(
        [r for r in theme_rows if r["avg_1m"] is not None],
        key=lambda x: x["avg_1m"], reverse=True
    )[:5]
    log.info("\n  TOP 5 THEMES (1M avg return):")
    for r in top5:
        log.info(f"    {r['theme']:<40} {r['avg_1m']:+.2f}%")
    log.info(f"\n  THEME PREP COMPLETE — {len(theme_rows)} themes, {len(ticker_rows)} tickers")


if __name__ == "__main__":
    main()
