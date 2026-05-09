"""
tactical_macro_prep.py — Macro Command Center Data Pipeline
============================================================
Fetches the macro universe via yfinance, computes all tactical macro
metrics, and writes parquets to data/ for the dashboard to read.

Outputs:
  data/tactical_macro_state.parquet   — flat metrics dict (serializable fields)
  data/pair_states.parquet           — one row per leadership pair
  data/narrative_history.parquet     — 60-day daily narrative state

Schedule: runs daily alongside other daily prep scripts (JOB_TYPE=daily).

Usage:
  python tactical_macro_prep.py          # from repo root
  USE_PARQUET=true streamlit run dashboard.py  # dashboard reads output
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

# ── Import from tactical data layer (no streamlit dependency needed here) ──
# We patch st.cache_data to be a no-op so we can import without Streamlit
import unittest.mock as mock
import sys

# Stub out streamlit.cache_data before importing tactical_data_layer
import types
fake_st = types.ModuleType("streamlit")
fake_st.cache_data = lambda *a, **kw: (lambda f: f)  # no-op decorator
fake_st.warning = log.warning
fake_st.error = log.error
sys.modules.setdefault("streamlit", fake_st)

from tactical_data_layer import (
    all_macro_tickers,
    fetch_universe,
    compute_macro_metrics,
)


def flatten_metrics(metrics: dict) -> dict:
    """
    Flatten the nested compute_macro_metrics dict into a JSON-serializable
    flat dict for parquet storage. Nested lists/series are stored as JSON strings.
    """
    flat = {
        "as_of":         str(metrics.get("as_of", "")),
        "regime_label":  metrics.get("regime_label", "MIXED / CHOPPY"),
        "generated_at":  datetime.now().isoformat(),
    }

    # Headline
    for k, v in metrics.get("headline", {}).items():
        flat[f"headline_{k}"] = v

    # Participation (scalar fields only — series stored separately)
    for ratio_key, d in metrics.get("participation", {}).items():
        for fk, fv in d.items():
            if fk != "series_60d":
                flat[f"participation_{ratio_key}_{fk}"] = fv

    # Style
    for sk, d in metrics.get("style", {}).items():
        if sk == "growth_premium":
            flat["style_growth_premium"] = d
        elif isinstance(d, dict):
            for fk, fv in d.items():
                if fk != "series_60d":
                    flat[f"style_{sk}_{fk}"] = fv

    # Sector rotation
    for k, v in metrics.get("sector_rotation", {}).items():
        if k in ("top_3_5d", "bottom_3_5d", "all_sectors_5d"):
            flat[k] = json.dumps(v)
        else:
            flat[f"sector_{k}"] = v

    # Group scores
    for grp, gdata in metrics.get("group_scores", {}).items():
        flat[f"group_{grp}_confirmed"] = gdata.get("confirmed", 0)
        flat[f"group_{grp}_total"]     = gdata.get("total", 0)
        flat[f"group_{grp}_pct"]       = gdata.get("pct", 0)

    # Stress tape
    for k, v in metrics.get("stress", {}).items():
        if k not in ("hyg_lqd_series_60d", "vix_series_60d"):
            flat[f"stress_{k}"] = v

    # Narrative (today's state + dominant, not history)
    nar = metrics.get("narrative")
    if nar:
        for k in ("today_state_id", "today_state_name", "today_sector_tilt",
                  "spx_dir", "dxy_dir", "rates_dir",
                  "dominant_id", "dominant_name", "dominant_tilt", "dominant_pct"):
            flat[f"narrative_{k}"] = nar.get(k)

    return flat


def main():
    log.info("Starting tactical_macro_prep.py...")

    # Fetch universe
    tickers = tuple(all_macro_tickers())
    log.info(f"Fetching {len(tickers)} macro tickers...")
    prices, failed = fetch_universe(tickers)

    if prices.empty:
        log.error("Universe fetch returned no data — aborting")
        return

    if failed:
        log.warning(f"Failed tickers ({len(failed)}): {', '.join(failed)}")

    # Compute metrics
    log.info("Computing macro metrics...")
    metrics = compute_macro_metrics(prices)

    # ── Save 1: Flat macro state ─────────────────────────────────────────
    flat = flatten_metrics(metrics)
    state_df = pd.DataFrame([flat])
    state_path = "data/tactical_macro_state.parquet"
    state_df.to_parquet(state_path, index=False)
    log.info(f"✅ {state_path}  ({len(flat)} fields)")

    # ── Save 2: Leadership pair states ───────────────────────────────────
    pair_rows = []
    for p in metrics.get("leadership_pairs", []):
        row = {
            "num":      p["num"],
            "den":      p["den"],
            "label":    p["label"],
            "group":    p["group"],
            "diff_5d":  p.get("diff_5d"),
            "diff_10d": p.get("diff_10d"),
            "diff_20d": p.get("diff_20d"),
            "state":    p.get("state", "Unknown"),
            # Store ratio history as JSON string
            "ratio_series_60d": json.dumps(p.get("ratio_series_60d") or []),
        }
        pair_rows.append(row)

    pairs_path = "data/pair_states.parquet"
    pd.DataFrame(pair_rows).to_parquet(pairs_path, index=False)
    log.info(f"✅ {pairs_path}  ({len(pair_rows)} pairs)")

    # ── Save 3: 60-day narrative history ─────────────────────────────────
    nar = metrics.get("narrative")
    if nar and nar.get("history_60d"):
        hist_df = pd.DataFrame(nar["history_60d"])
        hist_path = "data/narrative_history.parquet"
        hist_df.to_parquet(hist_path, index=False)
        log.info(f"✅ {hist_path}  ({len(hist_df)} days)")

    regime = metrics.get("regime_label", "—")
    log.info(f"\n  MACRO PREP COMPLETE — Regime: {regime}")
    log.info(f"  Narrative today: {nar.get('today_state_name','?') if nar else '?'}")


if __name__ == "__main__":
    main()
