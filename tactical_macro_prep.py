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
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.makedirs("data", exist_ok=True)


# =============================================================================
# HOLIDAY / MARKET-OPEN GUARDS
# =============================================================================

# US market holidays (NYSE schedule). Add each year as needed.
_US_MARKET_HOLIDAYS = {
    # 2025
    date(2025, 1, 1),   date(2025, 1, 20),  date(2025, 2, 17),
    date(2025, 4, 18),  date(2025, 5, 26),  date(2025, 6, 19),
    date(2025, 7, 4),   date(2025, 9, 1),   date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1),   date(2026, 1, 19),  date(2026, 2, 16),
    date(2026, 4, 3),   date(2026, 5, 25),  date(2026, 6, 19),
    date(2026, 7, 3),   date(2026, 9, 7),   date(2026, 11, 26),
    date(2026, 12, 25),
}


def is_market_open_today() -> bool:
    """
    Returns True if today is a NYSE trading day (Mon-Fri, not a holiday).
    Weekends and known US market holidays return False.
    """
    today = date.today()
    if today.weekday() >= 5:           # Saturday=5, Sunday=6
        return False
    if today in _US_MARKET_HOLIDAYS:
        return False
    return True


def has_new_market_data(prices: pd.DataFrame, proxy: str = "SPY") -> bool:
    """
    Verify the fetched universe contains data for today or the most recent
    trading day. Uses SPY (or first available column) as a proxy.

    Returns False if the most recent row is more than 4 calendar days old —
    a sign the fetch returned stale or empty data.
    """
    col = proxy if proxy in prices.columns else (
        prices.columns[0] if not prices.empty else None
    )
    if col is None:
        return False
    try:
        last_date = pd.to_datetime(prices[col].dropna().index[-1]).date()
        stale_threshold = date.today() - timedelta(days=4)
        if last_date < stale_threshold:
            log.warning(
                f"  Data freshness FAILED — last row is {last_date} "
                f"(threshold: {stale_threshold}). Likely stale holiday data."
            )
            return False
        log.info(f"  Data freshness OK — last row: {last_date}")
        return True
    except Exception as e:
        log.warning(f"  Data freshness check error: {e}")
        return False


def pair_states_are_healthy(pair_rows: list) -> bool:
    """
    Returns False if every leadership pair resolved to 'Unknown' —
    a sure sign the compute produced garbage (stale/holiday data).
    At least one pair must have a real state before the parquet is overwritten.
    """
    real_states = {"Confirmed", "Fresh", "Fading", "False Start", "Denied", "Mixed"}
    resolved = [r for r in pair_rows if r.get("state") in real_states]
    if not resolved:
        log.warning(
            f"  Pair health check FAILED — all {len(pair_rows)} pairs are 'Unknown'. "
            f"Skipping parquet write to preserve last good data."
        )
        return False
    log.info(
        f"  Pair health check OK — {len(resolved)}/{len(pair_rows)} pairs resolved "
        f"(e.g. {', '.join(r['state'] for r in resolved[:3])})"
    )
    return True


def fetch_real_rate_fred() -> dict:
    """
    Fetch 10-year TIPS real yield (DFII10) from FRED API.
    Returns dict with real_rate_10y, real_rate_direction, real_rate_label.
    Gracefully returns None values if FRED_API_KEY is missing or fetch fails.
    """
    result = {"real_rate_10y": None, "real_rate_direction": None, "real_rate_label": None}
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        log.warning("FRED_API_KEY not set — real rate signal unavailable")
        return result
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        series = fred.get_series("DFII10", observation_start=start).dropna()
        if len(series) < 6:
            return result
        real_rate_10y = round(float(series.iloc[-1]), 3)
        rr_5d_ago     = float(series.iloc[-6])
        rr_change     = real_rate_10y - rr_5d_ago
        if   rr_change <= -0.08: real_rate_direction = "FALLING"
        elif rr_change >=  0.08: real_rate_direction = "RISING"
        else:                     real_rate_direction = "FLAT"
        if   real_rate_10y < 0.0:  real_rate_label = "NEGATIVE"
        elif real_rate_10y < 0.50: real_rate_label = "NEAR ZERO"
        elif real_rate_10y < 1.50: real_rate_label = "POSITIVE"
        else:                       real_rate_label = "ELEVATED"
        log.info(f"  DFII10 real rate: {real_rate_10y:.2f}% [{real_rate_label}] {real_rate_direction}")
        return {
            "real_rate_10y":       real_rate_10y,
            "real_rate_direction": real_rate_direction,
            "real_rate_label":     real_rate_label,
        }
    except Exception as e:
        log.warning(f"FRED DFII10 fetch failed: {e}")
        return result

def fetch_fed_stance_fred() -> dict:
    """
    Fetch FEDTARMD (federal funds target rate, median) from FRED.
    Classifies the Fed stance over a rolling 120-day window.

    HIKING  = rate higher than 120 days ago by > 10bp
    CUTTING = rate lower than 120 days ago by > 10bp
    HOLDING = rate unchanged within ±10bp

    Returns dict with fed_rate, fed_stance, fed_rate_change.
    """
    result = {"fed_rate": None, "fed_stance": None, "fed_rate_change": None}
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return result
    try:
        from fredapi import Fred
        fred  = Fred(api_key=api_key)
        start = (datetime.now() - timedelta(days=130)).strftime("%Y-%m-%d")
        series = fred.get_series("FEDTARMD", observation_start=start).dropna()
        if len(series) < 2:
            return result
        fed_rate       = round(float(series.iloc[-1]), 3)
        rate_3m_ago    = round(float(series.iloc[0]),  3)
        rate_change    = round(fed_rate - rate_3m_ago, 3)
        if   rate_change >  0.10: fed_stance = "HIKING"
        elif rate_change < -0.10: fed_stance = "CUTTING"
        else:                      fed_stance = "HOLDING"
        result.update({
            "fed_rate":        fed_rate,
            "fed_stance":      fed_stance,
            "fed_rate_change": rate_change,
        })
        log.info(f"  FEDTARMD: {fed_rate:.2f}% [{fed_stance}] 3m_change={rate_change:+.2f}%")
    except Exception as e:
        log.warning(f"FRED FEDTARMD fetch failed: {e}")
    return result


def fetch_curve_regime_fred() -> dict:
    """
    Fetch 10Y-2Y Treasury spread (T10Y2Y) and 30Y-5Y spread (T30Y5Y) from FRED.
    Classifies the curve regime for use as an entry mode gate.

    Bear steepener = spread widening AND 10Y yield rising = multiple compression.
                     This gates BUY_RIP → TIGHT_MA (no chase, MA entries only).
    Bull steepener = spread widening AND 10Y yield falling (short rates dropping).
                     Recovery signal — positive for equities.
    Flattening     = spread narrowing (rate normalisation or growth concern).
    Inverted       = negative spread (growth scare / recession signal).
    Neutral        = less than 5bp move in spread over 5 days.

    Returns dict with curve_regime, spread_2s10s, spread_5s30s, curve_direction.
    """
    result = {
        "curve_regime":    None,
        "spread_2s10s":    None,
        "spread_5s30s":    None,
        "curve_direction": None,
    }
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        log.warning("FRED_API_KEY not set — curve regime signal unavailable")
        return result
    try:
        from fredapi import Fred
        fred  = Fred(api_key=api_key)
        start = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")

        # 10Y-2Y spread
        t10y2y = fred.get_series("T10Y2Y", observation_start=start).dropna()
        if len(t10y2y) < 6:
            log.warning("T10Y2Y: insufficient data for curve regime")
            return result

        spread_now = float(t10y2y.iloc[-1])
        spread_5d  = float(t10y2y.iloc[-6])
        spread_chg = round(spread_now - spread_5d, 3)   # positive = steepening

        # 10Y yield direction determines bear vs bull steepener
        dgs10 = fred.get_series("DGS10", observation_start=start).dropna()
        t10y_chg = None
        if len(dgs10) >= 6:
            t10y_chg = float(dgs10.iloc[-1]) - float(dgs10.iloc[-6])

        # Classify
        STEEPEN_THRESHOLD = 0.05   # 5bp change = meaningful
        if abs(spread_chg) < STEEPEN_THRESHOLD:
            curve_regime    = "INVERTED" if spread_now < 0 else "NEUTRAL"
            curve_direction = "FLAT"
        elif spread_chg > 0:
            curve_direction = "STEEPENING"
            if t10y_chg is not None and t10y_chg > 0.05:
                # 10Y rising AND spread widening = bear steepener
                curve_regime = "BEAR_STEEPENER"
            else:
                # Short rates falling faster OR 10Y flat = bull steepener
                curve_regime = "BULL_STEEPENER"
        else:
            curve_direction = "FLATTENING"
            curve_regime    = "INVERTED" if spread_now < 0 else "FLATTENING"

        result.update({
            "curve_regime":    curve_regime,
            "spread_2s10s":    round(spread_now, 3),
            "curve_direction": curve_direction,
        })

        # 30Y-5Y spread (T30Y5Y) — secondary curve read
        try:
            t30y5y = fred.get_series("T30Y5Y", observation_start=start).dropna()
            if not t30y5y.empty:
                result["spread_5s30s"] = round(float(t30y5y.iloc[-1]), 3)
        except Exception:
            pass

        log.info(
            f"  Curve regime: {curve_regime}  "
            f"2s10s={spread_now:+.2f}%  5d_chg={spread_chg:+.3f}  "
            f"direction={curve_direction}"
        )
    except Exception as e:
        log.warning(f"FRED curve regime fetch failed: {e}")
    return result


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

    # ── Guard 1: Holiday / non-trading day ───────────────────────────────
    # If today is a weekend or known US market holiday, the universe fetch
    # will return stale data (last Friday). Overwriting the parquets with
    # that stale compute would corrupt the dashboard until the next trading
    # day run. Exit early and leave the existing parquets intact.
    if not is_market_open_today():
        log.info(
            f"  Today ({date.today()}) is a non-trading day. "
            f"Skipping parquet write — existing data preserved."
        )
        return

    # Fetch universe
    tickers = tuple(all_macro_tickers())
    log.info(f"Fetching {len(tickers)} macro tickers...")
    prices, failed = fetch_universe(tickers)

    if prices.empty:
        log.error("Universe fetch returned no data — aborting")
        return

    if failed:
        log.warning(f"Failed tickers ({len(failed)}): {', '.join(failed)}")

    # ── Guard 2: Data freshness check ────────────────────────────────────
    # Confirm the fetched prices actually reflect recent market data.
    # If the most recent row is stale (e.g. holiday slipped through), abort
    # rather than overwrite good parquets with a bad compute.
    if not has_new_market_data(prices):
        log.warning(
            "  Stale data detected — aborting write to preserve last good parquets."
        )
        return

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
            # Store ratio history as JSON string — convert Timestamps to str
            "ratio_series_60d": json.dumps(
                [
                    {k: str(v) if hasattr(v, 'isoformat') else v
                     for k, v in item.items()}
                    if isinstance(item, dict) else item
                    for item in (p.get("ratio_series_60d") or [])
                ]
            ),
        }
        pair_rows.append(row)

    # ── Guard 3: Pair health check ───────────────────────────────────────
    # If every pair resolved to "Unknown" the compute was garbage —
    # don't overwrite the existing pair_states.parquet with bad data.
    pairs_path = "data/pair_states.parquet"
    if pair_states_are_healthy(pair_rows):
        pd.DataFrame(pair_rows).to_parquet(pairs_path, index=False)
        log.info(f"✅ {pairs_path}  ({len(pair_rows)} pairs)")
    else:
        log.warning(f"  ⚠️  {pairs_path} NOT updated — keeping last good version.")

    # ── Save 3: 60-day narrative history ─────────────────────────────────
    nar = metrics.get("narrative")
    if nar and nar.get("history_60d"):
        hist_df = pd.DataFrame(nar["history_60d"])
        hist_path = "data/narrative_history.parquet"
        hist_df.to_parquet(hist_path, index=False)
        log.info(f"✅ {hist_path}  ({len(hist_df)} days)")

    # ── Save 4: Narrative regime posture (Markov transition model) ────────
    try:
        from narrative_regime_model import run as run_narrative_regime
        regime_result = run_narrative_regime()
        if not regime_result:
            log.warning("Narrative regime model returned no result.")
        else:
            # Each injection step is independently try/excepted so that a failure
            # in any one step (e.g. FRED rate limit, missing env var) cannot
            # prevent the final save_regime_snapshot from running.
            # run_narrative_regime() above already wrote the parquet with None
            # placeholders — the goal here is to overwrite with real values.

            try:
                rr_data = fetch_real_rate_fred()
                regime_result.update(rr_data)
            except Exception as e:
                log.warning(f"Real rate inject failed: {e}")

            try:
                if nar:
                    for field in ("carry_jpy_5d", "carry_signal", "dxy_5d", "dxy_signal"):
                        if regime_result.get(field) is None and nar.get(field) is not None:
                            regime_result[field] = nar[field]
            except Exception as e:
                log.warning(f"Carry/DXY inject failed: {e}")

            try:
                fed_data = fetch_fed_stance_fred()
                regime_result.update(fed_data)
            except Exception as e:
                log.warning(f"Fed stance inject failed: {e}")

            try:
                curve_data = fetch_curve_regime_fred()
                regime_result.update(curve_data)
            except Exception as e:
                log.warning(f"Curve regime inject failed: {e}")

            try:
                from narrative_regime_model import compute_entry_mode
                entry_mode, entry_mode_reason = compute_entry_mode(regime_result)
                regime_result["entry_mode"]        = entry_mode
                regime_result["entry_mode_reason"] = entry_mode_reason
            except Exception as e:
                log.warning(f"Entry mode compute failed: {e}")

            # ── Final save — always runs regardless of which injections succeeded ──
            from narrative_regime_model import save_regime_snapshot
            save_regime_snapshot(regime_result)

            rr   = regime_result.get("real_rate_10y")
            rrl  = regime_result.get("real_rate_label", "?")
            rrd  = regime_result.get("real_rate_direction", "?")
            cs   = regime_result.get("carry_signal", "?")
            acl  = regime_result.get("acceleration_label", "?")
            crv  = regime_result.get("curve_regime", "?")
            em   = regime_result.get("entry_mode", "?")
            dxy  = regime_result.get("dxy_signal", "?")
            fed  = regime_result.get("fed_stance", "?")
            rr_str = f"real_rate={rr:.2f}% [{rrl} {rrd}]" if rr is not None else "real_rate=awaiting"
            log.info(
                f"✅ narrative_regime.parquet  "
                f"posture={regime_result['posture']} "
                f"({regime_result['posture_confidence']:.0%}) | "
                f"score={regime_result['regime_score']:.2f} | "
                f"accel={acl} | {rr_str} | carry={cs} | dxy={dxy} | "
                f"fed={fed} | curve={crv} | entry_mode={em}"
            )
    except Exception as e:
        log.warning(f"Narrative regime model failed (non-fatal): {e}")

    regime = metrics.get("regime_label", "—")
    log.info(f"\n  MACRO PREP COMPLETE — Regime: {regime}")
    log.info(f"  Narrative today: {nar.get('today_state_name','?') if nar else '?'}")


if __name__ == "__main__":
    main()
