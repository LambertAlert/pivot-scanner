"""
speculative_themes.py — The Speculative Themes tab.

Implements:
  • Theme Heat Map (sortable table, all themes ranked by composite score)
  • Rotation Detection (hottest, coldest, emerging, fading)
  • Macro Group View (themes filtered by active macro group)
  • Theme Detail Drill-Down (per-ticker breakdown + WL badge)
  • Short Interest Monitor (Most_Shorted — separate from main heat map)
  • Cross-Theme Movers (tickers in multiple themes)
  • Theme Membership Search

A1: Reads from pre-computed parquets (theme_state.parquet,
    theme_ticker_metrics.parquet) written by speculative_theme_prep.py.
    Falls back to live fetch if parquets are missing.
A2: Most_Shorted separated into standalone Short Interest Monitor section.
V2: Watchlist crossover badge (📋) on tickers present in daily watchlist
    as HIGH or MED conviction.
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from themes import (
    THEMES,
    THEME_TO_MACRO_GROUPS,
    MACRO_GROUPS,
    get_all_unique_tickers,
    get_themes_for_ticker,
)

from macro_view import (
    AMBER, COPPER, OBSIDIAN, GREEN, RED, GREY,
    PANEL_BG, GRID, STATE_COLORS,
    section_header, kpi_box, fmt_pct, fmt_num,
)


# ─────────────────────────────────────────────────────────────────────────────
#  PARQUET LOADERS  (A1)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_theme_state() -> list:
    """Load pre-computed per-theme aggregated records from parquet."""
    path = "data/theme_state.parquet"
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_parquet(path)
        records = df.replace({float("nan"): None}).to_dict("records")
        for r in records:
            if isinstance(r.get("macro_groups"), str):
                try:
                    r["macro_groups"] = json.loads(r["macro_groups"])
                except Exception:
                    r["macro_groups"] = []
        return records
    except Exception:
        return []


@st.cache_data(ttl=3600)
def load_ticker_metrics() -> dict:
    """Load pre-computed per-ticker metrics from parquet → {ticker: metrics}."""
    path = "data/theme_ticker_metrics.parquet"
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_parquet(path)
        result = {}
        for _, row in df.iterrows():
            tm = row.replace({float("nan"): None}).to_dict()
            if isinstance(tm.get("theme_memberships"), str):
                try:
                    tm["theme_memberships"] = json.loads(tm["theme_memberships"])
                except Exception:
                    tm["theme_memberships"] = []
            result[tm["ticker"]] = tm
        return result
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_watchlist_tickers() -> set:
    """Return set of HIGH/MED conviction tickers from daily watchlist (V2)."""
    try:
        from data_layer import get_latest_daily_watchlist
        data = get_latest_daily_watchlist()
        return {
            e["ticker"]
            for e in data.get("entries", [])
            if e.get("conviction") in ("HIGH", "MED")
        }
    except Exception:
        return set()


def build_ticker_details(theme_name: str, ticker_metrics: dict) -> list:
    """Reconstruct per-ticker detail list for theme drill-down from parquet data."""
    tickers_in_theme = THEMES.get(theme_name, [])
    details = []
    for t in tickers_in_theme:
        tm = ticker_metrics.get(t)
        if tm is None:
            continue
        details.append({
            "ticker":       t,
            "last":         tm.get("last"),
            "ret_1m":       tm.get("ret_1m"),
            "ret_3m":       tm.get("ret_3m"),
            "ret_6m":       tm.get("ret_6m"),
            "ret_ytd":      tm.get("ret_ytd"),
            "dist_52w_high": tm.get("dist_52w_high"),
            "above_50dma":  tm.get("above_50dma", False),
            "above_200dma": tm.get("above_200dma", False),
            "has_ep":       bool(tm.get("has_ep", False)),
            "has_vol_surge": bool(tm.get("has_vol_surge", False)),
            "accel_5d_21d": tm.get("accel_5d_21d"),
        })
    return sorted(details,
                  key=lambda x: x["ret_1m"] if x["ret_1m"] is not None else -999,
                  reverse=True)


def build_cross_theme_movers(ticker_metrics: dict) -> list:
    """Build cross-theme movers from parquet ticker_metrics (n_themes >= 3)."""
    rows = []
    for ticker, tm in ticker_metrics.items():
        memberships = tm.get("theme_memberships") or []
        if len(memberships) < 3:
            continue
        rows.append({
            "ticker":   ticker,
            "n_themes": len(memberships),
            "themes":   memberships,
            "ret_1m":   tm.get("ret_1m"),
            "ret_3m":   tm.get("ret_3m"),
            "last":     tm.get("last"),
        })
    rows.sort(key=lambda x: (x["n_themes"], x.get("ret_1m") or -999), reverse=True)
    return rows[:25]


# ─────────────────────────────────────────────────────────────────────────────
#  RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def render_heat_map(theme_records, wl_tickers: set):
    """Heat map — sorted by composite score (D5). Excludes Most_Shorted (A2)."""
    section_header("THEME HEAT MAP", "ALL NARRATIVES — SORTED BY COMPOSITE MOMENTUM SCORE")

    rows = []
    for r in theme_records:
        if r["theme"] == "Most_Shorted":
            continue
        groups = THEME_TO_MACRO_GROUPS.get(r["theme"], [])
        groups_str = ", ".join(g.replace("_", " ").title() for g in groups)

        # WL badge: count of theme tickers on watchlist (V2)
        theme_tickers = THEMES.get(r["theme"], [])
        wl_count = sum(1 for t in theme_tickers if t in wl_tickers)

        rows.append({
            "Score":      r.get("composite_score"),
            "Theme":      r["theme"].replace("_", " "),
            "Group":      groups_str or "—",
            "1M %":       r["avg_1m"],
            "3M %":       r["avg_3m"],
            "Accel":      r.get("acceleration"),
            "% Up":       r["pct_up_1m"],
            "EP 🔥":      r.get("ep_active_count") or 0,
            "Vol 📊":     r.get("vol_surge_count") or 0,
            "📋 WL":      wl_count or "",
            "Top Mover":  r["top_mover_1m"] or "—",
            "Best Ret":   r["top_mover_1m_ret"],
            "N":          r["n_with_data"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Score", ascending=False, na_position="last").reset_index(drop=True)

    def bg_pct(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v > 20:  return "background-color:#0d3320;color:#00ff88;font-weight:700"
        if v > 10:  return "background-color:#0a2a1a;color:#00cc66;font-weight:700"
        if v > 0:   return "background-color:#111a14;color:#00aa55"
        if v > -5:  return "background-color:#1a1408;color:#cc8800"
        if v > -15: return "background-color:#1a0e0e;color:#cc4444"
        return              "background-color:#200a0a;color:#ff4444;font-weight:700"

    def bg_score(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v >= 80: return "background-color:#0d3320;color:#00ff88;font-weight:700"
        if v >= 60: return "background-color:#0a2a1a;color:#00cc66"
        if v >= 40: return "background-color:#111111;color:#cccccc"
        if v >= 20: return "background-color:#1a1408;color:#cc8800"
        return              "background-color:#1a0e0e;color:#cc4444"

    def bg_accel(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v > 5:   return "background-color:#0d3320;color:#00ff88;font-weight:700"
        if v > 1:   return "background-color:#111a14;color:#00aa55"
        if v > -1:  return "background-color:#111111;color:#888"
        if v > -5:  return "background-color:#1a1408;color:#cc8800"
        return              "background-color:#1a0e0e;color:#cc4444"

    def bg_activity(v):
        try:
            v = int(v)
            if v >= 4: return "background-color:#0d3320;color:#00ff88;font-weight:700;text-align:center"
            if v >= 2: return "background-color:#111a14;color:#00aa55;text-align:center"
            if v >= 1: return "background-color:#1a1408;color:#FFA500;text-align:center"
            return "background-color:#111111;color:#333;text-align:center"
        except: return ""

    def fmt(x):
        return f"{x:+.1f}%" if pd.notna(x) and x is not None else "—"

    styled = (
        df.style
          .map(bg_score,    subset=["Score"])
          .map(bg_pct,      subset=["1M %", "3M %", "Best Ret"])
          .map(bg_accel,    subset=["Accel"])
          .map(bg_activity, subset=["EP 🔥", "Vol 📊"])
          .map(lambda v: "background-color:#111a14;color:#00aa55;font-weight:700"
               if pd.notna(v) and v != "" and int(v) > 0
               else "background-color:#111111;color:#333",
               subset=["📋 WL"])
          .map(lambda v: "background-color:#111a14;color:#00aa55;font-weight:700"
               if pd.notna(v) and v >= 80 else (
               "background-color:#1a1408;color:#cc8800" if pd.notna(v) and v >= 50 else
               "background-color:#1a0e0e;color:#cc4444" if pd.notna(v) else
               "background-color:#1a1a1a;color:#555"),
               subset=["% Up"])
          .format({
              "Score":    lambda x: f"{x:.0f}" if pd.notna(x) else "—",
              "1M %":     fmt,
              "3M %":     fmt,
              "Accel":    lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
              "Best Ret": fmt,
              "% Up":     lambda x: f"{x:.0f}%" if pd.notna(x) else "—",
              "EP 🔥":    lambda x: str(int(x)) if x else "—",
              "Vol 📊":   lambda x: str(int(x)) if x else "—",
              "📋 WL":    lambda x: str(int(x)) if x else "—",
          })
          .set_properties(**{
              "background-color": "#111111",
              "color":            "#cccccc",
              "border":           "1px solid #2a2a2a",
              "font-family":      "Fira Code, monospace",
              "font-size":        "12px",
              "padding":          "6px 10px",
          })
          .set_properties(subset=["Theme"], **{
              "color": "#FFA500", "font-family": "Fira Code, monospace", "font-weight": "500",
          })
          .set_properties(subset=["Group"], **{"color": "#B87333", "font-size": "11px"})
          .set_properties(subset=["Top Mover"], **{
              "color": "#00ff88", "font-family": "Orbitron, monospace", "font-size": "11px",
          })
          .set_properties(subset=["N"], **{"color": "#666", "font-size": "11px", "text-align": "center"})
          .set_table_styles([
              {"selector": "thead th", "props": [
                  ("background-color", "#0a0a0a"), ("color", "#FFA500"),
                  ("font-family", "Orbitron, monospace"), ("font-size", "10px"),
                  ("letter-spacing", "2px"), ("text-transform", "uppercase"),
                  ("border-bottom", "2px solid #FFA500"), ("padding", "8px 10px"),
              ]},
              {"selector": "tbody tr:hover td", "props": [("background-color", "#1a1a2a !important")]},
              {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#0f0f0f")]},
          ])
    )
    st.dataframe(styled, use_container_width=True, height=680, hide_index=True)
    st.caption("Score = composite momentum (0-100): 35% level · 25% breadth · 25% 5d/21d accel · 15% EP+vol activity   |   📋 WL = # theme tickers on daily watchlist (HIGH/MED)")


def render_rotation_detection(theme_records):
    section_header("ROTATION DETECTION", "EMERGING ⟷ FADING — WHERE CAPITAL IS MOVING")

    valid = [r for r in theme_records
             if r["theme"] != "Most_Shorted"
             and r["avg_1m"] is not None
             and r.get("acceleration") is not None]
    if not valid:
        st.markdown(f"<div style='color:{GREY}'>Insufficient data</div>", unsafe_allow_html=True)
        return

    by_1m_desc   = sorted(valid, key=lambda x: x["avg_1m"],      reverse=True)
    by_1m_asc    = sorted(valid, key=lambda x: x["avg_1m"])
    by_accel_desc = sorted(valid, key=lambda x: x["acceleration"], reverse=True)
    by_accel_asc  = sorted(valid, key=lambda x: x["acceleration"])

    c1, c2, c3, c4 = st.columns(4)

    def render_list(col, title, items, show_field, sub_field=None, color=AMBER):
        with col:
            st.markdown(
                f"<div style='background:{PANEL_BG};border:1px solid {GRID};"
                f"padding:14px;height:100%'>"
                f"<div style='color:{COPPER};font-family:Fira Code,monospace;"
                f"font-size:9px;letter-spacing:2px;margin-bottom:10px'>{title}</div>",
                unsafe_allow_html=True)
            for item in items[:5]:
                v  = item[show_field]
                vc = GREEN if v > 0 else RED
                sub = ""
                if sub_field:
                    sv = item.get(sub_field)
                    if sv is not None:
                        sub = f"<span style='color:{COPPER};font-size:10px;margin-left:6px'>(3M: {sv:+.1f}%)</span>"
                ep_badge  = " <span style='color:#FFA500;font-size:9px'>🔥EP</span>" if item.get("ep_active_count") else ""
                vol_badge = " <span style='color:#00aa55;font-size:9px'>📊</span>"   if item.get("vol_surge_count") else ""
                st.markdown(
                    f"<div style='font-family:Fira Code,monospace;padding:4px 0;"
                    f"border-bottom:1px solid {GRID}'>"
                    f"<div style='color:{color};font-size:11px'>"
                    f"{item['theme'].replace('_', ' ')}{ep_badge}{vol_badge}</div>"
                    f"<div style='color:{vc};font-size:13px;font-weight:700;margin-top:2px'>"
                    f"{v:+.2f}%{sub}</div></div>",
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    render_list(c1, "▲ HOTTEST (1M)",  by_1m_desc,    "avg_1m")
    render_list(c2, "▼ COLDEST (1M)",  by_1m_asc,     "avg_1m")
    render_list(c3, "⚡ EMERGING",      by_accel_desc, "acceleration", "avg_3m")
    render_list(c4, "⌫ FADING",        by_accel_asc,  "acceleration", "avg_3m")


def render_macro_group_view(theme_records, macro_metrics):
    section_header("MACRO GROUP VIEW", "THEMES ORGANIZED BY ACTIVE ROTATION GROUP")

    group_status = {}
    if macro_metrics:
        for grp, data in macro_metrics.get("group_scores", {}).items():
            confirmed = data.get("confirmed", 0)
            total     = data.get("total", 0)
            if total == 0:
                group_status[grp] = ("INACTIVE", GREY)
            elif confirmed / total >= 0.66:
                group_status[grp] = ("ACTIVE",   GREEN)
            elif confirmed / total >= 0.33:
                group_status[grp] = ("MIXED",    AMBER)
            else:
                group_status[grp] = ("DENIED",   RED)

    by_name = {r["theme"]: r for r in theme_records}
    group_meta = [
        ("Speculative_Risk_On",    "SPECULATIVE RISK-ON"),
        ("Cyclical_Expansion",     "CYCLICAL EXPANSION"),
        ("Commodity_Confirmation", "COMMODITY CONFIRMATION"),
        ("Idiosyncratic",          "IDIOSYNCRATIC"),
    ]

    cols = st.columns(2)
    for idx, (gkey, glabel) in enumerate(group_meta):
        col = cols[idx % 2]
        themes_in_group = MACRO_GROUPS.get(gkey, [])
        status, color   = group_status.get(gkey, ("INACTIVE", GREY))

        with col:
            html = (
                f"<div style='background:{PANEL_BG};border:2px solid {color};"
                f"padding:16px;margin-bottom:16px'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;margin-bottom:12px'>"
                f"<div style='color:{AMBER};font-family:Orbitron,monospace;"
                f"font-size:12px;font-weight:700;letter-spacing:3px'>{glabel}</div>"
                f"<div style='color:{color};font-family:Fira Code,monospace;"
                f"font-size:11px;font-weight:700'>&#9679; {status}</div>"
                f"</div>"
            )
            sorted_themes = sorted(
                [(t, by_name.get(t)) for t in themes_in_group if by_name.get(t)],
                key=lambda x: x[1].get("composite_score") or 0,
                reverse=True,
            )
            for tname, rec in sorted_themes:
                v    = rec["avg_1m"]
                vc   = GREEN if (v or 0) > 0 else RED
                vstr = f"{v:+.2f}%" if v is not None else "—"
                sc   = rec.get("composite_score")
                sc_str = f"<span style='color:{COPPER};font-size:10px;margin-left:8px'>({sc:.0f})</span>" if sc is not None else ""
                ep_b  = " 🔥" if rec.get("ep_active_count") else ""
                vol_b = " 📊" if rec.get("vol_surge_count") else ""
                html += (
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:6px 0;border-bottom:1px solid {GRID};"
                    f"font-family:Fira Code,monospace'>"
                    f"<span style='color:{COPPER};font-size:11px'>"
                    f"{tname.replace('_', ' ')}{ep_b}{vol_b}</span>"
                    f"<span style='color:{vc};font-size:12px;font-weight:700'>"
                    f"{vstr}{sc_str}</span></div>"
                )
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)


def render_theme_detail(theme_records, ticker_metrics: dict, wl_tickers: set):
    section_header("THEME DETAIL DRILL-DOWN", "INDIVIDUAL TICKERS WITHIN A THEME")

    theme_names = sorted([
        r["theme"] for r in theme_records
        if r["n_with_data"] > 0 and r["theme"] != "Most_Shorted"
    ])
    if not theme_names:
        st.markdown(f"<div style='color:{GREY}'>No themes with data</div>", unsafe_allow_html=True)
        return

    selected = st.selectbox("Select theme", theme_names, format_func=lambda x: x.replace("_", " "))
    record   = next((r for r in theme_records if r["theme"] == selected), None)
    if not record:
        return

    avg_1m  = record["avg_1m"]
    pct_up  = record["pct_up_1m"]
    color   = GREEN if (avg_1m or 0) > 0 else RED
    groups  = THEME_TO_MACRO_GROUPS.get(selected, [])
    grp_str = " / ".join(g.replace("_", " ").title() for g in groups) or "—"
    score   = record.get("composite_score")
    ep_cnt  = record.get("ep_active_count", 0)
    vol_cnt = record.get("vol_surge_count", 0)

    def _kv(label, val, col=AMBER, size="18px"):
        return (f"<div><div style='color:{COPPER};font-family:Fira Code,monospace;"
                f"font-size:9px;letter-spacing:2px'>{label}</div>"
                f"<div style='color:{col};font-family:Orbitron,monospace;"
                f"font-size:{size};font-weight:700'>{val}</div></div>")

    st.markdown(
        f"<div style='background:{PANEL_BG};border:1px solid {GRID};"
        f"padding:16px;margin-bottom:16px'>"
        f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:24px'>"
        + _kv("THEME",       selected.replace("_", " ").upper())
        + _kv("SCORE",       f"{score:.0f}/100" if score is not None else "—", AMBER)
        + _kv("1M AVG",      fmt_pct(avg_1m) if avg_1m is not None else "—", color)
        + _kv("% UP",        (fmt_num(pct_up, 0) + "%") if pct_up is not None else "—")
        + _kv("ACTIVE EP",   str(ep_cnt),  "#FFA500" if ep_cnt else GREY)
        + _kv("VOL SURGES",  str(vol_cnt), GREEN    if vol_cnt else GREY)
        + _kv("MACRO GROUP", grp_str, size="14px")
        + "</div></div>",
        unsafe_allow_html=True)

    # Build ticker details from parquet
    ticker_details = build_ticker_details(selected, ticker_metrics)
    if not ticker_details:
        st.markdown(f"<div style='color:{GREY}'>No ticker data available</div>", unsafe_allow_html=True)
        return

    rows = []
    for td in ticker_details:
        tk = td["ticker"]
        wl_badge = "📋" if tk in wl_tickers else ""
        ep_badge  = "🔥" if td.get("has_ep")       else ""
        vol_badge = "📊" if td.get("has_vol_surge") else ""
        rows.append({
            "Ticker":         tk,
            "WL":             wl_badge,
            "Signals":        f"{ep_badge}{vol_badge}",
            "Last":           td["last"],
            "1M":             td["ret_1m"],
            "3M":             td["ret_3m"],
            "6M":             td["ret_6m"],
            "YTD":            td["ret_ytd"],
            "Accel 5/21":     td.get("accel_5d_21d"),
            "% from 52W Hi":  td["dist_52w_high"],
            "> 50DMA":        "✓" if td["above_50dma"]  else "✗",
            "> 200DMA":       "✓" if td["above_200dma"] else "✗",
        })

    df = pd.DataFrame(rows)

    def bg_pct(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v > 10:  return "background-color:#0d3320;color:#00ff88;font-weight:700"
        if v > 0:   return "background-color:#111a14;color:#00aa55"
        if v > -10: return "background-color:#1a1408;color:#cc8800"
        return              "background-color:#1a0e0e;color:#cc4444;font-weight:700"

    def bg_accel(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v > 5:   return "background-color:#0d3320;color:#00ff88;font-weight:700"
        if v > 1:   return "background-color:#111a14;color:#00aa55"
        if v > -1:  return "background-color:#111111;color:#888"
        return              "background-color:#1a1408;color:#cc8800"

    def bg_dist(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v > -5:  return "background-color:#111a14;color:#00aa55"
        if v > -15: return "background-color:#1a1408;color:#cc8800"
        return              "background-color:#1a0e0e;color:#cc4444"

    def bg_ma(v):
        if v == "✓": return "background-color:#111a14;color:#00ff88;text-align:center"
        return "background-color:#1a0e0e;color:#ff4444;text-align:center"

    styled = (df.style
              .map(bg_pct,   subset=["1M", "3M", "6M", "YTD"])
              .map(bg_accel, subset=["Accel 5/21"])
              .map(bg_dist,  subset=["% from 52W Hi"])
              .map(bg_ma,    subset=["> 50DMA", "> 200DMA"])
              .map(lambda v: "color:#00ff88;font-weight:700;text-align:center" if v == "📋" else
                             "color:#333;text-align:center", subset=["WL"])
              .format({
                  "Last":         "{:.2f}",
                  "1M":           lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "3M":           lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "6M":           lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "YTD":          lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "Accel 5/21":   lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
                  "% from 52W Hi": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
              })
              .set_properties(**{
                  "background-color": "#111111", "color": "#cccccc",
                  "border": "1px solid #2a2a2a",
                  "font-family": "Fira Code, monospace", "font-size": "12px", "padding": "5px 10px",
              })
              .set_properties(subset=["Ticker"], **{
                  "color": "#FFA500", "font-weight": "600",
              })
              .set_table_styles([
                  {"selector": "thead th", "props": [
                      ("background-color", "#0a0a0a"), ("color", "#FFA500"),
                      ("font-family", "Orbitron, monospace"), ("font-size", "10px"),
                      ("letter-spacing", "2px"), ("border-bottom", "2px solid #FFA500"),
                      ("padding", "7px 10px"),
                  ]},
                  {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#0f0f0f")]},
              ]))
    st.dataframe(styled, use_container_width=True, height=400, hide_index=True)
    st.caption("📋 = on daily watchlist (HIGH/MED conviction)  |  🔥 = active EP event  |  📊 = fresh volume surge  |  Accel = 5d minus 21d return")


def render_short_interest_monitor(ticker_metrics: dict, wl_tickers: set):
    """A2: Standalone Short Interest Monitor — Most_Shorted as its own section."""
    section_header("SHORT INTEREST MONITOR", "MOST_SHORTED — POSITIONING RISK & SQUEEZE CANDIDATES")

    short_tickers = THEMES.get("Most_Shorted", [])
    rows = []
    for tk in short_tickers:
        tm = ticker_metrics.get(tk)
        if tm is None:
            continue
        rows.append({
            "Ticker":        tk,
            "WL":            "📋" if tk in wl_tickers else "",
            "Last":          tm.get("last"),
            "1M %":          tm.get("ret_1m"),
            "3M %":          tm.get("ret_3m"),
            "YTD %":         tm.get("ret_ytd"),
            "Accel 5/21":    tm.get("accel_5d_21d"),
            "% from 52W Hi": tm.get("dist_52w_high"),
            "> 50DMA":       "✓" if tm.get("above_50dma") else "✗",
            "EP 🔥":         "🔥" if tm.get("has_ep")       else "",
            "Vol 📊":        "📊" if tm.get("has_vol_surge") else "",
        })

    if not rows:
        st.markdown(f"<div style='color:{GREY}'>No data for Most_Shorted universe</div>",
                    unsafe_allow_html=True)
        return

    df = pd.DataFrame(rows).sort_values("1M %", ascending=False, na_position="last")

    def bg_pct(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v > 20:  return "background-color:#0d3320;color:#00ff88;font-weight:700"
        if v > 5:   return "background-color:#111a14;color:#00aa55"
        if v > -5:  return "background-color:#111111;color:#cccccc"
        if v > -20: return "background-color:#1a1408;color:#cc8800"
        return              "background-color:#1a0e0e;color:#cc4444;font-weight:700"

    def bg_accel(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v > 5:  return "background-color:#0d3320;color:#00ff88;font-weight:700"
        if v > 1:  return "background-color:#111a14;color:#00aa55"
        if v > -1: return "background-color:#111111;color:#888"
        return             "background-color:#1a1408;color:#cc8800"

    st.markdown(
        f"<div style='background:{PANEL_BG};border:1px solid #2a2a2a;"
        f"padding:10px 16px;margin-bottom:12px;font-family:Fira Code,monospace;"
        f"font-size:11px;color:{COPPER};'>"
        f"High short interest = positioning risk. Rising price + EP/vol surge = squeeze setup. "
        f"Treat as <span style='color:{AMBER};font-weight:700'>WATCH-ONLY</span> unless conviction score qualifies.</div>",
        unsafe_allow_html=True)

    styled = (df.style
              .map(bg_pct,   subset=["1M %", "3M %", "YTD %"])
              .map(bg_accel, subset=["Accel 5/21"])
              .format({
                  "Last":          lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                  "1M %":          lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "3M %":          lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "YTD %":         lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "Accel 5/21":    lambda x: f"{x:+.1f}" if pd.notna(x) else "—",
                  "% from 52W Hi": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
              })
              .set_properties(**{
                  "background-color": "#111111", "color": "#cccccc",
                  "border": "1px solid #2a2a2a",
                  "font-family": "Fira Code, monospace", "font-size": "12px",
                  "padding": "5px 10px",
              })
              .set_properties(subset=["Ticker"], **{"color": "#FFA500", "font-weight": "600"})
              .set_table_styles([
                  {"selector": "thead th", "props": [
                      ("background-color", "#0a0a0a"), ("color", "#FFA500"),
                      ("font-family", "Orbitron, monospace"), ("font-size", "10px"),
                      ("letter-spacing", "2px"), ("border-bottom", "2px solid #FFA500"),
                      ("padding", "7px 10px"),
                  ]},
                  {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#0f0f0f")]},
              ]))
    st.dataframe(styled, use_container_width=True, height=500, hide_index=True)


def render_cross_theme_movers(ticker_metrics: dict, wl_tickers: set):
    section_header("CROSS-THEME MOVERS", "TICKERS IN 3+ THEMES — MAX NARRATIVE TAILWINDS")

    rows = build_cross_theme_movers(ticker_metrics)
    if not rows:
        st.markdown(f"<div style='color:{GREY}'>No cross-theme movers found</div>", unsafe_allow_html=True)
        return

    table_rows = []
    for r in rows:
        themes_short = ", ".join(t.replace("_", " ") for t in r["themes"][:5])
        if len(r["themes"]) > 5:
            themes_short += f" +{len(r['themes']) - 5}"
        wl_badge = "📋" if r["ticker"] in wl_tickers else ""
        table_rows.append({
            "Ticker":   r["ticker"],
            "WL":       wl_badge,
            "Themes":   r["n_themes"],
            "1M":       r.get("ret_1m"),
            "3M":       r.get("ret_3m"),
            "Last":     r.get("last"),
            "Member of": themes_short,
        })

    df = pd.DataFrame(table_rows)

    def bg_pct(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v > 10:  return "background-color:#0d3320;color:#00ff88;font-weight:700"
        if v > 0:   return "background-color:#111a14;color:#00aa55"
        if v > -10: return "background-color:#1a1408;color:#cc8800"
        return              "background-color:#1a0e0e;color:#cc4444;font-weight:700"

    def bg_themes(v):
        if v >= 5: return "background-color:#0d3320;color:#00ff88;font-weight:700;text-align:center"
        if v >= 4: return "background-color:#1a1408;color:#FFA500;font-weight:700;text-align:center"
        return "background-color:#111111;color:#B87333;text-align:center"

    styled = (df.style
              .map(bg_pct,    subset=["1M", "3M"])
              .map(bg_themes, subset=["Themes"])
              .map(lambda v: "color:#00ff88;font-weight:700;text-align:center" if v == "📋"
                             else "color:#333;text-align:center", subset=["WL"])
              .format({
                  "Last": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                  "1M":   lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "3M":   lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
              })
              .set_properties(**{
                  "background-color": "#111111", "color": "#cccccc",
                  "border": "1px solid #2a2a2a",
                  "font-family": "Fira Code, monospace", "font-size": "12px",
                  "padding": "5px 10px",
              })
              .set_properties(subset=["Ticker"], **{
                  "color": "#FFA500", "font-family": "Orbitron, monospace",
                  "font-size": "11px", "font-weight": "700",
              })
              .set_properties(subset=["Member of"], **{"color": "#B87333", "font-size": "11px"})
              .set_table_styles([
                  {"selector": "thead th", "props": [
                      ("background-color", "#0a0a0a"), ("color", "#FFA500"),
                      ("font-family", "Orbitron, monospace"), ("font-size", "10px"),
                      ("letter-spacing", "2px"), ("border-bottom", "2px solid #FFA500"),
                      ("padding", "7px 10px"),
                  ]},
                  {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#0f0f0f")]},
              ]))
    st.dataframe(styled, use_container_width=True, height=500, hide_index=True)
    st.caption("📋 = on daily watchlist (HIGH/MED conviction)")


def render_membership_search():
    section_header("THEME MEMBERSHIP SEARCH", "FIND WHICH THEMES A TICKER BELONGS TO")

    query = st.text_input("Enter ticker", "", placeholder="e.g. BE, NVDA, OKLO").strip().upper()
    if not query:
        return

    matches = get_themes_for_ticker(query)
    if not matches:
        st.markdown(
            f"<div style='color:{GREY};font-family:Fira Code,monospace;padding:12px 0'>"
            f"No themes contain '{query}'</div>",
            unsafe_allow_html=True)
        return

    all_groups = set()
    for m in matches:
        for g in THEME_TO_MACRO_GROUPS.get(m, []):
            all_groups.add(g)
    groups_str = ", ".join(g.replace("_", " ").title() for g in all_groups) or "None"
    bullets = "<br>".join(
        f"<span style='color:{AMBER}'>&#8226;</span> <span style='color:#cccccc'>{m.replace('_', ' ')}</span>"
        for m in matches
    )
    st.markdown(
        f"<div style='background:{PANEL_BG};border:1px solid {AMBER};padding:16px;margin-top:8px'>"
        f"<div style='font-family:Fira Code,monospace'>"
        f"<span style='color:{COPPER};font-size:11px;letter-spacing:2px'>TICKER:</span>"
        f"<span style='color:{AMBER};font-family:Orbitron,monospace;font-size:18px;"
        f"font-weight:700;margin-left:8px'>{query}</span></div>"
        f"<div style='font-family:Fira Code,monospace;margin-top:12px'>"
        f"<span style='color:{COPPER};font-size:11px;letter-spacing:2px'>"
        f"FOUND IN {len(matches)} THEMES:</span></div>"
        f"<div style='margin-top:8px;font-family:Fira Code,monospace;font-size:13px'>{bullets}</div>"
        f"<div style='font-family:Fira Code,monospace;margin-top:14px'>"
        f"<span style='color:{COPPER};font-size:11px;letter-spacing:2px'>MACRO GROUPS:</span>"
        f"<span style='color:{AMBER};margin-left:8px;font-size:13px'>{groups_str}</span>"
        f"</div></div>",
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY  (A1: parquet-first, live fallback)
# ─────────────────────────────────────────────────────────────────────────────

def render():
    st.markdown(
        f"<h1 style='color:{AMBER};font-family:Orbitron,monospace;font-weight:900;"
        f"letter-spacing:8px;border-bottom:2px solid {AMBER};padding-bottom:12px'>"
        f"&#9685; SPECULATIVE THEMES</h1>",
        unsafe_allow_html=True)

    # ── A1: Try parquet reads first ───────────────────────────────────────
    theme_records  = load_theme_state()
    ticker_metrics = load_ticker_metrics()

    if not theme_records or not ticker_metrics:
        # Parquets missing — fall back to live fetch
        st.info("⚠ Pre-computed parquets not found — fetching live (may take ~30s). "
                "Run speculative_theme_prep.py to enable fast parquet reads.")
        from tactical_data_layer import fetch_universe, compute_theme_metrics, compute_macro_metrics
        with st.spinner("Pulling thematic universe..."):
            theme_tickers = tuple(get_all_unique_tickers())
            prices_themes, failed_themes = fetch_universe(theme_tickers)
        if prices_themes.empty:
            st.error("⚠ Theme universe fetch returned no data.")
            return
        if failed_themes:
            with st.expander(f"⚠ {len(failed_themes)} ticker(s) failed"):
                st.code(", ".join(failed_themes))
        theme_records, ticker_metrics = compute_theme_metrics(prices_themes, THEMES)
        macro_metrics = None
    else:
        as_of = theme_records[0].get("generated_at", "")[:16] if theme_records else "—"
        st.caption(f"Data as of: {as_of}  ·  {len(theme_records)} themes  ·  {len(ticker_metrics)} tickers  "
                   f"·  Next update: tonight 21:30 UTC")
        # Macro group view reads from tactical_macro_state.parquet (already fast)
        macro_metrics = None
        try:
            import pandas as _pd
            _mdf = _pd.read_parquet("data/tactical_macro_state.parquet")
            if not _mdf.empty:
                row = _mdf.iloc[0].to_dict()
                group_scores = {}
                for grp in ("Speculative_Risk_On", "Cyclical_Expansion",
                            "Commodity_Confirmation", "Idiosyncratic"):
                    c = row.get(f"group_{grp}_confirmed", 0)
                    t = row.get(f"group_{grp}_total", 0)
                    p = row.get(f"group_{grp}_pct",   0)
                    group_scores[grp] = {"confirmed": c, "total": t, "pct": p}
                macro_metrics = {"group_scores": group_scores}
        except Exception:
            pass

    # ── V2: Load watchlist crossover tickers ─────────────────────────────
    wl_tickers = load_watchlist_tickers()

    # ── Render all sections ───────────────────────────────────────────────
    render_heat_map(theme_records, wl_tickers)
    render_rotation_detection(theme_records)
    render_macro_group_view(theme_records, macro_metrics)
    render_theme_detail(theme_records, ticker_metrics, wl_tickers)
    render_short_interest_monitor(ticker_metrics, wl_tickers)
    render_cross_theme_movers(ticker_metrics, wl_tickers)
    render_membership_search()
