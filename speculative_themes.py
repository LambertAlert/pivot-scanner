"""
speculative_themes.py — The Speculative Themes tab.

Implements:
  • Theme Heat Map (sortable table, all themes ranked)
  • Rotation Detection (hottest, coldest, emerging, fading)
  • Macro Group View (themes filtered by active macro group)
  • Theme Detail Drill-Down (per-ticker breakdown)
  • Cross-Theme Movers (tickers in multiple themes)
  • Theme Membership Search
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tactical_data_layer import (
    fetch_universe,
    compute_macro_metrics,
    compute_theme_metrics,
    compute_cross_theme_movers,
    all_macro_tickers,
)
from themes import (
    THEMES,
    THEME_TO_MACRO_GROUPS,
    MACRO_GROUPS,
    get_all_unique_tickers,
    get_themes_for_ticker,
)

# Reuse colors from macro_view
from macro_view import (
    AMBER, COPPER, OBSIDIAN, GREEN, RED, GREY,
    PANEL_BG, GRID, STATE_COLORS,
    section_header, kpi_box, fmt_pct, fmt_num,
)


# =============================================================================
# RENDERERS
# =============================================================================

def render_heat_map(theme_records):
    section_header("THEME HEAT MAP", "ALL NARRATIVES — SORTED BY 1M PERFORMANCE")

    rows = []
    for r in theme_records:
        groups = THEME_TO_MACRO_GROUPS.get(r["theme"], [])
        groups_str = ", ".join(g.replace("_", " ").title() for g in groups)
        rows.append({
            "Theme":      r["theme"].replace("_", " "),
            "Group":      groups_str or "—",
            "1M %":       r["avg_1m"],
            "3M %":       r["avg_3m"],
            "6M %":       r["avg_6m"],
            "YTD %":      r["avg_ytd"],
            "% Up":       r["pct_up_1m"],
            "Top Mover":  r["top_mover_1m"] or "—",
            "Best Ret":   r["top_mover_1m_ret"],
            "N":          r["n_with_data"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("1M %", ascending=False, na_position="last").reset_index(drop=True)

    def bg_pct(v):
        """Background color for percentage cells."""
        if v is None or pd.isna(v):
            return "background-color:#1a1a1a;color:#555"
        if v > 20:  return "background-color:#0d3320;color:#00ff88;font-weight:700"
        if v > 10:  return "background-color:#0a2a1a;color:#00cc66;font-weight:700"
        if v > 0:   return "background-color:#111a14;color:#00aa55"
        if v > -5:  return "background-color:#1a1408;color:#cc8800"
        if v > -15: return "background-color:#1a0e0e;color:#cc4444"
        return              "background-color:#200a0a;color:#ff4444;font-weight:700"

    def fmt(x):
        return f"{x:+.1f}%" if pd.notna(x) else "—"

    styled = (
        df.style
          .map(bg_pct, subset=["1M %", "3M %", "6M %", "YTD %", "Best Ret"])
          .map(lambda v: "background-color:#111a14;color:#00aa55;font-weight:700"
               if pd.notna(v) and v >= 80 else (
               "background-color:#1a1408;color:#cc8800" if pd.notna(v) and v >= 50 else
               "background-color:#1a0e0e;color:#cc4444" if pd.notna(v) else
               "background-color:#1a1a1a;color:#555"),
               subset=["% Up"])
          .format({
              "1M %":    fmt,
              "3M %":    fmt,
              "6M %":    fmt,
              "YTD %":   fmt,
              "Best Ret": fmt,
              "% Up":    lambda x: f"{x:.0f}%" if pd.notna(x) else "—",
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
              "color":       "#FFA500",
              "font-family": "Fira Code, monospace",
              "font-weight": "500",
          })
          .set_properties(subset=["Group"], **{
              "color":     "#B87333",
              "font-size": "11px",
          })
          .set_properties(subset=["N"], **{
              "color":      "#666",
              "font-size":  "11px",
              "text-align": "center",
          })
          .set_properties(subset=["Top Mover"], **{
              "color":     "#00ff88",
              "font-family": "Orbitron, monospace",
              "font-size": "11px",
          })
          .set_table_styles([
              {"selector": "thead th", "props": [
                  ("background-color", "#0a0a0a"),
                  ("color", "#FFA500"),
                  ("font-family", "Orbitron, monospace"),
                  ("font-size", "10px"),
                  ("letter-spacing", "2px"),
                  ("text-transform", "uppercase"),
                  ("border-bottom", "2px solid #FFA500"),
                  ("padding", "8px 10px"),
              ]},
              {"selector": "tbody tr:hover td", "props": [
                  ("background-color", "#1a1a2a !important"),
              ]},
              {"selector": "tbody tr:nth-child(even) td", "props": [
                  ("background-color", "#0f0f0f"),
              ]},
          ])
    )

    st.dataframe(styled, use_container_width=True, height=680, hide_index=True)


def render_rotation_detection(theme_records):
    section_header("ROTATION DETECTION", "EMERGING ⟷ FADING — WHERE CAPITAL IS MOVING")

    # Filter to themes with valid data
    valid = [r for r in theme_records if r["avg_1m"] is not None and r["avg_3m"] is not None]
    if not valid:
        st.markdown(f"<div style='color:{GREY}'>Insufficient data</div>", unsafe_allow_html=True)
        return

    # Sort views
    by_1m_desc = sorted(valid, key=lambda x: x["avg_1m"], reverse=True)
    by_1m_asc = sorted(valid, key=lambda x: x["avg_1m"])

    # Emerging: 1M >> 3M (1M return well above 3M return = momentum building)
    # Compute "acceleration" as (1M return) − (3M return / 3) which gives monthly average rate of change
    for r in valid:
        r["acceleration"] = r["avg_1m"] - (r["avg_3m"] / 3.0)
    by_accel_desc = sorted(valid, key=lambda x: x["acceleration"], reverse=True)
    by_accel_asc = sorted(valid, key=lambda x: x["acceleration"])

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
                v = item[show_field]
                vc = GREEN if v > 0 else RED
                sub = ""
                if sub_field:
                    sv = item.get(sub_field)
                    if sv is not None:
                        sub = f"<span style='color:{COPPER};font-size:10px;margin-left:6px'>(3M: {sv:+.1f}%)</span>"
                st.markdown(
                    f"<div style='font-family:Fira Code,monospace;padding:4px 0;"
                    f"border-bottom:1px solid {GRID}'>"
                    f"<div style='color:{color};font-size:11px'>"
                    f"{item["theme"].replace("_", " ")}</div>"
                    f"<div style='color:{vc};font-size:13px;font-weight:700;margin-top:2px'>"
                    f"{v:+.2f}%{sub}</div></div>",
                    unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    render_list(c1, "▲ HOTTEST (1M)", by_1m_desc, "avg_1m")
    render_list(c2, "▼ COLDEST (1M)", by_1m_asc, "avg_1m")
    render_list(c3, "⚡ EMERGING", by_accel_desc, "acceleration", "avg_3m")
    render_list(c4, "⌫ FADING", by_accel_asc, "acceleration", "avg_3m")


def render_macro_group_view(theme_records, macro_metrics):
    section_header("MACRO GROUP VIEW", "THEMES ORGANIZED BY ACTIVE ROTATION GROUP")

    # Get group activity from macro metrics
    group_status = {}
    if macro_metrics:
        for grp, data in macro_metrics.get("group_scores", {}).items():
            confirmed = data.get("confirmed", 0)
            total = data.get("total", 0)
            if total == 0:
                group_status[grp] = ("INACTIVE", GREY)
            elif confirmed / total >= 0.66:
                group_status[grp] = ("ACTIVE", GREEN)
            elif confirmed / total >= 0.33:
                group_status[grp] = ("MIXED", AMBER)
            else:
                group_status[grp] = ("DENIED", RED)

    # Index theme records by name for fast lookup
    by_name = {r["theme"]: r for r in theme_records}

    group_meta = [
        ("Speculative_Risk_On",      "SPECULATIVE RISK-ON"),
        ("Cyclical_Expansion",       "CYCLICAL EXPANSION"),
        ("Commodity_Confirmation",   "COMMODITY CONFIRMATION"),
        ("Idiosyncratic",            "IDIOSYNCRATIC"),
    ]

    cols = st.columns(2)
    for idx, (gkey, glabel) in enumerate(group_meta):
        col = cols[idx % 2]
        themes_in_group = MACRO_GROUPS.get(gkey, [])
        status, color = group_status.get(gkey, ("INACTIVE", GREY))

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
                key=lambda x: x[1]["avg_1m"] if x[1]["avg_1m"] is not None else -999,
                reverse=True,
            )

            for tname, rec in sorted_themes:
                v = rec["avg_1m"]
                vc = GREEN if (v or 0) > 0 else RED
                vstr = f"{v:+.2f}%" if v is not None else "—"
                html += (
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:6px 0;border-bottom:1px solid {GRID};"
                    f"font-family:Fira Code,monospace'>"
                    f"<span style='color:{COPPER};font-size:11px'>"
                    f"{tname.replace('_', ' ')}</span>"
                    f"<span style='color:{vc};font-size:12px;font-weight:700'>"
                    f"{vstr}</span></div>"
                )

            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)


def render_theme_detail(theme_records):
    section_header("THEME DETAIL DRILL-DOWN", "INDIVIDUAL TICKERS WITHIN A THEME")

    theme_names = sorted([r["theme"] for r in theme_records if r["n_with_data"] > 0])

    if not theme_names:
        st.markdown(f"<div style='color:{GREY}'>No themes with data</div>", unsafe_allow_html=True)
        return

    selected = st.selectbox(
        "Select theme",
        theme_names,
        format_func=lambda x: x.replace("_", " "),
    )

    record = next((r for r in theme_records if r["theme"] == selected), None)
    if not record:
        return

    # Header strip
    avg_1m = record["avg_1m"]
    pct_up = record["pct_up_1m"]
    color = GREEN if (avg_1m or 0) > 0 else RED

    groups = THEME_TO_MACRO_GROUPS.get(selected, [])
    groups_str = " / ".join(g.replace("_", " ").title() for g in groups) or "—"

    def _kv(label, val, col=AMBER, size="18px"):
        return (f"<div><div style='color:{COPPER};font-family:Fira Code,monospace;"
                f"font-size:9px;letter-spacing:2px'>{label}</div>"
                f"<div style='color:{col};font-family:Orbitron,monospace;"
                f"font-size:{size};font-weight:700'>{val}</div></div>")

    st.markdown(
        f"<div style='background:{PANEL_BG};border:1px solid {GRID};"
        f"padding:16px;margin-bottom:16px'>"
        f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:24px'>"
        + _kv("THEME", selected.replace("_", " ").upper())
        + _kv("1M AVG", fmt_pct(avg_1m) if avg_1m is not None else "—", color)
        + _kv("% UP", (fmt_num(pct_up, 0) + "%") if pct_up is not None else "—")
        + _kv("TICKERS", f"{record['n_with_data']}/{record['n_tickers']}")
        + _kv("MACRO GROUP", groups_str, size="14px")
        + "</div></div>",
        unsafe_allow_html=True)

    # Per-ticker table
    rows = []
    for td in record["ticker_details"]:
        rows.append({
            "Ticker": td["ticker"],
            "Last": td["last"],
            "1M": td["ret_1m"],
            "3M": td["ret_3m"],
            "6M": td["ret_6m"],
            "YTD": td["ret_ytd"],
            "% from 52W High": td["dist_52w_high"],
            "> 50DMA": "✓" if td["above_50dma"] else "✗",
            "> 200DMA": "✓" if td["above_200dma"] else "✗",
        })

    df = pd.DataFrame(rows)

    def color_pct(v):
        if v is None or pd.isna(v):
            return f"color:{GREY}"
        if v > 0: return f"color:{GREEN}"
        return f"color:{RED}"

    def color_dist(v):
        if v is None or pd.isna(v):
            return f"color:{GREY}"
        if v > -5: return f"color:{GREEN}"
        if v > -15: return f"color:{AMBER}"
        return f"color:{RED}"

    def color_ma(v):
        if v == "✓": return f"color:{GREEN}"
        return f"color:{RED}"

    def bg_pct(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v > 10:  return "background-color:#0d3320;color:#00ff88;font-weight:700"
        if v > 0:   return "background-color:#111a14;color:#00aa55"
        if v > -10: return "background-color:#1a1408;color:#cc8800"
        return              "background-color:#1a0e0e;color:#cc4444;font-weight:700"

    def bg_dist(v):
        if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
        if v > -5:  return "background-color:#111a14;color:#00aa55"
        if v > -15: return "background-color:#1a1408;color:#cc8800"
        return              "background-color:#1a0e0e;color:#cc4444"

    def bg_ma(v):
        if v == "✓": return "background-color:#111a14;color:#00ff88;text-align:center"
        return "background-color:#1a0e0e;color:#ff4444;text-align:center"

    styled = (df.style
              .map(bg_pct, subset=["1M", "3M", "6M", "YTD"])
              .map(bg_dist, subset=["% from 52W High"])
              .map(bg_ma, subset=["> 50DMA", "> 200DMA"])
              .format({
                  "Last": "{:.2f}",
                  "1M": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "3M": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "6M": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "YTD": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "% from 52W High": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
              })
              .set_properties(**{
                  "background-color": "#111111",
                  "color":            "#cccccc",
                  "border":           "1px solid #2a2a2a",
                  "font-family":      "Fira Code, monospace",
                  "font-size":        "12px",
                  "padding":          "5px 10px",
              })
              .set_properties(subset=["Ticker"], **{"color": "#FFA500", "font-weight": "600"})
              .set_table_styles([
                  {"selector": "thead th", "props": [
                      ("background-color", "#0a0a0a"), ("color", "#FFA500"),
                      ("font-family", "Orbitron, monospace"), ("font-size", "10px"),
                      ("letter-spacing", "2px"), ("border-bottom", "2px solid #FFA500"),
                      ("padding", "7px 10px"),
                  ]},
                  {"selector": "tbody tr:nth-child(even) td", "props": [
                      ("background-color", "#0f0f0f"),
                  ]},
              ]))
    st.dataframe(styled, use_container_width=True, height=400, hide_index=True)


def render_cross_theme_movers(theme_records, ticker_metrics):
    section_header("CROSS-THEME MOVERS", "TICKERS IN MULTIPLE THEMES — MAX NARRATIVE TAILWINDS")

    rows = compute_cross_theme_movers(THEMES, ticker_metrics)
    # Filter: 3+ themes (single-theme tickers aren't interesting here)
    rows = [r for r in rows if r["n_themes"] >= 3]
    rows = rows[:25]

    if not rows:
        st.markdown(f"<div style='color:{GREY}'>No cross-theme movers found</div>", unsafe_allow_html=True)
        return

    table_rows = []
    for r in rows:
        themes_short = ", ".join(t.replace("_", " ") for t in r["themes"][:5])
        if len(r["themes"]) > 5:
            themes_short += f" + {len(r['themes']) - 5}"
        table_rows.append({
            "Ticker": r["ticker"],
            "Themes": r["n_themes"],
            "1M": r.get("ret_1m"),
            "3M": r.get("ret_3m"),
            "Last": r.get("last"),
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
              .map(bg_pct, subset=["1M", "3M"])
              .map(bg_themes, subset=["Themes"])
              .format({
                  "Last": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                  "1M": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                  "3M": lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
              })
              .set_properties(**{
                  "background-color": "#111111",
                  "color":            "#cccccc",
                  "border":           "1px solid #2a2a2a",
                  "font-family":      "Fira Code, monospace",
                  "font-size":        "12px",
                  "padding":          "5px 10px",
              })
              .set_properties(subset=["Ticker"], **{
                  "color": "#FFA500", "font-family": "Orbitron, monospace",
                  "font-size": "11px", "font-weight": "700",
              })
              .set_properties(subset=["Member of"], **{
                  "color": "#B87333", "font-size": "11px",
              })
              .set_table_styles([
                  {"selector": "thead th", "props": [
                      ("background-color", "#0a0a0a"), ("color", "#FFA500"),
                      ("font-family", "Orbitron, monospace"), ("font-size", "10px"),
                      ("letter-spacing", "2px"), ("border-bottom", "2px solid #FFA500"),
                      ("padding", "7px 10px"),
                  ]},
                  {"selector": "tbody tr:nth-child(even) td", "props": [
                      ("background-color", "#0f0f0f"),
                  ]},
              ]))

    st.dataframe(styled, use_container_width=True, height=600, hide_index=True)


def render_membership_search():
    section_header("THEME MEMBERSHIP SEARCH", "FIND WHICH THEMES A TICKER BELONGS TO")

    query = st.text_input("Enter ticker", "", placeholder="e.g. BE, NVDA, OKLO").strip().upper()
    if not query:
        return

    matches = get_themes_for_ticker(query)
    if not matches:
        st.markdown(f"<div style='color:{GREY};font-family:Fira Code,monospace;padding:12px 0'>No themes contain '{query}'</div>",
                    unsafe_allow_html=True)
        return

    # Get macro groups
    all_groups = set()
    for m in matches:
        for g in THEME_TO_MACRO_GROUPS.get(m, []):
            all_groups.add(g)

    groups_str = ", ".join(g.replace("_", " ").title() for g in all_groups) or "None"

    bullets_parts = []
    for m in matches:
        bullets_parts.append(
            "<span style=\"color:" + AMBER + "\">&#8226;</span> "
            "<span style=\"color:#cccccc\">" + m.replace("_", " ") + "</span>"
        )
    bullets = "<br>".join(bullets_parts)

    card_html = (
        "<div style=\"background:" + PANEL_BG + ";border:1px solid " + AMBER + ";"
        "padding:16px;margin-top:8px\">"
        "<div style=\"font-family:Fira Code,monospace\">"
        "<span style=\"color:" + COPPER + ";font-size:11px;letter-spacing:2px\">TICKER:</span>"
        "<span style=\"color:" + AMBER + ";font-family:Orbitron,monospace;font-size:18px;"
        "font-weight:700;margin-left:8px\">" + query + "</span></div>"
        "<div style=\"font-family:Fira Code,monospace;margin-top:12px\">"
        "<span style=\"color:" + COPPER + ";font-size:11px;letter-spacing:2px\">"
        "FOUND IN " + str(len(matches)) + " THEMES:</span></div>"
        "<div style=\"margin-top:8px;font-family:Fira Code,monospace;font-size:13px\">"
        + bullets + "</div>"
        "<div style=\"font-family:Fira Code,monospace;margin-top:14px\">"
        "<span style=\"color:" + COPPER + ";font-size:11px;letter-spacing:2px\">MACRO GROUPS:</span>"
        "<span style=\"color:" + AMBER + ";margin-left:8px;font-size:13px\">" + groups_str + "</span>"
        "</div></div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


# =============================================================================
# MAIN ENTRY
# =============================================================================

def render():
    st.markdown(
        f"<h1 style='color:{AMBER};font-family:Orbitron,monospace;font-weight:900;"
        f"letter-spacing:8px;border-bottom:2px solid {AMBER};padding-bottom:12px'>"
        f"&#9685; SPECULATIVE THEMES</h1>",
        unsafe_allow_html=True)

    # Theme universe — every unique ticker across all themes
    with st.spinner("Pulling thematic universe..."):
        theme_tickers = tuple(get_all_unique_tickers())
        prices_themes, failed_themes = fetch_universe(theme_tickers)

    if prices_themes.empty:
        st.error("⚠ Theme universe fetch returned no data.")
        return

    if failed_themes:
        with st.expander(f"⚠ {len(failed_themes)} ticker(s) failed to fetch (click to view)"):
            st.code(", ".join(failed_themes))
            st.caption("Themes containing only failed tickers will show '—'. Most likely delisted, ticker change, or yfinance hiccup.")

    # Compute theme metrics
    theme_records, ticker_metrics = compute_theme_metrics(prices_themes)

    # Also pull macro metrics for the bridge view (cached, so it's fast)
    with st.spinner("Pulling macro metrics for group view..."):
        prices_macro, failed_macro = fetch_universe(tuple(all_macro_tickers()))
    macro_metrics = compute_macro_metrics(prices_macro) if not prices_macro.empty else None

    render_heat_map(theme_records)
    render_rotation_detection(theme_records)
    render_macro_group_view(theme_records, macro_metrics)
    render_theme_detail(theme_records)
    render_cross_theme_movers(theme_records, ticker_metrics)
    render_membership_search()
