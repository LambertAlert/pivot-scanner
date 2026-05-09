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

    # Build a clean DataFrame for the heat map
    rows = []
    for r in theme_records:
        groups = THEME_TO_MACRO_GROUPS.get(r["theme"], [])
        groups_str = ", ".join(g.replace("_", " ").title() for g in groups)
        rows.append({
            "Theme": r["theme"].replace("_", " "),
            "Group": groups_str or "—",
            "1M": r["avg_1m"],
            "3M": r["avg_3m"],
            "6M": r["avg_6m"],
            "YTD": r["avg_ytd"],
            "% Up (1M)": r["pct_up_1m"],
            "Top Mover": r["top_mover_1m"] or "—",
            "Top Ret": r["top_mover_1m_ret"],
            "N": r["n_with_data"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("1M", ascending=False, na_position="last").reset_index(drop=True)

    # Style with conditional formatting
    def color_pct(v):
        if v is None or pd.isna(v):
            return f"color:{GREY}"
        if v > 10: return f"color:{GREEN};font-weight:700"
        if v > 0:  return f"color:{GREEN}"
        if v > -5: return f"color:{AMBER}"
        return f"color:{RED};font-weight:700"

    styled = (
        df.style
          .map(color_pct, subset=["1M", "3M", "6M", "YTD", "Top Ret"])
          .format({
              "1M": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
              "3M": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
              "6M": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
              "YTD": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
              "Top Ret": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
              "% Up (1M)": lambda x: f"{x:.0f}%" if pd.notna(x) else "—",
          })
    )

    st.dataframe(styled, use_container_width=True, height=720, hide_index=True)


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
            st.markdown(f"""
                <div style='background:{PANEL_BG};border:1px solid {GRID};padding:14px;height:100%'>
                    <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px;margin-bottom:10px'>
                        {title}
                    </div>
            """, unsafe_allow_html=True)

            for item in items[:5]:
                v = item[show_field]
                vc = GREEN if v > 0 else RED
                sub = ""
                if sub_field:
                    sv = item.get(sub_field)
                    if sv is not None:
                        sub = f"<span style='color:{COPPER};font-size:10px;margin-left:6px'>(3M: {sv:+.1f}%)</span>"
                st.markdown(f"""
                    <div style='font-family:Fira Code,monospace;padding:4px 0;border-bottom:1px solid {GRID}'>
                        <div style='color:{color};font-size:11px'>{item["theme"].replace("_", " ")}</div>
                        <div style='color:{vc};font-size:13px;font-weight:700;margin-top:2px'>
                            {v:+.2f}%{sub}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

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
            html_parts = [f"""
                <div style='background:{PANEL_BG};border:2px solid {color};padding:16px;margin-bottom:16px'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>
                        <div style='color:{AMBER};font-family:Orbitron,monospace;font-size:12px;font-weight:700;letter-spacing:3px'>
                            {glabel}
                        </div>
                        <div style='color:{color};font-family:Fira Code,monospace;font-size:11px;font-weight:700'>
                            ● {status}
                        </div>
                    </div>
            """]

            # Sort themes in this group by 1M return (descending)
            sorted_themes = sorted(
                [(t, by_name.get(t)) for t in themes_in_group if by_name.get(t)],
                key=lambda x: x[1]["avg_1m"] if x[1]["avg_1m"] is not None else -999,
                reverse=True,
            )

            for tname, rec in sorted_themes:
                v = rec["avg_1m"]
                vc = GREEN if (v or 0) > 0 else RED
                vstr = f"{v:+.2f}%" if v is not None else "—"
                html_parts.append(f"""
                    <div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid {GRID};font-family:Fira Code,monospace'>
                        <span style='color:{COPPER};font-size:11px'>{tname.replace("_", " ")}</span>
                        <span style='color:{vc};font-size:12px;font-weight:700'>{vstr}</span>
                    </div>
                """)

            html_parts.append("</div>")
            st.markdown("".join(html_parts), unsafe_allow_html=True)


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

    st.markdown(f"""
        <div style='background:{PANEL_BG};border:1px solid {GRID};padding:16px;margin-bottom:16px'>
            <div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:24px'>
                <div>
                    <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px'>THEME</div>
                    <div style='color:{AMBER};font-family:Orbitron,monospace;font-size:18px;font-weight:700'>
                        {selected.replace("_", " ").upper()}
                    </div>
                </div>
                <div>
                    <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px'>1M AVG</div>
                    <div style='color:{color};font-family:Orbitron,monospace;font-size:18px;font-weight:700'>
                        {fmt_pct(avg_1m) if avg_1m is not None else "—"}
                    </div>
                </div>
                <div>
                    <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px'>% UP</div>
                    <div style='color:{AMBER};font-family:Orbitron,monospace;font-size:18px;font-weight:700'>
                        {fmt_num(pct_up, 0) + "%" if pct_up is not None else "—"}
                    </div>
                </div>
                <div>
                    <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px'>TICKERS</div>
                    <div style='color:{AMBER};font-family:Orbitron,monospace;font-size:18px;font-weight:700'>
                        {record["n_with_data"]}/{record["n_tickers"]}
                    </div>
                </div>
                <div>
                    <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px'>MACRO GROUP</div>
                    <div style='color:{AMBER};font-family:Orbitron,monospace;font-size:14px;font-weight:700'>
                        {groups_str}
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

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

    styled = (df.style
              .map(color_pct, subset=["1M", "3M", "6M", "YTD"])
              .map(color_dist, subset=["% from 52W High"])
              .map(color_ma, subset=["> 50DMA", "> 200DMA"])
              .format({
                  "Last": "{:.2f}",
                  "1M": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                  "3M": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                  "6M": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                  "YTD": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                  "% from 52W High": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
              }))
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

    def color_pct(v):
        if v is None or pd.isna(v):
            return f"color:{GREY}"
        if v > 0: return f"color:{GREEN}"
        return f"color:{RED}"

    def color_themes(v):
        if v >= 5: return f"color:{GREEN};font-weight:700"
        if v >= 4: return f"color:{AMBER};font-weight:700"
        return f"color:{COPPER}"

    styled = (df.style
              .map(color_pct, subset=["1M", "3M"])
              .map(color_themes, subset=["Themes"])
              .format({
                  "Last": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                  "1M": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                  "3M": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
              }))

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

    st.markdown(f"""
        <div style='background:{PANEL_BG};border:1px solid {AMBER};padding:16px;margin-top:8px'>
            <div style='font-family:Fira Code,monospace'>
                <span style='color:{COPPER};font-size:11px;letter-spacing:2px'>TICKER:</span>
                <span style='color:{AMBER};font-family:Orbitron,monospace;font-size:18px;font-weight:700;margin-left:8px'>{query}</span>
            </div>
            <div style='font-family:Fira Code,monospace;margin-top:12px'>
                <span style='color:{COPPER};font-size:11px;letter-spacing:2px'>FOUND IN {len(matches)} THEMES:</span>
            </div>
            <div style='margin-top:8px;font-family:Fira Code,monospace;font-size:13px'>
                {"<br>".join(f"<span style='color:{AMBER}'>•</span> <span style='color:#cccccc'>{m.replace('_', ' ')}</span>" for m in matches)}
            </div>
            <div style='font-family:Fira Code,monospace;margin-top:14px'>
                <span style='color:{COPPER};font-size:11px;letter-spacing:2px'>MACRO GROUPS:</span>
                <span style='color:{AMBER};margin-left:8px;font-size:13px'>{groups_str}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN ENTRY
# =============================================================================

def render():
    st.markdown(f"""
        <h1 style='color:{AMBER};font-family:Orbitron,monospace;font-weight:900;letter-spacing:8px;border-bottom:2px solid {AMBER};padding-bottom:12px'>
            ◉ SPECULATIVE THEMES
        </h1>
    """, unsafe_allow_html=True)

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
