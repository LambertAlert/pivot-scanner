"""
macro_view.py — The Macro Command Center tab.

Renders all sections of the tactical macro framework:
  • Composite Regime Banner
  • Cross-Asset Narrative + Headline Breadth
  • Participation Panel (RSP/SPY family)
  • Rotation Map (leadership pair groups)
  • Style/Factor Box
  • Stress Tape
  • Regime History
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tactical_data_layer import (
    fetch_universe,
    compute_macro_metrics,
    all_macro_tickers,
    LEADERSHIP_PAIRS,
    NARRATIVE_STATES,
)

# =============================================================================
# COLOR PALETTE (war room aesthetic)
# =============================================================================
AMBER = "#FFA500"
COPPER = "#B87333"
OBSIDIAN = "#0a0a0a"
DEEP_AMBER = "#cc7a00"
GREEN = "#00ff88"
RED = "#ff4444"
GREY = "#666666"
PANEL_BG = "#141414"
GRID = "#2a2a2a"

STATE_COLORS = {
    "Confirmed": GREEN,
    "Fresh": "#88ddff",
    "Fading": AMBER,
    "False Start": COPPER,
    "Mixed": GREY,
    "Denied": RED,
    "Unknown": GREY,
}

NARRATIVE_COLORS = {
    1: GREEN,             # Goldilocks
    2: "#00cc88",         # Broad Risk-On
    3: "#88aaff",         # US Exceptionalism
    4: "#aacc88",         # Soft Landing Print
    5: "#ff8866",         # Flight to Safety
    6: RED,               # Hawkish Squeeze
    7: AMBER,             # Growth Scare
    8: "#cc4488",         # Supply Shock
    0: GREY,              # Unknown
}


# =============================================================================
# RENDERING HELPERS
# =============================================================================

def section_header(title, subtitle=None):
    """Render a war-room-styled section header."""
    sub = f"<div style='color:{COPPER};font-family:Fira Code,monospace;font-size:10px;letter-spacing:2px;margin-top:4px'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div style='border-left:3px solid {AMBER};padding:8px 16px;margin:24px 0 12px 0;'>
          <div style='color:{AMBER};font-family:Orbitron,monospace;font-size:14px;font-weight:700;letter-spacing:4px;'>
            {title}
          </div>
          {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_box(label, value, delta=None, delta_color="neutral", value_color=AMBER):
    """A single KPI cell with chamfered corners and amber data."""
    delta_html = ""
    if delta is not None:
        c = GREEN if delta_color == "good" else RED if delta_color == "bad" else GREY
        delta_html = f"<div style='color:{c};font-family:Fira Code,monospace;font-size:11px;margin-top:4px'>{delta}</div>"
    return f"""
        <div style='
            background:{PANEL_BG};
            border:1px solid {GRID};
            padding:14px 18px;
            clip-path:polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px));
            min-height:80px;
        '>
            <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px'>
                {label.upper()}
            </div>
            <div style='color:{value_color};font-family:Orbitron,monospace;font-size:22px;font-weight:700;margin-top:6px'>
                {value}
            </div>
            {delta_html}
        </div>
    """


def state_badge(state):
    color = STATE_COLORS.get(state, GREY)
    return f"""<span style='
        background:{color}22;
        color:{color};
        border:1px solid {color};
        padding:2px 8px;
        font-family:Fira Code,monospace;
        font-size:10px;
        letter-spacing:1px;
        border-radius:2px;
    '>{state.upper()}</span>"""


def fmt_pct(v, plus=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    s = f"+{v:.2f}%" if plus and v >= 0 else f"{v:.2f}%"
    return s


def fmt_num(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def spark_line(records, x_key="Date", y_key=None, color=AMBER, height=80, threshold=None):
    """Mini spark line for ratio charts."""
    if not records:
        return None
    df = pd.DataFrame(records)
    if y_key is None:
        # Heuristic — find the numeric column
        numeric_cols = [c for c in df.columns if c not in ("date", "Date") and pd.api.types.is_numeric_dtype(df[c])]
        y_key = numeric_cols[0] if numeric_cols else None
    if y_key is None or y_key not in df.columns:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.iloc[:, 0],
        y=df[y_key],
        mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy" if False else None,
        hoverinfo="skip",
    ))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=GREY, width=0.5, dash="dot"))

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=4, b=4),
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=PANEL_BG,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# =============================================================================
# SECTION RENDERERS
# =============================================================================

def render_regime_banner(metrics):
    """Top of tab — large composite regime label + 4 condition lights."""
    label = metrics.get("regime_label", "MIXED / CHOPPY")

    if label in ("HEALTHY RISK-ON", "GROWTH-LED RISK-ON", "CYCLICAL EXPANSION", "INFLATION REFLATION"):
        label_color = GREEN
    elif label in ("MEGA-CAP MIRAGE", "STEALTH RISK-OFF", "OVERBOUGHT DISTRIBUTION"):
        label_color = AMBER
    elif label == "CAPITULATION SETUP":
        label_color = "#88ddff"
    else:
        label_color = GREY

    g_spec = metrics["group_scores"].get("Speculative_Risk_On", {})
    g_cyc  = metrics["group_scores"].get("Cyclical_Expansion", {})
    g_com  = metrics["group_scores"].get("Commodity_Confirmation", {})
    g_idi  = metrics["group_scores"].get("Idiosyncratic", {})

    def light_html(label_text, score_dict):
        c = score_dict.get("confirmed", 0)
        t = score_dict.get("total", 0)
        if t == 0:
            color = GREY
        elif c / t >= 0.66:
            color = GREEN
        elif c / t >= 0.33:
            color = AMBER
        else:
            color = RED
        return (
            f"<div style='display:inline-block;margin:0 16px;text-align:center'>"
            f"<div style='font-family:Fira Code,monospace;font-size:9px;"
            f"color:{COPPER};letter-spacing:1px'>{label_text}</div>"
            f"<div style='color:{color};font-family:Orbitron,monospace;"
            f"font-size:18px;font-weight:700;margin-top:4px'>&#9679; {c}/{t}</div>"
            f"</div>"
        )

    as_of     = metrics.get("as_of")
    as_of_str = as_of.strftime("%Y-%m-%d") if as_of is not None else "—"

    lights = (
        light_html("SPECULATIVE", g_spec) +
        light_html("CYCLICAL",    g_cyc)  +
        light_html("COMMODITY",   g_com)  +
        light_html("IDIOSYNCRATIC", g_idi)
    )

    html = (
        f"<div style='background:{PANEL_BG};border:2px solid {AMBER};"
        f"padding:24px;margin-bottom:24px;'>"
        f"<div style='color:{COPPER};font-family:Fira Code,monospace;"
        f"font-size:10px;letter-spacing:3px;margin-bottom:8px'>"
        f"&#9685; COMPOSITE ROTATION REGIME &mdash; AS OF {as_of_str}</div>"
        f"<div style='color:{label_color};font-family:Orbitron,monospace;"
        f"font-size:36px;font-weight:900;letter-spacing:6px;margin-bottom:18px'>"
        f"{label}</div>"
        f"<div style='border-top:1px solid {GRID};padding-top:16px;text-align:center'>"
        f"{lights}</div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_narrative_and_breadth(metrics):
    """Side by side — Cross-Asset Narrative (left) + Headline Breadth (right)."""
    nar = metrics.get("narrative")

    col1, col2 = st.columns(2)

    # ── Narrative panel ────────────────────────────────────────────────
    with col1:
        section_header("CROSS-ASSET NARRATIVE", "SPX × DXY × RATES — DAILY DIRECTIONS")

        if not nar:
            st.markdown(f"<div style='color:{GREY};font-family:Fira Code,monospace'>Narrative data unavailable</div>", unsafe_allow_html=True)
        else:
            today_color = NARRATIVE_COLORS.get(nar["today_state_id"], GREY)
            dom_color = NARRATIVE_COLORS.get(nar["dominant_id"], GREY)

            spx_arrow = "↑" if nar["spx_dir"] > 0 else "↓"
            dxy_arrow = "↑" if nar["dxy_dir"] > 0 else "↓"
            rates_arrow = "↑" if nar["rates_dir"] > 0 else "↓"
            spx_color = GREEN if nar["spx_dir"] > 0 else RED
            dxy_color = GREEN if nar["dxy_dir"] > 0 else RED
            rates_color = GREEN if nar["rates_dir"] > 0 else RED

            st.markdown(f"""
                <div style='background:{PANEL_BG};border:1px solid {GRID};padding:18px;margin-bottom:12px'>
                    <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px;margin-bottom:6px'>
                        TODAY'S NARRATIVE
                    </div>
                    <div style='color:{today_color};font-family:Orbitron,monospace;font-size:22px;font-weight:700;letter-spacing:2px;margin-bottom:10px'>
                        STATE {nar["today_state_id"]} — {nar["today_state_name"].upper()}
                    </div>
                    <div style='font-family:Fira Code,monospace;font-size:13px;margin-bottom:8px'>
                        <span style='color:{spx_color}'>SPX {spx_arrow}</span>
                        &nbsp;•&nbsp;
                        <span style='color:{dxy_color}'>DXY {dxy_arrow}</span>
                        &nbsp;•&nbsp;
                        <span style='color:{rates_color}'>RATES {rates_arrow}</span>
                    </div>
                    <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:11px'>
                        Sector tilt: {nar["today_sector_tilt"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style='background:{PANEL_BG};border:1px solid {GRID};padding:18px'>
                    <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px;margin-bottom:6px'>
                        DOMINANT — LAST 10 DAYS
                    </div>
                    <div style='color:{dom_color};font-family:Orbitron,monospace;font-size:18px;font-weight:700;margin-bottom:6px'>
                        STATE {nar["dominant_id"]} — {nar["dominant_name"].upper()}
                    </div>
                    <div style='color:{AMBER};font-family:Fira Code,monospace;font-size:12px'>
                        {nar["dominant_pct"]}% frequency over 10 trading days
                    </div>
                    <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:11px;margin-top:8px'>
                        Tilt: {nar["dominant_tilt"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ── Headline Breadth panel ────────────────────────────────────────
    with col2:
        section_header("HEADLINE BREADTH", "BREADTH × VOL × SPY TREND")

        h = metrics.get("headline", {})

        c1, c2 = st.columns(2)

        with c1:
            pct20 = h.get("pct_sectors_above_20")
            color = GREEN if (pct20 or 0) > 50 else AMBER if (pct20 or 0) > 30 else RED
            st.markdown(kpi_box(
                "% SECTORS > 20DMA",
                fmt_num(pct20, 1) + "%" if pct20 is not None else "—",
                value_color=color,
            ), unsafe_allow_html=True)

            vix = h.get("vix_level")
            vix_color = GREEN if (vix or 99) < 16 else AMBER if (vix or 99) < 25 else RED
            st.markdown(kpi_box(
                "VIX",
                fmt_num(vix, 2) if vix is not None else "—",
                value_color=vix_color,
            ), unsafe_allow_html=True)

        with c2:
            pct50 = h.get("pct_sectors_above_50")
            color = GREEN if (pct50 or 0) > 50 else AMBER if (pct50 or 0) > 30 else RED
            st.markdown(kpi_box(
                "% SECTORS > 50DMA",
                fmt_num(pct50, 1) + "%" if pct50 is not None else "—",
                value_color=color,
            ), unsafe_allow_html=True)

            ratio = h.get("vix_term_ratio")
            term_color = GREEN if (ratio or 99) < 1.0 else RED
            term_label = "CONTANGO" if (ratio or 99) < 1.0 else "BACKWARDATION"
            st.markdown(kpi_box(
                "VIX TERM (1M/3M)",
                fmt_num(ratio, 3) if ratio is not None else "—",
                delta=term_label,
                delta_color="good" if term_color == GREEN else "bad",
                value_color=term_color,
            ), unsafe_allow_html=True)

        # SPY trend
        spy_above_50 = h.get("spy_above_50", False)
        spy_above_200 = h.get("spy_above_200", False)
        spy_color = GREEN if (spy_above_50 and spy_above_200) else AMBER if (spy_above_50 or spy_above_200) else RED
        spy_status = "BULL TREND" if (spy_above_50 and spy_above_200) else \
                     "MIXED" if (spy_above_50 or spy_above_200) else "BEAR TREND"
        st.markdown(f"""
            <div style='background:{PANEL_BG};border:1px solid {GRID};padding:14px;margin-top:12px'>
                <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px'>
                    SPY TREND STATUS
                </div>
                <div style='color:{spy_color};font-family:Orbitron,monospace;font-size:16px;font-weight:700;margin-top:4px'>
                    {spy_status}
                </div>
                <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:11px;margin-top:4px'>
                    > 50DMA: {"✓" if spy_above_50 else "✗"}    &nbsp; > 200DMA: {"✓" if spy_above_200 else "✗"}
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_participation(metrics):
    section_header("PARTICIPATION", "EQUAL-WEIGHT vs CAP-WEIGHT — IS THE RALLY BROAD?")

    p = metrics.get("participation", {})

    cols = st.columns(4)
    pairs = [
        ("RSP / SPY", "rsp_spy", "Equal-weight S&P"),
        ("QQQE / QQQ", "qqqe_qqq", "Equal-weight Nasdaq"),
        ("IWM / SPY", "iwm_spy", "Small caps"),
        ("MDY / SPY", "mdy_spy", "Mid caps"),
    ]
    for col, (label, key, sub) in zip(cols, pairs):
        with col:
            d = p.get(key)
            if not d:
                st.markdown(kpi_box(label, "—", value_color=GREY), unsafe_allow_html=True)
                continue
            chg = d["5d_change_pct"]
            color = GREEN if (chg or 0) > 0 else RED
            st.markdown(kpi_box(
                label,
                fmt_num(d["current"], 4),
                delta=f"5d: {fmt_pct(chg)}",
                delta_color="good" if (chg or 0) > 0 else "bad",
                value_color=color,
            ), unsafe_allow_html=True)
            # spark line
            fig = spark_line(d["series_60d"], color=color)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"part_{key}")


def render_rotation_map(metrics):
    section_header("ROTATION MAP — LEADERSHIP PAIR GROUPS",
                   "10 PAIRS × 5/10/20 DAY DIFFERENTIALS")

    group_meta = [
        ("Speculative_Risk_On",      "SPECULATIVE RISK-ON",
         "Aggressive money leading defensive money within sectors"),
        ("Cyclical_Expansion",       "CYCLICAL EXPANSION",
         "Real economy expanding — services, builders, transports"),
        ("Commodity_Confirmation",   "COMMODITY CONFIRMATION",
         "Inflation/commodity moves backed by miner conviction"),
        ("Idiosyncratic",            "IDIOSYNCRATIC",
         "Standalone narratives — defense, cyber"),
    ]

    for group_key, group_label, group_desc in group_meta:
        gdata = metrics["group_scores"].get(group_key, {})
        confirmed = gdata.get("confirmed", 0)
        total = gdata.get("total", 0)

        # Group header
        if total == 0:
            score_color = GREY
            verdict = "NO PAIRS"
        else:
            score_color = GREEN if confirmed / total >= 0.66 else AMBER if confirmed / total >= 0.33 else RED
            verdict = "CONFIRMED" if confirmed / total >= 0.66 else "MIXED" if confirmed / total >= 0.33 else "DENIED"

        with st.expander(f"  {group_label}    ●  {confirmed}/{total}  —  {verdict}", expanded=(confirmed >= 1)):
            st.markdown(f"<div style='color:{COPPER};font-family:Fira Code,monospace;font-size:11px;margin-bottom:12px'>{group_desc}</div>",
                        unsafe_allow_html=True)

            for p in gdata.get("pairs", []):
                col1, col2, col3 = st.columns([2, 3, 2])

                with col1:
                    st.markdown(f"""
                        <div style='padding:8px 0'>
                            <div style='color:{AMBER};font-family:Orbitron,monospace;font-size:14px;font-weight:700'>
                                {p["num"]} / {p["den"]}
                            </div>
                            <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:11px;margin-top:2px'>
                                {p["label"]}
                            </div>
                            <div style='margin-top:6px'>
                                {state_badge(p["state"])}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with col2:
                    d5, d10, d20 = p["diff_5d"], p["diff_10d"], p["diff_20d"]
                    def color_for(v):
                        if v is None or np.isnan(v): return GREY
                        if v > 0.005: return GREEN
                        if v < -0.005: return RED
                        return AMBER

                    def fmt_diff(v):
                        if v is None or np.isnan(v): return "—"
                        return f"{v*100:+.2f}%"

                    st.markdown(f"""
                        <div style='display:flex;justify-content:space-around;padding:14px 0;font-family:Fira Code,monospace'>
                            <div style='text-align:center'>
                                <div style='color:{COPPER};font-size:9px;letter-spacing:1px'>5D</div>
                                <div style='color:{color_for(d5)};font-size:14px;font-weight:700;margin-top:2px'>
                                    {fmt_diff(d5)}
                                </div>
                            </div>
                            <div style='text-align:center'>
                                <div style='color:{COPPER};font-size:9px;letter-spacing:1px'>10D</div>
                                <div style='color:{color_for(d10)};font-size:14px;font-weight:700;margin-top:2px'>
                                    {fmt_diff(d10)}
                                </div>
                            </div>
                            <div style='text-align:center'>
                                <div style='color:{COPPER};font-size:9px;letter-spacing:1px'>20D</div>
                                <div style='color:{color_for(d20)};font-size:14px;font-weight:700;margin-top:2px'>
                                    {fmt_diff(d20)}
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                with col3:
                    if p["ratio_series_60d"]:
                        fig = spark_line(p["ratio_series_60d"], color=STATE_COLORS.get(p["state"], AMBER), height=60)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True, key=f"pair_{p['num']}_{p['den']}")


def render_sector_rotation(metrics):
    section_header("SECTOR ROTATION", "DEFENSIVE vs OFFENSIVE — TOP/BOTTOM LEADERS")

    sr = metrics.get("sector_rotation", {})

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        diff = sr.get("def_off_diff_5d")
        if diff is None or np.isnan(diff):
            color = GREY; label_value = "—"
        else:
            color = RED if diff > 1.0 else AMBER if diff > 0 else GREEN
            label_value = f"{diff:+.2f}%"
        verdict = "DEF LEADING" if (diff or -99) > 1.0 else "MIXED" if (diff or -99) > -1.0 else "OFF LEADING"
        st.markdown(kpi_box(
            "DEF − OFF (5D)",
            label_value,
            delta=verdict,
            delta_color="bad" if color == RED else "good" if color == GREEN else "neutral",
            value_color=color,
        ), unsafe_allow_html=True)

    with c2:
        d_5d = sr.get("defensive_5d")
        o_5d = sr.get("offensive_5d")
        st.markdown(kpi_box(
            "DEFENSIVE 5D",
            fmt_pct(d_5d) if d_5d is not None else "—",
            value_color=GREEN if (d_5d or 0) > 0 else RED,
        ), unsafe_allow_html=True)
        st.markdown(kpi_box(
            "OFFENSIVE 5D",
            fmt_pct(o_5d) if o_5d is not None else "—",
            value_color=GREEN if (o_5d or 0) > 0 else RED,
        ), unsafe_allow_html=True)

    with c3:
        top = sr.get("top_3_5d", [])
        bot = sr.get("bottom_3_5d", [])

        top_html = "<br>".join([
            f"<span style='color:{AMBER};font-family:Orbitron,monospace'>{t}</span> "
            f"<span style='color:{GREEN};font-family:Fira Code,monospace'>{r:+.2f}%</span>"
            for t, r in top
        ])
        bot_html = "<br>".join([
            f"<span style='color:{AMBER};font-family:Orbitron,monospace'>{t}</span> "
            f"<span style='color:{RED};font-family:Fira Code,monospace'>{r:+.2f}%</span>"
            for t, r in bot
        ])

        st.markdown(f"""
            <div style='background:{PANEL_BG};border:1px solid {GRID};padding:14px;height:100%'>
                <div style='display:flex'>
                    <div style='flex:1;padding-right:12px;border-right:1px solid {GRID}'>
                        <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px;margin-bottom:8px'>
                            ▲ TOP 3 — 5D
                        </div>
                        <div style='font-size:13px;line-height:1.8'>{top_html or "—"}</div>
                    </div>
                    <div style='flex:1;padding-left:12px'>
                        <div style='color:{COPPER};font-family:Fira Code,monospace;font-size:9px;letter-spacing:2px;margin-bottom:8px'>
                            ▼ BOTTOM 3 — 5D
                        </div>
                        <div style='font-size:13px;line-height:1.8'>{bot_html or "—"}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_style_factor(metrics):
    section_header("STYLE / FACTOR BOX", "GROWTH vs VALUE ACROSS LARGE / MID / SMALL")

    s = metrics.get("style", {})

    cols = st.columns(4)
    items = [
        ("LARGE G/V", "large_gv", "IVW / IVE"),
        ("MID G/V", "mid_gv", "IJK / IJJ"),
        ("SMALL G/V", "small_gv", "IJT / IJS"),
    ]

    for col, (label, key, sub) in zip(cols[:3], items):
        with col:
            d = s.get(key)
            if not d:
                st.markdown(kpi_box(label, "—", value_color=GREY), unsafe_allow_html=True)
                continue
            chg = d["5d_change_pct"]
            color = GREEN if (chg or 0) > 0 else RED
            st.markdown(kpi_box(
                f"{label} — {sub}",
                fmt_num(d["current"], 4),
                delta=f"5d: {fmt_pct(chg)}",
                delta_color="good" if (chg or 0) > 0 else "bad",
                value_color=color,
            ), unsafe_allow_html=True)
            fig = spark_line(d["series_60d"], color=color)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"style_{key}")

    with cols[3]:
        gp = s.get("growth_premium")
        gp_color = GREEN if (gp or 0) > 1.0 else RED
        st.markdown(kpi_box(
            "GROWTH PREMIUM",
            fmt_num(gp, 4) if gp is not None else "—",
            delta="GROWTH-LED" if (gp or 0) > 1.0 else "VALUE-LED",
            delta_color="good" if (gp or 0) > 1.0 else "neutral",
            value_color=gp_color,
        ), unsafe_allow_html=True)
        st.markdown(f"<div style='color:{COPPER};font-family:Fira Code,monospace;font-size:10px;padding:8px 0'>Avg of L/M/S G/V ratios. >1.0 = growth-led market.</div>",
                    unsafe_allow_html=True)


def render_stress_tape(metrics):
    section_header("STRESS TAPE", "VOL × CREDIT × DURATION × SAFE-HAVEN FLOWS")

    stress = metrics.get("stress", {})

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        ratio = stress.get("hyg_lqd_ratio")
        chg = stress.get("hyg_lqd_5d_pct")
        color = GREEN if (chg or 0) > 0 else RED
        st.markdown(kpi_box(
            "HYG / LQD",
            fmt_num(ratio, 4) if ratio is not None else "—",
            delta=f"5d: {fmt_pct(chg)}",
            delta_color="good" if (chg or 0) > 0 else "bad",
            value_color=color,
        ), unsafe_allow_html=True)
        st.markdown(f"<div style='color:{COPPER};font-family:Fira Code,monospace;font-size:10px;padding:6px 0'>Credit appetite: rising = risk-on</div>", unsafe_allow_html=True)

    with c2:
        rv = stress.get("tlt_realized_vol_30d")
        st.markdown(kpi_box(
            "TLT REALIZED VOL 30D",
            fmt_num(rv, 1) + "%" if rv is not None else "—",
            delta="MOVE proxy",
            value_color=AMBER,
        ), unsafe_allow_html=True)
        st.markdown(f"<div style='color:{COPPER};font-family:Fira Code,monospace;font-size:10px;padding:6px 0'>Bond uncertainty / rates risk</div>", unsafe_allow_html=True)

    with c3:
        gld_5d = stress.get("gld_5d")
        gld_above = stress.get("gld_above_50", False)
        st.markdown(kpi_box(
            "GLD 5D",
            fmt_pct(gld_5d) if gld_5d is not None else "—",
            delta="> 50DMA" if gld_above else "< 50DMA",
            delta_color="bad" if gld_above else "neutral",  # gold rallying often signals risk-off
            value_color=GREEN if (gld_5d or 0) > 0 else RED,
        ), unsafe_allow_html=True)
        st.markdown(f"<div style='color:{COPPER};font-family:Fira Code,monospace;font-size:10px;padding:6px 0'>Safe-haven flow probe</div>", unsafe_allow_html=True)

    with c4:
        hyg_5d = stress.get("hyg_5d")
        hyg_above = stress.get("hyg_above_50", False)
        st.markdown(kpi_box(
            "HYG 5D",
            fmt_pct(hyg_5d) if hyg_5d is not None else "—",
            delta="> 50DMA" if hyg_above else "< 50DMA",
            delta_color="good" if hyg_above else "bad",
            value_color=GREEN if (hyg_5d or 0) > 0 else RED,
        ), unsafe_allow_html=True)
        st.markdown(f"<div style='color:{COPPER};font-family:Fira Code,monospace;font-size:10px;padding:6px 0'>Risk appetite: above 50 = healthy</div>", unsafe_allow_html=True)


def render_regime_history(metrics):
    section_header("NARRATIVE HISTORY", "60-DAY DAILY NARRATIVE STATE")

    nar = metrics.get("narrative")
    if not nar or not nar.get("history_60d"):
        st.markdown(f"<div style='color:{GREY};font-family:Fira Code,monospace'>History unavailable</div>", unsafe_allow_html=True)
        return

    df = pd.DataFrame(nar["history_60d"])
    df["color"] = df["state_id"].map(NARRATIVE_COLORS).fillna(GREY)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=[1] * len(df),
        mode="markers",
        marker=dict(
            color=df["color"],
            size=18,
            symbol="square",
            line=dict(width=0),
        ),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>State %{text}<extra></extra>",
        text=df["state_id"],
    ))
    fig.update_layout(
        height=110,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor=OBSIDIAN,
        paper_bgcolor=OBSIDIAN,
        showlegend=False,
        xaxis=dict(showgrid=False, color=COPPER, tickfont=dict(family="Fira Code", size=9)),
        yaxis=dict(visible=False, range=[0.5, 1.5]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    legend_html = "<div style='display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;font-family:Fira Code,monospace;font-size:10px'>"
    for sid, meta in sorted(NARRATIVE_STATES.items(), key=lambda x: x[1][0]):
        sname = meta[1]
        c = NARRATIVE_COLORS.get(meta[0], GREY)
        legend_html += f"<div><span style='display:inline-block;width:10px;height:10px;background:{c};margin-right:6px'></span><span style='color:{COPPER}'>{meta[0]}: {sname}</span></div>"
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)


# =============================================================================
# MAIN RENDER ENTRY POINT
# =============================================================================

def render():
    """Main entry — fetch data, compute metrics, render all sections."""
    st.markdown(f"""
        <h1 style='color:{AMBER};font-family:Orbitron,monospace;font-weight:900;letter-spacing:8px;border-bottom:2px solid {AMBER};padding-bottom:12px'>
            ◉ MACRO COMMAND CENTER
        </h1>
    """, unsafe_allow_html=True)

    with st.spinner("Pulling cross-asset universe..."):
        prices, failed = fetch_universe(tuple(all_macro_tickers()))

    if prices.empty:
        st.error("⚠ Universe fetch returned no data. Check yfinance connectivity.")
        return

    if failed:
        st.warning(f"⚠ {len(failed)} ticker(s) failed: {', '.join(failed)}. Sections relying on these may show '—'.")

    metrics = compute_macro_metrics(prices)

    render_regime_banner(metrics)
    render_narrative_and_breadth(metrics)
    render_participation(metrics)
    render_rotation_map(metrics)
    render_sector_rotation(metrics)
    render_style_factor(metrics)
    render_stress_tape(metrics)
    render_regime_history(metrics)
