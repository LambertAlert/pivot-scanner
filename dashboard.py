"""
Pivot Scanner — Unified Dashboard
====================================
Tabs:
  1. ◈ MACRO VIEW      — Regime, VIX, macro indicators, sector rotation
  2. ◈ SECTOR THEMES   — ETF thematic RS rankings + momentum engine v3.0
  3. ◈ ACTIVE TRIGGERS — Today's 30/65-min pivot alerts
  4. ◈ DAILY WATCHLIST — Conviction-tiered setups + near-resistance filter
  5. ◈ WEEKLY SCREEN   — Stage + BBUW from weekly screener
  6. ◈ TRIGGER HISTORY — SQLite-backed history

Aesthetic: Amber War Room — Orbitron + Fira Code / Obsidian / Amber accents
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_layer import (
    get_latest_weekly_watchlist,
    get_latest_daily_watchlist,
    get_today_triggers,
    get_trigger_history,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PIVOT SCANNER — WAR ROOM",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Fira+Code:wght@300;400;500&display=swap');
:root{--bg:#09090e;--panel:#0f0f17;--border:rgba(245,166,35,0.22);--border2:rgba(245,166,35,0.08);--text:#ede8d9;--muted:#7a7060;--accent:#f5a623;--accent2:#c97d1e;--green:#4caf7d;--red:#e05c5c;--blue:#5b8dee;--mono:'Fira Code',monospace;--display:'Orbitron',sans-serif;}
html,body,[class*="css"]{background-color:var(--bg)!important;color:var(--text)!important;font-family:var(--mono)!important;}
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:radial-gradient(ellipse at 50% 0%,rgba(245,166,35,.07) 0%,transparent 60%),repeating-linear-gradient(0deg,transparent,transparent 24px,rgba(245,166,35,.025) 24px,rgba(245,166,35,.025) 25px),repeating-linear-gradient(90deg,transparent,transparent 24px,rgba(245,166,35,.025) 24px,rgba(245,166,35,.025) 25px),var(--bg)!important;}
.block-container{padding:0 1.5rem 2rem!important;max-width:1800px!important;}
.masthead{background:linear-gradient(90deg,#0f0f17 0%,#161620 50%,#0f0f17 100%);border-bottom:1px solid var(--border);padding:0;margin:0 -1.5rem 1.4rem -1.5rem;position:relative;}
.masthead::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),var(--accent2),var(--accent),transparent);}
.masthead-inner{display:flex;align-items:center;justify-content:space-between;padding:.9rem 2rem;}
.masthead-title{font-family:var(--display);font-size:1.4rem;font-weight:900;letter-spacing:.18em;color:var(--accent);text-transform:uppercase;text-shadow:0 0 24px rgba(245,166,35,.35);}
.masthead-title span{color:var(--accent2);}
.masthead-meta{font-family:var(--display);font-size:.52rem;letter-spacing:.2em;color:var(--muted);text-transform:uppercase;display:flex;gap:1.8rem;align-items:center;}
.live-dot{color:var(--green);display:flex;align-items:center;gap:.4rem;}
.live-dot::before{content:'●';animation:pulse-dot 2s ease-in-out infinite;}
@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:.3}}
.kpi-strip{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:1.2rem;}
.kpi-cell{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--accent2);padding:.7rem 1rem;clip-path:polygon(0 0,calc(100% - 8px) 0,100% 8px,100% 100%,0 100%);}
.kpi-cell.green{border-left-color:var(--green);}.kpi-cell.red{border-left-color:var(--red);}.kpi-cell.blue{border-left-color:var(--blue);}.kpi-cell.amber{border-left-color:var(--accent);}
.kpi-lbl{font-family:var(--display);font-size:.5rem;letter-spacing:.18em;color:var(--muted);text-transform:uppercase;margin-bottom:.3rem;}
.kpi-val{font-family:var(--display);font-size:1.3rem;font-weight:700;color:var(--text);line-height:1;}
.kpi-val.amber{color:var(--accent);}.kpi-val.green{color:var(--green);}.kpi-val.red{color:var(--red);}
.kpi-sub{font-family:var(--mono);font-size:.6rem;color:var(--muted);margin-top:.2rem;}
.sec-bar{display:flex;align-items:center;gap:.8rem;margin:1.2rem 0 .8rem 0;}
.sec-bar-line{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}
.sec-bar-label{font-family:var(--display);font-size:.6rem;letter-spacing:.2em;color:var(--accent);text-transform:uppercase;white-space:nowrap;}
.trigger-card{background:linear-gradient(135deg,#0f0f17 0%,#09090e 100%);border:1px solid var(--border);border-left:3px solid var(--green);padding:1.1rem;margin-bottom:.8rem;border-radius:4px;}
.trigger-card-bear{border-left-color:var(--red)!important;}
.trigger-card-high{border-left-color:var(--accent)!important;box-shadow:0 0 12px rgba(245,166,35,.15);}
.t-ticker{font-family:var(--display);font-size:1.4rem;font-weight:700;color:var(--text);}
.t-meta{font-family:var(--display);font-size:.5rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin-top:.2rem;}
.t-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-top:.8rem;}
.t-field-lbl{font-family:var(--display);font-size:.47rem;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);}
.t-field-val{font-size:1rem;font-weight:600;color:var(--text);font-family:var(--mono);}
.t-field-val.red{color:var(--red);}.t-field-val.blue{color:var(--blue);}
.badge-high{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#000;padding:.2rem .6rem;font-family:var(--display);font-size:.5rem;font-weight:700;letter-spacing:.15em;}
.badge-med{background:rgba(245,166,35,.15);color:var(--accent);border:1px solid var(--border);padding:.2rem .6rem;font-family:var(--display);font-size:.5rem;font-weight:700;letter-spacing:.15em;}
.badge-low{background:rgba(255,255,255,.05);color:var(--muted);border:1px solid rgba(255,255,255,.08);padding:.2rem .6rem;font-family:var(--display);font-size:.5rem;letter-spacing:.15em;}
.badge-bull{color:var(--green);font-family:var(--display);font-size:.55rem;font-weight:700;}
.badge-bear{color:var(--red);font-family:var(--display);font-size:.55rem;font-weight:700;}
.stTabs [data-baseweb="tab-list"]{gap:0;background:transparent;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-family:var(--display);font-size:.62rem;text-transform:uppercase;letter-spacing:.12em;background:transparent;color:var(--muted);border-bottom:2px solid transparent;padding:.7rem 1.4rem;}
.stTabs [aria-selected="true"]{color:var(--accent);border-bottom-color:var(--accent);background:transparent;}
[data-testid="stMetricValue"]{font-family:var(--display)!important;font-weight:700!important;color:var(--accent)!important;}
[data-testid="stMetricLabel"]{font-family:var(--display)!important;font-size:.55rem!important;letter-spacing:.12em!important;text-transform:uppercase!important;color:var(--muted)!important;}
.stSelectbox label,.stMultiSelect label,.stTextInput label,.stNumberInput label{font-family:var(--display)!important;font-size:.55rem!important;letter-spacing:.15em!important;text-transform:uppercase!important;color:var(--muted)!important;}
.dash-footer{text-align:center;font-family:var(--display);color:var(--muted);font-size:.47rem;letter-spacing:.15em;text-transform:uppercase;border-top:1px solid var(--border2);padding-top:1rem;margin-top:2rem;}
</style>
""", unsafe_allow_html=True)

# ── Plotly defaults ───────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="#09090e", plot_bgcolor="#0f0f17",
    font=dict(family="Fira Code, monospace", color="#ede8d9", size=11),
    xaxis=dict(gridcolor="rgba(245,166,35,0.08)", linecolor="rgba(245,166,35,0.22)"),
    yaxis=dict(gridcolor="rgba(245,166,35,0.08)", linecolor="rgba(245,166,35,0.22)"),
    colorway=["#f5a623","#4caf7d","#5b8dee","#e05c5c","#c97d1e","#9b7ed1"],
    title_font=dict(family="Orbitron, sans-serif", size=11, color="#f5a623"),
    margin=dict(l=0,r=0,t=35,b=0),
)

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_macro():
    p = "data/macro_data.json"
    if not os.path.exists(p): return {}
    with open(p) as f: return json.load(f)

@st.cache_data(ttl=300)
def load_theme():
    p = "data/theme_data.json"
    if not os.path.exists(p): return {}
    with open(p) as f: return json.load(f)

@st.cache_data(ttl=120)
def load_radar():
    p = "data/radar_data.json"
    if not os.path.exists(p): return {}
    with open(p) as f: return json.load(f)

@st.cache_data(ttl=60)
def _triggers(): return get_today_triggers()

@st.cache_data(ttl=60)
def _daily(): return get_latest_daily_watchlist()

@st.cache_data(ttl=300)
def _weekly(): return get_latest_weekly_watchlist()


macro          = load_macro()
theme_data     = load_theme()
radar_data     = load_radar()
today_triggers = _triggers()
daily_data     = _daily()
weekly_data    = _weekly()
cur            = macro.get("current", {})
now            = datetime.now()

high_today = sum(1 for t in today_triggers if t.get("conviction")=="HIGH")
med_today  = sum(1 for t in today_triggers if t.get("conviction")=="MED")

# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='masthead'><div class='masthead-inner'>
  <div class='masthead-title'>◈ PIVOT SCANNER <span>WAR ROOM</span></div>
  <div class='masthead-meta'>
    <span class='live-dot'>LIVE</span>
    <span>REGIME: {str(cur.get('combined_regime','—'))[:28]}</span>
    <span>VIX {cur.get('VIX','—')}</span>
    <span>SPY ${cur.get('SPY','—')}</span>
    <span>{now.strftime('%Y-%m-%d %H:%M')}</span>
  </div>
</div></div>
""", unsafe_allow_html=True)

# ── Global KPIs ───────────────────────────────────────────────────────────────
spy_chg = cur.get("SPY_ret_1d", 0) or 0
qqq_chg = cur.get("QQQ_ret_1d", 0) or 0

st.markdown(f"""
<div class='kpi-strip'>
  <div class='kpi-cell {"green" if spy_chg>=0 else "red"}'>
    <div class='kpi-lbl'>SPY</div>
    <div class='kpi-val {"green" if spy_chg>=0 else "red"}'>${cur.get("SPY","—")}</div>
    <div class='kpi-sub'>{"+" if spy_chg>=0 else ""}{spy_chg:.2f}% 1D</div>
  </div>
  <div class='kpi-cell {"green" if qqq_chg>=0 else "red"}'>
    <div class='kpi-lbl'>QQQ</div>
    <div class='kpi-val {"green" if qqq_chg>=0 else "red"}'>${cur.get("QQQ","—")}</div>
    <div class='kpi-sub'>{"+" if qqq_chg>=0 else ""}{qqq_chg:.2f}% 1D</div>
  </div>
  <div class='kpi-cell amber'>
    <div class='kpi-lbl'>VIX</div>
    <div class='kpi-val amber'>{cur.get("VIX","—")}</div>
    <div class='kpi-sub'>{str(cur.get("vix_regime","—"))[:22]}</div>
  </div>
  <div class='kpi-cell amber'>
    <div class='kpi-lbl'>Triggers Today</div>
    <div class='kpi-val amber'>{len(today_triggers)}</div>
    <div class='kpi-sub'>🔥 {high_today} HIGH  ⭐ {med_today} MED</div>
  </div>
  <div class='kpi-cell blue'>
    <div class='kpi-lbl'>Daily Watchlist</div>
    <div class='kpi-val'>{daily_data.get("count",len(daily_data.get("entries",[])))}</div>
    <div class='kpi-sub'>setups qualified</div>
  </div>
  <div class='kpi-cell'>
    <div class='kpi-lbl'>Weekly Screen</div>
    <div class='kpi-val'>{weekly_data.get("count",len(weekly_data.get("entries",[])))}</div>
    <div class='kpi-sub'>stage 1/2 names</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab_radar,tab5,tab6 = st.tabs([
    "◈ MACRO VIEW","◈ SECTOR THEMES","◈ ACTIVE TRIGGERS",
    "◈ DAILY WATCHLIST","◈ SETUPS RADAR","◈ WEEKLY SCREEN","◈ TRIGGER HISTORY",
])

# ─── TAB 1: MACRO VIEW ────────────────────────────────────────────────────────
with tab1:
    if not macro:
        st.info("No macro data. Run macro_prep.py.")
    else:
        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Macro Indicators</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
        mc1,mc2,mc3,mc4,mc5 = st.columns(5)
        with mc1: st.metric("10Y Yield",f"{cur.get('US10Y','—')}%")
        with mc2: st.metric("DXY",str(cur.get("DXY","—")))
        with mc3: st.metric("Gold",f"${cur.get('GOLD','—')}")
        with mc4: st.metric("Oil (WTI)",f"${cur.get('OIL','—')}")
        with mc5: st.metric("Yield Curve 10Y-2Y",f"{cur.get('Yield_Curve','—')}%")

        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Current Regime</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
        rc1,rc2 = st.columns([1,2])
        with rc1:
            st.markdown(f"""
            <div style='background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--accent);padding:1.2rem;'>
              <div style='font-family:var(--display);font-size:.52rem;color:var(--muted);letter-spacing:.15em;text-transform:uppercase;'>Price Regime</div>
              <div style='font-family:var(--display);font-size:1.1rem;font-weight:700;color:var(--accent);margin:.4rem 0;'>{cur.get('price_regime','—')}</div>
              <div style='font-family:var(--display);font-size:.52rem;color:var(--muted);'>VIX: {str(cur.get('vix_regime','—'))[:30]}</div>
              <div style='font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-top:.6rem;'>SPY 20d: {"+" if cur.get("SPY_20d",0)>=0 else ""}{cur.get("SPY_20d",0):.1f}% &nbsp;|&nbsp; Vol21: {cur.get("SPY_vol_21",0):.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with rc2:
            sector_snap = macro.get("sector_snapshot",[])
            if sector_snap:
                sdf = pd.DataFrame(sector_snap).sort_values("vs_SPY_20d",ascending=False)
                fig = px.bar(sdf, x="ticker", y="vs_SPY_20d", color="vs_SPY_20d",
                             color_continuous_scale=["#e05c5c","#09090e","#4caf7d"],
                             color_continuous_midpoint=0,
                             title="Sector ETF vs SPY (20-day)")
                fig.update_layout(**PL, showlegend=False, coloraxis_showscale=False, height=280)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Recent Regime History</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
        rh = macro.get("regime_history",[])
        if rh:
            rh_df = pd.DataFrame(rh)
            fig2 = px.line(rh_df, x="date", y="SPY", color="combined_regime",
                           title="SPY Price — Colored by Regime")
            fig2.update_layout(**PL, height=280, legend=dict(font=dict(size=9)))
            st.plotly_chart(fig2, use_container_width=True)

        rs = macro.get("regime_stats",{})
        if rs:
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Regime-Sector Forward Returns</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
            rp = st.selectbox("Select regime", list(rs.keys()), key="regime_pick")
            if rp in rs:
                st.dataframe(pd.DataFrame(rs[rp]).T, use_container_width=True, height=350)

        tr = macro.get("transitions",{})
        if tr:
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Regime Transition Matrix</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
            horizon = st.radio("Horizon", ["1d","5d","10d"], horizontal=True)
            t_data = tr.get(horizon,{})
            if t_data:
                t_df = pd.DataFrame(t_data).T
                fig3 = px.imshow(t_df, text_auto=".2f", aspect="auto",
                                 color_continuous_scale=["#09090e","#f5a623"],
                                 title=f"Transition Probabilities ({horizon})")
                fig3.update_layout(**PL, height=350)
                st.plotly_chart(fig3, use_container_width=True)

# ─── TAB 2: SECTOR THEMES ─────────────────────────────────────────────────────
with tab2:
    if not theme_data:
        st.info("No theme data. Run theme_prep.py.")
    else:
        meta = theme_data.get("metadata",{})
        st.caption(f"As of {meta.get('as_of','—')} · {meta.get('total_etfs','—')} ETFs · {meta.get('total_themes','—')} themes · Momentum Engine v3.0")

        theme_df = pd.DataFrame(theme_data.get("theme_rankings",[]))
        etf_df   = pd.DataFrame(theme_data.get("etf_rankings",[]))

        if not theme_df.empty:
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Theme Rankings</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
            top20 = theme_df.head(20)
            if "Avg_Mom_Score" in top20.columns:
                fig_t = px.bar(top20, x="Theme", y="Avg_Mom_Score",
                               color="Avg_Mom_Score",
                               color_continuous_scale=["#e05c5c","#f5a623","#4caf7d"],
                               range_color=[0,10],
                               title="Top 20 Themes — Avg Momentum Score")
                fig_t.update_layout(**PL, showlegend=False, coloraxis_showscale=False,
                                    height=300, xaxis_tickangle=-45)
                st.plotly_chart(fig_t, use_container_width=True)

            dcols = [c for c in ["Theme","ETF_Count","Avg_Theme_Rank","Avg_Mom_Score",
                                   "Leading_Count","Improving_Count","Bull_Stack_Pct","MACD_Bull_Pct"]
                     if c in theme_df.columns]
            st.dataframe(theme_df[dcols], use_container_width=True, height=400, hide_index=True)

        if not etf_df.empty:
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>ETF Rankings</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
            ef1,ef2 = st.columns([1,2])
            with ef1:
                mom_f = st.multiselect("Momentum",["🔥 Leading","✅ Improving","➡ Flat","⚠ Fading","🔴 Broken"],
                                       default=["🔥 Leading","✅ Improving"], key="etf_mom")
            with ef2:
                th_f = st.multiselect("Theme", sorted(etf_df["Theme"].unique()) if "Theme" in etf_df.columns else [], key="etf_th")

            ef = etf_df.copy()
            if mom_f and "Momentum_Label" in ef.columns: ef = ef[ef["Momentum_Label"].isin(mom_f)]
            if th_f and "Theme" in ef.columns: ef = ef[ef["Theme"].isin(th_f)]

            ecols = [c for c in ["Ticker","Theme","Weighted_RS_Rank","Momentum_Score","Momentum_Label",
                                   "Score_A","Score_B","Score_C","Score_D","Score_E",
                                   "Full_Bull_Stack","Dist_52wHigh_%"] if c in ef.columns]
            st.dataframe(ef[ecols], use_container_width=True, height=500, hide_index=True)

# ─── TAB 3: ACTIVE TRIGGERS ───────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Today's Pivot Triggers</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
    if not today_triggers:
        st.markdown("<div style='font-family:var(--mono);color:var(--muted);padding:2rem;text-align:center;'>No triggers today — scanner runs at :01 and :31</div>", unsafe_allow_html=True)
    else:
        tc1,tc2,tc3 = st.columns(3)
        with tc1: conv_f = st.multiselect("Conviction",["HIGH","MED","LOW"],default=["HIGH","MED"],key="tc_conv")
        with tc2: dir_f  = st.multiselect("Direction",["bullish","bearish"],default=["bullish","bearish"],key="tc_dir")
        with tc3: tf_f   = st.multiselect("Timeframe",["30-MIN","65-MIN"],default=["30-MIN","65-MIN"],key="tc_tf")

        filtered = sorted(
            [t for t in today_triggers if t.get("conviction") in conv_f
             and t.get("direction") in dir_f and t.get("timeframe") in tf_f],
            key=lambda t: ({"HIGH":0,"MED":1,"LOW":2}.get(t.get("conviction"),3), t.get("scan_time",""))
        )

        for t in filtered:
            d    = t.get("direction","")
            conv = t.get("conviction","LOW")
            cc   = "trigger-card" + (" trigger-card-bear" if d=="bearish" else "") + (" trigger-card-high" if conv=="HIGH" else "")
            bc   = {"HIGH":"badge-high","MED":"badge-med","LOW":"badge-low"}.get(conv,"badge-low")
            db   = "<span class='badge-bull'>▲ BULL</span>" if d=="bullish" else "<span class='badge-bear'>▼ BEAR</span>"
            pk   = "pivot_high" if d=="bullish" else "pivot_low"

            st.markdown(f"""
<div class='{cc}'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
    <div><div class='t-ticker'>{t.get("ticker","")}</div><div class='t-meta'>{t.get("theme","")} · Rank {t.get("theme_rank","")} · Streak: {t.get("streak_len","")} bars</div></div>
    <div style='text-align:right;'><span class='{bc}'>{conv}</span><div style='margin-top:.4rem;'>{db} · {t.get("timeframe","")}</div></div>
  </div>
  <div class='t-grid'>
    <div><div class='t-field-lbl'>Trigger Close</div><div class='t-field-val'>${t.get("trigger_close",0):.2f}</div></div>
    <div><div class='t-field-lbl'>Pivot {"High" if d=="bullish" else "Low"}</div><div class='t-field-val'>${t.get(pk,0):.2f}</div></div>
    <div><div class='t-field-lbl'>Stop Level</div><div class='t-field-val red'>${t.get("stop_level",0):.2f}</div></div>
    <div><div class='t-field-lbl'>BBUW D/W</div><div class='t-field-val blue'>{t.get("daily_bbuw",0):.0f} / {t.get("weekly_bbuw",0):.0f}</div></div>
  </div>
  <div style='margin-top:.6rem;font-family:var(--mono);font-size:.7rem;color:var(--muted);'>{t.get("entry_note","")} · Stage W{t.get("weekly_stage","")}/D{t.get("daily_stage","")} · Trend {t.get("trend_template","")}/8 · {str(t.get("trigger_time",""))[:19]}</div>
</div>""", unsafe_allow_html=True)

# ─── TAB 4: DAILY WATCHLIST ───────────────────────────────────────────────────
with tab4:
    daily_entries = daily_data.get("entries",[])
    st.caption(f"Last updated: {daily_data.get('scan_time','—')}")

    if not daily_entries:
        st.info("No daily watchlist. Run daily_screener.py.")
    else:
        dl1,dl2,dl3,dl4 = st.columns([1,1,1.5,1])
        with dl1:
            conv_p = st.multiselect("Conviction",["HIGH","MED","LOW"],default=["HIGH","MED"],key="dl_conv")
        with dl2:
            stage_p = st.multiselect("Daily Stage",[1,2,3,4],default=[1,2],key="dl_stage")
        with dl3:
            pivot_p = st.multiselect("8W Pivot tier",
                                       ["STRONG","STANDARD","WEAK","PROXIMITY","NONE"],
                                       default=[],
                                       key="dl_pivot",
                                       help="Empty = show all")
        with dl4:
            nr_only = st.checkbox("Near resistance only", value=False,
                                   help="Show only tickers flagged near a resistance/flipped zone")

        df = pd.DataFrame(daily_entries)
        if conv_p  and "conviction"     in df.columns: df = df[df["conviction"].isin(conv_p)]
        if stage_p and "daily_stage"    in df.columns: df = df[df["daily_stage"].isin(stage_p)]
        if pivot_p and "pivot_8w_tier"  in df.columns: df = df[df["pivot_8w_tier"].isin(pivot_p)]
        if nr_only and "near_resistance" in df.columns: df = df[df["near_resistance"] == True]

        # 8W Pivot emoji column
        tier_emoji_map = {
            "STRONG":    "🔥 STRONG",
            "STANDARD":  "✅ STANDARD",
            "WEAK":      "⚠️ WEAK",
            "PROXIMITY": "👀 PROXIMITY",
            "NONE":      "—",
        }
        if "pivot_8w_tier" in df.columns:
            df["8W Pivot"] = df["pivot_8w_tier"].map(tier_emoji_map).fillna("—")

        dcols = [c for c in [
            "ticker", "conviction", "8W Pivot", "theme", "theme_rank",
            "weekly_stage", "daily_stage", "trend_template",
            "weekly_bbuw", "daily_bbuw",
            "ema8", "pct_from_ema8",
            "near_resistance", "resistance_level", "resistance_distance_pct",
        ] if c in df.columns]
        st.dataframe(df[dcols], use_container_width=True, height=550, hide_index=True)

        # BBUW drill-down
        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>BBUW Component Drill-Down</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
        ins = st.selectbox("Inspect ticker",[e["ticker"] for e in daily_entries],key="dl_ins")
        sel = next((e for e in daily_entries if e["ticker"]==ins),None)
        if sel and sel.get("daily_bbuw_components"):
            cd = pd.DataFrame([{"Component":k.replace("_"," ").title(),"Score":v}
                                for k,v in sel["daily_bbuw_components"].items()])
            fig_c = px.bar(cd, x="Component", y="Score", color="Score",
                           color_continuous_scale=["#e05c5c","#f5a623","#4caf7d"],
                           range_color=[0,100], title=f"BBUW Components — {ins}")
            fig_c.update_layout(**PL, showlegend=False, coloraxis_showscale=False, height=250)
            st.plotly_chart(fig_c, use_container_width=True)

# ─── TAB RADAR: SETUPS RADAR ──────────────────────────────────────────────────
with tab_radar:
    radar_entries = radar_data.get("entries", [])

    if not radar_entries:
        st.info("No radar data. Run radar_prep.py.")
    else:
        st.caption(f"Generated: {radar_data.get('generated_at','—')} · {len(radar_entries)} tickers scored")

        # Top-N selector + filters
        rc1, rc2, rc3 = st.columns([1, 1, 2])
        with rc1:
            top_n = st.number_input("Top N per section", min_value=3, max_value=50, value=10, step=1, key="radar_top_n")
        with rc2:
            min_score = st.slider("Min score threshold", 0, 100, 50, key="radar_min_score")
        with rc3:
            theme_filter = st.multiselect(
                "Theme filter",
                sorted({r.get("theme","Unclassified") for r in radar_entries}),
                key="radar_theme_filter",
            )

        # Apply filters
        filtered = radar_entries
        if theme_filter:
            filtered = [r for r in filtered if r.get("theme") in theme_filter]

        # Sort each section
        in_motion_sorted = sorted(
            [r for r in filtered if r["in_motion_score"] >= min_score],
            key=lambda r: r["in_motion_score"], reverse=True,
        )[:top_n]

        loading_sorted = sorted(
            [r for r in filtered if r["loading_score"] >= min_score],
            key=lambda r: r["loading_score"], reverse=True,
        )[:top_n]

        primed_sorted = sorted(
            [r for r in filtered if r["state"] == "MOMENTUM_PRIMED"],
            key=lambda r: r["in_motion_score"] + r["loading_score"], reverse=True,
        )[:top_n]

        # ── KPI summary ──────────────────────────────────────────────────────
        st.markdown(f"""
<div class='kpi-strip' style='grid-template-columns:repeat(3,1fr);'>
  <div class='kpi-cell red'>
    <div class='kpi-lbl'>🔥 In Motion</div>
    <div class='kpi-val red'>{len([r for r in filtered if r["state"] in ("IN_MOTION","RUNNING","MOMENTUM_PRIMED")])}</div>
    <div class='kpi-sub'>moving right now</div>
  </div>
  <div class='kpi-cell amber'>
    <div class='kpi-lbl'>⭐ Primed (both)</div>
    <div class='kpi-val amber'>{len([r for r in filtered if r["state"]=="MOMENTUM_PRIMED"])}</div>
    <div class='kpi-sub'>moving + still loaded</div>
  </div>
  <div class='kpi-cell blue'>
    <div class='kpi-lbl'>👀 Loaded</div>
    <div class='kpi-val'>{len([r for r in filtered if r["state"] in ("LOADED","DEVELOPING","MOMENTUM_PRIMED")])}</div>
    <div class='kpi-sub'>coiled — watch for trigger</div>
  </div>
</div>
        """, unsafe_allow_html=True)

        # ── Card renderer (shared between sections) ──────────────────────────
        def render_card(r, mode="motion"):
            """mode = 'motion', 'loading', or 'primed'"""
            ticker = r["ticker"]
            score  = r["in_motion_score"] if mode == "motion" else r["loading_score"]
            conv   = r.get("conviction","LOW")
            theme  = r.get("theme","")
            theme_rank = r.get("theme_rank","—")
            pivot_tier = r.get("pivot_8w_tier","NONE")

            tier_emoji = {"STRONG":"🔥","STANDARD":"✅","WEAK":"⚠️","PROXIMITY":"👀","NONE":""}.get(pivot_tier,"")
            badge_cls  = {"HIGH":"badge-high","MED":"badge-med","LOW":"badge-low"}.get(conv,"badge-low")

            # Border color by mode
            if mode == "motion":
                border_color = "var(--red)"
                section_emoji = "🔥"
            elif mode == "loading":
                border_color = "var(--blue)"
                section_emoji = "👀"
            else:  # primed
                border_color = "var(--accent)"
                section_emoji = "⭐"

            reasons = r.get("in_motion_reasons" if mode == "motion" else "loading_reasons", [])
            reasons_html = "  ·  ".join(reasons[:5])

            # Build metrics grid
            today_close = r.get("today_close", 0) or 0
            today_pct   = r.get("today_pct", 0) or 0
            pct_color   = "green" if today_pct >= 0 else "red"
            pct_sign    = "+" if today_pct >= 0 else ""

            # Mode-specific stat blocks
            if mode == "motion":
                stat_blocks = f"""
    <div><div class='t-field-lbl'>Today</div><div class='t-field-val {pct_color}'>{pct_sign}{today_pct:.2f}%</div></div>
    <div><div class='t-field-lbl'>Close</div><div class='t-field-val'>${today_close:.2f}</div></div>
    <div><div class='t-field-lbl'>RVOL</div><div class='t-field-val blue'>{r.get('rvol',0):.1f}×</div></div>
    <div><div class='t-field-lbl'>BBUW D/W</div><div class='t-field-val'>{r.get('daily_bbuw',0):.0f} / {r.get('weekly_bbuw',0):.0f}</div></div>
                """
            else:  # loading or primed — show coil metrics
                stat_blocks = f"""
    <div><div class='t-field-lbl'>ATR Ratio</div><div class='t-field-val'>{r.get('atr_ratio',0):.2f}×</div></div>
    <div><div class='t-field-lbl'>Vol Ratio</div><div class='t-field-val blue'>{r.get('vol_ratio',0):.2f}×</div></div>
    <div><div class='t-field-lbl'>21 EMA Dist</div><div class='t-field-val'>{r.get('ema21_dist_pct',0):.1f}%</div></div>
    <div><div class='t-field-lbl'>BBUW D/W</div><div class='t-field-val'>{r.get('daily_bbuw',0):.0f} / {r.get('weekly_bbuw',0):.0f}</div></div>
                """

            score_label = "IN MOTION" if mode == "motion" else "LOADING" if mode == "loading" else "PRIMED"
            score_color = "var(--red)" if mode == "motion" else "var(--blue)" if mode == "loading" else "var(--accent)"

            return f"""
<div class='trigger-card' style='border-left-color:{border_color};{"box-shadow:0 0 12px rgba(245,166,35,.15);" if mode == "primed" else ""}'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
    <div>
      <div class='t-ticker'>{section_emoji} {ticker}</div>
      <div class='t-meta'>{theme} · Rank {theme_rank} · {tier_emoji} {pivot_tier} · Stage W{r.get("weekly_stage","")}/D{r.get("daily_stage","")}</div>
    </div>
    <div style='text-align:right;'>
      <span class='{badge_cls}'>{conv}</span>
      <div style='margin-top:.4rem;font-family:var(--display);font-size:.65rem;letter-spacing:.1em;color:{score_color};font-weight:700;'>{score_label}: {score:.0f}</div>
    </div>
  </div>
  <div class='t-grid'>{stat_blocks}</div>
  <div style='margin-top:.6rem;font-family:var(--mono);font-size:.7rem;color:var(--muted);'>{reasons_html}</div>
</div>
"""

        # ── PRIMED section (highest conviction — moving + loaded) ────────────
        if primed_sorted:
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>⭐ Primed — Moving + Still Loaded (Top Conviction)</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
            for r in primed_sorted:
                st.markdown(render_card(r, mode="primed"), unsafe_allow_html=True)

        # ── IN MOTION section ────────────────────────────────────────────────
        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>🔥 In Motion — Moving Now</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
        if not in_motion_sorted:
            st.markdown(f"<div style='font-family:var(--mono);color:var(--muted);padding:1rem;text-align:center;font-size:.8rem;'>No tickers above {min_score} threshold.</div>", unsafe_allow_html=True)
        else:
            for r in in_motion_sorted:
                if r["state"] == "MOMENTUM_PRIMED":
                    continue   # already shown above
                st.markdown(render_card(r, mode="motion"), unsafe_allow_html=True)

        # ── LOADING section ──────────────────────────────────────────────────
        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>👀 Loading — Coiled, Watch for Trigger</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
        if not loading_sorted:
            st.markdown(f"<div style='font-family:var(--mono);color:var(--muted);padding:1rem;text-align:center;font-size:.8rem;'>No tickers above {min_score} threshold.</div>", unsafe_allow_html=True)
        else:
            for r in loading_sorted:
                if r["state"] == "MOMENTUM_PRIMED":
                    continue   # already shown above
                st.markdown(render_card(r, mode="loading"), unsafe_allow_html=True)

# ─── TAB 5: WEEKLY SCREEN ────────────────────────────────────────────────────
with tab5:
    weekly_entries = weekly_data.get("entries",[])
    st.caption(f"Last updated: {weekly_data.get('scan_time','—')}")

    if not weekly_entries:
        st.info("No weekly watchlist. Run weekly_screener.py.")
    else:
        wdf = pd.DataFrame(weekly_entries)
        stage_labels = {1:"1·Basing",2:"2·Advancing",3:"3·Topping",4:"4·Declining"}
        if "stage" in wdf.columns:
            wdf["Stage"] = wdf["stage"].map(stage_labels).fillna("—")

        # 8W Pivot tier emoji column
        tier_emoji_map = {
            "STRONG":    "🔥 STRONG",
            "STANDARD":  "✅ STANDARD",
            "WEAK":      "⚠️ WEAK",
            "PROXIMITY": "👀 PROXIMITY",
            "NONE":      "—",
        }
        if "pivot_8w_tier" in wdf.columns:
            wdf["8W Pivot"] = wdf["pivot_8w_tier"].map(tier_emoji_map).fillna("—")

        # ── Summary KPI row for 8W pivots ──────────────────────────────────────
        if "pivot_8w_tier" in wdf.columns:
            n_strong = int((wdf["pivot_8w_tier"] == "STRONG").sum())
            n_std    = int((wdf["pivot_8w_tier"] == "STANDARD").sum())
            n_weak   = int((wdf["pivot_8w_tier"] == "WEAK").sum())
            n_prox   = int((wdf["pivot_8w_tier"] == "PROXIMITY").sum())

            st.markdown(f"""
<div class='kpi-strip' style='grid-template-columns:repeat(4,1fr);'>
  <div class='kpi-cell amber'>
    <div class='kpi-lbl'>🔥 Strong 8W Pivot</div>
    <div class='kpi-val amber'>{n_strong}</div>
    <div class='kpi-sub'>pivot + EMA up + vol spike</div>
  </div>
  <div class='kpi-cell green'>
    <div class='kpi-lbl'>✅ Standard 8W Pivot</div>
    <div class='kpi-val green'>{n_std}</div>
    <div class='kpi-sub'>pivot + EMA rising</div>
  </div>
  <div class='kpi-cell blue'>
    <div class='kpi-lbl'>👀 Proximity</div>
    <div class='kpi-val'>{n_prox}</div>
    <div class='kpi-sub'>within 3% of rising 8W EMA</div>
  </div>
  <div class='kpi-cell red'>
    <div class='kpi-lbl'>⚠️ Weak Pivot</div>
    <div class='kpi-val red'>{n_weak}</div>
    <div class='kpi-sub'>EMA flat/falling — caution</div>
  </div>
</div>
            """, unsafe_allow_html=True)

        # ── Filter ─────────────────────────────────────────────────────────────
        if "pivot_8w_tier" in wdf.columns:
            tier_filter = st.multiselect(
                "8W Pivot tier",
                ["STRONG", "STANDARD", "WEAK", "PROXIMITY", "NONE"],
                default=["STRONG", "STANDARD", "PROXIMITY"],
                key="weekly_8w_filter",
            )
            if tier_filter:
                wdf = wdf[wdf["pivot_8w_tier"].isin(tier_filter)]

        # ── Table ──────────────────────────────────────────────────────────────
        wd = [c for c in [
            "ticker", "Stage", "8W Pivot", "ema8", "pct_from_ema8",
            "ema8_rising", "pivot_8w_volume_spike",
            "trend_template_score", "bbuw_score",
        ] if c in wdf.columns]
        st.dataframe(wdf[wd], use_container_width=True, height=500, hide_index=True)

        # ── Scatter chart ──────────────────────────────────────────────────────
        if "bbuw_score" in wdf.columns and "trend_template_score" in wdf.columns:
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>BBUW vs Trend Template (Colored by 8W Pivot Tier)</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
            tier_color_map = {
                "STRONG":    "#f5a623",
                "STANDARD":  "#4caf7d",
                "PROXIMITY": "#5b8dee",
                "WEAK":      "#e05c5c",
                "NONE":      "#7a7060",
            }
            color_arg = "pivot_8w_tier" if "pivot_8w_tier" in wdf.columns else "bbuw_score"
            color_kwargs = (
                {"color_discrete_map": tier_color_map}
                if color_arg == "pivot_8w_tier"
                else {"color_continuous_scale": ["#e05c5c","#f5a623","#4caf7d"]}
            )
            fig_s = px.scatter(
                wdf, x="trend_template_score", y="bbuw_score",
                text="ticker", color=color_arg,
                title="BBUW Score vs Minervini Trend Template",
                **color_kwargs,
            )
            fig_s.update_traces(textposition="top center", textfont_size=9)
            fig_s.update_layout(
                **PL, showlegend=(color_arg == "pivot_8w_tier"),
                coloraxis_showscale=False, height=420,
                xaxis_title="Trend Template (0-8)", yaxis_title="Weekly BBUW",
                legend=dict(font=dict(size=9), title_text="8W Pivot Tier"),
            )
            fig_s.add_vline(x=6, line_dash="dash", line_color="rgba(245,166,35,0.3)")
            fig_s.add_hline(y=60, line_dash="dash", line_color="rgba(245,166,35,0.3)")
            st.plotly_chart(fig_s, use_container_width=True)

# ─── TAB 6: TRIGGER HISTORY ───────────────────────────────────────────────────
with tab6:
    hc1,hc2,hc3,hc4 = st.columns(4)
    with hc1: days_b = st.selectbox("Days",[7,14,30,60,90],index=2)
    with hc2: conv_h = st.selectbox("Conviction",["All","HIGH","MED","LOW"])
    with hc3: tick_h = st.text_input("Ticker","").strip().upper()
    with hc4: them_h = st.text_input("Theme","").strip()

    history = get_trigger_history(
        days=days_b,
        conviction=None if conv_h=="All" else conv_h,
        ticker=tick_h or None,
        theme=them_h or None,
    )

    if not history:
        st.info(f"No triggers in the last {days_b} days.")
    else:
        hdf = pd.DataFrame(history)
        hs1,hs2,hs3,hs4,hs5 = st.columns(5)
        with hs1: st.metric("Total",len(hdf))
        with hs2: st.metric("Bullish",int((hdf["direction"]=="bullish").sum()))
        with hs3: st.metric("Bearish",int((hdf["direction"]=="bearish").sum()))
        with hs4: st.metric("HIGH",int((hdf["conviction"]=="HIGH").sum()))
        with hs5: st.metric("Unique Tickers",int(hdf["ticker"].nunique()))

        dc = hdf.groupby(["direction","conviction"]).size().reset_index(name="count")
        fh = px.bar(dc, x="conviction", y="count", color="direction", barmode="group",
                    color_discrete_map={"bullish":"#4caf7d","bearish":"#e05c5c"},
                    title="Triggers by Conviction + Direction")
        fh.update_layout(**PL, height=240)
        st.plotly_chart(fh, use_container_width=True)

        hcols = [c for c in ["scan_time","ticker","timeframe","direction","conviction",
                               "theme","trigger_close","stop_level","daily_bbuw","weekly_bbuw"] if c in hdf.columns]
        if "scan_time" in hdf.columns:
            hdf["scan_time"] = pd.to_datetime(hdf["scan_time"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(hdf[hcols], use_container_width=True, height=500, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-footer'>
  PIVOT SCANNER WAR ROOM · SCAN ALERT ONLY — NOT FINANCIAL ADVICE ·
  @1CHARTMASTER × WEINSTEIN × MINERVINI × @OHIAIN ·
  PRESSURE COOKER · STAGE ANALYSIS · BBUW · 30/65-MIN PIVOTS
</div>
""", unsafe_allow_html=True)
