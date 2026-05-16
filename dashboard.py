"""
Pivot Scanner — Unified Dashboard
====================================
Tabs:
  1. ◈ MACRO VIEW      — Regime, VIX, macro indicators, sector rotation
  2. ◈ THEME TRACKER   — ETF thematic RS rankings + momentum engine v3.0
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

# R-ratio thresholds (must match pivot_scanner.py)
R_ELITE   = 3.0
R_GOOD    = 2.0
R_MINIMUM = 1.0

from data_layer import (
    get_latest_weekly_watchlist,
    get_latest_daily_watchlist,
    get_today_triggers,
    get_trigger_history,
    get_industry_ranks_full,
    get_volume_surges,
    get_ep_events,
)

# ── New tactical macro + speculative theme modules ────────────────────────────
try:
    import macro_view
    import speculative_themes
    TACTICAL_MODULES_AVAILABLE = True
except ImportError as _e:
    TACTICAL_MODULES_AVAILABLE = False

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
.kpi-strip{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-bottom:1.2rem;}
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
    """
    Load price-tape macro data. macro_data.json is now optional —
    the new macro_prep.py is the v2 GIP engine. Fall back to
    vol_compression.json for VIX and return a minimal cur dict
    so the masthead KPI strip still renders.
    """
    p = "data/macro_data.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    # Graceful fallback — masthead will show placeholder values
    return {}

@st.cache_data(ttl=300)
def load_theme():
    p = "data/theme_data.json"
    if not os.path.exists(p): return {}
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}

@st.cache_data(ttl=120)
def load_radar():
    p = "data/radar_data.json"
    if not os.path.exists(p): return {}
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}

@st.cache_data(ttl=3600)
def load_gip():
    p = "data/gip_data.json"
    if not os.path.exists(p): return {}
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}

@st.cache_data(ttl=3600)
def load_narrative():
    p = "data/narrative_data.json"
    if not os.path.exists(p): return {}
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}

@st.cache_data(ttl=3600)
def load_gate():
    p = "data/gate_state.json"
    if not os.path.exists(p): return {}
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}

@st.cache_data(ttl=3600)
def load_vol_compression():
    p = "data/vol_compression.json"
    if not os.path.exists(p): return {}
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}

@st.cache_data(ttl=300)
def load_index_read():
    p = "data/index_read.json"
    if not os.path.exists(p): return {}
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}

@st.cache_data(ttl=3600)
def load_industry_ranks():
    return get_industry_ranks_full()

@st.cache_data(ttl=300)
def load_volume_surges():
    return get_volume_surges()

@st.cache_data(ttl=3600)
def load_ep_events():
    return get_ep_events()

@st.cache_data(ttl=300)
def load_posture() -> dict:
    """Load pre-computed narrative regime posture from parquet."""
    path = "data/narrative_regime.parquet"
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return {}
        row = df.iloc[0].to_dict()
        # top3_transitions may come back as numpy array — normalise to list
        raw = row.get("top3_transitions")
        if raw is not None and hasattr(raw, "tolist"):
            row["top3_transitions"] = raw.tolist()
        return row
    except Exception:
        return {}

@st.cache_data(ttl=60)
def _triggers(): return get_today_triggers()

@st.cache_data(ttl=60)
def _daily(): return get_latest_daily_watchlist()

@st.cache_data(ttl=300)
def _weekly(): return get_latest_weekly_watchlist()


macro          = load_macro()
theme_data     = load_theme()
radar_data     = load_radar()
index_read     = load_index_read()
industry_ranks = load_industry_ranks()
volume_data    = load_volume_surges()
ep_data        = load_ep_events()
gip_data       = load_gip()
narrative_data = load_narrative()
gate_data      = load_gate()
vol_data       = load_vol_compression()
posture_data   = load_posture()
today_triggers = _triggers()
daily_data     = _daily()
weekly_data    = _weekly()
cur            = macro.get("current", {})
now            = datetime.now()

high_today = sum(1 for t in today_triggers if t.get("conviction")=="HIGH")
med_today  = sum(1 for t in today_triggers if t.get("conviction")=="MED")

# Posture card values
_posture       = str(posture_data.get("posture", "—"))
_posture_label = _posture.replace("_", " ")
_posture_conf  = posture_data.get("posture_confidence") or 0.0
_regime_score  = posture_data.get("regime_score") or 0.0
_posture_class = {
    "BUY_RIP":  "green",
    "BUY_DIP":  "blue",
    "AVOID":    "red",
    "NEUTRAL":  "amber",
}.get(_posture, "amber") if posture_data else "amber"
_posture_sub = f"{float(_posture_conf):.0%} conf · score {float(_regime_score):.2f}" if posture_data else "awaiting data"

# ── Masthead ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='masthead'><div class='masthead-inner'>
  <div class='masthead-title'>◈ PIVOT SCANNER <span>WAR ROOM</span></div>
  <div class='masthead-meta'>
    <span class='live-dot'>LIVE</span>
    <span>REGIME: {gip_data.get('regime',{}).get('label', str(cur.get('combined_regime','—')))[:28]}</span>
    <span>VIX {vol_data.get('vix', cur.get('VIX','—'))}</span>
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
  <div class='kpi-cell {_posture_class}'>
    <div class='kpi-lbl'>Posture</div>
    <div class='kpi-val {_posture_class}'>{_posture_label}</div>
    <div class='kpi-sub'>{_posture_sub}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1,tab_index,tab2,tab_spec,tab3,tab4,tab_radar,tab5,tab6,tab_vol = st.tabs([
    "◈ MACRO COMMAND","◈ INDEX READ","◈ THEME TRACKER","◈ SPECULATIVE THEMES",
    "◈ ACTIVE TRIGGERS","◈ DAILY WATCHLIST","◈ SETUPS RADAR","◈ WEEKLY SCREEN",
    "◈ TRIGGER HISTORY","◈ VOLUME LOG",
])

# ─── TAB 1: MACRO COMMAND CENTER (tactical ETF-based framework) ───────────────
with tab1:
    if not TACTICAL_MODULES_AVAILABLE:
        st.info("macro_view.py or speculative_themes.py not found. "
                "Add the new module files to your repo.")
    else:
        macro_view.render()

# ─── TAB INDEX: KELL-STYLE INDEX READ ─────────────────────────────────────────
with tab_index:
    if not index_read:
        st.info("No index read data. Run index_read_prep.py.")
    else:
        st.caption(f"Generated: {index_read.get('generated_at','—')} · Rule-based technical structure analysis using the Cycle of Price Action framework")

        indices_data = index_read.get("indices", [])
        sectors_data = index_read.get("sectors", [])
        all_data = indices_data + sectors_data

        # ── Framework Reference (collapsible) ───────────────────────────────
        with st.expander("📖 Cycle of Price Action — Framework Reference", expanded=False):
            st.markdown("""
<div style='font-family:var(--mono);font-size:.85rem;color:var(--text);line-height:1.6;'>

<div style='font-family:var(--display);font-size:.7rem;color:var(--accent);letter-spacing:.15em;text-transform:uppercase;margin-bottom:.6rem;'>Core EMA Toolkit</div>

<b>Daily 10/20 EMA</b> — Kell's primary working levels. Trend filter, dynamic support, and risk tool. Hold above = trend intact. Loss of 10 EMA after extension = early warning. Loss of 20/21 EMA = deeper structural break.

<b>Weekly 10/20 EMA & 20-Week MA</b> — Major trend shift signals. The 20-week MA is the Weinstein reference for Stage 2 vs Stage 4 markets. Slope direction matters more than position.

<b>50/200 SMA</b> — Long-term trend filter. Below 200 SMA = long-term trend compromised. Crossover signals (death cross / golden cross) are confirmation, not entries.

<br>
<div style='font-family:var(--display);font-size:.7rem;color:var(--accent);letter-spacing:.15em;text-transform:uppercase;margin:1rem 0 .6rem 0;'>Four Phases of the Cycle</div>

<b>🌱 Wedge Pop / Base n' Break</b> — Bottoming or reversal forming. Falling wedge resolves up, base breaks out, bear trap reverses. Watch for confirmation; entries near breakout level. Stop below pattern low.

<b>↗️ EMA Crossback</b> — <i>Kell's preferred entry phase</i>. Price re-tests the 10 EMA (or deeper, the 21 EMA) from above and holds. Best in established uptrends with HH/HL structure. Lowest-risk entry with tight stop just below the EMA.

<b>🚀 Trend Continuation</b> — Riding the trend. Above 10/20 EMA, working levels intact, higher highs and higher lows. Hold the position; let it run. Avoid re-entering when extended >5-8% from 10 EMA.

<b>⚠️ Exhaustion / End of Cycle</b> — Distribution risk. Rising wedge, bull traps, failed breakouts, or loss of working levels after extension. Tighten stops; loss of 10 EMA = scale out. No new longs.

<br>
<div style='font-family:var(--display);font-size:.7rem;color:var(--accent);letter-spacing:.15em;text-transform:uppercase;margin:1rem 0 .6rem 0;'>Risk Management Rules</div>

<b>Entry rule:</b> EMA Crossback is the highest-quality setup — wait for it.<br>
<b>Hold rule:</b> Above 10 EMA daily / 10-week weekly = stay long.<br>
<b>Sell signal:</b> Downside cross of 10 EMA after extension, distribution candles, weekly structure flips to LH/LL.<br>
<b>Multi-timeframe:</b> Weekly structure rules > daily structure rules. Daily defines entries; weekly defines if you should be looking at all.

</div>
            """, unsafe_allow_html=True)

        # ── Bias + Phase Summary KPI Strip ──────────────────────────────────
        n_constructive = sum(1 for r in all_data if r["narrative"]["bias_color"] == "green")
        n_cautious     = sum(1 for r in all_data if r["narrative"]["bias_color"] == "amber")
        n_distribution = sum(1 for r in all_data if r["narrative"]["bias_color"] == "red")

        from collections import Counter
        phase_counts = Counter(r["phase"]["phase"] for r in all_data)

        st.markdown(f"""
<div class='kpi-strip' style='grid-template-columns:repeat(4,1fr);'>
  <div class='kpi-cell green'>
    <div class='kpi-lbl'>↗️ EMA Crossback</div>
    <div class='kpi-val green'>{phase_counts.get('EMA_CROSSBACK', 0)}</div>
    <div class='kpi-sub'>low-risk entry zone</div>
  </div>
  <div class='kpi-cell green'>
    <div class='kpi-lbl'>🚀 Trend Continuation</div>
    <div class='kpi-val green'>{phase_counts.get('TREND_CONTINUATION', 0)}</div>
    <div class='kpi-sub'>working levels intact</div>
  </div>
  <div class='kpi-cell blue'>
    <div class='kpi-lbl'>🌱 Wedge / Base Break</div>
    <div class='kpi-val'>{phase_counts.get('WEDGE_POP_BASE_BREAK', 0)}</div>
    <div class='kpi-sub'>reversal forming</div>
  </div>
  <div class='kpi-cell red'>
    <div class='kpi-lbl'>⚠️ Exhaustion</div>
    <div class='kpi-val red'>{phase_counts.get('EXHAUSTION', 0)}</div>
    <div class='kpi-sub'>distribution risk</div>
  </div>
</div>
        """, unsafe_allow_html=True)

        # ── Filters ─────────────────────────────────────────────────────────
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            view_pick = st.radio("View", ["Indices", "Sectors", "Both"], horizontal=True, key="index_view")
        with fc2:
            phase_filter = st.multiselect(
                "Filter by phase",
                ["EMA_CROSSBACK", "TREND_CONTINUATION", "WEDGE_POP_BASE_BREAK", "EXHAUSTION", "NEUTRAL"],
                default=[],
                key="index_phase_filter",
                help="Empty = show all phases",
            )

        # ── Card renderer ───────────────────────────────────────────────────
        def render_index_card(r):
            narr = r["narrative"]
            phase = r["phase"]
            bias = narr["bias"]
            bias_color = narr["bias_color"]
            today_pct = r.get("today_pct", 0)
            today_color = "var(--green)" if today_pct >= 0 else "var(--red)"

            # Phase styling
            phase_color_map = {
                "green": "var(--green)",
                "blue":  "var(--blue)",
                "amber": "var(--accent)",
                "red":   "var(--red)",
            }
            phase_bg_map = {
                "green": "rgba(76,175,125,0.15)",
                "blue":  "rgba(91,141,238,0.15)",
                "amber": "rgba(245,166,35,0.15)",
                "red":   "rgba(224,92,92,0.15)",
            }
            phase_color = phase_color_map[phase["phase_color"]]
            phase_bg = phase_bg_map[phase["phase_color"]]

            # Bias styling
            border_color = {"green": "var(--green)", "amber": "var(--accent)", "red": "var(--red)"}[bias_color]
            bias_bg = {"green": "rgba(76,175,125,0.15)", "amber": "rgba(245,166,35,0.15)", "red": "rgba(224,92,92,0.15)"}[bias_color]
            bias_txt = {"green": "var(--green)", "amber": "var(--accent)", "red": "var(--red)"}[bias_color]

            # Build daily/weekly read HTML
            daily_html = "".join(f"<li style='margin-bottom:.25rem;'>{line}</li>" for line in narr["daily_read"])
            weekly_html = "".join(f"<li style='margin-bottom:.25rem;'>{line}</li>" for line in narr["weekly_read"])

            # Cycle signals
            cycle_html = ""
            if narr["cycle_signals"]:
                cycle_html = "<div style='margin-top:.7rem;padding:.6rem;background:rgba(245,166,35,0.06);border-left:2px solid var(--accent);'>"
                cycle_html += "<div class='t-field-lbl' style='margin-bottom:.4rem;'>Cycle of Price Action — Active Signals</div>"
                for sig in narr["cycle_signals"]:
                    cycle_html += f"<div style='font-family:var(--mono);font-size:.78rem;color:var(--text);margin:.15rem 0;'>{sig}</div>"
                cycle_html += "</div>"

            # Bull / Bear cases
            bull_html = "".join(f"<div style='font-family:var(--mono);font-size:.75rem;margin:.15rem 0;color:var(--green);'>▲ {lvl}</div>" for lvl in narr["bull_levels"])
            bear_html = "".join(f"<div style='font-family:var(--mono);font-size:.75rem;margin:.15rem 0;color:var(--red);'>▼ {lvl}</div>" for lvl in narr["bear_levels"])

            return f"""
<div style='background:linear-gradient(135deg,#0f0f17 0%,#09090e 100%);border:1px solid var(--border);border-left:3px solid {border_color};padding:1.2rem;margin-bottom:1rem;border-radius:4px;'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.8rem;'>
    <div>
      <div class='t-ticker'>{r['ticker']} <span style='font-family:var(--display);font-size:.7rem;font-weight:400;color:var(--muted);letter-spacing:.1em;'>· {r['name']}</span></div>
      <div class='t-meta' style='margin-top:.3rem;'>
        Price: <span style='color:var(--text);font-family:var(--mono);'>${r['current_price']:.2f}</span> &nbsp;
        Today: <span style='color:{today_color};font-family:var(--mono);'>{"+" if today_pct>=0 else ""}{today_pct:.2f}%</span>
      </div>
    </div>
    <div style='display:flex;flex-direction:column;gap:.3rem;align-items:flex-end;'>
      <div style='background:{bias_bg};color:{bias_txt};padding:.4rem .8rem;font-family:var(--display);font-size:.6rem;font-weight:700;letter-spacing:.15em;clip-path:polygon(0 0,calc(100% - 6px) 0,100% 6px,100% 100%,0 100%);'>
        {bias}
      </div>
    </div>
  </div>

  <!-- Cycle Phase Banner -->
  <div style='background:{phase_bg};border-left:3px solid {phase_color};padding:.6rem .8rem;margin-bottom:.8rem;border-radius:3px;'>
    <div style='display:flex;justify-content:space-between;align-items:center;'>
      <div>
        <div style='font-family:var(--display);font-size:.55rem;letter-spacing:.15em;color:var(--muted);text-transform:uppercase;'>Cycle Phase</div>
        <div style='font-family:var(--display);font-size:.95rem;color:{phase_color};font-weight:700;margin-top:.15rem;'>{phase['phase_emoji']} {phase['phase_label']}</div>
      </div>
    </div>
    <div style='font-family:var(--mono);font-size:.78rem;color:var(--text);margin-top:.4rem;line-height:1.5;'>
      {phase['phase_action']}
    </div>
  </div>

  <div style='display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:.8rem;'>
    <div>
      <div class='t-field-lbl'>Daily Structure Read</div>
      <ul style='list-style:none;padding:0;margin:.4rem 0 0 0;font-family:var(--mono);font-size:.78rem;color:var(--text);'>
        {daily_html}
      </ul>
    </div>
    <div>
      <div class='t-field-lbl'>Weekly Structure Read</div>
      <ul style='list-style:none;padding:0;margin:.4rem 0 0 0;font-family:var(--mono);font-size:.78rem;color:var(--text);'>
        {weekly_html}
      </ul>
    </div>
  </div>

  {cycle_html}

  <div style='display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:.8rem;padding-top:.7rem;border-top:1px solid var(--border2);'>
    <div>
      <div class='t-field-lbl' style='color:var(--green);'>Bull Case</div>
      <div style='margin-top:.3rem;'>{bull_html}</div>
    </div>
    <div>
      <div class='t-field-lbl' style='color:var(--red);'>Bear Case</div>
      <div style='margin-top:.3rem;'>{bear_html}</div>
    </div>
  </div>
</div>
"""

        # Apply phase filter
        def filter_phase(items):
            if not phase_filter:
                return items
            return [r for r in items if r["phase"]["phase"] in phase_filter]

        filtered_indices = filter_phase(indices_data)
        filtered_sectors = filter_phase(sectors_data)

        # Render cards
        if view_pick in ("Indices", "Both"):
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Major Indices</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
            if not filtered_indices:
                st.markdown("<div style='font-family:var(--mono);color:var(--muted);padding:1rem;font-size:.8rem;'>No indices match the selected phase filter.</div>", unsafe_allow_html=True)
            else:
                for r in filtered_indices:
                    st.markdown(render_index_card(r), unsafe_allow_html=True)

        if view_pick in ("Sectors", "Both"):
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Sector ETFs</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
            if not filtered_sectors:
                st.markdown("<div style='font-family:var(--mono);color:var(--muted);padding:1rem;font-size:.8rem;'>No sectors match the selected phase filter.</div>", unsafe_allow_html=True)
            else:
                for r in filtered_sectors:
                    st.markdown(render_index_card(r), unsafe_allow_html=True)

        # ── Footer disclaimer / source pointer ──────────────────────────────
        st.markdown("""
<div style='margin-top:2rem;padding:1rem;background:rgba(245,166,35,0.04);border-left:2px solid var(--accent);font-family:var(--mono);font-size:.75rem;color:var(--muted);line-height:1.6;'>
  <strong style='color:var(--accent);'>About this analysis:</strong> Rule-based technical structure detection using the Cycle of Price Action framework popularized by Oliver Kell. <em>This is generated mechanically from price data — it is not Oliver Kell's analysis or current views.</em> For Kell's actual views and commentary, see his Substack at <strong>theweeklieswatch.substack.com</strong> or follow <strong>@OliverKell_</strong> and <strong>@TheSwingReport</strong> on X.
</div>
        """, unsafe_allow_html=True)

# ─── TAB 2: THEME TRACKER ─────────────────────────────────────────────────────
with tab2:
    if not theme_data:
        st.info("No theme data. Run theme_prep.py.")
    else:
        meta = theme_data.get("metadata", {})
        st.caption(f"As of {meta.get('as_of','—')} · {meta.get('total_etfs','—')} ETFs · "
                   f"{meta.get('total_themes','—')} themes · {meta.get('version','RS Engine v4.0')}")

        theme_df = pd.DataFrame(theme_data.get("theme_rankings", []))
        etf_df   = pd.DataFrame(theme_data.get("etf_rankings",   []))

        if not theme_df.empty:
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div>"
                        "<div class='sec-bar-label'>Theme Rankings — RS vs SPY (1–99)</div>"
                        "<div class='sec-bar-line'></div></div>", unsafe_allow_html=True)

            # ── RS bar chart ───────────────────────────────────────────────────
            rs_col = "Avg_RS_Pct" if "Avg_RS_Pct" in theme_df.columns else None
            if rs_col:
                top20 = theme_df.head(20).copy()
                fig_t = px.bar(
                    top20, x="Theme", y=rs_col,
                    color=rs_col,
                    color_continuous_scale=["#e05c5c", "#f5a623", "#4caf7d"],
                    range_color=[20, 80],
                    title="Top 20 Themes — RS Percentile vs SPY (higher = stronger)",
                )
                fig_t.add_hline(y=70, line_dash="dash",
                                line_color="rgba(245,166,35,0.4)",
                                annotation_text="RS 70",
                                annotation_font_color="rgba(245,166,35,0.7)",
                                annotation_font_size=9)
                fig_t.add_hline(y=80, line_dash="dash",
                                line_color="rgba(76,175,125,0.4)",
                                annotation_text="RS 80",
                                annotation_font_color="rgba(76,175,125,0.7)",
                                annotation_font_size=9)
                fig_t.update_layout(**PL, showlegend=False,
                                    coloraxis_showscale=False,
                                    height=320, xaxis_tickangle=-45,
                                    yaxis_title="RS Percentile vs SPY",
                                    yaxis_range=[0, 100])
                st.plotly_chart(fig_t, use_container_width=True)

            # ── Theme table ────────────────────────────────────────────────────
            tcols = [c for c in ["Theme","ETF_Count","Avg_RS_Pct","Median_RS",
                                  "Top_ETF","Top_ETF_RS","N_Above_80","N_Above_70",
                                  "Ret_1M_%","Ret_3M_%","Ret_12M_%"]
                     if c in theme_df.columns]

            def rs_bg(v):
                if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
                if v >= 90: return "background-color:#0d3320;color:#00ff88;font-weight:700"
                if v >= 80: return "background-color:#0a2a1a;color:#00cc66;font-weight:700"
                if v >= 70: return "background-color:#111a14;color:#00aa55"
                if v >= 50: return "background-color:#111111;color:#cccccc"
                if v >= 30: return "background-color:#1a1408;color:#cc8800"
                return              "background-color:#1a0e0e;color:#cc4444"

            def ret_bg(v):
                if v is None or pd.isna(v): return "background-color:#1a1a1a;color:#555"
                if v > 10:  return "background-color:#0d3320;color:#00ff88;font-weight:700"
                if v > 0:   return "background-color:#111a14;color:#00aa55"
                if v > -5:  return "background-color:#1a1408;color:#cc8800"
                return              "background-color:#1a0e0e;color:#cc4444"

            rs_styled_cols = [c for c in ["Avg_RS_Pct","Median_RS","Top_ETF_RS"] if c in theme_df.columns]
            ret_styled_cols = [c for c in ["Ret_1M_%","Ret_3M_%","Ret_12M_%"] if c in theme_df.columns]

            t_styled = (
                theme_df[tcols].style
                .map(rs_bg,  subset=rs_styled_cols  if rs_styled_cols  else [])
                .map(ret_bg, subset=ret_styled_cols if ret_styled_cols else [])
                .format({
                    "Avg_RS_Pct":  lambda x: f"{x:.0f}" if pd.notna(x) else "—",
                    "Median_RS":   lambda x: f"{x:.0f}" if pd.notna(x) else "—",
                    "Top_ETF_RS":  lambda x: f"{x:.0f}" if pd.notna(x) else "—",
                    "Ret_1M_%":    lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                    "Ret_3M_%":    lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                    "Ret_12M_%":   lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                })
                .set_properties(**{
                    "background-color": "#111111",
                    "color": "#cccccc",
                    "border": "1px solid #2a2a2a",
                    "font-family": "Fira Code, monospace",
                    "font-size": "12px",
                    "padding": "5px 10px",
                })
                .set_properties(subset=["Theme"], **{
                    "color": "#FFA500",
                    "font-family": "Fira Code, monospace",
                    "font-weight": "500",
                })
                .set_properties(subset=["Top_ETF"], **{
                    "color": "#00ff88",
                    "font-family": "Orbitron, monospace",
                    "font-size": "11px",
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
                ])
            )
            st.dataframe(t_styled, use_container_width=True, height=600, hide_index=True)

        if not etf_df.empty:
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div>"
                        "<div class='sec-bar-label'>ETF Rankings — RS vs SPY</div>"
                        "<div class='sec-bar-line'></div></div>", unsafe_allow_html=True)

            ef1, ef2 = st.columns([1, 2])
            with ef1:
                rs_filter = st.selectbox("Min RS Percentile",
                                         ["All", "≥ 70", "≥ 80", "≥ 90"],
                                         key="etf_rs_filter")
            with ef2:
                th_f = st.multiselect(
                    "Theme",
                    sorted(etf_df["Theme"].unique()) if "Theme" in etf_df.columns else [],
                    key="etf_th")

            ef = etf_df.copy()
            rs_min = {"≥ 70": 70, "≥ 80": 80, "≥ 90": 90}.get(rs_filter, 0)
            if rs_min and "RS_Pct" in ef.columns:
                ef = ef[ef["RS_Pct"] >= rs_min]
            if th_f and "Theme" in ef.columns:
                ef = ef[ef["Theme"].isin(th_f)]

            ecols = [c for c in ["Ticker","Theme","RS_Pct","RS_vs_SPY",
                                  "Ret_1M_%","Ret_3M_%","Ret_6M_%","Ret_12M_%",
                                  "Dist_52wHigh_%"] if c in ef.columns]

            e_styled = (
                ef[ecols].style
                .map(rs_bg,  subset=["RS_Pct"] if "RS_Pct" in ef.columns else [])
                .map(ret_bg, subset=[c for c in ["Ret_1M_%","Ret_3M_%","Ret_6M_%","Ret_12M_%"] if c in ef.columns])
                .format({
                    "RS_Pct":         lambda x: f"{x:.0f}" if pd.notna(x) else "—",
                    "RS_vs_SPY":      lambda x: f"{x:.3f}" if pd.notna(x) else "—",
                    "Ret_1M_%":       lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                    "Ret_3M_%":       lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                    "Ret_6M_%":       lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                    "Ret_12M_%":      lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                    "Dist_52wHigh_%": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                })
                .set_properties(**{
                    "background-color": "#111111",
                    "color": "#cccccc",
                    "border": "1px solid #2a2a2a",
                    "font-family": "Fira Code, monospace",
                    "font-size": "12px",
                    "padding": "5px 10px",
                })
                .set_properties(subset=["Ticker"], **{
                    "color": "#FFA500",
                    "font-family": "Orbitron, monospace",
                    "font-size": "11px",
                    "font-weight": "700",
                })
                .set_properties(subset=["Theme"], **{"color": "#B87333", "font-size": "11px"})
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
                ])
            )
            st.dataframe(e_styled, use_container_width=True, height=600, hide_index=True)

# ─── TAB SPEC: SPECULATIVE THEMES ─────────────────────────────────────────────
with tab_spec:
    if not TACTICAL_MODULES_AVAILABLE:
        st.info("speculative_themes.py not found. Add the new module files to your repo.")
    else:
        speculative_themes.render()

# ─── TAB 3: ACTIVE TRIGGERS ───────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Today's Pivot Triggers</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
    if not today_triggers:
        st.markdown("<div style='font-family:var(--mono);color:var(--muted);padding:2rem;text-align:center;'>No triggers today — scanner runs at :01 and :31</div>", unsafe_allow_html=True)
    else:
        tc1, tc2, tc3, tc4 = st.columns([1, 1, 1, 1])
        with tc1: conv_f = st.multiselect("Conviction", ["HIGH","MED","LOW"], default=["HIGH","MED"], key="tc_conv")
        with tc2: dir_f  = st.multiselect("Direction",  ["bullish","bearish"], default=["bullish","bearish"], key="tc_dir")
        with tc3: tf_f   = st.multiselect("Timeframe",  ["30-MIN","65-MIN"], default=["30-MIN","65-MIN"], key="tc_tf")
        with tc4: dedup  = st.checkbox("Deduplicate (65-MIN wins)", value=True, key="tc_dedup",
                                        help="If same ticker fires on both timeframes, show only 65-MIN")

        filtered = [
            t for t in today_triggers
            if t.get("conviction") in conv_f
            and t.get("direction") in dir_f
            and t.get("timeframe") in tf_f
        ]

        # Deduplication — 65-MIN wins over 30-MIN for same ticker+direction
        if dedup:
            seen_65 = {(t["ticker"], t.get("direction"))
                       for t in filtered if t.get("timeframe") == "65-MIN"}
            filtered = [
                t for t in filtered
                if not (t.get("timeframe") == "30-MIN" and
                        (t["ticker"], t.get("direction")) in seen_65)
            ]

        # Sort by composite trigger score descending
        filtered = sorted(filtered,
                          key=lambda t: t.get("trigger_score", 0),
                          reverse=True)

        st.caption(f"{len(filtered)} trigger(s) · Sorted by trigger score (conviction + streak + R-ratio + RVOL + timeframe)")

        for t in filtered:
            d    = t.get("direction", "")
            conv = t.get("conviction", "LOW")
            score = t.get("trigger_score", 0)
            r_ratio = t.get("r_ratio")
            rvol    = t.get("rvol_trigger", 1.0) or 1.0
            atr     = t.get("atr_14d")
            dist_sh = t.get("dist_session_high_%")
            dist_e8 = t.get("dist_ema8w_%")
            ep_tier = t.get("ep_tier", "NONE")
            rs_rat  = t.get("rs_rating")
            ttype   = t.get("trigger_type", "PIVOT")  # PIVOT or ORB

            # Card border — ORB gets distinct blue border
            if ttype == "ORB":
                border = "var(--blue)"
            elif score >= 10: border = "var(--green)"
            elif score >= 7:  border = "var(--accent)"
            else:             border = "var(--border)"

            dir_color = "var(--green)" if d == "bullish" else "var(--red)"
            dir_label = "▲ BULL" if d == "bullish" else "▼ BEAR"
            conv_color = {"HIGH": "var(--green)", "MED": "var(--accent)", "LOW": "var(--muted)"}.get(conv, "var(--muted)")

            # R-ratio display
            if r_ratio is None:
                r_display, r_color = "—", "var(--muted)"
            elif r_ratio >= R_ELITE:
                r_display, r_color = f"{r_ratio:.1f}R ⭐", "var(--green)"
            elif r_ratio >= R_GOOD:
                r_display, r_color = f"{r_ratio:.1f}R", "var(--green)"
            elif r_ratio >= R_MINIMUM:
                r_display, r_color = f"{r_ratio:.1f}R", "var(--accent)"
            else:
                r_display, r_color = f"{r_ratio:.1f}R ⚠", "var(--red)"

            rvol_color = "var(--green)" if rvol >= 2.0 else "var(--accent)" if rvol >= 1.5 else "var(--muted)"
            sh_str   = f"{dist_sh:+.1f}%" if dist_sh is not None else "—"
            sh_color = "var(--green)" if (dist_sh is not None and dist_sh >= -1) else "var(--muted)"
            e8_str   = f"{dist_e8:+.1f}%" if dist_e8 is not None else "—"
            e8_color = "var(--green)" if (dist_e8 is not None and 0 < dist_e8 < 8) else "var(--muted)"

            ep_badge = ""
            if ep_tier == "STRONG":   ep_badge = "<span style='color:var(--accent);font-size:.6rem;font-family:var(--display);'>🚀 EP STRONG</span>"
            elif ep_tier == "STANDARD": ep_badge = "<span style='color:var(--green);font-size:.6rem;font-family:var(--display);'>✅ EP</span>"

            # ── ORB-specific extra row ─────────────────────────────────────
            orb_row = ""
            if ttype == "ORB":
                or_h = t.get("or_high", 0)
                or_l = t.get("or_low", 0)
                or_r = t.get("or_range", 0)
                brk  = t.get("breakout_pct", 0)
                orb_row = (
                    f"<div style='background:rgba(91,141,238,0.08);border-left:2px solid var(--blue);"
                    f"padding:.5rem .8rem;margin:.5rem 0;font-family:var(--mono);font-size:.75rem;'>"
                    f"<span style='color:var(--blue);font-family:var(--display);font-size:.55rem;"
                    f"letter-spacing:.12em;'>🎯 OPENING RANGE BREAKOUT</span>&nbsp;&nbsp;"
                    f"OR: ${or_l:.2f} – ${or_h:.2f} ({or_r:.2f} range) &nbsp;·&nbsp;"
                    f"Breakout: <span style='color:{dir_color};font-weight:700;'>+{brk:.1f}%</span>"
                    f"</div>"
                )

            tf_label = t.get("timeframe", "")
            streak_info = f"Streak {t.get('streak_len','')} bars" if ttype != "ORB" else "Opening Range Break"

            st.markdown(f"""
<div style='background:linear-gradient(135deg,#0f0f17 0%,#09090e 100%);
     border:1px solid var(--border);border-left:3px solid {border};
     padding:1.2rem;margin-bottom:.8rem;border-radius:4px;'>

  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.8rem;'>
    <div>
      <div class='t-ticker'>{t.get("ticker","")}</div>
      <div class='t-meta' style='margin-top:.2rem;'>
        {t.get("theme","")} · {t.get("industry","")} · {streak_info} {ep_badge}
      </div>
    </div>
    <div style='text-align:right;display:flex;flex-direction:column;gap:.3rem;align-items:flex-end;'>
      <div style='font-family:var(--display);font-size:.65rem;font-weight:700;color:{conv_color};letter-spacing:.1em;'>{conv}</div>
      <div style='font-family:var(--display);font-size:.65rem;color:{dir_color};font-weight:700;'>{dir_label} · {tf_label}</div>
      <div style='font-family:var(--display);font-size:.55rem;color:{"var(--green)" if score>=10 else "var(--blue)" if ttype=="ORB" else "var(--accent)"};'>SCORE {score}</div>
    </div>
  </div>

  {orb_row}

  <div class='t-grid' style='margin-bottom:.6rem;'>
    <div><div class='t-field-lbl'>Trigger Close</div><div class='t-field-val'>${t.get("trigger_close",0):.2f}</div></div>
    <div><div class='t-field-lbl'>Stop Level</div><div class='t-field-val red'>${t.get("stop_level",0):.2f}</div></div>
    <div><div class='t-field-lbl'>R-Ratio</div><div class='t-field-val' style='color:{r_color};font-weight:700;'>{r_display}</div></div>
    <div><div class='t-field-lbl'>RVOL</div><div class='t-field-val' style='color:{rvol_color};'>{rvol:.2f}×</div></div>
  </div>

  <div class='t-grid' style='margin-bottom:.6rem;padding-top:.5rem;border-top:1px solid var(--border2);'>
    <div><div class='t-field-lbl'>vs Session High</div><div class='t-field-val' style='color:{sh_color};'>{sh_str}</div></div>
    <div><div class='t-field-lbl'>vs 8W EMA</div><div class='t-field-val' style='color:{e8_color};'>{e8_str}</div></div>
    <div><div class='t-field-lbl'>ATR (14d)</div><div class='t-field-val'>${atr:.2f}</div></div>
    <div><div class='t-field-lbl'>RS Rating</div><div class='t-field-val {"green" if (rs_rat or 0)>=80 else "amber" if (rs_rat or 0)>=60 else ""}'>{rs_rat or "—"}</div></div>
  </div>

  <div style='font-family:var(--mono);font-size:.7rem;color:var(--muted);padding-top:.4rem;border-top:1px solid var(--border2);'>
    {t.get("entry_note","")} · Stage W{t.get("weekly_stage","")} · Trend {t.get("trend_template","")}/8 · BBUW D{t.get("daily_bbuw",0):.0f}/W{t.get("weekly_bbuw",0):.0f} · {str(t.get("trigger_time",""))[:19]} UTC
  </div>
</div>
            """, unsafe_allow_html=True)

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
            stage_p = st.multiselect("Weekly Stage",[1,2,3,4],default=[1,2],key="dl_stage")
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
        if stage_p and "weekly_stage"   in df.columns: df = df[df["weekly_stage"].isin(stage_p)]
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
            "ticker", "conviction", "8W Pivot", "ep_tier", "ep_score",
            "theme", "industry", "industry_rank", "theme_rank",
            "weekly_stage", "trend_template",
            "weekly_bbuw", "daily_bbuw",
            "ema8", "pct_from_ema8",
            "near_resistance", "resistance_level", "resistance_distance_pct",
        ] if c in df.columns]

        conviction_colors = {
            "HIGH": "background-color:#0d3320;color:#00ff88;font-weight:700",
            "MED":  "background-color:#1a1408;color:#FFA500;font-weight:700",
            "LOW":  "background-color:#1a0e0e;color:#cc4444",
        }
        ep_tier_colors = {
            "STRONG":   "background-color:#1a1200;color:#FFA500;font-weight:700",
            "STANDARD": "background-color:#0d3320;color:#00ff88;font-weight:700",
            "WATCH":    "background-color:#111a14;color:#5b8dee",
            "NONE":     "background-color:#111111;color:#555",
        }
        def dl_bg(v):
            if pd.isna(v) or v is None: return "background-color:#111111;color:#555"
            try:
                v = float(v)
                if v >= 80: return "background-color:#0d3320;color:#00ff88;font-weight:700"
                if v >= 60: return "background-color:#111a14;color:#00aa55"
                if v >= 40: return "background-color:#111111;color:#cccccc"
                return "background-color:#1a0e0e;color:#cc4444"
            except: return ""
        dl_styled = (
            df[dcols].style
            .map(lambda v: conviction_colors.get(str(v), ""), subset=["conviction"] if "conviction" in df.columns else [])
            .map(lambda v: ep_tier_colors.get(str(v), ""), subset=["ep_tier"] if "ep_tier" in dcols else [])
            .map(dl_bg, subset=[c for c in ["weekly_bbuw","daily_bbuw","trend_template","industry_rank","ep_score"] if c in dcols])
            .set_properties(**{
                "background-color": "#111111",
                "color": "#cccccc",
                "border": "1px solid #2a2a2a",
                "font-family": "Fira Code, monospace",
                "font-size": "12px",
                "padding": "5px 10px",
            })
            .set_properties(subset=["ticker"] if "ticker" in dcols else [], **{
                "color": "#FFA500", "font-family": "Orbitron, monospace",
                "font-size": "11px", "font-weight": "700",
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
            ])
        )
        st.dataframe(dl_styled, use_container_width=True, height=550, hide_index=True)

        # ── Industry Clustering Charts ─────────────────────────────────────────
        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Industry Clustering</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)

        if "industry" in df.columns and not df.empty:
            ic1, ic2 = st.columns([1, 1])

            # Count by industry — filter out blanks
            ind_counts = (
                df[df["industry"].str.strip() != ""]["industry"]
                .value_counts()
                .reset_index()
            )
            ind_counts.columns = ["Industry", "Count"]

            # Count by theme
            theme_counts = (
                df[df["theme"].str.strip() != ""]["theme"]
                .value_counts()
                .reset_index()
            )
            theme_counts.columns = ["Theme", "Count"]

            with ic1:
                if not ind_counts.empty:
                    # Horizontal bar chart — easier to read with long industry names
                    fig_ind = px.bar(
                        ind_counts.head(20),
                        x="Count", y="Industry",
                        orientation="h",
                        color="Count",
                        color_continuous_scale=["#1a1a2e", "#f5a623", "#4caf7d"],
                        title=f"Industry Breakdown — Daily Watchlist ({len(df)} names)",
                    )
                    fig_ind.update_layout(
                        **PL,
                        showlegend=False,
                        coloraxis_showscale=False,
                        height=max(300, len(ind_counts.head(20)) * 28),
                    )
                    fig_ind.update_yaxes(categoryorder="total ascending")
                    st.plotly_chart(fig_ind, use_container_width=True)

            with ic2:
                if not theme_counts.empty:
                    # Donut chart for themes (fewer categories)
                    fig_theme = px.pie(
                        theme_counts,
                        values="Count",
                        names="Theme",
                        hole=0.55,
                        title=f"Theme Breakdown — Daily Watchlist",
                        color_discrete_sequence=[
                            "#f5a623", "#4caf7d", "#5b8dee", "#e05c5c",
                            "#c97d1e", "#9b7ed1", "#2ecc71", "#e74c3c",
                            "#3498db", "#f39c12", "#1abc9c", "#e91e63",
                        ],
                    )
                    fig_theme.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                        textfont=dict(family="Fira Code, monospace", size=10),
                    )
                    fig_theme.update_layout(
                        **PL,
                        height=max(300, len(ind_counts.head(20)) * 28),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_theme, use_container_width=True)

            # Clustering alert — flag any industry with ≥ 3 names
            clustered = ind_counts[ind_counts["Count"] >= 3].sort_values("Count", ascending=False)
            if not clustered.empty:
                cluster_items = "  ·  ".join(
                    f"<span style='color:var(--accent);font-weight:700;'>{r['Industry']}</span> "
                    f"<span style='color:var(--text);'>({r['Count']})</span>"
                    for _, r in clustered.iterrows()
                )
                st.markdown(f"""
<div style='background:rgba(245,166,35,0.08);border-left:3px solid var(--accent);
     padding:.7rem 1rem;font-family:var(--mono);font-size:.78rem;margin-top:.5rem;'>
  <span style='font-family:var(--display);font-size:.55rem;color:var(--muted);
       letter-spacing:.15em;text-transform:uppercase;'>Cluster Alert — Industries with ≥ 3 names: </span>
  {cluster_items}
</div>
                """, unsafe_allow_html=True)

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
      <div class='t-meta'>{theme} · Rank {theme_rank} · {tier_emoji} {pivot_tier} · Stage W{r.get("weekly_stage","")}</div>
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

        # ── RS Rating (1-99 percentile, IBD-style) ────────────────────────────
        # Computed on the FULL weekly universe BEFORE any tier/filter is applied
        # so the percentile reflects true relative strength across all Stage 1/2 names.
        # 99 = strongest RS vs SPY, 1 = weakest. Matches IBD RS Rating convention.
        if "bbuw_components" in wdf.columns:
            wdf["rs_vs_spy_score"] = wdf["bbuw_components"].apply(
                lambda c: c.get("rs_vs_spy", 50) if isinstance(c, dict) else 50
            )
            n_total = len(wdf)
            ordinal_rank = wdf["rs_vs_spy_score"].rank(
                ascending=True, method="average", na_option="bottom"
            )
            # Convert to 1-99 scale: percentile = rank / total * 99
            wdf["rs_rating"] = (ordinal_rank / n_total * 99).round(0).clip(1, 99).astype(int)
        else:
            wdf["rs_vs_spy_score"] = 50
            wdf["rs_rating"] = 50

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
            "ticker", "Stage", "8W Pivot", "rs_rating", "ema8", "pct_from_ema8",
            "ema8_rising", "pivot_8w_volume_spike",
            "trend_template_score", "bbuw_score",
        ] if c in wdf.columns]

        wk_styled = (
            wdf[wd].style
            .set_properties(**{
                "background-color": "#111111",
                "color": "#cccccc",
                "border": "1px solid #2a2a2a",
                "font-family": "Fira Code, monospace",
                "font-size": "12px",
                "padding": "5px 10px",
            })
            .set_properties(subset=["ticker"] if "ticker" in wdf.columns else [], **{
                "color": "#FFA500", "font-family": "Orbitron, monospace",
                "font-size": "11px", "font-weight": "700",
            })
            .map(lambda v: (
                "background-color:#1a1200;color:#FFA500;font-weight:700" if v == "STRONG" else
                "background-color:#0d3320;color:#00ff88;font-weight:700" if v == "STANDARD" else
                "background-color:#1a1408;color:#cc8800" if v == "WEAK" else
                "background-color:#111111;color:#5b8dee" if v == "PROXIMITY" else ""
            ), subset=["8W Pivot"] if "8W Pivot" in wdf.columns else [])
            .map(lambda v: (
                "background-color:#0d3320;color:#00ff88;font-weight:700" if isinstance(v,float) and v>=80 else
                "background-color:#111a14;color:#00aa55" if isinstance(v,float) and v>=60 else
                "background-color:#1a0e0e;color:#cc4444" if isinstance(v,float) and v<40 else ""
            ), subset=[c for c in ["bbuw_score","rs_rating"] if c in wdf.columns])
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
            ])
        )
        st.dataframe(wk_styled, use_container_width=True, height=500, hide_index=True)

        # ── Industry Clustering — Weekly Screen ───────────────────────────────
        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Industry Clustering — Stage 1/2 Names</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)

        # For weekly screen we pull industry from sector_themes via daily data
        # The wdf has "ticker" — look up industries from daily_data if available
        daily_ind_lookup = {
            e["ticker"]: e.get("industry", "")
            for e in daily_data.get("entries", [])
            if e.get("industry", "")
        }
        # Also check if weekly entries have industry from being in the full results
        weekly_industries = []
        for _, row in wdf.iterrows():
            ticker = row.get("ticker", "")
            ind = daily_ind_lookup.get(ticker, "")
            if ind:
                weekly_industries.append(ind)

        if weekly_industries:
            wc1, wc2 = st.columns([1, 1])
            wind_counts = (
                pd.Series(weekly_industries)
                .value_counts()
                .reset_index()
            )
            wind_counts.columns = ["Industry", "Count"]

            with wc1:
                fig_wind = px.bar(
                    wind_counts.head(20),
                    x="Count", y="Industry",
                    orientation="h",
                    color="Count",
                    color_continuous_scale=["#1a1a2e", "#f5a623", "#4caf7d"],
                    title=f"Industry Breakdown — Weekly Screen ({len(wdf)} names)",
                )
                fig_wind.update_layout(
                    **PL,
                    showlegend=False,
                    coloraxis_showscale=False,
                    height=max(300, len(wind_counts.head(20)) * 28),
                )
                fig_wind.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig_wind, use_container_width=True)

            with wc2:
                # Map industries to themes for the donut
                theme_lookup = {
                    e["ticker"]: e.get("theme", "Unclassified")
                    for e in daily_data.get("entries", [])
                }
                weekly_themes = [
                    theme_lookup.get(row.get("ticker", ""), "Unclassified")
                    for _, row in wdf.iterrows()
                ]
                wtheme_counts = (
                    pd.Series(weekly_themes)
                    .value_counts()
                    .reset_index()
                )
                wtheme_counts.columns = ["Theme", "Count"]
                wtheme_counts = wtheme_counts[wtheme_counts["Theme"] != "Unclassified"]

                if not wtheme_counts.empty:
                    fig_wtheme = px.pie(
                        wtheme_counts,
                        values="Count",
                        names="Theme",
                        hole=0.55,
                        title="Theme Breakdown — Weekly Screen",
                        color_discrete_sequence=[
                            "#f5a623", "#4caf7d", "#5b8dee", "#e05c5c",
                            "#c97d1e", "#9b7ed1", "#2ecc71", "#e74c3c",
                            "#3498db", "#f39c12", "#1abc9c", "#e91e63",
                        ],
                    )
                    fig_wtheme.update_traces(
                        textposition="inside",
                        textinfo="percent+label",
                        textfont=dict(family="Fira Code, monospace", size=10),
                    )
                    fig_wtheme.update_layout(
                        **PL,
                        height=max(300, len(wind_counts.head(20)) * 28),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_wtheme, use_container_width=True)

            # Cluster alert
            wclustered = wind_counts[wind_counts["Count"] >= 3].sort_values("Count", ascending=False)
            if not wclustered.empty:
                w_items = "  ·  ".join(
                    f"<span style='color:var(--accent);font-weight:700;'>{r['Industry']}</span> "
                    f"<span style='color:var(--text);'>({r['Count']})</span>"
                    for _, r in wclustered.iterrows()
                )
                st.markdown(f"""
<div style='background:rgba(245,166,35,0.08);border-left:3px solid var(--accent);
     padding:.7rem 1rem;font-family:var(--mono);font-size:.78rem;margin-top:.5rem;'>
  <span style='font-family:var(--display);font-size:.55rem;color:var(--muted);
       letter-spacing:.15em;text-transform:uppercase;'>Cluster Alert — Industries with ≥ 3 names: </span>
  {w_items}
</div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Industry data appears once daily screener has run.")

        # ── Episodic Pivot Radar ────────────────────────────────────────────────
        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Episodic Pivot Radar — Weekly Re-Rating Events</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)

        ep_events_list = ep_data.get("events", [])
        if not ep_events_list:
            st.caption("No EP data yet. Run the Weekly Screen schedule to populate.")
        else:
            st.caption(
                f"Generated: {ep_data.get('generated_at','—')[:16]} · "
                f"{len(ep_events_list)} EP events detected · "
                f"Score = Weekly % × Relative Volume · Manual catalyst check required"
            )

            # KPI strip
            n_strong   = sum(1 for e in ep_events_list if e.get("ep_tier") == "STRONG")
            n_standard = sum(1 for e in ep_events_list if e.get("ep_tier") == "STANDARD")
            n_watch    = sum(1 for e in ep_events_list if e.get("ep_tier") == "WATCH")
            top_score  = ep_events_list[0].get("ep_score", 0) if ep_events_list else 0

            st.markdown(f"""
<div class='kpi-strip' style='grid-template-columns:repeat(4,1fr);'>
  <div class='kpi-cell amber'>
    <div class='kpi-lbl'>🚀 Strong EPs</div>
    <div class='kpi-val amber'>{n_strong}</div>
    <div class='kpi-sub'>≥25% · ≥2.0× vol</div>
  </div>
  <div class='kpi-cell green'>
    <div class='kpi-lbl'>✅ Standard EPs</div>
    <div class='kpi-val green'>{n_standard}</div>
    <div class='kpi-sub'>≥15% · ≥1.5× vol</div>
  </div>
  <div class='kpi-cell blue'>
    <div class='kpi-lbl'>👀 Watch</div>
    <div class='kpi-val'>{n_watch}</div>
    <div class='kpi-sub'>≥8% · ≥1.3× vol</div>
  </div>
  <div class='kpi-cell amber'>
    <div class='kpi-lbl'>Top EP Score</div>
    <div class='kpi-val amber'>{top_score:.0f}</div>
    <div class='kpi-sub'>{ep_events_list[0]["ticker"] if ep_events_list else "—"}</div>
  </div>
</div>
            """, unsafe_allow_html=True)

            # Filter controls
            ep_f1, ep_f2 = st.columns([1, 2])
            with ep_f1:
                ep_tier_filter = st.multiselect(
                    "EP Tier",
                    ["STRONG", "STANDARD", "WATCH"],
                    default=["STRONG", "STANDARD"],
                    key="ep_tier_filter",
                )
            with ep_f2:
                ep_stage_only = st.checkbox(
                    "Stage 1/2 only (overlaps with weekly screen)",
                    value=False,
                    key="ep_stage_filter",
                    help="Filter to EPs that also passed the Stage + BBUW filter"
                )

            ep_filtered = [
                e for e in ep_events_list
                if e.get("ep_tier") in ep_tier_filter
                and (not ep_stage_only or e.get("stage") in [1, 2])
            ]

            # EP table
            if ep_filtered:
                ep_df = pd.DataFrame([{
                    "Ticker":     e["ticker"],
                    "EP Tier":    e.get("ep_tier", "—"),
                    "EP Score":   e.get("ep_score", 0),
                    "Week %":     e.get("ep_week_pct"),
                    "RVOL":       e.get("ep_rvol"),
                    "4W %":       e.get("ep_4w_pct"),
                    ">50W SMA":   "✓" if e.get("ep_above_50sma") else "✗",
                    "Stage":      e.get("stage", "—"),
                    "BBUW":       e.get("bbuw_score"),
                    "8W Pivot":   e.get("pivot_8w_tier", "—"),
                } for e in ep_filtered])

                def ep_tier_bg(v):
                    if v == "STRONG":   return "background-color:#1a1200;color:#FFA500;font-weight:700"
                    if v == "STANDARD": return "background-color:#0d3320;color:#00ff88;font-weight:700"
                    return "background-color:#111111;color:#5b8dee"

                def ep_score_bg(v):
                    try:
                        v = float(v)
                        if v >= 100: return "background-color:#1a1200;color:#FFA500;font-weight:700"
                        if v >= 50:  return "background-color:#0a2a1a;color:#00cc66"
                        if v >= 25:  return "background-color:#111a14;color:#00aa55"
                        return "background-color:#111111;color:#cccccc"
                    except: return ""

                def ep_pct_bg(v):
                    if v is None or pd.isna(v): return "color:#555"
                    if v >= 25:  return "background-color:#0d3320;color:#00ff88;font-weight:700"
                    if v >= 15:  return "background-color:#111a14;color:#00aa55"
                    if v >= 0:   return "background-color:#111111;color:#aaccaa"
                    return "background-color:#1a0e0e;color:#cc4444"

                ep_styled = (
                    ep_df.style
                    .map(ep_tier_bg, subset=["EP Tier"])
                    .map(ep_score_bg, subset=["EP Score"])
                    .map(ep_pct_bg, subset=["Week %", "4W %"])
                    .format({
                        "EP Score": "{:.1f}",
                        "Week %":   lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                        "4W %":     lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                        "RVOL":     lambda x: f"{x:.2f}×" if pd.notna(x) else "—",
                        "BBUW":     lambda x: f"{x:.0f}" if pd.notna(x) else "—",
                    })
                    .set_properties(**{
                        "background-color": "#111111",
                        "color": "#cccccc",
                        "border": "1px solid #2a2a2a",
                        "font-family": "Fira Code, monospace",
                        "font-size": "12px",
                        "padding": "5px 10px",
                    })
                    .set_properties(subset=["Ticker"], **{
                        "color": "#FFA500", "font-family": "Orbitron, monospace",
                        "font-size": "11px", "font-weight": "700",
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
                    ])
                )
                st.dataframe(ep_styled, use_container_width=True,
                             height=min(600, 50 + len(ep_filtered) * 35),
                             hide_index=True)

                st.markdown("""
<div style='font-family:var(--mono);font-size:.72rem;color:var(--muted);
     border-left:2px solid var(--accent);padding:.5rem .8rem;margin-top:.5rem;'>
  <strong style='color:var(--accent);'>EP Score = Weekly % × Relative Volume</strong> — 
  Higher score = stronger re-rating signal. This scanner finds the move.
  <em>Always verify the catalyst manually</em> — earnings beat, contract win, product launch, etc.
  EPs require patience: hold 2–8 weeks if weekly structure (10/20-week SMA) is respected.
</div>
                """, unsafe_allow_html=True)

        # ── Scatter chart ──────────────────────────────────────────────────────
        if "bbuw_score" in wdf.columns and "rs_rating" in wdf.columns:
            st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>BBUW vs RS Rating (Colored by 8W Pivot Tier)</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)
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
                wdf, x="rs_rating", y="bbuw_score",
                text="ticker", color=color_arg,
                hover_data={"rs_rating": True, "rs_vs_spy_score": True,
                            "trend_template_score": True, "bbuw_score": True},
                title="BBUW Score vs RS Rating (1–99, percentile vs SPY over 26W)",
                **color_kwargs,
            )
            fig_s.update_traces(textposition="top center", textfont_size=9)
            fig_s.update_layout(
                **PL, showlegend=(color_arg == "pivot_8w_tier"),
                coloraxis_showscale=False, height=420,
                xaxis_title="RS Rating (1–99  ·  higher = stronger RS vs SPY)",
                yaxis_title="Weekly BBUW",
                legend=dict(font=dict(size=9), title_text="8W Pivot Tier"),
            )
            # RS 80 = @1ChartMaster minimum threshold, RS 90 = elite tier
            fig_s.add_vline(x=80, line_dash="dash", line_color="rgba(245,166,35,0.3)",
                            annotation_text="RS 80", annotation_font_size=9,
                            annotation_font_color="rgba(245,166,35,0.6)")
            fig_s.add_vline(x=90, line_dash="dash", line_color="rgba(76,175,125,0.3)",
                            annotation_text="RS 90", annotation_font_size=9,
                            annotation_font_color="rgba(76,175,125,0.6)")
            fig_s.add_hline(y=60, line_dash="dash", line_color="rgba(245,166,35,0.3)",
                            annotation_text="BBUW 60", annotation_font_size=9,
                            annotation_font_color="rgba(245,166,35,0.6)")
            st.plotly_chart(fig_s, use_container_width=True)

        # ── Industry Rankings ──────────────────────────────────────────────────
        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Industry Rankings (Dynamic · Updated Weekly)</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)

        ind_ranks = industry_ranks.get("ranks", [])
        if not ind_ranks:
            st.info("No industry rank data yet. Run the Weekly Screen schedule to populate.")
        else:
            st.caption(f"Generated: {industry_ranks.get('generated_at','—')} · "
                       f"{len(ind_ranks)} industries ranked by composite score · "
                       f"Replaces static theme_rank in conviction scoring")

            # Top 10 callout strip
            top10 = ind_ranks[:10]
            top10_html = ""
            for r in top10:
                score = r.get("composite_score", 0)
                bar_w = int(score)
                top10_html += f"""
<div style='display:flex;align-items:center;gap:.8rem;padding:.35rem 0;border-bottom:1px solid var(--border2);font-family:var(--mono);font-size:.78rem;'>
  <div style='font-family:var(--display);font-size:.65rem;font-weight:700;color:var(--accent);width:2rem;text-align:right;'>#{r['rank']}</div>
  <div style='flex:1;color:var(--text);'>{r['industry']}</div>
  <div style='width:120px;background:var(--panel);border-radius:2px;overflow:hidden;'>
    <div style='width:{bar_w}%;height:6px;background:linear-gradient(90deg,var(--accent2),var(--accent));border-radius:2px;'></div>
  </div>
  <div style='color:var(--accent);font-weight:700;width:3rem;text-align:right;'>{score:.0f}</div>
  <div style='color:var(--muted);width:2rem;text-align:right;'>{r['ticker_count']}t</div>
</div>"""

            st.markdown(f"""
<div style='background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--accent);padding:1rem;margin-bottom:1rem;'>
  <div style='font-family:var(--display);font-size:.55rem;color:var(--muted);letter-spacing:.15em;text-transform:uppercase;margin-bottom:.6rem;'>Top 10 Industries This Week</div>
  {top10_html}
</div>
            """, unsafe_allow_html=True)

            # Full sortable table
            ir_df = pd.DataFrame(ind_ranks)
            ir_display_cols = [c for c in [
                "rank", "industry", "ticker_count", "composite_score",
                "avg_bbuw", "avg_rs_component", "pct_stage12",
                "pct_trend_template", "pct_8w_active",
            ] if c in ir_df.columns]
            st.dataframe(ir_df[ir_display_cols], use_container_width=True,
                         height=500, hide_index=True)

            # Bar chart of top 20
            top20 = ir_df.head(20)
            fig_ir = px.bar(
                top20, x="industry", y="composite_score",
                color="composite_score",
                color_continuous_scale=["#e05c5c", "#f5a623", "#4caf7d"],
                range_color=[40, 80],
                title="Top 20 Industries — Composite Rank Score",
            )
            fig_ir.update_layout(
                **PL, showlegend=False, coloraxis_showscale=False,
                height=320, xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_ir, use_container_width=True)

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

# ─── TAB VOLUME LOG ───────────────────────────────────────────────────────────
with tab_vol:
    vol_events = volume_data.get("events", [])

    st.caption(
        f"Generated: {volume_data.get('generated_at', '—')} · "
        f"{len(vol_events)} surge events in log · "
        f"Universe: Stage 1/2 names from weekly screen · "
        f"Retained 180 days"
    )

    if not vol_events:
        st.info("No volume surge data yet. Trigger the Daily Screen schedule to populate.")
    else:
        # ── Summary KPIs ─────────────────────────────────────────────────────
        n_daily   = sum(1 for e in vol_events if e.get("surge_type") == "DAILY")
        n_weekly  = sum(1 for e in vol_events if e.get("surge_type") == "WEEKLY")
        n_fresh   = sum(1 for e in vol_events if e.get("is_fresh"))
        n_up      = sum(1 for e in vol_events if e.get("price_chg_since_pct", 0) > 0)

        st.markdown(f"""
<div class='kpi-strip' style='grid-template-columns:repeat(4,1fr);'>
  <div class='kpi-cell amber'>
    <div class='kpi-lbl'>📊 Total Events</div>
    <div class='kpi-val amber'>{len(vol_events)}</div>
    <div class='kpi-sub'>{n_daily} daily · {n_weekly} weekly</div>
  </div>
  <div class='kpi-cell green'>
    <div class='kpi-lbl'>✅ Still Fresh</div>
    <div class='kpi-val green'>{n_fresh}</div>
    <div class='kpi-sub'>within staleness window</div>
  </div>
  <div class='kpi-cell blue'>
    <div class='kpi-lbl'>📈 Up Since Surge</div>
    <div class='kpi-val'>{n_up}</div>
    <div class='kpi-sub'>price higher than surge day</div>
  </div>
  <div class='kpi-cell red'>
    <div class='kpi-lbl'>📉 Down Since Surge</div>
    <div class='kpi-val red'>{len(vol_events) - n_up}</div>
    <div class='kpi-sub'>price below surge day</div>
  </div>
</div>
        """, unsafe_allow_html=True)

        # ── Filters ───────────────────────────────────────────────────────────
        vf1, vf2, vf3, vf4 = st.columns([1, 1, 1, 1])
        with vf1:
            type_filter = st.multiselect(
                "Surge type", ["DAILY", "WEEKLY"],
                default=["DAILY", "WEEKLY"], key="vol_type"
            )
        with vf2:
            fresh_only = st.checkbox("Fresh only", value=False, key="vol_fresh",
                                      help="Hide events older than staleness window")
        with vf3:
            pivot_filter = st.multiselect(
                "8W Pivot tier",
                ["STRONG", "STANDARD", "PROXIMITY", "WEAK", "NONE"],
                default=[], key="vol_pivot",
                help="Empty = show all"
            )
        with vf4:
            theme_filter_vol = st.multiselect(
                "Theme",
                sorted({e.get("theme", "Unclassified") for e in vol_events}),
                key="vol_theme"
            )

        # Apply filters
        filtered_events = [
            e for e in vol_events
            if e.get("surge_type") in type_filter
            and (not fresh_only or e.get("is_fresh"))
            and (not pivot_filter or e.get("pivot_8w_tier") in pivot_filter)
            and (not theme_filter_vol or e.get("theme") in theme_filter_vol)
        ]

        st.caption(f"Showing {len(filtered_events)} events")

        # ── Render cards ──────────────────────────────────────────────────────
        for e in filtered_events:
            surge_type  = e.get("surge_type", "")
            is_fresh    = e.get("is_fresh", False)
            price_chg   = e.get("price_chg_since_pct", 0) or 0
            rvol        = e.get("rvol", 0) or 0
            pivot_tier  = e.get("pivot_8w_tier", "NONE")
            stage       = e.get("stage")
            bbuw        = e.get("bbuw_score", 0) or 0

            # Card color by type
            border_color = "var(--accent)" if surge_type == "WEEKLY" else "var(--blue)"
            type_label   = "🗓 WEEKLY SURGE" if surge_type == "WEEKLY" else "📅 DAILY SURGE"
            type_color   = "var(--accent)" if surge_type == "WEEKLY" else "var(--blue)"

            # Price change color
            chg_color = "var(--green)" if price_chg >= 0 else "var(--red)"
            chg_sign  = "+" if price_chg >= 0 else ""

            # Fresh badge
            freshness = "✅ FRESH" if is_fresh else "⏳ AGING"
            fresh_col = "var(--green)" if is_fresh else "var(--muted)"

            # Staleness details
            if surge_type == "WEEKLY":
                age_str = f"{e.get('weeks_since_surge', 0)}w ago"
            else:
                age_str = f"{e.get('days_since_surge', 0)}d ago"

            # Candle quality
            candle_pos = e.get("candle_position", 0.5) or 0.5
            if candle_pos >= 0.8:
                candle_lbl = "Closed strong (top 20%)"
            elif candle_pos >= 0.6:
                candle_lbl = "Closed upper half"
            elif candle_pos >= 0.4:
                candle_lbl = "Mid-range close"
            else:
                candle_lbl = "Closed weak"

            # 8W pivot
            pivot_emoji = {
                "STRONG": "🔥", "STANDARD": "✅",
                "PROXIMITY": "👀", "WEAK": "⚠️", "NONE": "—"
            }.get(pivot_tier, "—")

            stage_labels = {1: "1·Basing", 2: "2·Advancing",
                            3: "3·Topping", 4: "4·Declining"}
            stage_lbl = stage_labels.get(stage, "—")

            st.markdown(f"""
<div style='background:linear-gradient(135deg,#0f0f17 0%,#09090e 100%);
     border:1px solid var(--border);border-left:3px solid {border_color};
     padding:1.1rem;margin-bottom:.8rem;border-radius:4px;'>

  <!-- Header row -->
  <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
    <div>
      <div class='t-ticker'>{e.get("ticker","")}</div>
      <div class='t-meta'>{e.get("theme","")} · {e.get("industry","")} · Stage {stage_lbl}</div>
    </div>
    <div style='text-align:right;'>
      <div style='font-family:var(--display);font-size:.65rem;font-weight:700;
           color:{type_color};letter-spacing:.12em;'>{type_label}</div>
      <div style='font-family:var(--display);font-size:.55rem;
           color:{fresh_col};margin-top:.25rem;'>{freshness} · {age_str}</div>
    </div>
  </div>

  <!-- Metrics grid -->
  <div class='t-grid' style='margin-top:.9rem;'>
    <div>
      <div class='t-field-lbl'>Surge Date</div>
      <div class='t-field-val'>{e.get("surge_date","—")}</div>
    </div>
    <div>
      <div class='t-field-lbl'>RVOL (vs avg)</div>
      <div class='t-field-val' style='color:{type_color};'>{rvol:.2f}×</div>
    </div>
    <div>
      <div class='t-field-lbl'>Surge Price</div>
      <div class='t-field-val'>${e.get("surge_price",0):.2f}</div>
    </div>
    <div>
      <div class='t-field-lbl'>Price Now vs Surge</div>
      <div class='t-field-val' style='color:{chg_color};'>
        ${e.get("current_price",0):.2f} ({chg_sign}{price_chg:.1f}%)
      </div>
    </div>
  </div>

  <!-- Secondary row -->
  <div class='t-grid' style='margin-top:.6rem;'>
    <div>
      <div class='t-field-lbl'>Candle Quality</div>
      <div class='t-field-val' style='font-size:.8rem;'>{candle_lbl}</div>
    </div>
    <div>
      <div class='t-field-lbl'>21 EMA Dist</div>
      <div class='t-field-val'>{e.get("pct_from_21ema",0):+.1f}%</div>
    </div>
    <div>
      <div class='t-field-lbl'>8W Pivot</div>
      <div class='t-field-val'>{pivot_emoji} {pivot_tier}</div>
    </div>
    <div>
      <div class='t-field-lbl'>BBUW Score</div>
      <div class='t-field-val blue'>{bbuw:.0f}</div>
    </div>
  </div>

  <!-- Interpretation note -->
  <div style='margin-top:.7rem;font-family:var(--mono);font-size:.72rem;
       color:var(--muted);border-top:1px solid var(--border2);padding-top:.5rem;'>
    {"🔥 <span style='color:var(--accent);'>WEEKLY surge</span> — rare, high-weight institutional signal. Watch for coil + 30-min pivot to fire." if surge_type == "WEEKLY" else "📅 Daily volume climax — institutional accumulation signal. Monitor for BBUW compression and pivot trigger."}
    &nbsp;|&nbsp; Trend Template: {e.get("trend_template","—")}/8
  </div>
</div>
            """, unsafe_allow_html=True)

        # ── Full table view ───────────────────────────────────────────────────
        st.markdown("<div class='sec-bar'><div class='sec-bar-line'></div><div class='sec-bar-label'>Full Log — Sortable Table</div><div class='sec-bar-line'></div></div>", unsafe_allow_html=True)

        table_cols = [
            "ticker", "surge_type", "surge_date", "rvol", "is_fresh",
            "surge_price", "current_price", "price_chg_since_pct",
            "days_since_surge", "pct_from_21ema",
            "pivot_8w_tier", "bbuw_score", "stage",
            "theme", "industry",
        ]
        vol_df = pd.DataFrame(filtered_events)
        disp_cols = [c for c in table_cols if c in vol_df.columns]
        st.dataframe(vol_df[disp_cols], use_container_width=True,
                     height=400, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-footer'>
  PIVOT SCANNER WAR ROOM · SCAN ALERT ONLY — NOT FINANCIAL ADVICE ·
  @1CHARTMASTER × WEINSTEIN × MINERVINI × @OHIAIN ·
  PRESSURE COOKER · STAGE ANALYSIS · BBUW · 30/65-MIN PIVOTS
</div>
""", unsafe_allow_html=True)
