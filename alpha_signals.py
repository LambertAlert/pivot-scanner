"""
alpha_signals.py — ◈ ALPHA SIGNALS tab (WorldQuant 101 formulaic alphas)
================================================================================
Renders the alpha-signals story view, top to bottom:

  1. VALIDATION PANEL   — composite alpha percentile vs. Burst Readiness Score
  2. CATEGORY OVERVIEW  — 6 taxonomy categories, watchlist-wide averages,
                          expandable to show every alpha in the category
  3. FULL ALPHA REFERENCE — all 57 alphas, filterable by category
  4. TICKER STORIES     — top tickers as cards: 6 category meters + a
                          plain-language Read line
  5. FULL SIGNALS TABLE — composite + 6 category percentiles per ticker
  6. TICKER DRILL-DOWN  — per-ticker narrative: 57-bar profile, combination
                          patterns (rule shown), confirmations/dissents,
                          annotated 57-row table, synthesized read

Taxonomy (MOM/MRV/VOL/CMP/CND/IND) is read from alpha_features.json's
"alpha_meta" block — written by alpha_features.py — so this module needs no
import of the compute pipeline.

Reads:
  data/alpha_features.json   (written by alpha_features.py)
  data/daily_watchlist.json  (written by daily_screener.py)
  data/burst_watch.json      (written by momentum_burst_prep.py)

⚠ CONFIRM BEFORE FIRST RUN
  BURST_WATCH_JSON and BURST_SCORE_FIELD below are a best guess at
  momentum_burst_prep.py's output path and score field. If the Validation
  Panel shows the "not found" warning, check that module's actual output
  and update the two constants — everything else works unchanged.

Integration (dashboard.py):

    try:
        import alpha_signals
        ALPHA_SIGNALS_AVAILABLE = True
    except ImportError:
        ALPHA_SIGNALS_AVAILABLE = False

    # add "◈ ALPHA SIGNALS" to the st.tabs([...]) list, capture as tab_alpha

    with tab_alpha:
        if not ALPHA_SIGNALS_AVAILABLE:
            st.info("alpha_signals.py not found. Add the module to your repo.")
        else:
            alpha_signals.render()
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── CONFIG — confirm the burst constants against momentum_burst_prep.py ─────
ALPHA_FEATURES_JSON  = "data/alpha_features.json"
DAILY_WATCHLIST_JSON = "data/daily_watchlist.json"
BURST_WATCH_JSON     = "data/burst_watch.json"     # ⚠ best guess — confirm path
BURST_SCORE_FIELD    = "readiness_score"           # ⚠ best guess — confirm field

N_STORY_CARDS = 3        # how many top-composite tickers get story cards
EXTREME_HI, EXTREME_LO = 85, 25   # per-alpha extremes for the annotated table
DISSENT_GAP = 35         # pts from category avg to count as a dissent

# ── Category presentation metadata (taxonomy itself comes from the JSON) ────
CAT_ORDER = ["MOM", "MRV", "VOL", "CMP", "CND", "IND"]
CATEGORIES = {
    "MOM": {"name": "Momentum & Trend",          "color": "#f5a623",
            "question": "Is the move continuing, accelerating, or decelerating?"},
    "MRV": {"name": "Mean-Reversion Risk",       "color": "#e05c5c",
            "question": "Is this stretched and due to snap back or stall?"},
    "VOL": {"name": "Volume Confirmation",       "color": "#5b8dee",
            "question": "Is volume validating the price action, or diverging?"},
    "CMP": {"name": "Volatility Compression",    "color": "#4caf7d",
            "question": "Is the range tightening (coiling) or loosening?"},
    "CND": {"name": "Candle & Range Position",   "color": "#c97d1e",
            "question": "Accumulation or distribution within the day's range?"},
    "IND": {"name": "Industry-Relative Strength","color": "#9b7ed1",
            "question": "Beating industry peers, sector moves subtracted out?"},
}

# ── Combination pattern library ─────────────────────────────────────────────
# Each: (id, name, rule text, condition fn on {cat: pct}, narrative, kind)
# kind: "good" (green), "warn" (amber). Inactive patterns render greyed.
PATTERNS = [
    ("COILED_LEADER", "Coiled Leader",
     "MOM ≥ 70 AND CMP ≥ 70 AND IND ≥ 80",
     lambda c: _ge(c, "MOM", 70) and _ge(c, "CMP", 70) and _ge(c, "IND", 80),
     "Accelerating trend inside a tightening range, while leading its industry "
     "peers — the pre-burst profile the Bonde methodology targets.", "good"),
    ("VOLUME_CONFIRMED", "Volume-Confirmed Trend",
     "MOM ≥ 70 AND VOL ≥ 70",
     lambda c: _ge(c, "MOM", 70) and _ge(c, "VOL", 70),
     "Participation is validating the move — the momentum isn't running on air.", "good"),
    ("ACCUMULATION_CLOSES", "Accumulation Closes",
     "CND ≥ 80 AND MRV ≤ 35",
     lambda c: _ge(c, "CND", 80) and _le(c, "MRV", 35),
     "Price keeps finishing near the top of its daily range without getting "
     "statistically stretched — buyers in control into the close.", "good"),
    ("UNCONFIRMED_MOVE", "Unconfirmed Move",
     "MOM ≥ 70 AND VOL ≤ 45",
     lambda c: _ge(c, "MOM", 70) and _le(c, "VOL", 45),
     "Price running ahead of participation — would flag waiting for volume "
     "before adding size.", "warn"),
    ("EXHAUSTION_RISK", "Exhaustion Risk",
     "MRV ≥ 75 AND CND ≤ 30",
     lambda c: _ge(c, "MRV", 75) and _le(c, "CND", 30),
     "Statistically stretched with weak closes — distribution profile; would "
     "flag avoid-chase.", "warn"),
]


def _ge(c, k, v):
    return c.get(k) is not None and c[k] >= v


def _le(c, k, v):
    return c.get(k) is not None and c[k] <= v


def _section(label):
    st.markdown(
        "<div class='sec-bar'><div class='sec-bar-line'></div>"
        "<div class='sec-bar-label'>" + label + "</div>"
        "<div class='sec-bar-line'></div></div>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
def _load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


# ═══════════════════════════════════════════════════════════════════════════
#  DATA ASSEMBLY  (pure — no Streamlit calls; unit-testable)
# ═══════════════════════════════════════════════════════════════════════════

def get_alpha_meta(alpha_data):
    """{int_id: {cat, tier, desc}} from the JSON's alpha_meta block."""
    raw = alpha_data.get("alpha_meta", {})
    return {int(k): v for k, v in raw.items()}


def ticker_alpha_pcts(alpha_data, ticker):
    """{alpha_id: pct} for one ticker."""
    vals = alpha_data.get("features", {}).get(ticker, {})
    out = {}
    for k, v in vals.items():
        if k.endswith("_pct") and v is not None:
            out[int(k.split("_")[1])] = float(v)
    return out


def category_pcts(pcts, meta):
    """{cat: avg pct} from per-alpha percentiles + taxonomy meta."""
    buckets = {c: [] for c in CAT_ORDER}
    for aid, p in pcts.items():
        cat = meta.get(aid, {}).get("cat")
        if cat in buckets:
            buckets[cat].append(p)
    return {c: (float(np.mean(v)) if v else None) for c, v in buckets.items()}


def detect_patterns(cat_pcts):
    """[(id, name, rule, narrative, kind, active_bool)] for every library pattern."""
    return [(pid, name, rule, narr, kind, bool(cond(cat_pcts)))
            for pid, name, rule, cond, narr, kind in PATTERNS]


def find_dissents(pcts, cat_pcts, meta, gap=DISSENT_GAP):
    """Alphas deviating ≥ gap pts from their category average.
    Returns [(aid, pct, cat, gap_pts, direction)] sorted by gap desc."""
    out = []
    for aid, p in pcts.items():
        cat = meta.get(aid, {}).get("cat")
        avg = cat_pcts.get(cat)
        if avg is None:
            continue
        d = p - avg
        if abs(d) >= gap:
            out.append((aid, p, cat, abs(d), "above" if d > 0 else "below"))
    return sorted(out, key=lambda x: -x[3])


def supportive_score(aid, pct, meta):
    """Higher = more supportive of a long setup. MRV is inverted (high MRV = risk)."""
    cat = meta.get(aid, {}).get("cat")
    return 100 - pct if cat == "MRV" else pct


def top_bottom_alphas(pcts, meta, n=5):
    """(top_n, bottom_n) by supportive score: [(aid, pct, cat, desc)]."""
    scored = sorted(
        [(aid, p, meta.get(aid, {}).get("cat", "?"), meta.get(aid, {}).get("desc", ""))
         for aid, p in pcts.items()],
        key=lambda x: -supportive_score(x[0], x[1], meta),
    )
    return scored[:n], scored[-n:][::-1]


def build_read(ticker, cat_pcts, patterns, dissents, meta):
    """Templated two-part synthesized read: (strengths_text, watch_text)."""
    strong = [c for c in CAT_ORDER
              if cat_pcts.get(c) is not None
              and ((c != "MRV" and cat_pcts[c] >= 70)
                   or (c == "MRV" and cat_pcts[c] <= 30))]
    weak = [c for c in CAT_ORDER
            if cat_pcts.get(c) is not None
            and ((c != "MRV" and cat_pcts[c] <= 40)
                 or (c == "MRV" and cat_pcts[c] >= 70))]
    active = [name for _, name, _, _, _, on in patterns if on]
    active_ids = {pid for pid, _, _, _, _, on in patterns if on}

    if strong:
        names = ", ".join(CATEGORIES[c]["name"] for c in strong)
        p1 = names + (" all confirm." if len(strong) > 1 else " confirms.")
    else:
        p1 = "No category is strongly confirming."
    if active:
        p1 += " Active patterns: " + ", ".join(active) + "."

    parts = []
    if weak:
        parts.append("Soft: " + ", ".join(CATEGORIES[c]["name"] for c in weak) + ".")
    below = [d for d in dissents if d[4] == "below"][:2]
    if below:
        parts.append("Dissenting alphas: "
                     + ", ".join("#%d (%d, %s avg %d)" % (a, p, c, cat_pcts.get(c) or 0)
                                 for a, p, c, _, _ in below) + ".")
    if "EXHAUSTION_RISK" in active_ids:
        stance = "The alphas read distribution — avoid chasing."
    elif "UNCONFIRMED_MOVE" in active_ids:
        stance = "The alphas favor waiting for volume to confirm before adding size."
    elif "COILED_LEADER" in active_ids:
        stance = "The strongest combination in the library is active — constructive profile."
    elif len(strong) >= 3:
        stance = "Broadly constructive; no gating pattern active."
    else:
        stance = "Mixed profile — no clear combination read."
    parts.append(stance)
    return p1, " ".join(parts)


def build_signals_df(alpha_data, daily_data, burst_data):
    """One row per ticker: composite + 6 category pcts + merged context.
    Returns (df, meta_info)."""
    features = alpha_data.get("features", {})
    meta = get_alpha_meta(alpha_data)
    if not features:
        return pd.DataFrame(), {"has_burst": False, "as_of": "—"}

    rows = []
    for ticker in features:
        pcts = ticker_alpha_pcts(alpha_data, ticker)
        if not pcts:
            continue
        cats = category_pcts(pcts, meta)
        row = {"ticker": ticker,
               "composite_alpha_pct": float(np.mean(list(pcts.values()))),
               "n_alphas": len(pcts)}
        for c in CAT_ORDER:
            row[c] = cats.get(c)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"has_burst": False, "as_of": alpha_data.get("as_of_date", "—")}

    daily_lookup = {e["ticker"]: e for e in daily_data.get("entries", []) if "ticker" in e}
    df["conviction"] = df["ticker"].map(lambda t: daily_lookup.get(t, {}).get("conviction", "—"))
    df["theme"]      = df["ticker"].map(lambda t: daily_lookup.get(t, {}).get("theme", "—"))
    df["industry"]   = df["ticker"].map(lambda t: daily_lookup.get(t, {}).get("industry", "—"))

    burst_entries = burst_data.get("entries", burst_data.get("watchlist", []))
    burst_lookup = {}
    if isinstance(burst_entries, list):
        for e in burst_entries:
            tk = e.get("ticker")
            if tk and BURST_SCORE_FIELD in e:
                burst_lookup[tk] = e.get(BURST_SCORE_FIELD)
    has_burst = len(burst_lookup) > 0
    df["burst_readiness_score"] = df["ticker"].map(burst_lookup) if has_burst else np.nan

    return df, {"has_burst": has_burst, "as_of": alpha_data.get("as_of_date", "—")}


# ═══════════════════════════════════════════════════════════════════════════
#  HTML BUILDERS  (string concatenation only — no HTML vars nested in
#  f-strings inside st.columns, per the dashboard's hard rule)
# ═══════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
.as-pattern-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:1rem;}
.as-pattern{background:#0f0f17;border:1px solid rgba(245,166,35,0.22);border-left:3px solid #4caf7d;padding:.85rem 1rem;}
.as-pattern.warn{border-left-color:#f5a623;}
.as-pattern.off{border-left-color:rgba(255,255,255,.12);opacity:.5;}
.as-pname{font-family:Orbitron,sans-serif;font-size:.62rem;font-weight:700;letter-spacing:.1em;color:#ede8d9;text-transform:uppercase;display:flex;justify-content:space-between;}
.as-pstat{font-size:.5rem;color:#4caf7d;}
.as-pattern.warn .as-pstat{color:#f5a623;}
.as-pattern.off .as-pstat{color:#7a7060;}
.as-prule{font-family:'Fira Code',monospace;font-size:.62rem;color:#7a7060;margin:.3rem 0 .45rem;}
.as-pnarr{font-family:'Fira Code',monospace;font-size:.72rem;color:#ede8d9;line-height:1.45;opacity:.9;}
.as-story{background:linear-gradient(135deg,#0f0f17 0%,#09090e 100%);border:1px solid rgba(245,166,35,0.22);border-left:3px solid #4caf7d;padding:1.1rem 1.3rem;margin-bottom:1rem;border-radius:4px;}
.as-story.warn{border-left-color:#e05c5c;}
.as-story.watch{border-left-color:#5b8dee;}
.as-tkr{font-family:Orbitron,sans-serif;font-size:1.3rem;font-weight:700;color:#ede8d9;}
.as-meta{font-family:Orbitron,sans-serif;font-size:.5rem;color:#7a7060;text-transform:uppercase;letter-spacing:.1em;margin-top:.15rem;}
.as-meters{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:.8rem 0;}
.as-m{display:flex;flex-direction:column;gap:.3rem;}
.as-mlbl{font-family:Orbitron,sans-serif;font-size:.46rem;letter-spacing:.08em;color:#7a7060;text-transform:uppercase;display:flex;justify-content:space-between;}
.as-mtrack{background:#1a1a1a;border-radius:2px;height:8px;overflow:hidden;}
.as-read{margin-top:.7rem;font-family:'Fira Code',monospace;font-size:.78rem;color:#ede8d9;border-top:1px solid rgba(245,166,35,0.08);padding-top:.6rem;line-height:1.5;}
.as-read b{color:#f5a623;}
.as-agree{background:#0f0f17;border:1px solid rgba(245,166,35,0.22);padding:.9rem 1.1rem;height:100%;}
.as-atitle{font-family:Orbitron,sans-serif;font-size:.55rem;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.55rem;}
.as-aitem{display:flex;gap:.6rem;padding:.3rem 0;border-bottom:1px dashed rgba(255,255,255,.05);font-family:'Fira Code',monospace;font-size:.72rem;color:#ede8d9;line-height:1.4;}
.as-aitem:last-child{border-bottom:none;}
.as-anum{font-family:Orbitron,sans-serif;font-size:.62rem;font-weight:700;min-width:2.8em;}
</style>
"""


def _meter_html(code, val, color):
    v = 0.0 if val is None else max(0.0, min(100.0, float(val)))
    vs = "%.0f" % v
    return ("<div class='as-m'><div class='as-mlbl'>" + code
            + " <span style='color:" + color + ";font-weight:700'>" + vs + "</span></div>"
            + "<div class='as-mtrack'><div style='width:" + vs + "%;height:100%;"
            + "border-radius:2px;background:" + color + ";'></div></div></div>")


def _story_card_html(row, cat_pcts, read1, read2, border_cls):
    meters = "".join(_meter_html(c, cat_pcts.get(c), CATEGORIES[c]["color"])
                     for c in CAT_ORDER)
    meta_line = str(row.get("industry") or "—") + " · " + str(row.get("theme") or "—")
    return ("<div class='as-story " + border_cls + "'>"
            + "<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
            + "<div><div class='as-tkr'>" + str(row["ticker"]) + "</div>"
            + "<div class='as-meta'>" + meta_line + "</div></div>"
            + "<div class='as-meta'>" + str(row.get("conviction") or "—")
            + " · composite " + ("%.0f" % row["composite_alpha_pct"]) + "</div></div>"
            + "<div class='as-meters'>" + meters + "</div>"
            + "<div class='as-read'><b>Read:</b> " + read1 + " " + read2 + "</div></div>")


def _pattern_html(name, rule, narr, kind, active):
    cls = "as-pattern" + ("" if (active and kind == "good") else
                          " warn" if (active and kind == "warn") else " off")
    stat = ("● ACTIVE" if (active and kind == "good") else
            "◆ FLAG" if (active and kind == "warn") else "○ NOT ACTIVE")
    return ("<div class='" + cls + "'><div class='as-pname'>" + name
            + " <span class='as-pstat'>" + stat + "</span></div>"
            + "<div class='as-prule'>RULE: " + rule + "</div>"
            + "<div class='as-pnarr'>" + narr + "</div></div>")


def _agree_item_html(aid, pct, cat, note):
    color = CATEGORIES.get(cat, {}).get("color", "#7a7060")
    return ("<div class='as-aitem'><span class='as-anum' style='color:" + color + ";'>#"
            + str(aid) + "</span><span>" + ("%.0f" % pct) + " — " + note + "</span></div>")


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE STYLING  (matches dashboard.py's conventions)
# ═══════════════════════════════════════════════════════════════════════════

def _pct_bg(v):
    if pd.isna(v):
        return "background-color:#111111;color:#555"
    v = float(v)
    if v >= 80: return "background-color:#0d3320;color:#00ff88;font-weight:700"
    if v >= 60: return "background-color:#111a14;color:#00aa55"
    if v >= 40: return "background-color:#111111;color:#cccccc"
    return "background-color:#1a0e0e;color:#cc4444"


_CONVICTION_COLORS = {
    "HIGH": "background-color:#0d3320;color:#00ff88;font-weight:700",
    "MED":  "background-color:#1a1408;color:#FFA500;font-weight:700",
    "LOW":  "background-color:#1a0e0e;color:#cc4444",
}

_TABLE_STYLES = [
    {"selector": "thead th", "props": [
        ("background-color", "#0a0a0a"), ("color", "#FFA500"),
        ("font-family", "Orbitron, monospace"), ("font-size", "10px"),
        ("letter-spacing", "2px"), ("border-bottom", "2px solid #FFA500"),
        ("padding", "7px 10px")]},
    {"selector": "tbody tr:nth-child(even) td", "props": [("background-color", "#0f0f0f")]},
]

_BASE_PROPS = {
    "background-color": "#111111", "color": "#cccccc",
    "border": "1px solid #2a2a2a", "font-family": "Fira Code, monospace",
    "font-size": "12px", "padding": "5px 10px",
}

_TICKER_PROPS = {
    "color": "#FFA500", "font-family": "Orbitron, monospace",
    "font-size": "11px", "font-weight": "700",
}


def _style_table(df, pct_cols, ticker_col=None, conviction_col=None, fmt_pct=True):
    styled = df.style
    if conviction_col and conviction_col in df.columns:
        styled = styled.map(lambda v: _CONVICTION_COLORS.get(str(v), ""),
                            subset=[conviction_col])
    present_pct = [c for c in pct_cols if c in df.columns]
    if present_pct:
        styled = styled.map(_pct_bg, subset=present_pct)
        if fmt_pct:
            styled = styled.format({c: "{:.0f}" for c in present_pct}, na_rep="—")
    styled = styled.set_properties(**_BASE_PROPS)
    if ticker_col and ticker_col in df.columns:
        styled = styled.set_properties(subset=[ticker_col], **_TICKER_PROPS)
    return styled.set_table_styles(_TABLE_STYLES)


# ═══════════════════════════════════════════════════════════════════════════
#  RENDER SECTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _render_validation(df, meta_info):
    _section("Validation — Alpha Composite vs. Burst Readiness Score")
    if not meta_info["has_burst"]:
        st.warning(
            "Burst Readiness Score not found at `" + BURST_WATCH_JSON + "` (field `"
            + BURST_SCORE_FIELD + "`). Confirm momentum_burst_prep.py's actual "
            "output path/field and update the CONFIG constants at the top of "
            "alpha_signals.py — this panel will populate automatically.")
        return
    valid = df.dropna(subset=["composite_alpha_pct", "burst_readiness_score"])
    if len(valid) < 5:
        st.info("Not enough overlapping tickers yet for a meaningful correlation.")
        return

    corr = valid["composite_alpha_pct"].corr(valid["burst_readiness_score"])
    strength = ("Weak/none" if abs(corr) < 0.2 else
                "Moderate" if abs(corr) < 0.5 else "Strong")
    v1, v2, v3 = st.columns(3)
    v1.metric("Correlation (Pearson r)", "%+.2f" % corr)
    v2.metric("Tickers compared", len(valid))
    v3.metric("Read", strength)

    fig = go.Figure()
    for conv, color in [("HIGH", "#4caf7d"), ("MED", "#f5a623"), ("LOW", "#7a7060")]:
        sub = valid[valid["conviction"] == conv]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["composite_alpha_pct"], y=sub["burst_readiness_score"],
            mode="markers", name=conv,
            marker=dict(color=color, size=8, opacity=0.75), text=sub["ticker"],
            hovertemplate="%{text}<br>Alpha pct: %{x:.1f}<br>Burst: %{y:.1f}<extra></extra>"))
    x, y = valid["composite_alpha_pct"].values, valid["burst_readiness_score"].values
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        tx = np.linspace(x.min(), x.max(), 50)
        fig.add_trace(go.Scatter(x=tx, y=m * tx + b, mode="lines", name="Trend",
                                 line=dict(color="#5b8dee", dash="dash", width=1.5)))
    fig.update_layout(
        paper_bgcolor="#09090e", plot_bgcolor="#0f0f17",
        font=dict(family="Fira Code, monospace", color="#ede8d9", size=11),
        xaxis=dict(title="Composite Alpha Percentile",
                   gridcolor="rgba(245,166,35,0.08)"),
        yaxis=dict(title="Burst Readiness Score",
                   gridcolor="rgba(245,166,35,0.08)"),
        height=420, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "⚠ Composite = naive equal-weight average of all alpha percentiles, not "
        "sign-aligned. Read as 'is there any overlap at all,' not a validated "
        "score. Low |r| may mean the alphas capture something existing scoring "
        "doesn't — an argument FOR the ML layer, not against it.")


def _render_category_overview(df, meta):
    _section("Category Overview — Watchlist-Wide Averages")
    st.caption("Expand any category to see exactly which alphas make up its average.")
    cols = st.columns(3)
    for i, cat in enumerate(CAT_ORDER):
        info = CATEGORIES[cat]
        avg = df[cat].mean() if cat in df.columns else None
        avg_s = "—" if avg is None or pd.isna(avg) else "%.0f" % avg
        members = sorted(aid for aid, m in meta.items() if m.get("cat") == cat)
        with cols[i % 3]:
            card = ("<div style='background:#0f0f17;border:1px solid "
                    "rgba(245,166,35,0.22);border-top:3px solid " + info["color"]
                    + ";padding:1rem 1.1rem;margin-bottom:.6rem;'>"
                    + "<div style='font-family:Orbitron,sans-serif;font-size:.55rem;"
                    + "color:" + info["color"] + ";letter-spacing:.15em;'>" + cat
                    + " · " + str(len(members)) + " ALPHAS</div>"
                    + "<div style='font-family:Orbitron,sans-serif;font-size:.68rem;"
                    + "font-weight:700;color:#ede8d9;text-transform:uppercase;'>"
                    + info["name"] + "</div>"
                    + "<div style='font-family:Fira Code,monospace;font-size:.72rem;"
                    + "color:#7a7060;margin:.5rem 0;'>" + info["question"] + "</div>"
                    + "<div style='background:#1a1a1a;border-radius:2px;height:8px;"
                    + "overflow:hidden;'><div style='width:" + (avg_s if avg_s != "—" else "0")
                    + "%;height:100%;background:" + info["color"] + ";'></div></div>"
                    + "<div style='font-family:Orbitron,sans-serif;font-size:1rem;"
                    + "font-weight:700;color:" + info["color"] + ";margin-top:.3rem;'>"
                    + avg_s + " <span style='font-size:.6rem;color:#7a7060;'>avg pct"
                    + (" · thin sample" if len(members) <= 3 else "") + "</span></div>"
                    + "</div>")
            st.markdown(card, unsafe_allow_html=True)
            with st.expander("Show alphas"):
                for aid in members:
                    st.markdown("**#" + str(aid) + "** — "
                                + meta[aid].get("desc", ""))


def _render_reference_table(meta):
    _section("Full Alpha Reference — All 57")
    cat_filter = st.multiselect(
        "Category", CAT_ORDER, default=[], key="as_ref_cat",
        help="Empty = show all",
        format_func=lambda c: c + " · " + CATEGORIES[c]["name"])
    rows = [{"Alpha": "#" + str(aid), "Category": m.get("cat", "?"),
             "Tier": "Tier " + str(m.get("tier", "?")),
             "What it measures": m.get("desc", "")}
            for aid, m in sorted(meta.items())]
    ref = pd.DataFrame(rows)
    if cat_filter:
        ref = ref[ref["Category"].isin(cat_filter)]
    cat_style = {c: ("background-color:" + CATEGORIES[c]["color"]
                     + ";color:#000;font-weight:700") for c in CAT_ORDER}
    styled = (ref.style
              .map(lambda v: cat_style.get(str(v), ""), subset=["Category"])
              .set_properties(**_BASE_PROPS)
              .set_properties(subset=["Alpha"], **_TICKER_PROPS)
              .set_table_styles(_TABLE_STYLES))
    st.dataframe(styled, use_container_width=True, height=420, hide_index=True)


def _render_ticker_stories(df, alpha_data, meta):
    _section("Ticker Stories")
    top = df.sort_values("composite_alpha_pct", ascending=False).head(N_STORY_CARDS)
    for _, row in top.iterrows():
        pcts = ticker_alpha_pcts(alpha_data, row["ticker"])
        cats = category_pcts(pcts, meta)
        patterns = detect_patterns(cats)
        dissents = find_dissents(pcts, cats, meta)
        r1, r2 = build_read(row["ticker"], cats, patterns, dissents, meta)
        active_ids = {pid for pid, _, _, _, _, on in patterns if on}
        border = ("warn" if "EXHAUSTION_RISK" in active_ids else
                  "watch" if "UNCONFIRMED_MOVE" in active_ids else "")
        st.markdown(_story_card_html(row, cats, r1, r2, border),
                    unsafe_allow_html=True)


def _render_signals_table(df):
    _section("Full Signals Table")
    s1, s2 = st.columns([1, 2])
    with s1:
        conv_f = st.multiselect("Conviction", ["HIGH", "MED", "LOW"],
                                default=["HIGH", "MED"], key="as_conv")
    with s2:
        min_pct = st.slider("Min composite alpha percentile", 0, 100, 0, key="as_min")

    view = df.copy()
    if conv_f:
        view = view[view["conviction"].isin(conv_f)]
    view = view[view["composite_alpha_pct"] >= min_pct]
    view = view.sort_values("composite_alpha_pct", ascending=False)

    show = [c for c in ["ticker", "conviction", "composite_alpha_pct"] + CAT_ORDER
            + ["burst_readiness_score", "theme", "industry"] if c in view.columns]
    pct_cols = ["composite_alpha_pct"] + CAT_ORDER + ["burst_readiness_score"]
    styled = _style_table(view[show], pct_cols, ticker_col="ticker",
                          conviction_col="conviction")
    st.dataframe(styled, use_container_width=True, height=460, hide_index=True)
    st.caption("Showing " + str(len(view)) + " of " + str(len(df)) + " tickers")


def _render_drilldown(df, alpha_data, meta):
    _section("Ticker Drill-Down — Alpha Narrative")
    tickers = sorted(df["ticker"].tolist())
    if not tickers:
        return
    sel = st.selectbox("Ticker", tickers, key="as_drill")
    pcts = ticker_alpha_pcts(alpha_data, sel)
    if not pcts:
        st.info("No alpha values found for " + sel + ".")
        return
    cats = category_pcts(pcts, meta)
    patterns = detect_patterns(cats)
    dissents = find_dissents(pcts, cats, meta)
    top5, bottom5 = top_bottom_alphas(pcts, meta)

    # ── 57-bar profile colored by category ────────────────────────────────
    order = [aid for c in CAT_ORDER
             for aid in sorted(a for a, m in meta.items() if m.get("cat") == c)
             if aid in pcts]
    fig = go.Figure(go.Bar(
        x=["#" + str(a) for a in order],
        y=[pcts[a] for a in order],
        marker_color=[CATEGORIES[meta[a]["cat"]]["color"] for a in order],
        hovertext=[meta[a].get("desc", "") for a in order]))
    fig.update_layout(
        paper_bgcolor="#09090e", plot_bgcolor="#0f0f17",
        font=dict(family="Fira Code, monospace", color="#ede8d9", size=10),
        xaxis=dict(tickangle=-90, gridcolor="rgba(245,166,35,0.08)"),
        yaxis=dict(title="Percentile", gridcolor="rgba(245,166,35,0.08)",
                   range=[0, 100]),
        height=300, margin=dict(l=0, r=0, t=15, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Combination patterns ──────────────────────────────────────────────
    st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:.6rem;"
                "letter-spacing:.18em;color:#f5a623;text-transform:uppercase;"
                "margin:.6rem 0;'>Combination Patterns — " + sel + "</div>",
                unsafe_allow_html=True)
    dissent_cards = ""
    for aid, p, cat, gap_v, direction in dissents[:1]:
        avg = cats.get(cat)
        narr = ("#" + str(aid) + " sits at " + ("%.0f" % p) + " while its " + cat
                + " category averages " + ("%.0f" % (avg or 0)) + " ("
                + ("%.0f" % gap_v) + " pts " + direction + "). "
                + meta[aid].get("desc", ""))
        dissent_cards += _pattern_html(
            "Dissent — Alpha #" + str(aid),
            "any single alpha ≥ " + str(DISSENT_GAP) + " pts from its category average",
            narr, "warn", True)
    pattern_cards = "".join(
        _pattern_html(name, rule, narr, kind, active)
        for _, name, rule, narr, kind, active in patterns)
    st.markdown(_CSS + "<div class='as-pattern-grid'>" + pattern_cards
                + dissent_cards + "</div>", unsafe_allow_html=True)

    # ── Confirmations & dissents ──────────────────────────────────────────
    conf_items = "".join(
        _agree_item_html(aid, p, cat, desc) for aid, p, cat, desc in top5)
    soft_items = "".join(
        _agree_item_html(aid, p, cat, desc) for aid, p, cat, desc in bottom5)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='as-agree'><div class='as-atitle' "
                    "style='color:#4caf7d;'>Strongest Confirmations</div>"
                    + conf_items + "</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='as-agree'><div class='as-atitle' "
                    "style='color:#f5a623;'>Soft Spots</div>"
                    + soft_items + "</div>", unsafe_allow_html=True)

    # ── Annotated 57-row table ────────────────────────────────────────────
    st.markdown("<div style='font-family:Orbitron,sans-serif;font-size:.6rem;"
                "letter-spacing:.18em;color:#f5a623;text-transform:uppercase;"
                "margin:1rem 0 .4rem;'>" + sel + " — All Alphas, Annotated</div>",
                unsafe_allow_html=True)
    t1, t2 = st.columns([2, 1])
    with t1:
        cat_f = st.multiselect("Category", CAT_ORDER, default=[], key="as_drill_cat",
                               help="Empty = show all")
    with t2:
        extremes = st.checkbox("Extremes only", value=False, key="as_extremes",
                               help="Only rows ≥ " + str(EXTREME_HI) + " / ≤ "
                                    + str(EXTREME_LO) + " pct, or dissents")

    dissent_ids = {d[0] for d in dissents}
    rows = []
    for aid in order:
        p, cat = pcts[aid], meta[aid]["cat"]
        note = ""
        if aid in dissent_ids:
            note = "DISSENT — " + ("%.0f" % abs(p - (cats.get(cat) or 0))) \
                   + " pts from " + cat + " avg"
        elif cat == "MRV" and p <= EXTREME_LO:
            note = "Low snap-back risk"
        elif cat != "MRV" and p >= EXTREME_HI:
            note = "Strong confirmation"
        elif cat == "MRV" and p >= 75:
            note = "Elevated reversal risk"
        is_extreme = p >= EXTREME_HI or p <= EXTREME_LO or aid in dissent_ids
        if extremes and not is_extreme:
            continue
        if cat_f and cat not in cat_f:
            continue
        rows.append({"Alpha": "#" + str(aid), "Cat": cat,
                     "Pct": round(p, 0), "Tier": "T" + str(meta[aid].get("tier", "?")),
                     "What it measures": meta[aid].get("desc", ""), "Note": note})
    tdf = pd.DataFrame(rows)
    if tdf.empty:
        st.info("No rows match the current filters.")
    else:
        cat_style = {c: ("background-color:" + CATEGORIES[c]["color"]
                         + ";color:#000;font-weight:700") for c in CAT_ORDER}
        styled = (tdf.style
                  .map(lambda v: cat_style.get(str(v), ""), subset=["Cat"])
                  .map(_pct_bg, subset=["Pct"])
                  .format({"Pct": "{:.0f}"})
                  .set_properties(**_BASE_PROPS)
                  .set_properties(subset=["Alpha"], **_TICKER_PROPS)
                  .set_table_styles(_TABLE_STYLES))
        st.dataframe(styled, use_container_width=True, height=440, hide_index=True)

    # ── Synthesized read ──────────────────────────────────────────────────
    r1, r2 = build_read(sel, cats, patterns, dissents, meta)
    n_active = sum(1 for *_, on in patterns if on)
    read_html = ("<div class='as-story'>"
                 + "<div class='as-tkr' style='font-size:.9rem;'>SYNTHESIZED READ — "
                 + sel + "</div>"
                 + "<div class='as-meta'>" + str(n_active) + " active patterns · "
                 + str(len(dissents)) + " dissents</div>"
                 + "<div class='as-read'><b>The combined story:</b> " + r1
                 + "<br><br><b>The threads to watch:</b> " + r2 + "</div></div>")
    st.markdown(_CSS + read_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

def render():
    alpha_data = _load_json(ALPHA_FEATURES_JSON, default={})
    daily_data = _load_json(DAILY_WATCHLIST_JSON, default={"entries": []})
    burst_data = _load_json(BURST_WATCH_JSON, default={})

    if not alpha_data or not alpha_data.get("features"):
        st.info("No alpha feature data yet. Run alpha_features.py after "
                "daily_screener.py.")
        return
    meta = get_alpha_meta(alpha_data)
    if not meta:
        st.warning("alpha_features.json has no alpha_meta block — re-run the "
                   "updated alpha_features.py so the taxonomy is written into "
                   "the output.")
        return

    df, meta_info = build_signals_df(alpha_data, daily_data, burst_data)
    if df.empty:
        st.info("alpha_features.json loaded but produced no rows.")
        return

    st.markdown(_CSS, unsafe_allow_html=True)
    st.caption("As of " + str(meta_info["as_of"]) + " · " + str(len(df))
               + " tickers · " + str(len(meta)) + " alphas in 6 story categories")

    _render_validation(df, meta_info)
    _render_category_overview(df, meta)
    _render_reference_table(meta)
    _render_ticker_stories(df, alpha_data, meta)
    _render_signals_table(df)
    _render_drilldown(df, alpha_data, meta)
