"""
data_layer.py — Shared persistence for the pivot scanning system.

Handles:
  • JSON files for current state (latest watchlists, latest triggers)
  • SQLite database for historical triggers (full backtest history)

All other scripts import from here so we have one place to change schemas.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional


DB_PATH                  = "pivot_data.db"
WEEKLY_JSON              = "data/weekly_watchlist.json"
DAILY_JSON               = "data/daily_watchlist.json"
LATEST_TRIGGERS_JSON     = "data/latest_triggers.json"
SECTOR_THEMES_JSON       = "data/sector_themes.json"
INDUSTRY_RANKS_JSON      = "data/industry_ranks.json"
EP_EVENTS_JSON           = "data/ep_events.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  SQLITE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

def init_db(db_path: str = DB_PATH):
    """Create tables if they don't exist."""
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Historical pivot triggers — every alert ever fired
    c.execute("""
        CREATE TABLE IF NOT EXISTS triggers (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_time       TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            timeframe       TEXT NOT NULL,
            direction       TEXT NOT NULL,
            conviction      TEXT NOT NULL,
            theme           TEXT,
            theme_rank      INTEGER,
            weekly_stage    INTEGER,
            daily_stage     INTEGER,
            trend_template  INTEGER,
            weekly_bbuw     REAL,
            daily_bbuw      REAL,
            streak_len      INTEGER,
            pivot_open      REAL,
            pivot_high      REAL,
            pivot_low       REAL,
            pivot_close     REAL,
            pivot_time      TEXT,
            trigger_open    REAL,
            trigger_high    REAL,
            trigger_low     REAL,
            trigger_close   REAL,
            trigger_time    TEXT,
            stop_level      REAL,
            entry_note      TEXT
        )
    """)

    # Index for fast filtering on dashboard
    c.execute("CREATE INDEX IF NOT EXISTS idx_scan_time ON triggers(scan_time)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON triggers(ticker)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_conviction ON triggers(conviction)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_theme ON triggers(theme)")

    # Watchlist snapshots (one row per ticker per screen run)
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_time       TEXT NOT NULL,
            scan_type       TEXT NOT NULL,  -- 'weekly' or 'daily'
            ticker          TEXT NOT NULL,
            conviction      TEXT,
            weekly_stage    INTEGER,
            daily_stage     INTEGER,
            trend_template  INTEGER,
            weekly_bbuw     REAL,
            daily_bbuw      REAL,
            theme           TEXT,
            theme_rank      INTEGER
        )
    """)

    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_json(path: str, data):
    """Write JSON with pretty formatting."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: str, default=None):
    """Read JSON file, return default if missing."""
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


# ═══════════════════════════════════════════════════════════════════════════════
#  WATCHLIST WRITES (called by screeners)
# ═══════════════════════════════════════════════════════════════════════════════

def save_weekly_watchlist(entries: list):
    """
    Save weekly screener output to both JSON (current state) and SQLite (history).
    entries: list of dicts with keys ticker, stage, trend_template_score, bbuw_score, bbuw_components
    """
    scan_time = datetime.now().isoformat()

    # JSON — current state
    write_json(WEEKLY_JSON, {
        "scan_time": scan_time,
        "count": len(entries),
        "entries": entries,
    })

    # SQLite — history
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for e in entries:
        c.execute("""
            INSERT INTO watchlist_history
            (scan_time, scan_type, ticker, weekly_stage, trend_template, weekly_bbuw)
            VALUES (?, 'weekly', ?, ?, ?, ?)
        """, (
            scan_time,
            e["ticker"],
            e.get("stage"),
            e.get("trend_template_score"),
            e.get("bbuw_score"),
        ))
    conn.commit()
    conn.close()


def save_daily_watchlist(entries: list):
    """
    Save daily screener output to both JSON and SQLite.
    entries: list of dicts produced by daily_screener.py
    """
    scan_time = datetime.now().isoformat()

    write_json(DAILY_JSON, {
        "scan_time": scan_time,
        "count": len(entries),
        "entries": entries,
    })

    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for e in entries:
        c.execute("""
            INSERT INTO watchlist_history
            (scan_time, scan_type, ticker, conviction, weekly_stage, daily_stage,
             trend_template, weekly_bbuw, daily_bbuw, theme, theme_rank)
            VALUES (?, 'daily', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_time,
            e["ticker"],
            e.get("conviction"),
            e.get("weekly_stage"),
            e.get("daily_stage"),
            e.get("trend_template"),
            e.get("weekly_bbuw"),
            e.get("daily_bbuw"),
            e.get("theme"),
            e.get("theme_rank"),
        ))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRIGGER WRITES (called by intraday scanner)
# ═══════════════════════════════════════════════════════════════════════════════

def save_trigger(trigger: dict):
    """
    Persist a pivot trigger to SQLite (history) and append to today's JSON
    (current-state for dashboard fast reads).

    trigger dict keys:
      ticker, timeframe, direction, conviction, theme, theme_rank,
      weekly_stage, daily_stage, trend_template, weekly_bbuw, daily_bbuw,
      streak_len,
      pivot_open, pivot_high, pivot_low, pivot_close, pivot_time,
      trigger_open, trigger_high, trigger_low, trigger_close, trigger_time,
      stop_level, entry_note
    """
    scan_time = datetime.now().isoformat()
    trigger["scan_time"] = scan_time

    # ── SQLite write ──────────────────────────────────────────────────────
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO triggers
        (scan_time, ticker, timeframe, direction, conviction, theme, theme_rank,
         weekly_stage, daily_stage, trend_template, weekly_bbuw, daily_bbuw,
         streak_len, pivot_open, pivot_high, pivot_low, pivot_close, pivot_time,
         trigger_open, trigger_high, trigger_low, trigger_close, trigger_time,
         stop_level, entry_note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_time,
        trigger.get("ticker"),
        trigger.get("timeframe"),
        trigger.get("direction"),
        trigger.get("conviction"),
        trigger.get("theme"),
        trigger.get("theme_rank"),
        trigger.get("weekly_stage"),
        trigger.get("daily_stage"),
        trigger.get("trend_template"),
        trigger.get("weekly_bbuw"),
        trigger.get("daily_bbuw"),
        trigger.get("streak_len"),
        trigger.get("pivot_open"),
        trigger.get("pivot_high"),
        trigger.get("pivot_low"),
        trigger.get("pivot_close"),
        str(trigger.get("pivot_time")),
        trigger.get("trigger_open"),
        trigger.get("trigger_high"),
        trigger.get("trigger_low"),
        trigger.get("trigger_close"),
        str(trigger.get("trigger_time")),
        trigger.get("stop_level"),
        trigger.get("entry_note"),
    ))
    conn.commit()
    conn.close()

    # ── JSON append ───────────────────────────────────────────────────────
    today_str = datetime.now().strftime("%Y-%m-%d")
    current = read_json(LATEST_TRIGGERS_JSON, default={"date": today_str, "triggers": []})

    # Reset if it's a new day
    if current.get("date") != today_str:
        current = {"date": today_str, "triggers": []}

    current["triggers"].append(trigger)
    write_json(LATEST_TRIGGERS_JSON, current)


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD READS (called by Streamlit app)
# ═══════════════════════════════════════════════════════════════════════════════

def get_latest_weekly_watchlist():
    return read_json(WEEKLY_JSON, default={"entries": [], "scan_time": None})


def get_latest_daily_watchlist():
    return read_json(DAILY_JSON, default={"entries": [], "scan_time": None})


def get_today_triggers():
    today_str = datetime.now().strftime("%Y-%m-%d")
    data = read_json(LATEST_TRIGGERS_JSON, default={"date": today_str, "triggers": []})
    if data.get("date") != today_str:
        return []
    return data.get("triggers", [])


def save_industry_ranks(ranks: list):
    """
    Persist weekly industry ranking to JSON.
    """
    write_json(INDUSTRY_RANKS_JSON, {
        "generated_at": datetime.now().isoformat(),
        "count":        len(ranks),
        "ranks":        ranks,
    })


def get_industry_ranks() -> dict:
    """Fast {industry_name: rank_int} lookup for conviction scoring."""
    data = read_json(INDUSTRY_RANKS_JSON, default={"ranks": []})
    return {r["industry"]: r["rank"] for r in data.get("ranks", [])}


def get_industry_ranks_full() -> dict:
    """Full industry rank records for dashboard display."""
    return read_json(INDUSTRY_RANKS_JSON, default={"ranks": [], "generated_at": None})


def save_ep_events(events: list):
    """
    Persist weekly Episodic Pivot events.
    events: list of ticker result dicts with ep_* fields, sorted by ep_score desc.
    """
    write_json(EP_EVENTS_JSON, {
        "generated_at": datetime.now().isoformat(),
        "count":        len(events),
        "events":       events,
    })


def get_ep_events() -> dict:
    """Load the latest EP events for dashboard display."""
    return read_json(EP_EVENTS_JSON, default={
        "generated_at": None,
        "count":        0,
        "events":       [],
    })


def save_volume_surges(events: list):
    """
    Persist the full volume surge event log.
    events: list of dicts — each is a daily or weekly surge record.
    Append-only pattern is handled in volume_surge_prep.py.
    """
    write_json(VOLUME_SURGES_JSON, {
        "generated_at": datetime.now().isoformat(),
        "count":        len(events),
        "events":       events,
    })


def get_volume_surges() -> dict:
    """Load the full volume surge log for dashboard display."""
    return read_json(VOLUME_SURGES_JSON, default={
        "generated_at": None,
        "count":        0,
        "events":       [],
    })


def get_trigger_history(days: int = 30, conviction: Optional[str] = None,
                        ticker: Optional[str] = None, theme: Optional[str] = None):
    """Query SQLite for historical triggers."""
    if not os.path.exists(DB_PATH):
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    query = "SELECT * FROM triggers WHERE scan_time >= datetime('now', ?)"
    params = [f"-{days} days"]

    if conviction:
        query += " AND conviction = ?"
        params.append(conviction)
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    if theme:
        query += " AND theme = ?"
        params.append(theme)

    query += " ORDER BY scan_time DESC"

    rows = c.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]
