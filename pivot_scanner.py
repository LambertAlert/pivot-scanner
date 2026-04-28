"""
30-Minute Pivot Scanner — Based on @1ChartMaster / Elite Swing Traders Strategy
================================================================================
DETECTION ONLY — no order execution.

Strategy:
  BULLISH: streak of ≥ MIN_STREAK consecutive RED candles → first GREEN = pivot candle
           → if NEXT candle closes ABOVE pivot HIGH → ALERT (calls setup)
  BEARISH: streak of ≥ MIN_STREAK consecutive GREEN candles → first RED = pivot candle
           → if NEXT candle closes BELOW pivot LOW → ALERT (puts setup)

Scheduling: run this script once after each 30-min bar closes (e.g., via cron at :01 and :31)
            or call scan_all_tickers() from your own scheduler loop.

Dependencies:
    pip install polygon-api-client yfinance pandas numpy

Gmail setup:
    1. Enable 2-Step Verification on your Google account
    2. Generate an App Password at: myaccount.google.com/apppasswords
    3. Set GMAIL_SENDER, GMAIL_APP_PASS, GMAIL_RECIPIENT as env vars or edit the config block below
"""

import os
import time
import logging
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import pandas as pd

# ── Try importing optional data providers ────────────────────────────────────
try:
    from polygon import RESTClient as PolygonClient
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURABLE PARAMETERS  (edit these)
# ═══════════════════════════════════════════════════════════════════════════════

TICKERS = [
    "ORCL", "LRCX", "PANW", "CRWD", "CVNA", "CCJ", "ALAB", "CRDO", "Q", "BWXT",
    "MKSI", "ROKU", "IONQ", "IREN", "STRL", "RMBS", "MOD", "HL", "MP", "FLS",
    "CAVA", "ARWR", "SANM", "PRIM", "PRAX", "WFRD", "AUGO", "ECG", "SITE", "CELC",
    "ALM", "CGON", "FLY", "ONDS", "SIMO", "SEI", "KNF", "LMND", "LGND", "DNTH",
    "TDW", "LASR", "AXTI", "SYNA", "PII", "PARR", "AMPX", "SEDG", "SEZL", "KWR",
    "NGL", "LAR", "NBTX", "NBR", "BKSY", "GPRE", "LPTH", "ANRO", "RLMD", "FET",
    "ARMP", "IRD", "OPTX", "ASYS", "SLGL", "AP", "OCC", "NRXS"
]

# Core strategy parameters
MIN_STREAK      = 3       # minimum consecutive same-color candles before pivot
BARS_TO_FETCH   = 100     # how many 30-min bars of history to pull

# Data source: "polygon" | "yfinance" | "auto"
# "auto" tries Polygon first, falls back to yfinance
DATA_SOURCE     = "yfinance"

# Polygon.io API key — set here OR as env var POLYGON_API_KEY
# Free key: https://polygon.io/dashboard/signup
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "AVQOVtvsBKFlRp5HBjdysH0oMos4geGc")

# Alert verbosity: "full" prints full details; "minimal" prints ticker + direction only
ALERT_STYLE     = "full"

# Pause between tickers to respect free-tier rate limits (seconds)
# Polygon free tier = 5 calls/min → 12s between calls is safe
RATE_LIMIT_PAUSE = 1

# ── Gmail config ──────────────────────────────────────────────────────────────
GMAIL_SENDER    = os.getenv("GMAIL_SENDER",    "neil.lambert1214@gmail.com")   # your Gmail address
GMAIL_APP_PASS  = os.getenv("GMAIL_APP_PASS",  "yzvp axky tani icpc")               # 16-char App Password
GMAIL_RECIPIENT = os.getenv("GMAIL_RECIPIENT", "neil.lambert1214@gmail.com")   # alert destination
SEND_EMAIL      = True    # set False to disable email and only print to console

# Logging level
LOG_LEVEL       = logging.INFO


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_bars_polygon(ticker: str, n_bars: int = BARS_TO_FETCH) -> Optional[pd.DataFrame]:
    """
    Fetch the last n_bars × 30-minute OHLC bars from Polygon.io.
    Returns a DataFrame with columns [open, high, low, close, volume]
    indexed by UTC datetime, sorted oldest → newest.
    Returns None on failure.
    """
    if not POLYGON_AVAILABLE:
        log.warning("polygon-api-client not installed. Run: pip install polygon-api-client")
        return None
    if POLYGON_API_KEY == "YOUR_POLYGON_API_KEY_HERE":
        log.warning("Polygon API key not set. Set POLYGON_API_KEY env var or edit the script.")
        return None

    try:
        client = PolygonClient(POLYGON_API_KEY)

        # Go back enough calendar days to guarantee n_bars of trading data.
        # ~13 trading hours/day × 2 bars/hour = ~26 bars/day. Add buffer for weekends/holidays.
        calendar_days_back = max(10, int((n_bars / 26) * 2.5) + 7)
        end_dt   = datetime.now(tz=timezone.utc)
        start_dt = end_dt - timedelta(days=calendar_days_back)

        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=30,
            timespan="minute",
            from_=start_dt.strftime("%Y-%m-%d"),
            to=end_dt.strftime("%Y-%m-%d"),
            adjusted=True,
            sort="asc",
            limit=n_bars + 50,   # fetch a little extra, trim below
        )

        if not aggs:
            log.warning(f"[{ticker}] Polygon returned no data.")
            return None

        rows = []
        for bar in aggs:
            rows.append({
                "timestamp": pd.Timestamp(bar.timestamp, unit="ms", tz="UTC"),
                "open":   bar.open,
                "high":   bar.high,
                "low":    bar.low,
                "close":  bar.close,
                "volume": bar.volume,
            })

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df = df.tail(n_bars)   # keep only the most recent n_bars
        log.debug(f"[{ticker}] Polygon: fetched {len(df)} 30-min bars.")
        return df

    except Exception as e:
        log.error(f"[{ticker}] Polygon fetch error: {e}")
        return None


def fetch_bars_yfinance(ticker: str, n_bars: int = BARS_TO_FETCH) -> Optional[pd.DataFrame]:
    """
    Fetch 30-minute bars from yfinance.
    yfinance provides up to 60 days of 30-min history.
    Returns a DataFrame with columns [open, high, low, close, volume]
    sorted oldest → newest. Returns None on failure.
    """
    if not YFINANCE_AVAILABLE:
        log.warning("yfinance not installed. Run: pip install yfinance")
        return None

    try:
        tkr = yf.Ticker(ticker)
        df  = tkr.history(period="60d", interval="30m", auto_adjust=True)

        if df.empty:
            log.warning(f"[{ticker}] yfinance returned no data.")
            return None

        # Normalize timezone and column names
        df.index   = df.index.tz_convert("UTC") if df.index.tzinfo else df.index.tz_localize("UTC")
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        df = df.tail(n_bars)
        log.debug(f"[{ticker}] yfinance: fetched {len(df)} 30-min bars.")
        return df

    except Exception as e:
        log.error(f"[{ticker}] yfinance fetch error: {e}")
        return None


def fetch_bars(ticker: str) -> Optional[pd.DataFrame]:
    """
    Route to the configured data source.
    "auto" tries Polygon first, falls back to yfinance.
    """
    source = DATA_SOURCE.lower()

    if source == "polygon":
        return fetch_bars_polygon(ticker)

    if source == "yfinance":
        return fetch_bars_yfinance(ticker)

    # "auto" mode — try Polygon, silently fall back to yfinance
    df = fetch_bars_polygon(ticker)
    if df is not None and not df.empty:
        return df

    log.info(f"[{ticker}] Falling back to yfinance...")
    return fetch_bars_yfinance(ticker)


# ═══════════════════════════════════════════════════════════════════════════════
#  CANDLE COLOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def is_green(row: pd.Series) -> bool:
    """Green (bullish) candle: close strictly above open."""
    return row["close"] > row["open"]


def is_red(row: pd.Series) -> bool:
    """
    Red (bearish) candle: close at or below open.
    Doji/flat candles count as red — no follow-through = no bullish conviction.
    """
    return row["close"] <= row["open"]


# ═══════════════════════════════════════════════════════════════════════════════
#  PIVOT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_pivot(df: pd.DataFrame, min_streak: int = MIN_STREAK) -> Optional[dict]:
    """
    Scan the tail of df to detect a freshly triggered 30-min pivot.

    Bar roles (from the end of df):
        df.iloc[-1]  = trigger candle  (the bar that just closed)
        df.iloc[-2]  = pivot candle    (first candle breaking the streak)
        df.iloc[-3]  and earlier = streak candles

    Only fires on the very latest bar — no look-back re-alerts.

    Returns a dict on trigger, None otherwise:
        direction    : "bullish" | "bearish"
        streak_len   : actual number of streak candles found
        pivot_bar    : pd.Series
        trigger_bar  : pd.Series
        pivot_time   : timestamp of pivot candle
        trigger_time : timestamp of trigger candle
    """
    if len(df) < min_streak + 2:
        log.debug(f"Not enough bars to evaluate pivot (need {min_streak + 2}, have {len(df)}).")
        return None

    trigger_bar = df.iloc[-1]   # candle that just closed — this is what we're evaluating
    pivot_bar   = df.iloc[-2]   # the candle immediately before the trigger

    # ── BULLISH check ─────────────────────────────────────────────────────────
    # Pivot candle must be GREEN (first green after a red streak)
    if is_green(pivot_bar):
        # Walk backwards from pivot_bar to count consecutive RED candles
        streak = 0
        for i in range(3, len(df) + 1):
            bar = df.iloc[-i]
            if is_red(bar):
                streak += 1
            else:
                break   # streak ends at first non-red bar

        if streak >= min_streak:
            # Trigger fires if the latest bar closes ABOVE the pivot candle's HIGH
            if trigger_bar["close"] > pivot_bar["high"]:
                return {
                    "direction":    "bullish",
                    "streak_len":   streak,
                    "pivot_bar":    pivot_bar,
                    "trigger_bar":  trigger_bar,
                    "pivot_time":   df.index[-2],
                    "trigger_time": df.index[-1],
                }

    # ── BEARISH check ─────────────────────────────────────────────────────────
    # Pivot candle must be RED (first red after a green streak)
    if is_red(pivot_bar):
        # Walk backwards from pivot_bar to count consecutive GREEN candles
        streak = 0
        for i in range(3, len(df) + 1):
            bar = df.iloc[-i]
            if is_green(bar):
                streak += 1
            else:
                break

        if streak >= min_streak:
            # Trigger fires if the latest bar closes BELOW the pivot candle's LOW
            if trigger_bar["close"] < pivot_bar["low"]:
                return {
                    "direction":    "bearish",
                    "streak_len":   streak,
                    "pivot_bar":    pivot_bar,
                    "trigger_bar":  trigger_bar,
                    "pivot_time":   df.index[-2],
                    "trigger_time": df.index[-1],
                }

    return None   # no trigger on the latest bar


# ═══════════════════════════════════════════════════════════════════════════════
#  ALERT FORMATTING  (console)
# ═══════════════════════════════════════════════════════════════════════════════

def format_console_alert(ticker: str, result: dict, style: str = ALERT_STYLE) -> str:
    """Build the console alert string from a detect_pivot() result dict."""
    d      = result["direction"]
    pb     = result["pivot_bar"]
    tb     = result["trigger_bar"]
    p_time = result["pivot_time"]
    t_time = result["trigger_time"]
    streak = result["streak_len"]

    pivot_str = (f"O:{pb['open']:.2f}  H:{pb['high']:.2f}  "
                 f"L:{pb['low']:.2f}  C:{pb['close']:.2f}")
    trig_str  = (f"O:{tb['open']:.2f}  H:{tb['high']:.2f}  "
                 f"L:{tb['low']:.2f}  C:{tb['close']:.2f}")

    if d == "bullish":
        emoji      = "🚨🟢"
        label      = "BULLISH 30-MIN PIVOT TRIGGERED"
        streak_txt = f"{streak} consecutive RED bars → GREEN pivot → trigger close above pivot HIGH"
        entry_txt  = "Entry: ITM/ATM weekly calls (consider next weekly expiry)"
        stop_txt   = f"Stop: below pivot candle LOW ({pb['low']:.2f})"
        broken_txt = (f"Trigger close {tb['close']:.2f} > Pivot high {pb['high']:.2f}")
    else:
        emoji      = "🚨🔴"
        label      = "BEARISH 30-MIN PIVOT TRIGGERED"
        streak_txt = f"{streak} consecutive GREEN bars → RED pivot → trigger close below pivot LOW"
        entry_txt  = "Entry: ITM/ATM weekly puts (consider next weekly expiry)"
        stop_txt   = f"Stop: above pivot candle HIGH ({pb['high']:.2f})"
        broken_txt = (f"Trigger close {tb['close']:.2f} < Pivot low {pb['low']:.2f}")

    if style == "minimal":
        return f"{emoji} {label} on {ticker} | {broken_txt}"

    lines = [
        "",
        "=" * 70,
        f"  {emoji}  {label} on {ticker}",
        "=" * 70,
        f"  Streak        : {streak_txt}",
        f"  Pivot candle  : {pivot_str}  @ {p_time}",
        f"  Trigger candle: {trig_str}  @ {t_time}",
        f"  Signal        : {broken_txt}",
        f"  {entry_txt}",
        f"  {stop_txt}",
        "=" * 70,
        "",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  EMAIL ALERT  (Gmail)
# ═══════════════════════════════════════════════════════════════════════════════

def format_email_alert(ticker: str, result: dict) -> tuple:
    """
    Build the (subject, body) tuple for the Gmail alert.
    Subject is kept short so it reads clearly as a phone banner notification.
    Body has full OHLC + trade parameters.
    """
    d      = result["direction"]
    pb     = result["pivot_bar"]
    tb     = result["trigger_bar"]
    streak = result["streak_len"]
    p_time = result["pivot_time"]
    t_time = result["trigger_time"]

    direction_label = "BULLISH 🟢" if d == "bullish" else "BEARISH 🔴"
    streak_color    = "RED"   if d == "bullish" else "GREEN"
    entry_txt       = "ITM/ATM weekly calls" if d == "bullish" else "ITM/ATM weekly puts"

    if d == "bullish":
        stop_txt   = f"Below pivot low  ({pb['low']:.2f})"
        broken_txt = f"Trigger close {tb['close']:.2f}  >  Pivot high {pb['high']:.2f}"