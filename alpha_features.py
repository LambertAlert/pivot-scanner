"""
alpha_features.py — WorldQuant "101 Formulaic Alphas" (Kakushadze, 2015)
================================================================================
Implements the alphas that are computable from data this pipeline already
collects, split into three tiers:

  TIER 1 (39 alphas) — plain OHLCV + daily returns. No new data needed.
      #1,2,3,4,6,8,9,10,12,13,14,15,16,18,19,20,22,23,24,26,29,30,33,34,35,
      37,38,40,44,45,46,49,51,52,53,54,55,60,101

  TIER 2 (13 alphas) — needs adv{d} (average daily dollar volume), a
      one-line derived column: (close * volume).rolling(d).mean().
      #7,17,21,28,31,39,43,68,85,88,92,95,99

  TIER 4 (5 alphas) — needs industry-neutralization (indneutralize in the
      paper). Uses sector_themes.csv's Industry column as the IndClass
      grouping — no new data collection required.
      #48,80,82,90,100

  57 alphas total. The remaining 44 are deferred:
    - 30 need vwap only (#5,11,25,27,32,36,41,42,47,50,57,61,62,64,65,66,
      71,72,73,74,75,77,78,81,83,84,86,94,96,98)
    - 13 need vwap AND industry-neutralization together (#58,59,63,67,69,
      70,76,79,87,89,91,93,97)
    - 1 needs market cap (#56) — skipped; not worth the added yfinance
      serial-call overhead for a single alpha.

Formulas and operator definitions are from Appendix A of Kakushadze, Z.
"101 Formulaic Alphas" (2015), used here as an independent Python
re-implementation for feature engineering, not a reproduction of the paper.

Output: data/alpha_features.json — one row per ticker in the current daily
watchlist universe, holding each alpha's raw value AND its cross-sectional
percentile rank as of the latest trading day. The percentile rank is the
more useful of the two for dashboard display since raw alpha values are
unitless and not comparable across alphas.

Runs: after daily_screener.py in the daily pipeline, or standalone.
Dependencies: yfinance, pandas, numpy
"""

import os
import csv
import time
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from data_layer import get_latest_daily_watchlist, write_json

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_JSON = "data/alpha_features.json"
THEMES_CSV  = "sector_themes.csv"
BATCH_CHUNK = 200
LOOKBACK    = "1y"     # matches daily_screener.py's fetch window
MIN_ROWS    = 260       # need ~1y of history for the longest-lookback alphas (#19, #37, #39 use 200-250d windows)

os.makedirs("data", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH DATA FETCHING  (same pattern as daily_screener.py / radar_prep.py)
# ═══════════════════════════════════════════════════════════════════════════════

def batch_fetch_daily(tickers: list, period: str = LOOKBACK) -> dict:
    """Download daily OHLCV bars for all tickers in one yf.download() call per chunk."""
    result = {}
    tickers = sorted(set(tickers))

    for i in range(0, len(tickers), BATCH_CHUNK):
        chunk = tickers[i: i + BATCH_CHUNK]
        try:
            raw = yf.download(
                chunk, period=period, interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )
            if raw.empty:
                log.warning(f"alpha_features batch chunk {i//BATCH_CHUNK + 1}: empty result")
                continue

            for tk in chunk:
                try:
                    df = raw.xs(tk, axis=1, level=1).copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
                    df.columns = [c.lower() for c in df.columns]
                    needed = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                    df = df[needed].dropna(how="all")
                    df = df[df["close"] > 0]
                    if len(df) >= 60:
                        result[tk] = df
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"alpha_features batch fetch error: {e}")

        if i + BATCH_CHUNK < len(tickers):
            time.sleep(2)

    log.info(f"Batch daily fetch: {len(result)}/{len(tickers)} tickers with usable history")
    return result


def load_industry_groups(csv_path: str = THEMES_CSV) -> dict:
    """
    {ticker: industry_string} — this is the IndClass grouping used in place
    of the paper's GICS/BICS/etc. classification for indneutralize().
    Tickers missing an Industry value fall into 'Unclassified' rather than
    being dropped, consistent with the rest of the pipeline's convention.
    """
    groups = {}
    if not os.path.exists(csv_path):
        log.warning(f"Sector themes file not found: {csv_path}")
        return groups
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row.get("Symbol", "").strip().upper()
            industry = row.get("Industry", "").strip() or "Unclassified"
            if sym:
                groups[sym] = industry
    log.info(f"Loaded {len(groups)} industry classifications.")
    return groups


def build_panels(bars: dict) -> dict:
    """
    Assemble wide DataFrames (index=date, columns=ticker) for each OHLCV
    field, aligned on the union of all tickers' dates.
    """
    tickers = sorted(bars.keys())
    P = {}
    for f in ["open", "high", "low", "close", "volume"]:
        P[f] = pd.DataFrame({tk: bars[tk][f] for tk in tickers if f in bars[tk].columns}).sort_index()
    P["returns"] = P["close"].pct_change()
    return P


# ═══════════════════════════════════════════════════════════════════════════════
#  OPERATOR LIBRARY  (Appendix A.2 of Kakushadze 2015)
# ═══════════════════════════════════════════════════════════════════════════════

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank (0-1], computed per date across tickers."""
    return df.rank(axis=1, pct=True)


def delay(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.shift(int(d))


def delta(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df - df.shift(int(d))


def ts_min(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(int(d)).min()


def ts_max(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(int(d)).max()


def sum_(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(int(d)).sum()


def stddev(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(int(d)).std()


def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(int(d)).corr(y)


def covariance(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(int(d)).cov(y)


def signedpower(df: pd.DataFrame, a: float) -> pd.DataFrame:
    return np.sign(df) * np.abs(df) ** a


def scale(df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """Rescale each date's cross-section so sum(abs(x)) == a."""
    denom = df.abs().sum(axis=1).replace(0, np.nan)
    return df.div(denom, axis=0) * a


def _ts_rank_1d(arr: np.ndarray) -> float:
    if np.isnan(arr[-1]):
        return np.nan
    valid = arr[~np.isnan(arr)]
    if len(valid) <= 1:
        return np.nan
    return float((valid < arr[-1]).sum()) / (len(valid) - 1)


def ts_rank(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Percentile rank (0-1) of today's value within the trailing d-day window."""
    return df.rolling(int(d)).apply(_ts_rank_1d, raw=True)


def _argmax_1d(arr: np.ndarray) -> float:
    if np.all(np.isnan(arr)):
        return np.nan
    return float(len(arr) - 1 - np.nanargmax(arr))  # 0 = today, d-1 = d-1 days ago


def _argmin_1d(arr: np.ndarray) -> float:
    if np.all(np.isnan(arr)):
        return np.nan
    return float(len(arr) - 1 - np.nanargmin(arr))


def ts_argmax(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(int(d)).apply(_argmax_1d, raw=True)


def ts_argmin(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(int(d)).apply(_argmin_1d, raw=True)


def _decay_linear_1d(arr: np.ndarray) -> float:
    arr = np.nan_to_num(arr, nan=0.0)
    d = len(arr)
    weights = np.arange(1, d + 1)  # oldest=1 ... newest=d, rescaled to sum=1 by the division below
    return float(np.dot(arr, weights) / weights.sum())


def decay_linear(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Weighted moving average with linearly decaying weights (most recent day heaviest)."""
    return df.rolling(int(d)).apply(_decay_linear_1d, raw=True)


def adv(volume_df: pd.DataFrame, close_df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Average daily dollar volume over the past d days."""
    return (close_df * volume_df).rolling(int(d)).mean()


def indneutralize(df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """
    Cross-sectionally demean df within each industry group, per date.
    groups: {ticker: industry_string} — sourced from sector_themes.csv.
    """
    group_map = pd.Series({tk: groups.get(tk, "Unclassified") for tk in df.columns})
    out = df.copy()
    for _, cols in group_map.groupby(group_map).groups.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            continue
        sub = df[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    return out


def iif(cond: pd.DataFrame, a, b) -> pd.DataFrame:
    """Ternary: cond ? a : b, elementwise. a/b may be DataFrames or scalars."""
    a_df = a if isinstance(a, pd.DataFrame) else pd.DataFrame(float(a), index=cond.index, columns=cond.columns)
    b_df = b if isinstance(b, pd.DataFrame) else pd.DataFrame(float(b), index=cond.index, columns=cond.columns)
    return a_df.where(cond.astype(bool), b_df)


# ═══════════════════════════════════════════════════════════════════════════════
#  TIER 1 — OHLCV + returns only (39 alphas)
# ═══════════════════════════════════════════════════════════════════════════════

def alpha_1(P):
    ret, close = P["returns"], P["close"]
    x = iif(ret < 0, stddev(ret, 20), close)
    x = signedpower(x, 2.0)
    return rank(ts_argmax(x, 5)) - 0.5


def alpha_2(P):
    volume, close, open_ = P["volume"], P["close"], P["open"]
    a = rank(delta(np.log(volume), 2))
    b = rank((close - open_) / open_)
    return -1 * correlation(a, b, 6)


def alpha_3(P):
    return -1 * correlation(rank(P["open"]), rank(P["volume"]), 10)


def alpha_4(P):
    return -1 * ts_rank(rank(P["low"]), 9)


def alpha_6(P):
    return -1 * correlation(P["open"], P["volume"], 10)


def alpha_8(P):
    open_, ret = P["open"], P["returns"]
    x = sum_(open_, 5) * sum_(ret, 5)
    return -1 * rank(x - delay(x, 10))


def alpha_9(P):
    close = P["close"]
    d1 = delta(close, 1)
    cond1 = 0 < ts_min(d1, 5)
    cond2 = ts_max(d1, 5) < 0
    return iif(cond1, d1, iif(cond2, d1, -1 * d1))


def alpha_10(P):
    close = P["close"]
    d1 = delta(close, 1)
    cond1 = 0 < ts_min(d1, 4)
    cond2 = ts_max(d1, 4) < 0
    return rank(iif(cond1, d1, iif(cond2, d1, -1 * d1)))


def alpha_12(P):
    volume, close = P["volume"], P["close"]
    return np.sign(delta(volume, 1)) * (-1 * delta(close, 1))


def alpha_13(P):
    close, volume = P["close"], P["volume"]
    return -1 * rank(covariance(rank(close), rank(volume), 5))


def alpha_14(P):
    open_, volume, ret = P["open"], P["volume"], P["returns"]
    return (-1 * rank(delta(ret, 3))) * correlation(open_, volume, 10)


def alpha_15(P):
    high, volume = P["high"], P["volume"]
    return -1 * sum_(rank(correlation(rank(high), rank(volume), 3)), 3)


def alpha_16(P):
    high, volume = P["high"], P["volume"]
    return -1 * rank(covariance(rank(high), rank(volume), 5))


def alpha_18(P):
    close, open_ = P["close"], P["open"]
    return -1 * rank((stddev((close - open_).abs(), 5) + (close - open_)) + correlation(close, open_, 10))


def alpha_19(P):
    close, ret = P["close"], P["returns"]
    a = -1 * np.sign((close - delay(close, 7)) + delta(close, 7))
    b = 1 + rank(1 + sum_(ret, 250))
    return a * b


def alpha_20(P):
    open_, high, close, low = P["open"], P["high"], P["close"], P["low"]
    a = -1 * rank(open_ - delay(high, 1))
    b = rank(open_ - delay(close, 1))
    c = rank(open_ - delay(low, 1))
    return a * b * c


def alpha_22(P):
    high, volume, close = P["high"], P["volume"], P["close"]
    return -1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))


def alpha_23(P):
    high = P["high"]
    cond = (sum_(high, 20) / 20) < high
    a = -1 * delta(high, 2)
    return iif(cond, a, 0.0)


def alpha_24(P):
    close = P["close"]
    m100 = sum_(close, 100) / 100
    chg = delta(m100, 100) / delay(close, 100)
    cond = chg <= 0.05
    a = -1 * (close - ts_min(close, 100))
    b = -1 * delta(close, 3)
    return iif(cond, a, b)


def alpha_26(P):
    volume, high = P["volume"], P["high"]
    return -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)


def alpha_29(P):
    """
    Faithful simplification: the paper's sum(x,1) and product(x,1) wrap
    trivial 1-day windows (identity operations), so they're omitted below —
    the math is unchanged, just fewer no-op rolling calls.
    """
    close, ret = P["close"], P["returns"]
    d = delta(close - 1, 5)
    r1 = rank(rank(-1 * rank(d)))
    m2 = ts_min(r1, 2)
    sc1 = scale(np.log(m2))
    r4 = rank(rank(sc1))
    part1 = ts_min(r4, 5)
    part2 = ts_rank(delay(-1 * ret, 6), 5)
    return part1 + part2


def alpha_30(P):
    close, volume = P["close"], P["volume"]
    s = (np.sign(close - delay(close, 1)) + np.sign(delay(close, 1) - delay(close, 2))
         + np.sign(delay(close, 2) - delay(close, 3)))
    return ((1.0 - rank(s)) * sum_(volume, 5)) / sum_(volume, 20)


def alpha_33(P):
    open_, close = P["open"], P["close"]
    return rank(-1 * (1 - (open_ / close)))


def alpha_34(P):
    ret, close = P["returns"], P["close"]
    return rank((1 - rank(stddev(ret, 2) / stddev(ret, 5))) + (1 - rank(delta(close, 1))))


def alpha_35(P):
    volume, close, high, low, ret = P["volume"], P["close"], P["high"], P["low"], P["returns"]
    return (ts_rank(volume, 32) * (1 - ts_rank((close + high) - low, 16))) * (1 - ts_rank(ret, 32))


def alpha_37(P):
    open_, close = P["open"], P["close"]
    return rank(correlation(delay(open_ - close, 1), close, 200)) + rank(open_ - close)


def alpha_38(P):
    close, open_ = P["close"], P["open"]
    return (-1 * rank(ts_rank(close, 10))) * rank(close / open_)


def alpha_40(P):
    high, volume = P["high"], P["volume"]
    return (-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)


def alpha_44(P):
    high, volume = P["high"], P["volume"]
    return -1 * correlation(high, rank(volume), 5)


def alpha_45(P):
    close, volume = P["close"], P["volume"]
    a = rank(sum_(delay(close, 5), 20) / 20)
    b = correlation(close, volume, 2)
    c = rank(correlation(sum_(close, 5), sum_(close, 20), 2))
    return -1 * (a * b * c)


def alpha_46(P):
    close = P["close"]
    term = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    cond1 = term > 0.25
    cond2 = term < 0
    fallback = -1.0 * (close - delay(close, 1))
    return iif(cond1, -1.0, iif(cond2, 1.0, fallback))


def alpha_49(P):
    close = P["close"]
    term = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    cond = term < -0.1
    fallback = -1 * (close - delay(close, 1))
    return iif(cond, 1.0, fallback)


def alpha_51(P):
    close = P["close"]
    term = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    cond = term < -0.05
    fallback = -1 * (close - delay(close, 1))
    return iif(cond, 1.0, fallback)


def alpha_52(P):
    low, ret, volume = P["low"], P["returns"], P["volume"]
    a = (-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)
    b = rank((sum_(ret, 240) - sum_(ret, 20)) / 220)
    c = ts_rank(volume, 5)
    return a * b * c


def alpha_53(P):
    close, low, high = P["close"], P["low"], P["high"]
    return -1 * delta(((close - low) - (high - close)) / (close - low), 9)


def alpha_54(P):
    low, close, open_, high = P["low"], P["close"], P["open"], P["high"]
    return (-1 * ((low - close) * (open_ ** 5))) / ((low - high) * (close ** 5))


def alpha_55(P):
    close, low, high, volume = P["close"], P["low"], P["high"], P["volume"]
    x = (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))
    return -1 * correlation(rank(x), rank(volume), 6)


def alpha_60(P):
    close, low, high, volume = P["close"], P["low"], P["high"], P["volume"]
    a = 2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume))
    b = scale(rank(ts_argmax(close, 10)))
    return -1 * (a - b)


def alpha_101(P):
    close, open_, high, low = P["close"], P["open"], P["high"], P["low"]
    return (close - open_) / ((high - low) + 0.001)


# ═══════════════════════════════════════════════════════════════════════════════
#  TIER 2 — needs adv{d} (13 alphas)
# ═══════════════════════════════════════════════════════════════════════════════

def alpha_7(P):
    close, volume = P["close"], P["volume"]
    adv20 = P["adv20"]
    cond = adv20 < volume
    a = (-1 * ts_rank(delta(close, 7).abs(), 60)) * np.sign(delta(close, 7))
    return iif(cond, a, -1.0)


def alpha_17(P):
    close, volume = P["close"], P["volume"]
    adv20 = P["adv20"]
    a = -1 * rank(ts_rank(close, 10))
    b = rank(delta(delta(close, 1), 1))
    c = rank(ts_rank(volume / adv20, 5))
    return a * b * c


def alpha_21(P):
    close, volume = P["close"], P["volume"]
    adv20 = P["adv20"]
    m8, sd8, m2 = sum_(close, 8) / 8, stddev(close, 8), sum_(close, 2) / 2
    cond1 = (m8 + sd8) < m2
    cond2 = m2 < (m8 - sd8)
    cond3 = (volume / adv20) >= 1
    inner = iif(cond3, 1.0, -1.0)
    return iif(cond1, -1.0, iif(cond2, 1.0, inner))


def alpha_28(P):
    low, high, close = P["low"], P["high"], P["close"]
    adv20 = P["adv20"]
    return scale(correlation(adv20, low, 5) + ((high + low) / 2) - close)


def alpha_31(P):
    close, low = P["close"], P["low"]
    adv20 = P["adv20"]
    a = rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, 10))), 10))))
    b = rank(-1 * delta(close, 3))
    c = np.sign(scale(correlation(adv20, low, 12)))
    return a + b + c


def alpha_39(P):
    close, volume, ret = P["close"], P["volume"], P["returns"]
    adv20 = P["adv20"]
    a = -1 * rank(delta(close, 7) * (1 - rank(decay_linear(volume / adv20, 9))))
    b = 1 + rank(sum_(ret, 250))
    return a * b


def alpha_43(P):
    volume, close = P["volume"], P["close"]
    adv20 = P["adv20"]
    return ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)


def alpha_68(P):
    high, adv15, close, low = P["high"], P["adv15"], P["close"], P["low"]
    a = ts_rank(correlation(rank(high), rank(adv15), 8), 13)
    b = rank(delta((close * 0.518371) + (low * (1 - 0.518371)), 1))
    return (a < b).astype(float) * -1


def alpha_85(P):
    high, close, low, volume, adv30 = P["high"], P["close"], P["low"], P["volume"], P["adv30"]
    a = rank(correlation((high * 0.876703) + (close * (1 - 0.876703)), adv30, 9))
    b = rank(correlation(ts_rank((high + low) / 2, 3), ts_rank(volume, 10), 7))
    return a ** b


def alpha_88(P):
    open_, low, high, close, adv60 = P["open"], P["low"], P["high"], P["close"], P["adv60"]
    a = rank(decay_linear((rank(open_) + rank(low)) - (rank(high) + rank(close)), 8))
    b = ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 20), 8), 6), 2)
    return np.minimum(a, b)


def alpha_92(P):
    high, low, close, open_, adv30 = P["high"], P["low"], P["close"], P["open"], P["adv30"]
    cond = ((high + low) / 2 + close) < (low + open_)
    a = ts_rank(decay_linear(cond.astype(float), 14), 18)
    b = ts_rank(decay_linear(correlation(rank(low), rank(adv30), 7), 6), 6)
    return np.minimum(a, b)


def alpha_95(P):
    open_, high, low, adv40 = P["open"], P["high"], P["low"], P["adv40"]
    a = rank(open_ - ts_min(open_, 12))
    b = ts_rank(rank(correlation(sum_((high + low) / 2, 19), sum_(adv40, 19), 12)) ** 5, 11)
    return (a < b).astype(float)


def alpha_99(P):
    high, low, volume, adv60 = P["high"], P["low"], P["volume"], P["adv60"]
    a = rank(correlation(sum_((high + low) / 2, 19), sum_(adv60, 19), 8))
    b = rank(correlation(low, volume, 6))
    return (a < b).astype(float) * -1


# ═══════════════════════════════════════════════════════════════════════════════
#  TIER 4 — needs industry-neutralization via sector_themes.csv (5 alphas)
# ═══════════════════════════════════════════════════════════════════════════════

def alpha_48(P, groups):
    close = P["close"]
    d1, dd1 = delta(close, 1), delta(delay(close, 1), 1)
    num = indneutralize((correlation(d1, dd1, 250) * d1) / close, groups)
    denom = sum_((delta(close, 1) / delay(close, 1)) ** 2, 250)
    return num / denom


def alpha_80(P, groups):
    open_, high, adv10 = P["open"], P["high"], P["adv10"]
    x = indneutralize((open_ * 0.868128) + (high * (1 - 0.868128)), groups)
    a = rank(np.sign(delta(x, 4)))
    b = ts_rank(correlation(high, adv10, 5), 5)
    return (a ** b) * -1


def alpha_82(P, groups):
    open_, volume = P["open"], P["volume"]
    a = rank(decay_linear(delta(open_, 1), 14))
    x = indneutralize(volume, groups)
    b = ts_rank(decay_linear(correlation(x, open_, 17), 6), 13)
    return np.minimum(a, b) * -1


def alpha_90(P, groups):
    close, low, adv40 = P["close"], P["low"], P["adv40"]
    a = rank(close - ts_max(close, 4))
    x = indneutralize(adv40, groups)
    b = ts_rank(correlation(x, low, 5), 3)
    return (a ** b) * -1


def alpha_100(P, groups):
    close, low, high, volume, adv20 = P["close"], P["low"], P["high"], P["volume"], P["adv20"]
    x = ((close - low) - (high - close)) / (high - low) * volume
    part_a = 1.5 * scale(indneutralize(indneutralize(rank(x), groups), groups))
    part_b = scale(indneutralize(correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30)), groups))
    return -1 * ((part_a - part_b) * (volume / adv20))


# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

TIER1_IDS = [1, 2, 3, 4, 6, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24,
             26, 29, 30, 33, 34, 35, 37, 38, 40, 44, 45, 46, 49, 51, 52, 53, 54, 55, 60, 101]
TIER2_IDS = [7, 17, 21, 28, 31, 39, 43, 68, 85, 88, 92, 95, 99]
TIER4_IDS = [48, 80, 82, 90, 100]

ALPHA_FUNCS = {
    1: alpha_1, 2: alpha_2, 3: alpha_3, 4: alpha_4, 6: alpha_6, 7: alpha_7,
    8: alpha_8, 9: alpha_9, 10: alpha_10, 12: alpha_12, 13: alpha_13,
    14: alpha_14, 15: alpha_15, 16: alpha_16, 17: alpha_17, 18: alpha_18,
    19: alpha_19, 20: alpha_20, 21: alpha_21, 22: alpha_22, 23: alpha_23,
    24: alpha_24, 26: alpha_26, 28: alpha_28, 29: alpha_29, 30: alpha_30,
    31: alpha_31, 33: alpha_33, 34: alpha_34, 35: alpha_35, 37: alpha_37,
    38: alpha_38, 39: alpha_39, 40: alpha_40, 43: alpha_43, 44: alpha_44,
    45: alpha_45, 46: alpha_46, 49: alpha_49, 51: alpha_51, 52: alpha_52,
    53: alpha_53, 54: alpha_54, 55: alpha_55, 60: alpha_60, 68: alpha_68,
    85: alpha_85, 88: alpha_88, 92: alpha_92, 95: alpha_95, 99: alpha_99,
    101: alpha_101,
}
# Tier 4 alphas need the extra `groups` argument
ALPHA_FUNCS_NEEDS_GROUPS = {48: alpha_48, 80: alpha_80, 82: alpha_82, 90: alpha_90, 100: alpha_100}

ADV_WINDOWS = (10, 15, 20, 30, 40, 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  CATEGORY TAXONOMY  (see alpha_category_taxonomy.md for the classification rule)
#  MOM = Momentum & Trend      MRV = Mean-Reversion / Exhaustion
#  VOL = Volume Confirmation   CMP = Volatility Compression
#  CND = Candle & Range Pos.   IND = Industry-Relative Strength
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_IDS = {
    "MOM": [8, 9, 10, 19, 46, 49, 51, 52],
    "MRV": [4, 23, 24, 29, 30, 38],
    "VOL": [2, 3, 6, 7, 12, 13, 14, 15, 16, 17, 21, 22, 26, 28, 31, 35,
            39, 40, 43, 44, 45, 55, 68, 85, 88, 92, 95, 99],
    "CMP": [1, 18, 34],
    "CND": [20, 33, 37, 53, 54, 60, 101],
    "IND": [48, 80, 82, 90, 100],
}

ALPHA_DESC = {
    8:   "Acceleration of a 5-day open × return compound measure",
    9:   "Continues a 5-day one-directional move; fades only when choppy",
    10:  "Same as #9, ranked, 4-day window",
    19:  "Pullback-within-trend, scaled by 1-year trend strength",
    46:  "Trend curvature — recent vs. prior 10-day slope (wide threshold)",
    49:  "Same curvature check, tighter threshold",
    51:  "Same curvature check, tightest threshold",
    52:  "Trend durability — how much of the 1-yr gain was outside the last month",
    4:   "Today's relative low-rank vs. its own 9-day extreme",
    23:  "Fades 2-day high move, only when already extended vs. 20d avg",
    24:  "Trend-conditional fade of distance from a 100-day low",
    29:  "Rank-cascade fade of a 5-day price change + lagged reversal term",
    30:  "Fades a 3-day directional streak, volume-weighted",
    38:  "Fades a 10-day closing-rank extreme, weighted by intraday strength",
    2:   "Correlation between volume change and intraday return",
    3:   "Correlation between opening price and volume (ranked)",
    6:   "Correlation between open and volume (raw)",
    7:   "RVOL-gated momentum — continues 7-day trend if volume > 20d avg, else fades",
    12:  "Rewards gains on falling volume — quiet accumulation vs. loud churn",
    13:  "Rank covariance of price level vs. volume",
    14:  "Return-acceleration fade, weighted by open-volume correlation",
    15:  "Correlation of high-rank vs. volume-rank, summed over 3 days",
    16:  "Rank covariance of high vs. volume",
    17:  "10-day close-rank fade, weighted by RVOL rank",
    21:  "Bollinger-position fade with RVOL-≥-1 tiebreaker fallback",
    22:  "Change in the high-volume correlation itself, over time",
    26:  "Rolling max of volume-rank / high-rank correlation",
    28:  "Correlation of adv20 with low, plus range-midpoint vs. close",
    31:  "Momentum-fade decay + reversal fade + sign of adv-low correlation",
    35:  "Volume-rank extremity × weak range position × weak returns",
    39:  "7-day price-change fade, scaled by inverse RVOL-decay rank",
    40:  "Volatility-of-highs fade × high-volume correlation",
    43:  "RVOL rank × price-reversal rank",
    44:  "Correlation of high vs. volume-rank",
    45:  "Lagged price-level rank × close-volume corr. × trend-agreement rank",
    55:  "Stochastic-style range position, correlated with volume",
    68:  "Correlation of high-rank/adv15-rank vs. blended price delta",
    85:  "Blended price/adv30 corr. × range-position/volume corr.",
    88:  "Candle-shape rank vs. close-rank/adv60-rank correlation",
    92:  "Gap-weakness condition vs. low-rank/adv30-rank correlation",
    95:  "Open-extension rank vs. midpoint/adv40 correlation",
    99:  "Midpoint/adv60 corr. vs. low/volume correlation",
    1:   "Recency of a volatility spike (down days) or price extreme (up days)",
    18:  "Volatility of the daily open-close range, ranked",
    34:  "2-day vs. 5-day return volatility — closest of the 57 to BBUW",
    20:  "Pure overnight gap — open vs. yesterday's high/close/low",
    33:  "Intraday strength — close vs. open, ranked",
    37:  "Long-horizon (200-day) version of the open-close relationship",
    53:  "Change in candle-close position within the day's range, over 9 days",
    54:  "Candle-shape ratio (low-close vs. low-high spreads)",
    60:  "Candle-position × volume vs. recency of a 10-day high",
    101: "The purest version — simple candle-body-to-range ratio",
    48:  "Industry-neutralized price-change persistence",
    80:  "Industry-neutralized open/high blend sign-change vs. RVOL(adv10)",
    82:  "Industry-neutralized volume vs. open-delta decay",
    90:  "Industry-neutralized adv40 vs. low, vs. close extension",
    100: "Industry-neutralized candle-position/volume composite (double-neutralized)",
}


def build_alpha_meta() -> dict:
    """{alpha_id_str: {cat, tier, desc}} — written into the output JSON so the
    dashboard reads taxonomy from data instead of importing this module."""
    id_to_cat = {aid: cat for cat, ids in CATEGORY_IDS.items() for aid in ids}
    id_to_tier = {}
    for aid in TIER1_IDS: id_to_tier[aid] = 1
    for aid in TIER2_IDS: id_to_tier[aid] = 2
    for aid in TIER4_IDS: id_to_tier[aid] = 4
    return {
        str(aid): {
            "cat":  id_to_cat.get(aid, "?"),
            "tier": id_to_tier.get(aid),
            "desc": ALPHA_DESC.get(aid, ""),
        }
        for aid in sorted(id_to_cat)
    }


def compute_all_alphas(P: dict, groups: dict) -> dict:
    """Run every Tier 1/2/4 alpha. Returns {alpha_id: DataFrame(date x ticker)}."""
    results = {}
    for aid, fn in ALPHA_FUNCS.items():
        try:
            results[aid] = fn(P)
        except Exception as e:
            log.warning(f"Alpha #{aid} failed: {e}")
    for aid, fn in ALPHA_FUNCS_NEEDS_GROUPS.items():
        try:
            results[aid] = fn(P, groups)
        except Exception as e:
            log.warning(f"Alpha #{aid} failed: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Starting alpha_features — Tier 1 + Tier 2 + Tier 4 (57 alphas)...")

    daily_data = get_latest_daily_watchlist()
    entries = daily_data.get("entries", [])
    if not entries:
        log.error("No entries in daily watchlist. Run daily_screener.py first.")
        return

    tickers = sorted({e["ticker"] for e in entries})
    log.info(f"Fetching {len(tickers)} tickers for alpha computation...")
    bars = batch_fetch_daily(tickers)
    if not bars:
        log.error("No usable bars fetched — aborting.")
        return

    groups = load_industry_groups()

    P = build_panels(bars)
    for d in ADV_WINDOWS:
        P[f"adv{d}"] = adv(P["volume"], P["close"], d)

    log.info(f"Computing 57 alphas across {len(P['close'].columns)} tickers...")
    alpha_dfs = compute_all_alphas(P, groups)

    # ── Latest cross-sectional snapshot + percentile rank ─────────────────
    last_date = P["close"].index[-1]
    features = {tk: {} for tk in P["close"].columns}

    for aid, df in alpha_dfs.items():
        if last_date not in df.index:
            continue
        raw_row = df.loc[last_date]
        rank_row = raw_row.rank(pct=True)
        for tk in df.columns:
            v, r = raw_row.get(tk), rank_row.get(tk)
            if pd.notna(v):
                features.setdefault(tk, {})[f"alpha_{aid}"] = round(float(v), 6)
                features[tk][f"alpha_{aid}_pct"] = round(float(r) * 100, 1) if pd.notna(r) else None

    output = {
        "generated_at": datetime.now().isoformat(),
        "as_of_date": str(last_date.date()) if hasattr(last_date, "date") else str(last_date),
        "universe_size": len(P["close"].columns),
        "tiers": {
            "tier1_ohlcv_only":                    TIER1_IDS,
            "tier2_needs_adv":                      TIER2_IDS,
            "tier4_needs_industry_neutralization":  TIER4_IDS,
        },
        "categories": {cat: ids for cat, ids in CATEGORY_IDS.items()},
        "alpha_meta": build_alpha_meta(),
        "features": features,
    }
    write_json(OUTPUT_JSON, output)

    computed = len(alpha_dfs)
    log.info(f"\n  ALPHA FEATURES COMPLETE — {computed}/57 alphas computed for "
             f"{len(P['close'].columns)} tickers as of {output['as_of_date']}")


if __name__ == "__main__":
    main()
