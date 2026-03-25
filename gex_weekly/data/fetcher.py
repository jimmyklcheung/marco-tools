"""
Options chain + macro data fetcher.
All functions handle yfinance failures gracefully (return empty DataFrame / None).
"""
import logging
import os
import time
from datetime import datetime, date
from typing import Optional

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def _load_config(config_path: str = _CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


# ── Spot price ───────────────────────────────────────────────────────────────

def get_spot(ticker: str) -> float:
    """Fetch latest close price for ticker. Returns 0.0 on failure."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        if hist.empty:
            logger.warning(f"No history for {ticker}")
            return 0.0
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"Failed to fetch spot for {ticker}: {e}")
        return 0.0


# ── Expirations ──────────────────────────────────────────────────────────────

def get_expirations(ticker: str) -> list:
    """Return sorted list of option expiry strings (YYYY-MM-DD)."""
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return []
        return sorted(exps)
    except Exception as e:
        logger.warning(f"Failed to fetch expirations for {ticker}: {e}")
        return []


def get_next_opex(ticker: str) -> Optional[str]:
    """
    Return the next monthly OPEX date (3rd Friday heuristic: day 15–21, weekday==4).
    Returns None if no expirations found.
    """
    exps = get_expirations(ticker)
    today = date.today()
    for exp_str in exps:
        try:
            exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
            if exp < today:
                continue
            # 3rd Friday: day between 15–21, weekday == 4 (Friday)
            if 15 <= exp.day <= 21 and exp.weekday() == 4:
                return exp_str
        except ValueError:
            continue
    return None


# ── Single expiry chain ───────────────────────────────────────────────────────

def get_chain(
    ticker: str,
    expiry: str,
    spot: float,
    spot_range: float = 0.10,
    cache_ttl: int = 300,
) -> pd.DataFrame:
    """
    Fetch options chain for one expiry with local parquet caching.
    Returns empty DataFrame on failure.
    """
    _ensure_cache_dir()
    safe_ticker = ticker.replace("^", "").replace("=", "_").replace("-", "_")
    cache_file = os.path.join(_CACHE_DIR, f"{safe_ticker}_{expiry}.parquet")

    # Cache hit check
    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < cache_ttl:
            try:
                df = pd.read_parquet(cache_file)
                logger.debug(f"Cache hit: {cache_file} (age={age:.0f}s)")
                return df
            except Exception as e:
                logger.warning(f"Cache read failed for {cache_file}: {e}")

    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
    except Exception as e:
        logger.warning(f"Failed to fetch {ticker} {expiry}: {e}")
        return pd.DataFrame()

    frames = []
    exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    dte = max((exp_date - date.today()).days, 0)

    for opt_type, df_raw in [("call", chain.calls), ("put", chain.puts)]:
        if df_raw is None or df_raw.empty:
            continue
        df_raw = df_raw.copy()
        df_raw["type"] = opt_type
        df_raw["expiry"] = expiry
        df_raw["dte"] = dte
        frames.append(df_raw)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Standardise columns
    rename = {
        "strike": "strike",
        "lastPrice": "lastPrice",
        "bid": "bid",
        "ask": "ask",
        "impliedVolatility": "impliedVolatility",
        "openInterest": "openInterest",
        "volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Filter to ±spot_range of spot
    if spot > 0 and spot_range > 0:
        lo = spot * (1 - spot_range)
        hi = spot * (1 + spot_range)
        df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]

    # Fill NaN openInterest and impliedVolatility
    for col in ["openInterest", "impliedVolatility"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Save cache
    try:
        df.to_parquet(cache_file, index=False)
    except Exception as e:
        logger.warning(f"Cache write failed for {cache_file}: {e}")

    return df


# ── All expiries ──────────────────────────────────────────────────────────────

def get_all_chains(
    ticker: str,
    spot: float,
    max_expiries: int = 8,
    spot_range: float = 0.10,
    cache_ttl: int = 300,
) -> pd.DataFrame:
    """
    Fetch and stack options chains across up to max_expiries expiries.
    Returns empty DataFrame if all fail.
    """
    exps = get_expirations(ticker)
    if not exps:
        logger.warning(f"No expirations found for {ticker}")
        return pd.DataFrame()

    exps = exps[:max_expiries]
    frames = []
    for exp in exps:
        df = get_chain(ticker, exp, spot, spot_range, cache_ttl)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.warning(f"No chain data fetched for {ticker}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Fetched {len(combined)} option rows for {ticker} across {len(frames)} expiries")
    return combined


# ── Macro snapshot ────────────────────────────────────────────────────────────

def get_macro_snapshot(macro_tickers: dict, period: str = "3mo") -> pd.DataFrame:
    """
    Download closing prices for macro tickers.
    macro_tickers: dict of {label: yahoo_ticker}
    Returns DataFrame with columns = labels, index = dates.
    """
    ticker_map = {v: k for k, v in macro_tickers.items()}
    tickers = list(macro_tickers.values())
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            logger.warning("Macro snapshot returned empty data")
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        else:
            df = raw[["Close"]] if "Close" in raw.columns else raw
        df = df.rename(columns=ticker_map)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch macro snapshot: {e}")
        return pd.DataFrame()


def get_vix_data(vix_ticker: str = "^VIX", period: str = "5d") -> dict:
    """
    Return dict with current VIX level and 1-day change.
    """
    try:
        hist = yf.Ticker(vix_ticker).history(period=period)
        if hist.empty or len(hist) < 2:
            return {"level": None, "change_1d": None}
        level = float(hist["Close"].iloc[-1])
        change_1d = float(hist["Close"].iloc[-1] - hist["Close"].iloc[-2])
        return {"level": level, "change_1d": change_1d}
    except Exception as e:
        logger.warning(f"Failed to fetch VIX data: {e}")
        return {"level": None, "change_1d": None}
