"""
Black-Scholes Greeks engine — fully vectorised (numpy arrays).
All public functions accept and return numpy arrays or scalars.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ── Black-Scholes primitives ─────────────────────────────────────────────────

def _d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 with safe clipping."""
    T = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_delta(S, K, T, r, sigma, option_type_sign: np.ndarray) -> np.ndarray:
    """
    Black-Scholes delta (vectorised).
    option_type_sign: +1 for calls, -1 for puts.
    """
    T = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1.0
    return np.where(option_type_sign > 0, call_delta, put_delta)


def bs_gamma(S, K, T, r, sigma) -> np.ndarray:
    """Black-Scholes gamma (vectorised, same for calls and puts)."""
    T = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma) -> np.ndarray:
    """Black-Scholes vega (vectorised). Returns vega per 1 vol point (not %)."""
    T = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_vanna(S, K, T, r, sigma) -> np.ndarray:
    """
    Black-Scholes vanna = dDelta/dSigma = dVega/dS (vectorised).
    """
    T = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return -norm.pdf(d1) * d2 / sigma


def bs_charm(S, K, T, r, sigma, option_type_sign: np.ndarray) -> np.ndarray:
    """
    Black-Scholes charm = dDelta/dT (vectorised).

    By put-call parity (C - P = S - Ke^{-rT}), differentiating twice gives
    dDelta_call/dT == dDelta_put/dT, so calls and puts share the same formula.
    The result is the annualised rate of delta change per year of time-to-expiry.
    Divide by 365 in compute_greeks() to get daily dollar exposure (charmex_$).

    option_type_sign is accepted for API consistency but does not alter the output.
    """
    T = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    charm = norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    return charm


# ── Full Greeks computation on a chain DataFrame ─────────────────────────────

def compute_greeks(df: pd.DataFrame, spot: float, r: float, multiplier: int) -> pd.DataFrame:
    """
    Compute all Greeks and $ exposures on a raw options chain DataFrame.

    Required input columns: strike, dte, impliedVolatility, openInterest, type
    Returns df with added columns: delta, gamma, vanna, charm, vega,
                                   gex_$, dex_$, vannex_$, charmex_$
    """
    if df.empty:
        return df

    df = df.copy()

    # Safe numeric conversion
    for col in ["strike", "dte", "impliedVolatility", "openInterest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    K = df["strike"].values.astype(float)
    T = (df["dte"].values.astype(float) / 365.0)
    sigma = df["impliedVolatility"].values.astype(float)
    OI = np.where(np.isnan(df["openInterest"].values), 0, df["openInterest"].values).astype(float)

    # Clip — use 1 calendar day minimum for T to prevent charm blowup on 0-DTE options.
    # 1e-6 years ≈ 31 seconds, which causes charm to explode via 1/(2*T*sigma*sqrt(T)).
    T = np.maximum(T, 1.0 / 365.0)
    sigma = np.maximum(sigma, 1e-6)

    # Option type sign: +1 call, -1 put
    is_call = (df["type"].str.lower() == "call").values
    opt_sign = np.where(is_call, 1.0, -1.0)

    # Dealer sign convention:
    # Market makers are assumed net long calls (retail sells covered calls) and
    # net short puts (institutions buy put protection from dealers).
    # => Long calls: positive gamma → positive GEX contribution.
    # => Short puts: negative gamma → negative GEX contribution.
    dealer_sign = np.where(is_call, 1.0, -1.0)

    S = float(spot)

    df["delta"] = bs_delta(S, K, T, r, sigma, opt_sign)
    df["gamma"] = bs_gamma(S, K, T, r, sigma)
    df["vega"] = bs_vega(S, K, T, r, sigma)
    df["vanna"] = bs_vanna(S, K, T, r, sigma)
    df["charm"] = bs_charm(S, K, T, r, sigma, opt_sign)

    df["gex_$"] = df["gamma"] * OI * multiplier * S ** 2 * dealer_sign
    df["dex_$"] = df["delta"] * OI * multiplier * S
    df["vannex_$"] = df["vanna"] * OI * multiplier * S * dealer_sign
    # charm (B-S) is annualised (dDelta/dT where T is in years).
    # Divide by 365 to convert to a per-calendar-day delta flow in dollars.
    df["charmex_$"] = df["charm"] * OI * multiplier * S * dealer_sign / 365.0

    return df


# ── Aggregations ──────────────────────────────────────────────────────────────

def aggregate_by_strike(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Greeks by strike across all expiries.
    Returns columns: strike, call_oi, put_oi, call_gex, put_gex, net_gex,
                     call_dex, put_dex, net_dex, net_vannex, net_charmex, put_call_oi_ratio
    """
    if df.empty:
        return pd.DataFrame()

    calls = df[df["type"].str.lower() == "call"].copy()
    puts = df[df["type"].str.lower() == "put"].copy()

    def agg(frame, prefix):
        g = frame.groupby("strike").agg(
            oi=("openInterest", "sum"),
            gex=("gex_$", "sum"),
            dex=("dex_$", "sum"),
            vannex=("vannex_$", "sum"),
            charmex=("charmex_$", "sum"),
        ).rename(columns={
            "oi": f"{prefix}_oi",
            "gex": f"{prefix}_gex",
            "dex": f"{prefix}_dex",
            "vannex": f"{prefix}_vannex",
            "charmex": f"{prefix}_charmex",
        })
        return g

    c_agg = agg(calls, "call")
    p_agg = agg(puts, "put")

    result = c_agg.join(p_agg, how="outer").fillna(0)
    result["net_gex"] = result["call_gex"] + result["put_gex"]
    result["net_dex"] = result["call_dex"] + result["put_dex"]
    result["net_vannex"] = result.get("call_vannex", 0) + result.get("put_vannex", 0)
    result["net_charmex"] = result.get("call_charmex", 0) + result.get("put_charmex", 0)
    # PCR: require minimum call OI of 10 contracts to avoid ratio spikes from
    # near-zero denominators.  Strikes with too-thin call OI are excluded (NaN).
    min_call_oi = 10
    result["put_call_oi_ratio"] = np.where(
        result["call_oi"] >= min_call_oi,
        result["put_oi"] / result["call_oi"],
        np.nan
    )

    result = result.reset_index().sort_values("strike")
    return result


def aggregate_by_expiry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Greeks by expiry.
    Returns columns: expiry, dte, net_gex, net_dex, net_oi
    """
    if df.empty:
        return pd.DataFrame()

    result = df.groupby(["expiry", "dte"]).agg(
        net_gex=("gex_$", "sum"),
        net_dex=("dex_$", "sum"),
        net_oi=("openInterest", "sum"),
    ).reset_index().sort_values("dte")
    return result


def aggregate_by_book(
    df: pd.DataFrame,
    dte_max: int,
    dte_min: int = 0,
) -> pd.DataFrame:
    """
    Aggregate by strike for a specific DTE bucket (a 'book').

    Parameters
    ----------
    df      : full options chain with dte column
    dte_max : inclusive upper bound on DTE (e.g. 5 for tactical front-end)
    dte_min : inclusive lower bound (default 0)

    Returns same schema as aggregate_by_strike for the filtered chain.
    """
    if df.empty:
        return pd.DataFrame()
    sub = df[(df["dte"] >= dte_min) & (df["dte"] <= dte_max)].copy()
    if sub.empty:
        return pd.DataFrame()
    return aggregate_by_strike(sub)


# ── Key levels ────────────────────────────────────────────────────────────────

def find_gamma_flip(agg: pd.DataFrame, spot: float) -> float:
    """
    Find the gamma flip level: strike where cumulative net GEX crosses zero.
    Uses linear interpolation. Fallback: strike with minimum |net_gex|.
    """
    if agg.empty:
        return spot

    df = agg.sort_values("strike").copy()
    df["cumgex"] = df["net_gex"].cumsum()

    # Find sign crossings
    signs = np.sign(df["cumgex"].values)
    crossings = np.where(np.diff(signs))[0]

    if len(crossings) == 0:
        # Fallback: strike nearest to spot with smallest |net_gex|
        idx = df["net_gex"].abs().idxmin()
        return float(df.loc[idx, "strike"])

    i = crossings[0]
    K1 = df["strike"].iloc[i]
    K2 = df["strike"].iloc[i + 1]
    G1 = df["cumgex"].iloc[i]
    G2 = df["cumgex"].iloc[i + 1]

    if G2 == G1:
        return float(K1)

    # Linear interpolation
    flip = K1 + (0 - G1) * (K2 - K1) / (G2 - G1)
    return float(flip)


def find_gamma_walls(agg: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Identify gamma walls.
    Call walls: top N strikes by call_gex (most positive).
    Put walls: top N strikes by put_gex (most negative).
    Returns DataFrame: strike, gex_$, type (call_wall / put_wall).
    """
    if agg.empty:
        return pd.DataFrame(columns=["strike", "gex_$", "type"])

    call_walls = (
        agg.nlargest(top_n, "call_gex")[["strike", "call_gex"]]
        .rename(columns={"call_gex": "gex_$"})
        .assign(type="call_wall")
    )
    put_walls = (
        agg.nsmallest(top_n, "put_gex")[["strike", "put_gex"]]
        .rename(columns={"put_gex": "gex_$"})
        .assign(type="put_wall")
    )
    return pd.concat([call_walls, put_walls], ignore_index=True)


def find_max_pain(df: pd.DataFrame) -> float:
    """
    Compute max pain from the nearest expiry only.
    Returns the strike that minimises total intrinsic value for all options (put + call OI).
    """
    if df.empty:
        return 0.0

    nearest_dte = df["dte"].min()
    near = df[df["dte"] == nearest_dte].copy()

    strikes = sorted(near["strike"].unique())
    if not strikes:
        return 0.0

    total_pain = []
    for s in strikes:
        calls = near[near["type"].str.lower() == "call"]
        puts = near[near["type"].str.lower() == "put"]
        call_pain = ((s - calls["strike"]).clip(lower=0) * calls["openInterest"]).sum()
        put_pain = ((puts["strike"] - s).clip(lower=0) * puts["openInterest"]).sum()
        total_pain.append(call_pain + put_pain)

    min_idx = int(np.argmin(total_pain))
    return float(strikes[min_idx])


def opex_expiring_gex(df: pd.DataFrame, opex_date: str) -> dict:
    """
    Summarise GEX expiring on the given OPEX date.
    Returns dict with call/put/net GEX in $M and total OI.
    """
    if df.empty or opex_date is None:
        return {}

    opex_df = df[df["expiry"] == opex_date].copy()
    if opex_df.empty:
        return {"expiry": opex_date, "dte": None, "call_gex_$M": 0, "put_gex_$M": 0,
                "net_gex_$M": 0, "call_dex_$M": 0, "put_dex_$M": 0, "total_oi": 0}

    calls = opex_df[opex_df["type"].str.lower() == "call"]
    puts = opex_df[opex_df["type"].str.lower() == "put"]

    return {
        "expiry": opex_date,
        "dte": int(opex_df["dte"].iloc[0]) if not opex_df.empty else None,
        "call_gex_$M": round(calls["gex_$"].sum() / 1e6, 2),
        "put_gex_$M": round(puts["gex_$"].sum() / 1e6, 2),
        "net_gex_$M": round(opex_df["gex_$"].sum() / 1e6, 2),
        "call_dex_$M": round(calls["dex_$"].sum() / 1e6, 2),
        "put_dex_$M": round(puts["dex_$"].sum() / 1e6, 2),
        "total_oi": int(opex_df["openInterest"].sum()),
    }
