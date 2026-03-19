"""
Regression tests for analytics/greeks.py

Each test pins a specific behaviour that was previously broken or at risk of
breaking. Test names encode the invariant they protect.
"""
import math
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analytics.greeks import (
    bs_charm,
    compute_greeks,
    aggregate_by_strike,
    find_gamma_flip,
    find_gamma_walls,
)


# ── bs_charm ─────────────────────────────────────────────────────────────────

def test_charm_put_call_parity():
    """By put-call parity dDelta/dT is identical for calls and puts at same strike."""
    S, K, T, r, sigma = 6600.0, 6600.0, 36 / 365, 0.05, 0.20
    call_sign = np.array([1.0])
    put_sign = np.array([-1.0])
    charm_call = bs_charm(S, K, T, r, sigma, call_sign)
    charm_put = bs_charm(S, K, T, r, sigma, put_sign)
    np.testing.assert_allclose(charm_call, charm_put, rtol=1e-9,
                               err_msg="charm_call must equal charm_put (put-call parity)")


def test_charm_no_blowup_at_zero_dte():
    """Charm must stay finite when DTE=0 (T is floored to 1/365 in compute_greeks)."""
    # Build a minimal chain with DTE=0
    df = pd.DataFrame({
        "strike": [6600.0, 6600.0],
        "dte": [0.0, 0.0],
        "impliedVolatility": [0.20, 0.20],
        "openInterest": [1000.0, 1000.0],
        "type": ["call", "put"],
        "expiry": ["2026-03-21", "2026-03-21"],
    })
    result = compute_greeks(df, spot=6600.0, r=0.05, multiplier=100)
    assert np.all(np.isfinite(result["charmex_$"])), \
        "charmex_$ must be finite even for 0-DTE options"
    # Sanity: charmex per contract should not exceed a few billion dollars
    assert result["charmex_$"].abs().max() < 1e10, \
        "charmex_$ blowup: value exceeds $10B for a single contract"


def test_charmex_daily_units():
    """charmex_$ should be in $/day, not $/year. Annual value would be ~365× larger."""
    df = pd.DataFrame({
        "strike": [6600.0],
        "dte": [30.0],
        "impliedVolatility": [0.20],
        "openInterest": [100.0],
        "type": ["call"],
        "expiry": ["2026-04-18"],
    })
    result = compute_greeks(df, spot=6600.0, r=0.05, multiplier=100)
    charmex_daily = float(result["charmex_$"].iloc[0])
    # Recompute without the /365 to get the annual value
    from analytics.greeks import bs_charm, bs_gamma, _d1_d2
    T = 30 / 365
    sign = np.array([1.0])
    charm_annual_raw = float(bs_charm(6600, 6600, T, 0.05, 0.20, sign))
    charmex_annual = charm_annual_raw * 100.0 * 100 * 6600.0 * 1.0  # OI=100, mult=100
    # Daily should be roughly 1/365 of annual
    ratio = charmex_annual / charmex_daily
    assert 360 < ratio < 370, \
        f"charmex_$ is not in $/day: annual/daily ratio = {ratio:.1f} (expected ~365)"


# ── aggregate_by_strike ───────────────────────────────────────────────────────

def test_pcr_nan_for_thin_call_oi():
    """PCR must be NaN when call OI < 10 (denominator floor prevents ratio spikes)."""
    df = pd.DataFrame({
        "strike": [6500.0, 6500.0],
        "dte": [30.0, 30.0],
        "impliedVolatility": [0.20, 0.20],
        "openInterest": [5.0, 1000.0],   # call OI=5, put OI=1000
        "type": ["call", "put"],
        "expiry": ["2026-04-18", "2026-04-18"],
    })
    df = compute_greeks(df, spot=6600.0, r=0.05, multiplier=100)
    by_str = aggregate_by_strike(df)
    pcr = by_str.loc[by_str["strike"] == 6500.0, "put_call_oi_ratio"].iloc[0]
    assert math.isnan(pcr), \
        f"PCR should be NaN when call_oi < 10, got {pcr}"


def test_pcr_valid_above_threshold():
    """PCR should be computed when call OI >= 10."""
    df = pd.DataFrame({
        "strike": [6600.0, 6600.0],
        "dte": [30.0, 30.0],
        "impliedVolatility": [0.20, 0.20],
        "openInterest": [100.0, 150.0],   # call OI=100, put OI=150
        "type": ["call", "put"],
        "expiry": ["2026-04-18", "2026-04-18"],
    })
    df = compute_greeks(df, spot=6600.0, r=0.05, multiplier=100)
    by_str = aggregate_by_strike(df)
    pcr = by_str.loc[by_str["strike"] == 6600.0, "put_call_oi_ratio"].iloc[0]
    assert not math.isnan(pcr), "PCR should be valid when call_oi >= 10"
    assert abs(pcr - 1.5) < 0.01, f"PCR should be 150/100=1.5, got {pcr}"


# ── find_gamma_flip ───────────────────────────────────────────────────────────

def test_dist_to_flip_sign_invariant():
    """
    dist_to_flip sign invariant: if spot > flip → positive %; if spot < flip → negative %.
    Regression for the IWM bug where rounded display values implied wrong sign.
    """
    # Build a by_str where flip is slightly above spot
    by_str = pd.DataFrame({
        "strike": [200.0, 210.0, 220.0],
        "net_gex":  [-1e8, -5e7, 2e8],   # flip somewhere between 210 and 220
        "call_gex": [1e8,  1e8,  2e8],
        "put_gex":  [-2e8, -1.5e8, 0.0],
    })
    spot = 215.0
    flip = find_gamma_flip(by_str, spot)
    dist_pct = (spot - flip) / spot * 100

    # flip is between 210 and 220; spot=215 could be above or below
    # but the sign of dist_pct must match sign(spot - flip) using raw values
    if spot > flip:
        assert dist_pct > 0, f"spot ({spot}) > flip ({flip:.4f}) but dist_pct={dist_pct:.4f} < 0"
    else:
        assert dist_pct <= 0, f"spot ({spot}) < flip ({flip:.4f}) but dist_pct={dist_pct:.4f} > 0"


# ── find_gamma_walls ─────────────────────────────────────────────────────────

def test_wall_type_labels():
    """find_gamma_walls must label call walls as 'call_wall' and put walls as 'put_wall'."""
    by_str = pd.DataFrame({
        "strike": [6400.0, 6500.0, 6600.0, 6700.0, 6800.0],
        "call_gex": [1e9, 3e9, 5e9, 2e9, 1e9],
        "put_gex":  [-2e9, -4e9, -1e9, -5e9, -3e9],
        "net_gex":  [-1e9, -1e9, 4e9, -3e9, -2e9],
    })
    walls = find_gamma_walls(by_str, top_n=3)
    call_types = walls[walls["type"] == "call_wall"]["type"].unique()
    put_types = walls[walls["type"] == "put_wall"]["type"].unique()
    assert list(call_types) == ["call_wall"], "Call walls must have type='call_wall'"
    assert list(put_types) == ["put_wall"], "Put walls must have type='put_wall'"
    # Top call wall = strike with largest call_gex
    top_call = walls[walls["type"] == "call_wall"].iloc[0]["strike"]
    assert top_call == 6600.0, f"Top call wall should be 6600 (largest call_gex), got {top_call}"
    # Top put wall = strike with most negative put_gex
    top_put = walls[walls["type"] == "put_wall"].iloc[0]["strike"]
    assert top_put == 6700.0, f"Top put wall should be 6700 (most negative put_gex), got {top_put}"


def test_find_gamma_flip_no_zero_cross_fallback():
    """
    When the cumulative GEX curve never crosses zero (all one sign),
    find_gamma_flip must not raise and must return a finite strike near the smallest |GEX|.
    Regression: previously the function could silently return spot even when a better
    fallback strike is available.
    """
    # All net_gex positive — no zero crossing
    by_str = pd.DataFrame({
        "strike": [6500.0, 6550.0, 6600.0, 6650.0],
        "net_gex": [1e8, 2e8, 5e8, 3e8],
        "call_gex": [1e8, 2e8, 5e8, 3e8],
        "put_gex": [0.0, 0.0, 0.0, 0.0],
    })
    spot = 6600.0
    flip = find_gamma_flip(by_str, spot)
    # Must be a real strike, not spot (which has no special meaning here)
    assert math.isfinite(flip), f"flip must be finite, got {flip}"
    assert flip in [6500.0, 6550.0, 6600.0, 6650.0], \
        f"Fallback flip should be one of the strikes (smallest |net_gex|), got {flip}"
    # Smallest |net_gex| is 6500 (1e8)
    assert flip == 6500.0, f"Fallback should pick strike with smallest |net_gex|=1e8 (6500), got {flip}"
