"""
Scenario engine: compute dealer GEX and delta as a function of hypothetical spot.

This replaces the cumulative-by-strike flip heuristic with a physically grounded
model that evaluates net dealer gamma exposure across a configurable spot grid.

Core idea
---------
GEX(S_hyp) = Σ_i  gamma(S_hyp, K_i, T_i, σ_i) * OI_i * mult * S_hyp² * dealer_sign_i

When GEX(S_hyp) > 0  → dealers are net long gamma at that price.
                         Flows are compressive: buy dips, sell rips → "sticky" tape.
When GEX(S_hyp) < 0  → dealers are net short gamma at that price.
                         Flows are expansive: sell dips, buy rips → "slippery" tape.

The zero-crossings of GEX(S_hyp) are the modeled flip levels.

Level classification
--------------------
For each key wall strike K_w:
  - Evaluate GEX(K_w) from the scenario curve
  - Positive GEX → compressive at K_w → PIN candidate (if near spot and front-OI is strong)
  - Negative GEX → expansive at K_w:
      - Below current spot: downside accelerator (slippery zone on dips)
      - At or above spot: resistance / sell-on-rip supply (GEX causes dealers to sell rallies)
  - Near zero: ambiguous / low-confidence

Sign-model sensitivity
-----------------------
The street sign model (dealer long calls, short puts) is a convention.
A stress model flips the signs to check whether the key conclusions are robust.
Conclusions that flip under the stress model are labelled SIGN-SENSITIVE.

Book decomposition
------------------
structural_book : all expiries
tactical_book   : configurable short-dated window (default 0–5 DTE)
opex_book       : next monthly OPEX expiry only
event_book      : configurable event window (if configured)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_GRID_POINTS = 60          # Number of hypothetical spot values in grid
DEFAULT_GRID_HALF_RANGE = 0.08    # ±8% around current spot
DEFAULT_TACTICAL_MAX_DTE = 5      # Front-end / tactical book: 0–5 DTE
DEFAULT_PIN_MAX_DIST_PCT = 0.5    # Max % from spot to label a level "pin"
DEFAULT_PIN_MIN_FRONT_OI = 500    # Min front-expiry OI at strike to confirm pin
AMBIGUOUS_GEX_THRESHOLD = 0.05   # Fraction of max|GEX| below which a level is ambiguous


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """Output of the scenario engine for one instrument + sign model."""
    spot: float
    spot_grid: np.ndarray               # Hypothetical spot values
    net_gex_curve: np.ndarray           # GEX(S_hyp) in dollars
    net_delta_curve: np.ndarray         # Net dealer delta(S_hyp) in dollars
    flip_levels: list[float]            # Zero-crossings (may be empty)
    primary_flip: Optional[float]       # Nearest crossing to current spot (or None)
    flip_status: str                    # "modeled" | "none_in_range" | "undefined"
    flip_confidence: str                # "HIGH" | "MODERATE" | "LOW"
    regime_at_spot: str                 # "LONG_GAMMA" | "SHORT_GAMMA" | "NEUTRAL"
    gex_at_spot: float                  # GEX(spot) in dollars
    sign_model: str                     # "street" | "stress"


@dataclass
class BookResult:
    """GEX aggregation for a specific expiry bucket."""
    label: str                          # "structural" | "tactical" | "opex" | "event"
    expiry_filter: str                  # description of DTE range / expiry
    net_gex: float
    flip_status: str
    primary_flip: Optional[float]
    dominant_expiry: Optional[str]      # Expiry with largest |net_gex| in this book
    dte_range: Tuple[int, int]          # (min_dte, max_dte) in this book
    regime: str
    confidence: str


@dataclass
class LevelClassification:
    """Scenario-derived classification for one key wall strike."""
    strike: float
    wall_type: str                      # "call_wall" | "put_wall"
    gex_at_strike: float                # GEX(strike) from scenario curve
    gex_pct_of_max: float               # |gex_at_strike| / max|gex| in range
    classification: str                 # see CLASSIFICATIONS below
    confidence: str                     # "HIGH" | "MODERATE" | "LOW"
    sign_sensitive: bool                # True if classification changes under stress model
    dist_pct: float                     # (strike - spot) / spot * 100
    front_oi: float                     # OI in the tactical/front bucket at this strike
    interpretation: str                 # Human-readable sentence


# Classification labels (source of truth)
CLASSIFICATIONS = {
    "pin":              "Pin / magnet",
    "resistance":       "Upside resistance / sell-on-rip supply",
    "accelerator":      "Downside accelerator / slippery zone",
    "support":          "Support / dealer buyback zone",
    "concentration":    "Concentration node",
    "ambiguous":        "Ambiguous / low-confidence",
}


# ── Core scenario computation ─────────────────────────────────────────────────

def _bs_gamma_vec(S: float, K: np.ndarray, T: np.ndarray,
                  r: float, sigma: np.ndarray) -> np.ndarray:
    """Vectorised B-S gamma at a single hypothetical spot S."""
    T = np.maximum(T, 1.0 / 365.0)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def _bs_delta_vec(S: float, K: np.ndarray, T: np.ndarray,
                  r: float, sigma: np.ndarray,
                  opt_sign: np.ndarray) -> np.ndarray:
    """Vectorised B-S delta at a single hypothetical spot S."""
    T = np.maximum(T, 1.0 / 365.0)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1.0
    return np.where(opt_sign > 0, call_delta, put_delta)


def _compute_gex_at_spot(
    S_hyp: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
    OI: np.ndarray,
    multiplier: int,
    dealer_sign: np.ndarray,
) -> float:
    """Net dealer GEX if spot were at S_hyp, in dollars."""
    gamma = _bs_gamma_vec(S_hyp, K, T, r, sigma)
    return float(np.sum(gamma * OI * multiplier * S_hyp ** 2 * dealer_sign))


def _compute_delta_at_spot(
    S_hyp: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    sigma: np.ndarray,
    OI: np.ndarray,
    multiplier: int,
    dealer_sign: np.ndarray,
    opt_sign: np.ndarray,
) -> float:
    """Net dealer delta exposure if spot were at S_hyp, in dollars."""
    delta = _bs_delta_vec(S_hyp, K, T, r, sigma, opt_sign)
    return float(np.sum(delta * OI * multiplier * S_hyp * dealer_sign))


def compute_scenario(
    chain: pd.DataFrame,
    spot: float,
    r: float,
    multiplier: int,
    grid_points: int = DEFAULT_GRID_POINTS,
    grid_half_range: float = DEFAULT_GRID_HALF_RANGE,
    sign_model: str = "street",
) -> ScenarioResult:
    """
    Compute GEX(S_hyp) and delta(S_hyp) across a hypothetical spot grid.

    Parameters
    ----------
    chain       : options chain DataFrame (output of compute_greeks)
    spot        : current spot price
    r           : risk-free rate
    multiplier  : contract multiplier
    grid_points : number of points in the spot grid
    grid_half_range : ±fraction of spot to span in the grid
    sign_model  : "street" (standard) or "stress" (reversed signs)
    """
    if chain.empty:
        return _empty_scenario(spot, sign_model)

    K = chain["strike"].values.astype(float)
    T = np.maximum(chain["dte"].values.astype(float) / 365.0, 1.0 / 365.0)
    sigma = np.maximum(chain["impliedVolatility"].values.astype(float), 1e-6)
    OI = np.where(np.isnan(chain["openInterest"].values), 0,
                  chain["openInterest"].values).astype(float)
    is_call = (chain["type"].str.lower() == "call").values
    opt_sign = np.where(is_call, 1.0, -1.0)

    # Sign model: street = standard (calls +, puts -); stress = flipped
    if sign_model == "street":
        dealer_sign = np.where(is_call, 1.0, -1.0)
    else:
        dealer_sign = np.where(is_call, -1.0, 1.0)

    # Build spot grid — finer resolution near current spot
    lo = spot * (1 - grid_half_range)
    hi = spot * (1 + grid_half_range)
    spot_grid = np.linspace(lo, hi, grid_points)

    # Compute GEX and delta at each grid point
    gex_curve = np.array([
        _compute_gex_at_spot(s, K, T, r, sigma, OI, multiplier, dealer_sign)
        for s in spot_grid
    ])
    delta_curve = np.array([
        _compute_delta_at_spot(s, K, T, r, sigma, OI, multiplier, dealer_sign, opt_sign)
        for s in spot_grid
    ])

    # Find zero crossings in GEX curve
    flip_levels = _find_zero_crossings(spot_grid, gex_curve)
    primary_flip, flip_status = _select_primary_flip(flip_levels, spot, spot_grid)
    flip_confidence = _flip_confidence(flip_levels, gex_curve, spot_grid, spot)

    # Regime at current spot
    gex_at_spot = _interpolate_at_spot(spot_grid, gex_curve, spot)
    if abs(gex_at_spot) < 0.5e9:
        regime_at_spot = "NEUTRAL"
    elif gex_at_spot > 0:
        regime_at_spot = "LONG_GAMMA"
    else:
        regime_at_spot = "SHORT_GAMMA"

    return ScenarioResult(
        spot=spot,
        spot_grid=spot_grid,
        net_gex_curve=gex_curve,
        net_delta_curve=delta_curve,
        flip_levels=flip_levels,
        primary_flip=primary_flip,
        flip_status=flip_status,
        flip_confidence=flip_confidence,
        regime_at_spot=regime_at_spot,
        gex_at_spot=gex_at_spot,
        sign_model=sign_model,
    )


def _find_zero_crossings(grid: np.ndarray, curve: np.ndarray) -> list[float]:
    """Linear interpolation of all zero-crossings."""
    crossings = []
    signs = np.sign(curve)
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1] and signs[i] != 0 and signs[i + 1] != 0:
            # Linear interpolation
            g1, g2 = curve[i], curve[i + 1]
            s1, s2 = grid[i], grid[i + 1]
            cross = s1 + (0 - g1) * (s2 - s1) / (g2 - g1)
            crossings.append(float(cross))
    return crossings


def _select_primary_flip(
    flip_levels: list[float], spot: float, grid: np.ndarray
) -> Tuple[Optional[float], str]:
    """Select the nearest flip to spot; classify status."""
    if not flip_levels:
        return None, "none_in_range"
    primary = min(flip_levels, key=lambda f: abs(f - spot))
    return primary, "modeled"


def _flip_confidence(
    flip_levels: list[float],
    gex_curve: np.ndarray,
    grid: np.ndarray,
    spot: float,
) -> str:
    """
    Confidence in the primary flip:
    - HIGH   : single crossing, GEX changes sign cleanly, spot not near grid edge
    - MODERATE : multiple crossings or GEX magnitude low near crossing
    - LOW    : no crossing, or crossing very close to grid boundary
    """
    if not flip_levels:
        return "LOW"
    if len(flip_levels) > 2:
        return "LOW"
    # Check that spot is not within 10% of grid edge
    lo, hi = grid[0], grid[-1]
    edge_margin = (hi - lo) * 0.10
    if spot < lo + edge_margin or spot > hi - edge_margin:
        return "MODERATE"
    # Check GEX magnitude on both sides of the crossing
    max_abs = np.abs(gex_curve).max()
    if max_abs < 0.1e9:
        return "LOW"
    if len(flip_levels) > 1:
        return "MODERATE"
    return "HIGH"


def _interpolate_at_spot(
    grid: np.ndarray, curve: np.ndarray, spot: float
) -> float:
    """Linear interpolation of curve value at spot."""
    if spot <= grid[0]:
        return float(curve[0])
    if spot >= grid[-1]:
        return float(curve[-1])
    idx = np.searchsorted(grid, spot) - 1
    idx = max(0, min(idx, len(grid) - 2))
    frac = (spot - grid[idx]) / (grid[idx + 1] - grid[idx])
    return float(curve[idx] + frac * (curve[idx + 1] - curve[idx]))


def _empty_scenario(spot: float, sign_model: str) -> ScenarioResult:
    return ScenarioResult(
        spot=spot,
        spot_grid=np.array([spot]),
        net_gex_curve=np.array([0.0]),
        net_delta_curve=np.array([0.0]),
        flip_levels=[],
        primary_flip=None,
        flip_status="undefined",
        flip_confidence="LOW",
        regime_at_spot="NEUTRAL",
        gex_at_spot=0.0,
        sign_model=sign_model,
    )


# ── Sign-model sensitivity ────────────────────────────────────────────────────

@dataclass
class SignSensitivityResult:
    """Comparison of street vs stress sign model conclusions."""
    regime_stable: bool         # Regime is same under both sign models
    flip_stable: bool           # Flip level within 1% under both models
    street: ScenarioResult
    stress: ScenarioResult
    unstable_fields: list[str]


def check_sign_sensitivity(
    chain: pd.DataFrame,
    spot: float,
    r: float,
    multiplier: int,
) -> SignSensitivityResult:
    """
    Run scenario under street AND stress sign model.
    Compare: regime at spot, flip level, flip status.
    """
    street = compute_scenario(chain, spot, r, multiplier, sign_model="street")
    stress = compute_scenario(chain, spot, r, multiplier, sign_model="stress")

    unstable = []
    regime_stable = (street.regime_at_spot == stress.regime_at_spot)
    if not regime_stable:
        unstable.append("regime")

    flip_stable = True
    if street.flip_status == "modeled" and stress.flip_status == "modeled":
        pf_s = street.primary_flip
        pf_st = stress.primary_flip
        if pf_s and pf_st and abs(pf_s - pf_st) / spot > 0.01:
            flip_stable = False
            unstable.append("flip_level")
    elif street.flip_status != stress.flip_status:
        flip_stable = False
        unstable.append("flip_status")

    return SignSensitivityResult(
        regime_stable=regime_stable,
        flip_stable=flip_stable,
        street=street,
        stress=stress,
        unstable_fields=unstable,
    )


# ── Level classification ──────────────────────────────────────────────────────

def classify_level(
    strike: float,
    wall_type: str,           # "call_wall" or "put_wall"
    scenario: ScenarioResult,
    stress_scenario: ScenarioResult,
    by_str: pd.DataFrame,
    spot: float,
    tactical_max_dte: int = DEFAULT_TACTICAL_MAX_DTE,
    pin_max_dist_pct: float = DEFAULT_PIN_MAX_DIST_PCT,
) -> LevelClassification:
    """
    Classify a key wall strike using the scenario GEX curve.

    Classification logic (derived from local GEX sign, not hardcoded):
    ─────────────────────────────────────────────────────────────────
    GEX(strike) > 0  → dealers are NET LONG GAMMA at this level
                        flows are compressive (buy dips / sell rips)
      + near spot + strong front OI → PIN (pinning/magnetic pull)
      + above spot                  → can still be compressive resistance
      + otherwise                   → concentration node (compressive, not a pin)

    GEX(strike) < 0  → dealers are NET SHORT GAMMA at this level
                        flows are expansive (sell dips / buy rips)
      + below spot → DOWNSIDE ACCELERATOR (slippery on dips)
      + above spot → RESISTANCE / sell-on-rip supply (expansive if spot rallies here)

    Note: "above-spot put wall" has NEGATIVE GEX (short gamma at that level).
    Under our sign model, this creates SELL-ON-RIP pressure if spot rallies to it,
    NOT overhead support / buyback. The old "near-ATM pin" label for above-spot put
    walls was sign-inconsistent — only label PIN if GEX(strike) > 0.

    If GEX(strike) ≈ 0 or classification differs between street/stress → AMBIGUOUS.
    """
    dist_pct = (strike - spot) / spot * 100

    # Evaluate GEX at the key strike from both scenario models
    gex_at_str = _interpolate_at_spot(scenario.spot_grid, scenario.net_gex_curve, strike)
    gex_stress = _interpolate_at_spot(stress_scenario.spot_grid, stress_scenario.net_gex_curve, strike)

    max_abs_gex = float(np.abs(scenario.net_gex_curve).max()) or 1.0
    gex_pct = abs(gex_at_str) / max_abs_gex

    # Get front-end OI at this strike
    front_oi = _front_oi_at_strike(by_str, strike)

    # Classify under street model
    cls_street = _classify_single(
        strike, gex_at_str, dist_pct, front_oi, gex_pct,
        pin_max_dist_pct, scenario,
    )
    # Classify under stress model
    cls_stress = _classify_single(
        strike, gex_stress, dist_pct, front_oi, gex_pct,
        pin_max_dist_pct, stress_scenario,
    )

    sign_sensitive = (cls_street != cls_stress)

    # If sign-sensitive, downgrade to concentration/ambiguous
    if sign_sensitive:
        final_cls = "ambiguous" if gex_pct < 0.3 else "concentration"
        confidence = "LOW"
    elif gex_pct < AMBIGUOUS_GEX_THRESHOLD:
        final_cls = "ambiguous"
        confidence = "LOW"
    else:
        final_cls = cls_street
        confidence = "HIGH" if gex_pct > 0.4 else "MODERATE"

    interpretation = _build_interpretation(
        final_cls, strike, dist_pct, gex_at_str, sign_sensitive, front_oi,
    )

    return LevelClassification(
        strike=strike,
        wall_type=wall_type,
        gex_at_strike=gex_at_str,
        gex_pct_of_max=gex_pct,
        classification=final_cls,
        confidence=confidence,
        sign_sensitive=sign_sensitive,
        dist_pct=dist_pct,
        front_oi=front_oi,
        interpretation=interpretation,
    )


def _classify_single(
    strike: float,
    gex_at_strike: float,
    dist_pct: float,
    front_oi: float,
    gex_pct: float,
    pin_max_dist_pct: float,
    scenario: ScenarioResult,
) -> str:
    """Core classification logic for one sign model."""
    long_gamma = gex_at_strike > 0

    if long_gamma:
        # Positive GEX at this strike = compressive / mean-reverting
        is_near = abs(dist_pct) <= pin_max_dist_pct
        has_front_oi = front_oi >= DEFAULT_PIN_MIN_FRONT_OI
        if is_near and has_front_oi:
            return "pin"
        return "concentration"
    else:
        # Negative GEX = expansive / momentum-following
        if dist_pct < 0:
            # Strike is BELOW current spot → slippery on dips
            return "accelerator"
        else:
            # Strike is ABOVE current spot → resistance / sell-on-rip
            # IMPORTANT: This applies to BOTH call walls and put walls above spot.
            # A put wall above spot has NEGATIVE GEX (short gamma) under street convention.
            # Short gamma above spot = if spot rallies here, dealers sell into the rally.
            # This is RESISTANCE, not a pin or overhead support.
            return "resistance"


def _front_oi_at_strike(by_str: pd.DataFrame, strike: float) -> float:
    """Total OI at the strike in the front-end bucket (if available)."""
    if by_str.empty:
        return 0.0
    # Look for the nearest strike (floating point comparison)
    tolerance = 0.5
    mask = (by_str["strike"] - strike).abs() <= tolerance
    if not mask.any():
        return 0.0
    row = by_str[mask].iloc[0]
    return float(row.get("call_oi", 0) + row.get("put_oi", 0))


def _build_interpretation(
    cls: str, strike: float, dist_pct: float,
    gex_at_strike: float, sign_sensitive: bool, front_oi: float,
) -> str:
    """
    Build a concise, trader-language interpretation sentence.
    All language is derived from the classification, not hardcoded templates.
    """
    loc = f"{dist_pct:+.1f}% from spot"
    gex_b = gex_at_strike / 1e9

    qual = " [sign-sensitive — low confidence]" if sign_sensitive else ""

    if cls == "pin":
        return (
            f"Pin / magnet ({loc}) — strong front-end OI ({front_oi:,.0f} contracts); "
            f"long-gamma at this strike creates mean-reversion pull near expiry{qual}"
        )
    elif cls == "resistance":
        return (
            f"Upside resistance / sell-on-rip supply ({loc}) — "
            f"short-gamma at this level; dealers sell rallies into this strike"
            f"{qual}"
        )
    elif cls == "accelerator":
        return (
            f"Downside accelerator / slippery zone ({loc}) — "
            f"short-gamma below spot; dealers amplify moves through this level"
            f"{qual}"
        )
    elif cls == "support":
        return (
            f"Support / dealer buyback zone ({loc}) — "
            f"long-gamma; dealer hedging creates buying pressure on approach"
            f"{qual}"
        )
    elif cls == "concentration":
        return (
            f"Concentration node ({loc}) — "
            f"high GEX concentration but {'sign-sensitive' if sign_sensitive else 'low front-end OI'}; "
            f"may act as a pivot but directional flow uncertain{qual}"
        )
    else:
        return (
            f"Ambiguous ({loc}) — low GEX magnitude or sign-sensitive; "
            f"insufficient confidence to classify directional flow{qual}"
        )


# ── Book decomposition ────────────────────────────────────────────────────────

def compute_books(
    chain: pd.DataFrame,
    by_str: pd.DataFrame,
    spot: float,
    r: float,
    multiplier: int,
    tactical_max_dte: int = DEFAULT_TACTICAL_MAX_DTE,
    opex_date: Optional[str] = None,
    event_dte_range: Optional[Tuple[int, int]] = None,
    grid_points: int = 40,
    grid_half_range: float = 0.06,
) -> dict[str, BookResult]:
    """
    Compute structural, tactical, OPEX, and (optionally) event books.

    Returns dict keyed by book label.
    """
    books = {}

    # Structural: all expiries
    books["structural"] = _compute_book(
        chain, spot, r, multiplier, label="structural",
        dte_range=(0, 9999), expiry_filter="all expiries",
        grid_points=grid_points, grid_half_range=grid_half_range,
    )

    # Tactical: front-end
    books["tactical"] = _compute_book(
        chain, spot, r, multiplier, label="tactical",
        dte_range=(0, tactical_max_dte),
        expiry_filter=f"0–{tactical_max_dte} DTE",
        grid_points=grid_points, grid_half_range=grid_half_range,
    )

    # OPEX: next monthly expiry
    if opex_date:
        opex_chain = chain[chain["expiry"] == opex_date] if "expiry" in chain.columns else pd.DataFrame()
        if not opex_chain.empty:
            dte_val = int(opex_chain["dte"].iloc[0]) if not opex_chain.empty else 0
            books["opex"] = _compute_book(
                opex_chain, spot, r, multiplier, label="opex",
                dte_range=(dte_val, dte_val),
                expiry_filter=f"OPEX {opex_date}",
                grid_points=grid_points, grid_half_range=grid_half_range,
            )

    # Event: configurable window
    if event_dte_range:
        lo_dte, hi_dte = event_dte_range
        books["event"] = _compute_book(
            chain, spot, r, multiplier, label="event",
            dte_range=(lo_dte, hi_dte),
            expiry_filter=f"{lo_dte}–{hi_dte} DTE (event window)",
            grid_points=grid_points, grid_half_range=grid_half_range,
        )

    return books


def _compute_book(
    chain: pd.DataFrame,
    spot: float,
    r: float,
    multiplier: int,
    label: str,
    dte_range: Tuple[int, int],
    expiry_filter: str,
    grid_points: int,
    grid_half_range: float,
) -> BookResult:
    lo_dte, hi_dte = dte_range
    if "dte" in chain.columns:
        sub = chain[(chain["dte"] >= lo_dte) & (chain["dte"] <= hi_dte)]
    else:
        sub = chain

    if sub.empty:
        return BookResult(
            label=label, expiry_filter=expiry_filter,
            net_gex=0.0, flip_status="undefined", primary_flip=None,
            dominant_expiry=None, dte_range=(lo_dte, hi_dte),
            regime="NEUTRAL", confidence="LOW",
        )

    # Net GEX is already computed in chain
    if "gex_$" in sub.columns:
        net_gex = float(sub["gex_$"].sum())
    else:
        net_gex = 0.0

    # Scenario for this book
    sc = compute_scenario(sub, spot, r, multiplier,
                          grid_points=grid_points, grid_half_range=grid_half_range)

    # Dominant expiry
    if "expiry" in sub.columns and "gex_$" in sub.columns:
        dom = sub.groupby("expiry")["gex_$"].apply(lambda x: x.abs().sum())
        dominant_expiry = str(dom.idxmax()) if not dom.empty else None
    else:
        dominant_expiry = None

    actual_dte_range = (int(sub["dte"].min()), int(sub["dte"].max())) if "dte" in sub.columns else (lo_dte, hi_dte)

    if abs(net_gex) < 0.5e9:
        regime = "NEUTRAL"
    elif net_gex > 0:
        regime = "LONG_GAMMA"
    else:
        regime = "SHORT_GAMMA"

    return BookResult(
        label=label,
        expiry_filter=expiry_filter,
        net_gex=net_gex,
        flip_status=sc.flip_status,
        primary_flip=sc.primary_flip,
        dominant_expiry=dominant_expiry,
        dte_range=actual_dte_range,
        regime=regime,
        confidence=sc.flip_confidence,
    )


# ── Roll detection ────────────────────────────────────────────────────────────

@dataclass
class RollStatus:
    """Expiry-roll awareness: did the dominant front-end expiry change?"""
    roll_detected: bool
    prior_front_expiry: Optional[str]
    current_front_expiry: Optional[str]
    roll_note: str


def detect_roll(
    current_front_expiry: Optional[str],
    prior_front_expiry: Optional[str],
    prior_flip: Optional[float],
    current_flip: Optional[float],
    spot: float,
) -> RollStatus:
    """
    Detect if the front-end expiry has rolled (expired options dropped from chain).

    When a roll is detected, level shifts should be attributed to roll mechanics,
    not new positioning.
    """
    roll_detected = (
        prior_front_expiry is not None
        and current_front_expiry is not None
        and prior_front_expiry != current_front_expiry
        and prior_front_expiry < current_front_expiry
    )

    if roll_detected:
        flip_shift = ""
        if prior_flip is not None and current_flip is not None:
            delta_pct = (current_flip - prior_flip) / spot * 100
            flip_shift = (
                f" Flip shifted {delta_pct:+.1f}% — likely roll attribution, "
                f"not new positioning."
            )
        note = (
            f"Expiry roll detected: {prior_front_expiry} → {current_front_expiry}."
            f" Level changes may reflect expired contracts, not fresh hedging changes."
            f"{flip_shift}"
        )
    elif prior_front_expiry is None:
        note = "No prior run available for roll comparison."
    else:
        note = ""

    return RollStatus(
        roll_detected=roll_detected,
        prior_front_expiry=prior_front_expiry,
        current_front_expiry=current_front_expiry,
        roll_note=note,
    )
