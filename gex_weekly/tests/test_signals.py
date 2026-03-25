"""
Regression tests for analytics/signals.py
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analytics.signals import (
    signal_put_skew,
    signal_vanna_flow,
    signal_flip_proximity,
    signal_charm_flow,
)


# ── signal_put_skew ───────────────────────────────────────────────────────────

def test_put_skew_zero_call_gex_is_bearish():
    """
    Regression: when call_gex == 0, the old code short-circuited to ratio=0 →
    NEUTRAL_BALANCED → score=+0.2 (mildly bullish). With heavy put GEX and zero
    call GEX the correct result is maximum bearish.
    """
    sig = signal_put_skew(
        total_put_gex=-5e9,
        total_call_gex=0.0,
        spot=6600.0,
        put_wall=6500.0,
    )
    assert sig.score < 0, \
        f"Zero call_gex + heavy put_gex must produce negative (bearish) score, got {sig.score}"
    assert sig.value in ("BEARISH_SKEW_HEAVY", "BEARISH_SKEW_MODERATE"), \
        f"Expected bearish skew value, got '{sig.value}'"


def test_put_skew_balanced_book():
    """Balanced call/put GEX should give neutral or mildly positive score."""
    sig = signal_put_skew(
        total_put_gex=-1e9,
        total_call_gex=1e9,
        spot=6600.0,
        put_wall=6500.0,
    )
    assert sig.value == "NEUTRAL_BALANCED", \
        f"Balanced book should be NEUTRAL_BALANCED, got '{sig.value}'"
    assert sig.score > 0, f"Balanced book score should be positive (0.2), got {sig.score}"


def test_put_skew_heavy_bearish():
    """Put GEX > 1.5× call GEX should trigger BEARISH_SKEW_HEAVY."""
    sig = signal_put_skew(
        total_put_gex=-3e9,
        total_call_gex=1e9,
        spot=6600.0,
        put_wall=6400.0,
    )
    assert sig.value == "BEARISH_SKEW_HEAVY", \
        f"3:1 put/call GEX should be BEARISH_SKEW_HEAVY, got '{sig.value}'"
    assert sig.score == -0.8


# ── signal_vanna_flow ─────────────────────────────────────────────────────────

def test_vanna_score_dampened_by_rising_vix():
    """Rising VIX must dampen a positive vanna score."""
    sig_no_vix = signal_vanna_flow(total_vannex=5e8, vix_change_1d=None)
    sig_rising_vix = signal_vanna_flow(total_vannex=5e8, vix_change_1d=+2.5)
    assert sig_rising_vix.score < sig_no_vix.score, \
        "Rising VIX should dampen positive vanna score"
    assert "_DAMPENED" in sig_rising_vix.value, \
        "Dampened vanna value should include '_DAMPENED' suffix"


def test_vanna_score_not_dampened_when_aligned():
    """Falling VIX should NOT dampen a positive vanna score."""
    sig_no_vix = signal_vanna_flow(total_vannex=5e8, vix_change_1d=None)
    sig_falling_vix = signal_vanna_flow(total_vannex=5e8, vix_change_1d=-1.5)
    assert sig_falling_vix.score == sig_no_vix.score, \
        "Aligned VIX direction should not alter vanna score"
    assert "_DAMPENED" not in sig_falling_vix.value


# ── signal_flip_proximity ────────────────────────────────────────────────────

def test_flip_proximity_sign_above():
    """Spot above flip → ABOVE FLIP: STABLE, positive score."""
    sig = signal_flip_proximity(spot=6600.0, flip_level=6500.0)
    assert "ABOVE" in sig.value
    assert sig.score > 0


def test_flip_proximity_sign_below():
    """Spot below flip → BELOW FLIP: BEARISH, negative score."""
    sig = signal_flip_proximity(spot=6400.0, flip_level=6500.0)
    assert "BELOW" in sig.value
    assert sig.score < 0


def test_flip_proximity_at_flip():
    """Spot within 0.5% of flip → WARNING: AT FLIP, score = -1.0."""
    sig = signal_flip_proximity(spot=6600.0, flip_level=6602.0)  # 0.03% away
    assert "AT FLIP" in sig.value
    assert sig.score == -1.0


# ── signal_charm_flow ─────────────────────────────────────────────────────────

def test_charm_score_scale_is_daily():
    """
    Charm score normaliser is /50 ($50M/day = score 1.0).
    With $50M/day charm exposure, score should be exactly 1.0.
    """
    sig = signal_charm_flow(total_charmex=50e6, dte_nearest=10)
    assert sig.score == pytest.approx(1.0), \
        f"$50M/day charm should give score=1.0, got {sig.score}"


def test_charm_negative_is_outflow():
    """Negative charm exposure → DAILY_DELTA_OUTFLOW value."""
    sig = signal_charm_flow(total_charmex=-30e6, dte_nearest=20)
    assert sig.value == "DAILY_DELTA_OUTFLOW"
    assert sig.score < 0
