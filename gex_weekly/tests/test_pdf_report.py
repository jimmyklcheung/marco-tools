"""
Regression tests for reports/pdf_report.py — wall classification and display.
"""
import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reports.pdf_report import _fmt_gex


# ── _fmt_gex ─────────────────────────────────────────────────────────────────

def test_fmt_gex_large_shows_B():
    """Values >= $1B should display as B, not raw M (prevents 6-digit strings)."""
    result = _fmt_gex(391_342e6)   # $391.342B
    assert result.endswith("B"), f"Large GEX should show as B: got '{result}'"
    assert "391.3" in result, f"Expected '391.3B', got '{result}'"


def test_fmt_gex_small_shows_M():
    """Values < $1B should display as M."""
    result = _fmt_gex(450e6)   # $450M
    assert result.endswith("M"), f"Sub-billion GEX should show as M: got '{result}'"


def test_fmt_gex_negative():
    """Negative GEX should include a minus sign."""
    result = _fmt_gex(-2.5e9)
    assert "-" in result, f"Negative GEX should show minus sign: got '{result}'"


# ── Wall interpretation logic (via build_template_vars helper patterns) ───────

class MockReport:
    flip_level = 7000.0
    gamma_regime = "LONG_GAMMA"
    risk_level = "LOW"
    composite_score = 0.3
    summary = "Test summary"
    signals = []


def _build_walls(put_strike, spot):
    """Build minimal by_str and call build_template_vars to get key_levels_rows."""
    import pandas as pd
    from reports.pdf_report import build_template_vars

    # Minimal by_str with one call wall (above spot) and one put wall
    by_str = pd.DataFrame({
        "strike": [spot + 100, put_strike],
        "call_gex": [5e9, 0.0],
        "put_gex": [0.0, -5e9],
        "net_gex": [5e9, -5e9],
        "call_oi": [1000.0, 0.0],
        "put_oi": [0.0, 1000.0],
        "net_dex": [0.0, 0.0],
        "call_dex": [0.0, 0.0],
        "put_dex": [0.0, 0.0],
        "net_vannex": [0.0, 0.0],
        "net_charmex": [0.0, 0.0],
        "put_call_oi_ratio": [float("nan"), float("nan")],
    })

    config = {"report": {"title": "Test", "author": "Test"}}
    tvars = build_template_vars(
        spot=spot,
        by_str=by_str,
        by_exp=pd.DataFrame(),
        chain=pd.DataFrame(),
        report=MockReport(),
        opex_data={},
        multi_data=[],
        chart_paths={},
        config=config,
    )
    return tvars["key_levels_rows"]


def test_put_wall_below_spot_is_downside_accelerator():
    """Put wall below spot must be labelled as downside accelerator."""
    rows = _build_walls(put_strike=6400.0, spot=6600.0)
    put_row = next(r for r in rows if r["type"] == "Put Wall")
    assert "accelerator" in put_row["interpretation"].lower() or \
           "below spot" in put_row["interpretation"].lower(), \
        f"Below-spot put wall interpretation wrong: '{put_row['interpretation']}'"


def test_put_wall_near_atm_above_spot_is_pin():
    """Put wall within 1% above spot must be labelled as near-ATM pin/magnet."""
    # put_strike = spot + 0.5% = just above spot
    spot = 6600.0
    put_strike = spot * 1.005   # +0.5%, within near_atm_pct=1%
    rows = _build_walls(put_strike=put_strike, spot=spot)
    put_row = next(r for r in rows if r["type"] == "Put Wall")
    assert "pin" in put_row["interpretation"].lower() or \
           "magnet" in put_row["interpretation"].lower(), \
        f"Near-ATM put wall should be pin/magnet: '{put_row['interpretation']}'"


def test_put_wall_far_above_spot_is_overhead():
    """Put wall > 1% above spot must be labelled as overhead concentration."""
    spot = 6600.0
    put_strike = spot * 1.03   # +3%, beyond near_atm_pct
    rows = _build_walls(put_strike=put_strike, spot=spot)
    put_row = next(r for r in rows if r["type"] == "Put Wall")
    assert "overhead" in put_row["interpretation"].lower(), \
        f"Far above-spot put wall should be 'overhead': '{put_row['interpretation']}'"


def test_call_wall_above_spot_is_resistance():
    """Call wall above spot must be labelled as resistance."""
    rows = _build_walls(put_strike=6400.0, spot=6600.0)
    call_row = next(r for r in rows if r["type"] == "Call Wall")
    assert "resistance" in call_row["interpretation"].lower(), \
        f"Above-spot call wall should be resistance: '{call_row['interpretation']}'"


def test_gex_display_uses_B_for_large_values():
    """Key levels rows must use auto B/M formatting, not raw millions."""
    rows = _build_walls(put_strike=6400.0, spot=6600.0)
    for row in rows:
        # gex_fmt should never show a 6-digit number followed by M (the old bug)
        gex_str = row["gex_fmt"]
        assert len(gex_str) < 15, f"GEX display string too long (raw M not auto-scaled?): '{gex_str}'"
        # Values >= 1B should show B
        if abs(row["gex_m"]) >= 1000:   # gex_m is in $M, so 1000M = $1B
            assert "B" in gex_str, f"GEX ≥ $1B should show as B: '{gex_str}'"


def test_dual_sided_concentration_is_pin_node():
    """
    When the top call wall and top put wall are at the same strike (within 0.5%),
    both should be labelled as pin/straddle concentration, not resistance/accelerator.
    """
    import pandas as pd
    from reports.pdf_report import build_template_vars

    spot = 6600.0
    atm_strike = 6610.0   # 0.15% above spot — within 0.5% tolerance

    # Same strike dominates both call_gex and put_gex
    by_str = pd.DataFrame({
        "strike": [atm_strike, spot + 200],
        "call_gex": [8e9, 1e9],
        "put_gex":  [-8e9, -1e9],
        "net_gex":  [0.0, 0.0],
        "call_oi":  [2000.0, 500.0],
        "put_oi":   [2000.0, 500.0],
        "net_dex":  [0.0, 0.0],
        "call_dex": [0.0, 0.0],
        "put_dex":  [0.0, 0.0],
        "net_vannex":   [0.0, 0.0],
        "net_charmex":  [0.0, 0.0],
        "put_call_oi_ratio": [float("nan"), float("nan")],
    })

    config = {"report": {"title": "Test", "author": "Test"}}
    tvars = build_template_vars(
        spot=spot,
        by_str=by_str,
        by_exp=pd.DataFrame(),
        chain=pd.DataFrame(),
        report=MockReport(),
        opex_data={},
        multi_data=[],
        chart_paths={},
        config=config,
    )
    rows = tvars["key_levels_rows"]

    call_row = next(r for r in rows if r["type"] == "Call Wall")
    put_row = next(r for r in rows if r["type"] == "Put Wall")

    assert "pin" in call_row["interpretation"].lower() or \
           "straddle" in call_row["interpretation"].lower(), \
        f"Dual-sided call wall should be pin/straddle node: '{call_row['interpretation']}'"
    assert "pin" in put_row["interpretation"].lower() or \
           "straddle" in put_row["interpretation"].lower(), \
        f"Dual-sided put wall should be pin/straddle node: '{put_row['interpretation']}'"
