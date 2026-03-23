"""
Instrument metadata registry.

Every instrument analysed by this system must be registered here.
Registration enforces explicit metadata — unknown instruments fail loudly
rather than silently inheriting SPX assumptions.

Sign model conventions
----------------------
The "street" sign model (Spotgamma / standard market convention):
  dealer_sign_calls = +1   (dealers are assumed net long calls)
  dealer_sign_puts  = -1   (dealers are assumed net short puts)

An alternative "stress" sign model flips both signs to stress-test
whether the top-level conclusions (regime, flip, walls) are sign-stable.
Sign-unstable conclusions are explicitly labelled LOW-CONFIDENCE.

Settlement
----------
SPX / NDX: AM settlement on the 3rd Friday (opening print, cash-settled).
            Standard Fridays and weeklies = PM settlement.
SPY / QQQ / IWM: PM settlement (ETF options, exercise-or-expire at 4 PM ET).
ES / NQ / RTY: futures options — monthly = 3rd Friday, quarterly = March/June/Sep/Dec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Instrument dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class InstrumentMeta:
    key: str                    # Short key used in config / reports (e.g. "SPX")
    ticker: str                 # Yahoo Finance ticker
    label: str                  # Human-readable label
    multiplier: int             # Option contract multiplier (dollars per index point)
    underlying_type: str        # "index" | "etf" | "futures"
    settlement: str             # "AM" | "PM"
    monthly_opex_rule: str      # "3rd_friday_am" | "3rd_friday_pm" | "quarterly"
    futures_equivalent: Optional[str] = None   # e.g. "ES=F" for SPX
    futures_multiplier: Optional[int] = None   # e.g. 50 for ES (50x index)
    dividend_yield: float = 0.0                # Annualised continuous dividend yield
    carry_rate: Optional[float] = None         # If None, use risk_free_rate from config
    notes: str = ""

    def forward_price(self, spot: float, T: float, r: float) -> float:
        """Compute option-model forward price F = S * e^{(r - q) * T}."""
        q = self.dividend_yield
        effective_r = self.carry_rate if self.carry_rate is not None else r
        import math
        return spot * math.exp((effective_r - q) * T)


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, InstrumentMeta] = {}


def register(meta: InstrumentMeta) -> None:
    _REGISTRY[meta.key] = meta


def get(key: str) -> InstrumentMeta:
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown instrument '{key}'. "
            f"Register it in core/instruments.py before use. "
            f"Known instruments: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def get_or_default(key: str, multiplier: int) -> InstrumentMeta:
    """Return registered meta or create a minimal default (with loud warning)."""
    if key in _REGISTRY:
        return _REGISTRY[key]
    import logging
    logging.getLogger(__name__).warning(
        f"Instrument '{key}' not in registry — using default metadata with "
        f"multiplier={multiplier}. Register it in core/instruments.py for accuracy."
    )
    return InstrumentMeta(
        key=key,
        ticker=key,
        label=key,
        multiplier=multiplier,
        underlying_type="unknown",
        settlement="PM",
        monthly_opex_rule="3rd_friday_pm",
        notes="AUTO-DEFAULT: not in instrument registry",
    )


def all_keys() -> list[str]:
    return sorted(_REGISTRY.keys())


# ── Known instruments ─────────────────────────────────────────────────────────

register(InstrumentMeta(
    key="SPX",
    ticker="^SPX",
    label="S&P 500 Index",
    multiplier=100,
    underlying_type="index",
    settlement="AM",          # Monthly OPEX = AM-settled (opening print)
    monthly_opex_rule="3rd_friday_am",
    futures_equivalent="ES=F",
    futures_multiplier=50,    # ES = $50 × index
    dividend_yield=0.013,     # ~1.3% S&P 500 dividend yield
    notes="Monthly OPEX = 3rd Friday AM. Weekly OPEX = PM. "
          "ES multiplier is 50 (not 100); mini = MES at 5.",
))

register(InstrumentMeta(
    key="SPY",
    ticker="SPY",
    label="SPDR S&P 500 ETF",
    multiplier=100,
    underlying_type="etf",
    settlement="PM",
    monthly_opex_rule="3rd_friday_pm",
    futures_equivalent="ES=F",
    futures_multiplier=50,
    dividend_yield=0.013,
    notes="ETF shares ~1/10 SPX price. PM settlement on all expirations.",
))

register(InstrumentMeta(
    key="QQQ",
    ticker="QQQ",
    label="Invesco QQQ ETF",
    multiplier=100,
    underlying_type="etf",
    settlement="PM",
    monthly_opex_rule="3rd_friday_pm",
    futures_equivalent="NQ=F",
    futures_multiplier=20,    # NQ = $20 × NDX
    dividend_yield=0.006,
    notes="Tracks NDX. NQ futures multiplier is 20; MNQ = 2.",
))

register(InstrumentMeta(
    key="IWM",
    ticker="IWM",
    label="iShares Russell 2000 ETF",
    multiplier=100,
    underlying_type="etf",
    settlement="PM",
    monthly_opex_rule="3rd_friday_pm",
    futures_equivalent="RTY=F",
    futures_multiplier=50,    # RTY = $50 × RUT
    dividend_yield=0.010,
    notes="Tracks RUT. RTY futures multiplier is 50.",
))

register(InstrumentMeta(
    key="NDX",
    ticker="^NDX",
    label="Nasdaq 100 Index",
    multiplier=100,
    underlying_type="index",
    settlement="AM",
    monthly_opex_rule="3rd_friday_am",
    futures_equivalent="NQ=F",
    futures_multiplier=20,
    dividend_yield=0.006,
    notes="Monthly OPEX = AM. NQ multiplier is 20.",
))

register(InstrumentMeta(
    key="VIX",
    ticker="^VIX",
    label="CBOE Volatility Index",
    multiplier=1000,
    underlying_type="index",
    settlement="AM",
    monthly_opex_rule="3rd_friday_am",
    dividend_yield=0.0,
    notes="VIX options settle Wednesday before 3rd Friday. "
          "Futures-based, not spot-based. High model risk.",
))
