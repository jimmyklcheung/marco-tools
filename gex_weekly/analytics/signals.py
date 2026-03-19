"""
Macro signal engine: 6 signals + composite scoring + MacroReport generation.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Signal:
    name: str
    value: str                      # Human-readable direction label
    score: float                    # Float -1 to +1
    rationale: str
    magnitude: str                  # STRONG / MODERATE / WEAK
    asset_implications: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "score": round(self.score, 3),
            "rationale": self.rationale,
            "magnitude": self.magnitude,
            "asset_implications": self.asset_implications,
        }


@dataclass
class MacroReport:
    timestamp: datetime
    spot: float
    gamma_regime: str               # LONG_GAMMA / SHORT_GAMMA / NEUTRAL
    flip_level: float
    signals: List[Signal]
    composite_score: float
    risk_level: str                 # LOW / MEDIUM / HIGH / EXTREME
    summary: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "spot": self.spot,
            "gamma_regime": self.gamma_regime,
            "flip_level": self.flip_level,
            "signals": [s.to_dict() for s in self.signals],
            "composite_score": round(self.composite_score, 3),
            "risk_level": self.risk_level,
            "summary": self.summary,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _magnitude(score: float) -> str:
    abs_s = abs(score)
    if abs_s >= 0.65:
        return "STRONG"
    if abs_s >= 0.35:
        return "MODERATE"
    return "WEAK"


# ── 6 Signal functions ────────────────────────────────────────────────────────

def signal_gamma_regime(total_net_gex: float, flip_level: float, spot: float) -> Signal:
    """
    Determine the dealer gamma regime.
    GEX > 0 → LONG GAMMA (bullish vol-sell environment).
    GEX < 0 → SHORT GAMMA (bearish vol-buy environment).
    """
    score = float(np.clip(total_net_gex / 2e9, -1, 1))
    is_long = total_net_gex >= 0
    value = "BULLISH_VOL_SELL" if is_long else "BEARISH_VOL_BUY"
    regime_label = "LONG GAMMA" if is_long else "SHORT GAMMA"

    dist_to_flip_pct = ((spot - flip_level) / spot * 100) if spot > 0 else 0
    gex_bn = total_net_gex / 1e9

    rationale = (
        f"Net GEX = ${gex_bn:+.2f}B → dealers are {regime_label}. "
        f"Spot is {abs(dist_to_flip_pct):.1f}% {'above' if dist_to_flip_pct > 0 else 'below'} "
        f"gamma flip at {flip_level:,.0f}."
    )

    if is_long:
        implications = [
            "Equities: expect mean-reversion, sell strength into resistance",
            "Vol: VIX likely to fade on spikes — sell vol spikes",
            "Options: credit spreads / iron condors favoured",
            "Sizing: normal to increased — regime supports range-bound positioning",
        ]
    else:
        implications = [
            "Equities: trending moves more likely, reduce net delta exposure",
            "Vol: VIX spikes may persist — avoid short vol",
            "Options: long gamma / straddles for directional flexibility",
            "Sizing: reduce 30–50% — elevated tail risk",
        ]

    return Signal(
        name="Gamma Regime",
        value=value,
        score=score,
        rationale=rationale,
        magnitude=_magnitude(score),
        asset_implications=implications,
    )


def signal_flip_proximity(spot: float, flip_level: float) -> Signal:
    """
    Proximity of spot to gamma flip — the closer, the more dangerous.
    """
    if flip_level == 0 or spot == 0:
        return Signal("Flip Proximity", "UNKNOWN", 0.0, "Flip level unavailable.", "WEAK")

    dist_pct = (spot - flip_level) / spot * 100

    if abs(dist_pct) < 0.5:
        score = -1.0
        magnitude = "STRONG"
        value = "WARNING: AT FLIP"
        rationale = (
            f"Spot ({spot:,.0f}) is within {abs(dist_pct):.2f}% of gamma flip ({flip_level:,.0f}). "
            "Regime change imminent — elevated intraday vol risk."
        )
    elif abs(dist_pct) < 1.5:
        score = -0.7
        magnitude = "MODERATE"
        value = "WARNING: NEAR FLIP"
        rationale = (
            f"Spot ({spot:,.0f}) is {abs(dist_pct):.2f}% from gamma flip ({flip_level:,.0f}). "
            "Risk of regime flip within the week."
        )
    elif dist_pct > 0:
        score = 0.2
        magnitude = "WEAK"
        value = "ABOVE FLIP: STABLE"
        rationale = (
            f"Spot ({spot:,.0f}) is {dist_pct:.1f}% above gamma flip ({flip_level:,.0f}). "
            "Dealer support likely on dips toward flip."
        )
    else:
        score = -0.5
        magnitude = "MODERATE"
        value = "BELOW FLIP: BEARISH"
        rationale = (
            f"Spot ({spot:,.0f}) is {abs(dist_pct):.1f}% below gamma flip ({flip_level:,.0f}). "
            "Dealers in short-gamma — trend-following mode, no natural support."
        )

    implications = [
        f"Watch {flip_level:,.0f} as key inflection — sustained close above = regime shift",
        "Volatility clustering risk increases near flip",
    ]
    return Signal(
        name="Flip Proximity",
        value=value,
        score=score,
        rationale=rationale,
        magnitude=magnitude,
        asset_implications=implications,
    )


def signal_opex_countdown(opex_gex_net: float, dte: Optional[int], total_net_gex: float) -> Signal:
    """
    How urgently is OPEX GEX expiring and what % of the book does it represent?
    """
    if dte is None:
        return Signal("OPEX Countdown", "UNKNOWN", 0.0, "No OPEX data available.", "WEAK")

    pct_of_book = (abs(opex_gex_net) / max(abs(total_net_gex), 1)) * 100

    if dte <= 3:
        magnitude = "STRONG"
        abs_score = 0.7
        urgency = "CRITICAL"
    elif dte <= 7:
        magnitude = "MODERATE"
        abs_score = 0.4
        urgency = "ELEVATED"
    else:
        magnitude = "WEAK"
        abs_score = 0.1
        urgency = "LOW"

    # Direction: net negative OPEX GEX expiring = bearish momentum post-OPEX
    score = abs_score if opex_gex_net >= 0 else -abs_score

    rationale = (
        f"OPEX in {dte} DTE ({urgency} urgency). "
        f"${opex_gex_net / 1e6:+.0f}M net GEX expiring = {pct_of_book:.1f}% of total book. "
        f"{'Pin behaviour likely near OPEX strikes.' if dte <= 5 else 'Post-OPEX vol reset expected.'}"
    )

    implications = [
        f"Pin risk: market may gravitate to max-pain strike ±0.5% as OPEX approaches",
        "Post-OPEX: expect vol reset once large GEX concentration expires",
        f"Book impact: {pct_of_book:.1f}% of dealer delta hedging unwind at OPEX",
    ]

    return Signal(
        name="OPEX Countdown",
        value=f"OPEX {dte}DTE",
        score=score,
        rationale=rationale,
        magnitude=magnitude,
        asset_implications=implications,
    )


def signal_vanna_flow(total_vannex: float, vix_change_1d: Optional[float] = None) -> Signal:
    """
    Vanna exposure signal: how vol changes mechanically drive equity delta.
    Positive vanna → vol falls = equity bid. Negative → vol rises = equity offered.
    """
    score = float(np.clip(total_vannex / 5e8, -1, 1))
    vannex_mm = total_vannex / 1e6

    is_positive = total_vannex >= 0
    value = "VOL_FALL_BID" if is_positive else "VOL_RISE_OFFER"

    rationale = (
        f"Vanna exposure = ${vannex_mm:+.0f}M. "
        f"{'Vol compression → mechanical equity bid via delta re-hedging.' if is_positive else 'Vol expansion → mechanical equity selling via delta re-hedging.'}"
    )

    if vix_change_1d is not None:
        vix_direction = "falling" if vix_change_1d < 0 else "rising"
        vix_aligned = (vix_change_1d < 0) == is_positive  # falling VIX aligns with positive vanna
        alignment = "aligned" if vix_aligned else "diverging"
        rationale += f" VIX 1d change: {vix_change_1d:+.2f} ({vix_direction}) — {alignment} with vanna signal."

        # Adjust score: if VIX is moving against the vanna signal, reduce conviction.
        # Rising VIX (vol expansion) contradicts a positive vanna tailwind, and vice versa.
        if not vix_aligned:
            dampening = min(abs(vix_change_1d) / 5.0, 0.5)  # cap dampening at 0.5
            score = float(np.clip(score * (1.0 - dampening), -1, 1))
            value = value + "_DAMPENED"

    implications = [
        "Vol decline → dealer delta re-hedging creates systematic equity buying",
        "Watch VIX 1d direction as leading indicator of vanna flow direction",
        "Strong positive vanna + falling VIX = powerful mechanical tailwind",
    ]

    return Signal(
        name="Vanna Flow",
        value=value,
        score=score,
        rationale=rationale,
        magnitude=_magnitude(score),
        asset_implications=implications,
    )


def signal_charm_flow(total_charmex: float, dte_nearest: Optional[int]) -> Signal:
    """
    Charm (delta decay) flow: daily delta P&L from time passage.
    Near expiry = charm accelerates (delta bleeding).
    """
    # total_charmex is already in $/day (greeks.py divides by 365).
    # Normalise: ±$50M/day = full score (±1).  Adjust threshold if needed.
    charmex_mm = total_charmex / 1e6
    score = float(np.clip(charmex_mm / 50, -1, 1))

    if dte_nearest is not None and dte_nearest <= 5:
        magnitude = "STRONG"
        urgency_note = f"DTE={dte_nearest}: charm acceleration is sharp near expiry."
    elif dte_nearest is not None and dte_nearest <= 14:
        magnitude = "MODERATE"
        urgency_note = f"DTE={dte_nearest}: moderate charm decay."
    else:
        magnitude = "WEAK"
        urgency_note = "Charm effect is subdued with distant expiry."

    value = "DAILY_DELTA_INFLOW" if total_charmex >= 0 else "DAILY_DELTA_OUTFLOW"

    rationale = (
        f"Charm exposure = ${charmex_mm:+.0f}M/day. {urgency_note} "
        f"{'Positive charm = net delta accrual as time passes.' if total_charmex >= 0 else 'Negative charm = net delta drain as time passes.'}"
    )

    implications = [
        "Charm creates mechanical daily delta flows even with no price movement",
        "Near OPEX, charm accelerates and can distort intraday price action",
    ]

    return Signal(
        name="Charm Flow",
        value=value,
        score=score,
        rationale=rationale,
        magnitude=magnitude,
        asset_implications=implications,
    )


def signal_put_skew(
    total_put_gex: float,
    total_call_gex: float,
    spot: float,
    put_wall: Optional[float],
) -> Signal:
    """
    Put-call GEX skew: heavy put GEX = bearish dealer positioning.
    """
    if total_call_gex == 0:
        ratio = 0.0
    else:
        ratio = abs(total_put_gex) / max(abs(total_call_gex), 1)

    if ratio > 1.5:
        score = -0.8
        magnitude = "STRONG"
        value = "BEARISH_SKEW_HEAVY"
    elif ratio > 1.1:
        score = -0.4
        magnitude = "MODERATE"
        value = "BEARISH_SKEW_MODERATE"
    else:
        score = 0.2
        magnitude = "WEAK"
        value = "NEUTRAL_BALANCED"

    dist_to_put_wall = ""
    if put_wall and spot > 0:
        pct = (put_wall - spot) / spot * 100
        dist_to_put_wall = f" Put wall at {put_wall:,.0f} ({pct:+.1f}% from spot)."

    rationale = (
        f"Put/Call GEX ratio = {ratio:.2f}. "
        f"{'Heavy put protection in dealer books — bearish defensive positioning.' if ratio > 1.1 else 'Balanced put/call book — no strong directional skew.'}"
        f"{dist_to_put_wall}"
    )

    implications = [
        "High put skew: dealers short puts → forced to sell into weakness, amplifying downside moves",
        "Hedging demand exceeds upside speculation — institutional risk-off signal",
    ]

    return Signal(
        name="Put/Call Skew",
        value=value,
        score=score,
        rationale=rationale,
        magnitude=magnitude,
        asset_implications=implications,
    )


# ── Composite report generator ────────────────────────────────────────────────

def generate_report(
    spot: float,
    flip_level: float,
    total_net_gex: float,
    total_put_gex: float,
    total_call_gex: float,
    total_vannex: float,
    total_charmex: float,
    opex_gex_net: float,
    opex_dte: Optional[int],
    dte_nearest: Optional[int],
    put_wall: Optional[float],
    vix_change_1d: Optional[float] = None,
) -> MacroReport:
    """
    Run all 6 signals and produce a composite MacroReport.
    """
    s1 = signal_gamma_regime(total_net_gex, flip_level, spot)
    s2 = signal_flip_proximity(spot, flip_level)
    s3 = signal_opex_countdown(opex_gex_net, opex_dte, total_net_gex)
    s4 = signal_vanna_flow(total_vannex, vix_change_1d)
    s5 = signal_charm_flow(total_charmex, dte_nearest)
    s6 = signal_put_skew(total_put_gex, total_call_gex, spot, put_wall)

    signals = [s1, s2, s3, s4, s5, s6]
    scores = [s.score for s in signals]
    composite = float(np.mean(scores))

    # Count warnings (magnitude == STRONG and score < -0.6)
    warnings = sum(1 for s in signals if s.magnitude == "STRONG" and s.score < -0.6)

    # Risk level
    if warnings >= 2 or composite < -0.6:
        risk_level = "EXTREME"
    elif warnings >= 1 or abs(composite) > 0.4:
        risk_level = "HIGH"
    elif abs(composite) > 0.2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Gamma regime label
    if total_net_gex > 0.5e9:
        gamma_regime = "LONG_GAMMA"
    elif total_net_gex < -0.5e9:
        gamma_regime = "SHORT_GAMMA"
    else:
        gamma_regime = "NEUTRAL"

    # Summary
    dist_pct = ((spot - flip_level) / spot * 100) if spot > 0 else 0
    direction = "above" if dist_pct > 0 else "below"
    summary = (
        f"Dealers are in {gamma_regime.replace('_', ' ')} with net GEX of "
        f"${total_net_gex / 1e9:+.2f}B. Spot ({spot:,.0f}) sits {abs(dist_pct):.1f}% {direction} "
        f"the gamma flip level ({flip_level:,.0f}). Composite macro score: "
        f"{composite:+.2f} → Risk Level: {risk_level}. "
        f"Key risks: {', '.join(s.value for s in signals if s.score < -0.4) or 'none identified'}."
    )

    return MacroReport(
        timestamp=datetime.now(),
        spot=spot,
        gamma_regime=gamma_regime,
        flip_level=flip_level,
        signals=signals,
        composite_score=composite,
        risk_level=risk_level,
        summary=summary,
    )
