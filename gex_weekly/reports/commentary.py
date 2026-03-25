"""
Commentary framework: generate structured trader-language text from analytics.

Produces:
  - Street Take (3–5 concise lines, trader language)
  - Today's Map (sticky zones, slippery zones, triggers, magnets)
  - What Changed (delta from prior state)
  - Invalidation (what breaks the current read)
  - Confidence note

Design rules:
  - Confidence drives language precision: HIGH → directional, LOW → descriptive only
  - Never overclaim: when sign-sensitive, say "sign-sensitive / low-confidence"
  - Always include an invalidation statement
  - Distinguish intraday / near-term from swing / structural
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Output structures ─────────────────────────────────────────────────────────

@dataclass
class StreetTake:
    """
    3–5 concise lines for the desk.
    Lines are written in trader language: sticky, slippery, pin, air pocket, etc.
    """
    headline: str               # One-line regime summary
    tape_character: str         # What the tape is doing mechanically
    key_level_note: str         # Most important level in one line
    what_to_watch: str          # Key trigger / invalidation
    confidence_note: str        # Caveats or confidence context

    def as_bullets(self) -> list[str]:
        return [
            self.headline,
            self.tape_character,
            self.key_level_note,
            self.what_to_watch,
            self.confidence_note,
        ]

    def as_paragraph(self) -> str:
        return " ".join(self.as_bullets())


@dataclass
class TodaysMap:
    """Key level summary for today's session."""
    regime: str
    primary_magnet: Optional[str]         # Pin / sticky level if any
    downside_slippery_zone: Optional[str] # Accelerator / air pocket below spot
    upside_supply_zone: Optional[str]     # Resistance / sell-on-rip above spot
    dominant_expiry: Optional[str]        # Front-end expiry driving the picture
    invalidation: str                     # What breaks the map
    confidence: str                       # HIGH / MODERATE / LOW / VERY LOW
    confidence_int: int                   # 0–100


@dataclass
class WhatChanged:
    """Attribution of changes vs prior run."""
    flip_changed: bool
    regime_changed: bool
    roll_detected: bool
    flip_note: str
    regime_note: str
    roll_note: str
    dominant_drivers: list[str]          # ["spot_move", "vol_move", "roll", "oi_change"]

    def has_changes(self) -> bool:
        return self.flip_changed or self.regime_changed or self.roll_detected

    def summary(self) -> str:
        if not self.has_changes():
            return "No significant changes vs prior run."
        parts = []
        if self.roll_detected:
            parts.append(self.roll_note)
        if self.regime_changed:
            parts.append(self.regime_note)
        if self.flip_changed:
            parts.append(self.flip_note)
        return " ".join(parts)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_street_take(
    regime: str,
    composite_score: float,
    flip_status: str,
    primary_flip: Optional[float],
    spot: float,
    dist_to_flip_pct: float,
    vix: Optional[float],
    vix_change_1d: Optional[float],
    level_map: TodaysMap,
    confidence: str,
    confidence_int: int,
    sign_stable: bool,
    qa_publish_status: str,
) -> StreetTake:
    """
    Generate the Street Take from computed state.
    Language precision is governed by confidence level.
    """
    is_long = "LONG" in regime
    is_short = "SHORT" in regime
    is_neutral = "NEUTRAL" in regime
    low_conf = confidence in ("LOW", "VERY LOW")
    flip_undefined = flip_status in ("none_in_range", "undefined")

    # Headline: one sentence on regime
    if flip_undefined:
        headline = (
            f"Regime: {'long' if composite_score > 0 else 'short'}-gamma leaning "
            f"(score {composite_score:+.2f}), but gamma flip is undefined — no modeled zero-crossing. "
            "One-sided book."
        )
    elif is_long:
        dist_word = "well " if abs(dist_to_flip_pct) > 2 else ""
        headline = (
            f"Dealers in LONG GAMMA — spot is {dist_word}{abs(dist_to_flip_pct):.1f}% "
            f"{'above' if dist_to_flip_pct > 0 else 'below'} the modeled flip at "
            f"{primary_flip:,.0f}. Tape is sticky; moves compress near key strikes."
        )
    elif is_short:
        headline = (
            f"Dealers in SHORT GAMMA — spot is {abs(dist_to_flip_pct):.1f}% "
            f"{'above' if dist_to_flip_pct > 0 else 'below'} flip at {primary_flip:,.0f} "
            f"(if applicable). Tape is slippery; moves can extend."
        )
    else:
        headline = (
            f"Dealers near gamma-neutral (score {composite_score:+.2f}). "
            "Regime is ambiguous — monitor flip level for confirmation."
        )

    # Tape character: what dealers are mechanically doing
    if low_conf:
        tape_character = (
            "Mechanical flow direction is low-confidence — sign model is sensitive or data quality is limited. "
            "Use market structure (VIX, price action) as primary guide."
        )
    elif is_long:
        tape_character = (
            "Dealer hedging dampens moves: expect buy-the-dip and sell-the-rip flows near key strikes. "
            "Sharp directional moves require breaking out of the long-gamma zone."
        )
    else:
        tape_character = (
            "Dealer hedging amplifies moves: expect momentum persistence once key levels break. "
            "No mechanical cushion — trending behavior is the base case."
        )

    # Key level note
    if level_map.primary_magnet and level_map.downside_slippery_zone:
        key_level_note = (
            f"Pin/magnet: {level_map.primary_magnet}. "
            f"Slippery zone below: {level_map.downside_slippery_zone}. "
            f"Supply/resistance above: {level_map.upside_supply_zone or 'N/A'}."
        )
    elif level_map.primary_magnet:
        key_level_note = (
            f"Tape sticky near {level_map.primary_magnet}. "
            f"Resistance above: {level_map.upside_supply_zone or 'N/A'}."
        )
    elif level_map.downside_slippery_zone:
        key_level_note = (
            f"No clear pin — air pocket below at {level_map.downside_slippery_zone}. "
            "Downside moves can accelerate through this zone."
        )
    else:
        key_level_note = "No dominant pin or slippery zone identified near spot."

    # What to watch: trigger / invalidation
    what_to_watch = level_map.invalidation

    # Confidence note
    sign_note = " Sign-sensitive — conclusions may reverse under alternative dealer convention." if not sign_stable else ""
    qa_note = ""
    if qa_publish_status == "BLOCK":
        qa_note = " DATA QUALITY ISSUE: report is blocked — review QA diagnostics before acting."
    elif qa_publish_status == "WARN":
        qa_note = " Data quality warnings present — apply additional discretion."

    confidence_note = (
        f"Confidence: {confidence} ({confidence_int}%).{sign_note}{qa_note}"
    )

    if low_conf:
        confidence_note += " Treat directional levels as indicative only."

    return StreetTake(
        headline=headline,
        tape_character=tape_character,
        key_level_note=key_level_note,
        what_to_watch=what_to_watch,
        confidence_note=confidence_note,
    )


def build_todays_map(
    regime: str,
    level_classifications: list,  # list of LevelClassification from scenario.py
    spot: float,
    dominant_expiry: Optional[str],
    primary_flip: Optional[float],
    flip_status: str,
    confidence_label: str,
    confidence_int: int,
) -> TodaysMap:
    """
    Build Today's Map from classified levels.
    """
    # Find primary magnet: nearest PIN with HIGH or MODERATE confidence, closest to spot
    pins = [
        lc for lc in level_classifications
        if lc.classification == "pin" and lc.confidence in ("HIGH", "MODERATE")
    ]
    pins.sort(key=lambda lc: abs(lc.dist_pct))
    primary_magnet = f"{pins[0].strike:,.0f}" if pins else None

    # Downside slippery zone: nearest accelerator below spot
    accs = [lc for lc in level_classifications
            if lc.classification == "accelerator" and lc.dist_pct < 0]
    accs.sort(key=lambda lc: abs(lc.dist_pct))
    downside_slippery = f"{accs[0].strike:,.0f}" if accs else None

    # Upside supply zone: nearest resistance above spot
    res = [lc for lc in level_classifications
           if lc.classification == "resistance" and lc.dist_pct > 0]
    res.sort(key=lambda lc: lc.dist_pct)
    upside_supply = f"{res[0].strike:,.0f}" if res else None

    # Invalidation
    invalidation = _build_invalidation(
        regime, primary_flip, spot, flip_status,
        primary_magnet, downside_slippery, upside_supply,
    )

    return TodaysMap(
        regime=regime,
        primary_magnet=primary_magnet,
        downside_slippery_zone=downside_slippery,
        upside_supply_zone=upside_supply,
        dominant_expiry=dominant_expiry,
        invalidation=invalidation,
        confidence=confidence_label,
        confidence_int=confidence_int,
    )


def _build_invalidation(
    regime: str,
    primary_flip: Optional[float],
    spot: float,
    flip_status: str,
    primary_magnet: Optional[str],
    downside_slippery: Optional[str],
    upside_supply: Optional[str],
) -> str:
    """Derive an invalidation statement from the computed state."""
    parts = []

    if "LONG" in regime:
        if flip_status == "modeled" and primary_flip:
            pct = abs(spot - primary_flip) / spot * 100
            parts.append(
                f"Long-gamma read breaks if spot closes below modeled flip "
                f"at {primary_flip:,.0f} ({pct:.1f}% below). "
                "A sustained close below flip = regime shift to short-gamma."
            )
    elif "SHORT" in regime:
        if flip_status == "modeled" and primary_flip:
            pct = abs(primary_flip - spot) / spot * 100
            parts.append(
                f"Short-gamma read breaks if spot reclaims flip at {primary_flip:,.0f} "
                f"({pct:.1f}% above spot). "
                "Sustained close above flip = regime shift to long-gamma / move compression."
            )

    if downside_slippery:
        parts.append(
            f"Downside: break below {downside_slippery} opens slippery zone — "
            "dealer hedging amplifies moves, no mechanical support."
        )

    if upside_supply:
        parts.append(
            f"Upside: rallying through {upside_supply} hits dealer supply — "
            "watch for rejection or slowdown at that level."
        )

    if not parts:
        parts.append(
            "No clear invalidation level near spot. "
            "Monitor VIX direction and key strike OI changes for regime clues."
        )

    return " | ".join(parts)


def build_what_changed(
    prior_state: dict,
    current_regime: str,
    current_flip: Optional[float],
    current_flip_status: str,
    current_front_expiry: Optional[str],
    spot: float,
    roll_status,   # RollStatus from scenario.py
) -> WhatChanged:
    """
    Build a What Changed summary comparing current run to prior_state dict.
    """
    if not prior_state:
        return WhatChanged(
            flip_changed=False, regime_changed=False, roll_detected=False,
            flip_note="No prior run available.",
            regime_note="",
            roll_note="",
            dominant_drivers=[],
        )

    prior_regime = prior_state.get("regime")
    prior_flip = prior_state.get("flip_level")
    prior_front = prior_state.get("front_expiry")

    # Regime change
    regime_changed = (prior_regime is not None and prior_regime != current_regime)
    regime_note = (
        f"Regime changed: {prior_regime} → {current_regime}."
        if regime_changed else ""
    )

    # Flip change
    flip_changed = False
    flip_note = ""
    if prior_flip and current_flip and current_flip_status == "modeled":
        delta_pct = (current_flip - prior_flip) / spot * 100
        if abs(delta_pct) > 0.5:
            flip_changed = True
            flip_note = (
                f"Flip moved {delta_pct:+.1f}% "
                f"({prior_flip:,.0f} → {current_flip:,.0f})."
            )
    elif current_flip_status in ("none_in_range", "undefined") and prior_flip:
        flip_changed = True
        flip_note = "Gamma flip no longer modeled in range."

    # Roll
    roll_detected = roll_status.roll_detected if roll_status else False
    roll_note = roll_status.roll_note if roll_status else ""

    # Attribution
    drivers = []
    if roll_detected:
        drivers.append("expiry_roll")
    if regime_changed:
        drivers.append("regime_shift")
    if flip_changed and not roll_detected:
        drivers.append("positioning_change")

    return WhatChanged(
        flip_changed=flip_changed,
        regime_changed=regime_changed,
        roll_detected=roll_detected,
        flip_note=flip_note,
        regime_note=regime_note,
        roll_note=roll_note,
        dominant_drivers=drivers,
    )


def format_book_summary(books: dict) -> list[dict]:
    """
    Format book results (structural/tactical/OPEX/event) for template rendering.
    Returns a list of dicts suitable for a Jinja2 loop.
    """
    rows = []
    for label, book in books.items():
        flip_str = (
            f"{book.primary_flip:,.0f}"
            if book.primary_flip and book.flip_status == "modeled"
            else book.flip_status.replace("_", " ").title()
        )
        rows.append({
            "label": label.upper(),
            "expiry_filter": book.expiry_filter,
            "regime": book.regime.replace("_", " "),
            "net_gex_b": book.net_gex / 1e9,
            "flip": flip_str,
            "flip_status": book.flip_status,
            "dominant_expiry": book.dominant_expiry or "N/A",
            "dte_range": f"{book.dte_range[0]}–{book.dte_range[1]}",
            "confidence": book.confidence,
        })
    return rows


def format_cross_asset_row(key: str, res: dict, qa: object) -> dict:
    """Build a cross-asset comparison row for template rendering."""
    if res is None:
        return {
            "key": key,
            "label": key,
            "spot": "N/A",
            "regime": "DATA UNAVAILABLE",
            "flip": "N/A",
            "dist_pct": None,
            "nearest_magnet": "N/A",
            "nearest_slippery": "N/A",
            "futures_equivalent": "N/A",
            "confidence": "VERY LOW",
            "qa_status": "BLOCK",
        }

    flip_str = (
        f"{res.get('primary_flip', 0):,.0f}"
        if res.get("flip_status") == "modeled" and res.get("primary_flip")
        else "undefined"
    )

    # Find nearest magnet and slippery zone from classified levels
    classified = res.get("classified_levels", [])
    pins = [lc for lc in classified if lc.classification == "pin"]
    pins.sort(key=lambda x: abs(x.dist_pct))
    accs = [lc for lc in classified if lc.classification == "accelerator"]
    accs.sort(key=lambda x: abs(x.dist_pct))

    return {
        "key": key,
        "label": res.get("label", key),
        "spot": f"{res['spot']:,.2f}",
        "regime": res.get("regime", "N/A").replace("_", " "),
        "flip": flip_str,
        "dist_pct": res.get("dist_pct"),
        "nearest_magnet": f"{pins[0].strike:,.0f}" if pins else "none",
        "nearest_slippery": f"{accs[0].strike:,.0f}" if accs else "none",
        "futures_equivalent": res.get("futures_equivalent", "N/A"),
        "confidence": res.get("confidence_label", "MODERATE"),
        "qa_status": qa.publish_status if qa else "PASS",
    }
