"""
Validation / QA layer: health checks, publish blockers, and confidence scoring.

Every report run produces a QAReport with:
  - A list of QACheck results (PASS / WARN / BLOCK)
  - A publish_status: PASS | WARN | BLOCK
  - A confidence score: 0–100

Publish blocking rules
----------------------
- Any BLOCK check → publish_status = BLOCK
- Any WARN check  → publish_status = WARN (unless already BLOCK)
- All PASS        → publish_status = PASS

Confidence scoring
------------------
Starts at 100. Each failing check deducts:
  BLOCK: 30 pts
  WARN:  10 pts
Minimum is 0. Displayed as a percentage.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds (configurable; defaults here as fallback) ─────────────────────

DEFAULT_STALE_SPOT_HOURS = 24       # Spot data older than this = WARN
DEFAULT_STALE_CHAIN_HOURS = 4       # Chain cache older than this = WARN
DEFAULT_MIN_IV_VALID_FRACTION = 0.5 # Fraction of chain rows with IV > 0
DEFAULT_MIN_TOTAL_OI = 1_000        # Min total OI across chain
DEFAULT_MAX_CHARM_PER_CONTRACT = 1e10   # Charm blowup guard
DEFAULT_MAX_GEX_PER_CONTRACT = 1e13    # GEX blowup guard
DEFAULT_MAX_ALLOWED_FLIP_JUMP_PCT = 5.0 # Flip moved more than 5% vs prior = WARN


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class QACheck:
    name: str
    status: str       # "PASS" | "WARN" | "BLOCK"
    message: str
    detail: str = ""

    @property
    def is_passing(self) -> bool:
        return self.status == "PASS"


@dataclass
class QAReport:
    checks: list[QACheck] = field(default_factory=list)
    publish_status: str = "PASS"    # "PASS" | "WARN" | "BLOCK"
    confidence: int = 100           # 0–100
    confidence_label: str = "HIGH"  # "HIGH" | "MODERATE" | "LOW" | "VERY LOW"
    blocked_by: list[str] = field(default_factory=list)
    warned_by: list[str] = field(default_factory=list)

    def add(self, check: QACheck) -> None:
        self.checks.append(check)
        if check.status == "BLOCK":
            self.blocked_by.append(check.name)
        elif check.status == "WARN":
            self.warned_by.append(check.name)

    def finalise(self) -> None:
        """Compute publish_status and confidence after all checks are added."""
        score = 100
        for c in self.checks:
            if c.status == "BLOCK":
                score -= 30
            elif c.status == "WARN":
                score -= 10
        self.confidence = max(0, score)

        if self.blocked_by:
            self.publish_status = "BLOCK"
        elif self.warned_by:
            self.publish_status = "WARN"
        else:
            self.publish_status = "PASS"

        if self.confidence >= 80:
            self.confidence_label = "HIGH"
        elif self.confidence >= 60:
            self.confidence_label = "MODERATE"
        elif self.confidence >= 40:
            self.confidence_label = "LOW"
        else:
            self.confidence_label = "VERY LOW"

    def summary_lines(self) -> list[str]:
        lines = [f"Publish: {self.publish_status} | Confidence: {self.confidence}% ({self.confidence_label})"]
        if self.blocked_by:
            lines.append(f"BLOCKED by: {', '.join(self.blocked_by)}")
        if self.warned_by:
            lines.append(f"WARNINGS: {', '.join(self.warned_by)}")
        return lines


# ── Individual checks ─────────────────────────────────────────────────────────

def check_spot_validity(spot: float, key: str) -> QACheck:
    if spot is None or spot <= 0:
        return QACheck("spot_validity", "BLOCK",
                       f"{key}: spot price is zero or missing.",
                       f"spot={spot}")
    if spot > 1_000_000 or spot < 0.01:
        return QACheck("spot_validity", "BLOCK",
                       f"{key}: spot price ({spot}) is implausible.",
                       f"spot={spot}")
    return QACheck("spot_validity", "PASS", f"{key}: spot OK ({spot:.2f})")


def check_chain_not_empty(chain: pd.DataFrame, key: str) -> QACheck:
    if chain is None or chain.empty:
        return QACheck("chain_not_empty", "BLOCK",
                       f"{key}: options chain is empty. Cannot compute GEX.",
                       "chain.empty=True")
    return QACheck("chain_not_empty", "PASS",
                   f"{key}: chain has {len(chain):,} rows")


def check_iv_quality(chain: pd.DataFrame, key: str,
                     min_fraction: float = DEFAULT_MIN_IV_VALID_FRACTION) -> QACheck:
    if chain is None or chain.empty:
        return QACheck("iv_quality", "WARN", f"{key}: cannot check IV — chain empty")
    if "impliedVolatility" not in chain.columns:
        return QACheck("iv_quality", "WARN", f"{key}: impliedVolatility column missing")
    valid = (chain["impliedVolatility"] > 0.001).mean()
    if valid < 0.1:
        return QACheck("iv_quality", "BLOCK",
                       f"{key}: nearly all IV values are zero ({valid:.0%} valid). "
                       "Greeks will be meaningless.",
                       f"valid_fraction={valid:.3f}")
    if valid < min_fraction:
        return QACheck("iv_quality", "WARN",
                       f"{key}: only {valid:.0%} of IV values are valid (threshold {min_fraction:.0%}). "
                       "Greeks may be unreliable.",
                       f"valid_fraction={valid:.3f}")
    return QACheck("iv_quality", "PASS",
                   f"{key}: {valid:.0%} of IV values valid")


def check_oi_sufficiency(chain: pd.DataFrame, key: str,
                         min_oi: int = DEFAULT_MIN_TOTAL_OI) -> QACheck:
    if chain is None or chain.empty:
        return QACheck("oi_sufficiency", "WARN", f"{key}: chain empty, cannot check OI")
    if "openInterest" not in chain.columns:
        return QACheck("oi_sufficiency", "WARN", f"{key}: openInterest column missing")
    total_oi = chain["openInterest"].fillna(0).sum()
    if total_oi < min_oi:
        return QACheck("oi_sufficiency", "WARN",
                       f"{key}: total OI={total_oi:,.0f} is suspiciously low (threshold {min_oi:,}).",
                       f"total_oi={total_oi:.0f}")
    return QACheck("oi_sufficiency", "PASS",
                   f"{key}: total OI={total_oi:,.0f}")


def check_no_duplicate_contracts(chain: pd.DataFrame, key: str) -> QACheck:
    if chain is None or chain.empty:
        return QACheck("no_duplicates", "PASS", f"{key}: chain empty")
    key_cols = [c for c in ["strike", "expiry", "type"] if c in chain.columns]
    if len(key_cols) < 3:
        return QACheck("no_duplicates", "WARN", f"{key}: missing columns for duplicate check")
    dup_count = chain.duplicated(subset=key_cols).sum()
    if dup_count > 0:
        return QACheck("no_duplicates", "WARN",
                       f"{key}: {dup_count} duplicate contracts detected (same strike+expiry+type). "
                       "OI may be double-counted.",
                       f"duplicate_rows={dup_count}")
    return QACheck("no_duplicates", "PASS",
                   f"{key}: no duplicate contracts")


def check_front_expiry_present(chain: pd.DataFrame, key: str) -> QACheck:
    if chain is None or chain.empty:
        return QACheck("front_expiry", "BLOCK", f"{key}: chain empty — no front expiry")
    if "dte" not in chain.columns:
        return QACheck("front_expiry", "WARN", f"{key}: DTE column missing")
    min_dte = chain["dte"].min()
    if min_dte > 30:
        return QACheck("front_expiry", "WARN",
                       f"{key}: nearest expiry is {min_dte} DTE — no front-end data. "
                       "Tactical book and pin analysis may be unreliable.")
    return QACheck("front_expiry", "PASS",
                   f"{key}: front expiry at {min_dte} DTE")


def check_greeks_finite(chain: pd.DataFrame, key: str) -> QACheck:
    """Check for charm/GEX blowup in computed Greeks."""
    issues = []
    for col, limit, label in [
        ("charmex_$", DEFAULT_MAX_CHARM_PER_CONTRACT, "charmex"),
        ("gex_$", DEFAULT_MAX_GEX_PER_CONTRACT, "gex"),
    ]:
        if col in chain.columns:
            worst = chain[col].abs().max()
            if not np.isfinite(worst):
                issues.append(f"{label} has non-finite values")
            elif worst > limit:
                issues.append(f"{label} max={worst:.2e} exceeds {limit:.1e}")
    if issues:
        return QACheck("greeks_finite", "BLOCK",
                       f"{key}: Greek blowup detected: {'; '.join(issues)}.",
                       "Check T floor and sigma floor in compute_greeks()")
    return QACheck("greeks_finite", "PASS", f"{key}: all computed Greeks are finite")


def check_flip_status(flip_status: str, flip_confidence: str, key: str) -> QACheck:
    """Check modeled gamma flip has a valid result."""
    if flip_status == "undefined":
        return QACheck("flip_status", "BLOCK",
                       f"{key}: gamma flip is undefined. Chain may be empty or IV all-zero.",
                       f"flip_status={flip_status}")
    if flip_status == "none_in_range":
        return QACheck("flip_status", "WARN",
                       f"{key}: no gamma flip modeled within ±8% of spot. "
                       "Report 'flip undefined' not a fallback strike. "
                       "Regime is one-sided.",
                       f"flip_status={flip_status}")
    if flip_confidence == "LOW":
        return QACheck("flip_status", "WARN",
                       f"{key}: gamma flip modeled but confidence is LOW. "
                       "Multiple crossings or low GEX magnitude near crossing.",
                       f"flip_confidence={flip_confidence}")
    return QACheck("flip_status", "PASS",
                   f"{key}: flip modeled ({flip_confidence} confidence)")


def check_sign_stability(stable: bool, unstable_fields: list[str], key: str) -> QACheck:
    """Warn if key conclusions are sign-sensitive."""
    if not stable:
        return QACheck("sign_stability", "WARN",
                       f"{key}: key conclusions ({', '.join(unstable_fields)}) "
                       "change under alternative sign model. "
                       "Report should label these as LOW-CONFIDENCE.",
                       f"unstable={unstable_fields}")
    return QACheck("sign_stability", "PASS",
                   f"{key}: conclusions stable across street/stress sign models")


def check_gex_jump(
    current_net_gex: float,
    prior_net_gex: Optional[float],
    spot: float,
    key: str,
) -> QACheck:
    """Warn if GEX jumped abnormally vs prior run (may indicate data error or major event)."""
    if prior_net_gex is None:
        return QACheck("gex_jump", "PASS", f"{key}: no prior GEX for comparison")
    if prior_net_gex == 0:
        return QACheck("gex_jump", "PASS", f"{key}: prior GEX was zero")
    jump_pct = abs(current_net_gex - prior_net_gex) / max(abs(prior_net_gex), 1) * 100
    if jump_pct > 80:
        return QACheck("gex_jump", "WARN",
                       f"{key}: net GEX changed {jump_pct:.0f}% vs prior run. "
                       "Verify: large OI roll, data error, or major event.",
                       f"prior={prior_net_gex/1e9:.2f}B current={current_net_gex/1e9:.2f}B")
    return QACheck("gex_jump", "PASS", f"{key}: GEX change vs prior within normal range")


def check_flip_jump(
    current_flip: Optional[float],
    prior_flip: Optional[float],
    spot: float,
    key: str,
    max_jump_pct: float = DEFAULT_MAX_ALLOWED_FLIP_JUMP_PCT,
) -> QACheck:
    if current_flip is None or prior_flip is None:
        return QACheck("flip_jump", "PASS", f"{key}: no prior flip for comparison")
    jump_pct = abs(current_flip - prior_flip) / spot * 100
    if jump_pct > max_jump_pct:
        return QACheck("flip_jump", "WARN",
                       f"{key}: flip level moved {jump_pct:.1f}% vs prior "
                       f"(threshold {max_jump_pct:.1f}%). "
                       "Likely roll or data change — verify before relying on flip.",
                       f"prior_flip={prior_flip:.1f} current_flip={current_flip:.1f}")
    return QACheck("flip_jump", "PASS",
                   f"{key}: flip movement within normal range ({jump_pct:.1f}%)")


def check_cache_freshness(
    cache_dir: str,
    ticker: str,
    expiry: str,
    max_age_hours: float = DEFAULT_STALE_CHAIN_HOURS,
) -> QACheck:
    """Check if the cached chain data is fresh."""
    safe_ticker = ticker.replace("^", "").replace("=", "_").replace("-", "_")
    cache_file = os.path.join(cache_dir, f"{safe_ticker}_{expiry}.parquet")
    if not os.path.exists(cache_file):
        return QACheck("cache_freshness", "WARN",
                       f"No cache file found for {ticker} {expiry}. "
                       "Data may be freshly fetched or unavailable.",
                       f"path={cache_file}")
    age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
    if age_hours > max_age_hours:
        return QACheck("cache_freshness", "WARN",
                       f"{ticker} {expiry} chain cache is {age_hours:.1f}h old "
                       f"(threshold {max_age_hours:.0f}h). Data may be stale.",
                       f"age_hours={age_hours:.1f}")
    return QACheck("cache_freshness", "PASS",
                   f"{ticker} {expiry} cache age={age_hours:.1f}h")


def check_pcr_axis_clipping(by_str: pd.DataFrame, key: str, cap: float = 5.0) -> QACheck:
    """Warn if significant PCR values are being silently clipped by the y-axis cap."""
    if by_str.empty or "put_call_oi_ratio" not in by_str.columns:
        return QACheck("pcr_clipping", "PASS", f"{key}: no PCR data to check")
    pcr = by_str["put_call_oi_ratio"].dropna()
    clipped = (pcr > cap).sum()
    if clipped > 0:
        max_val = pcr.max()
        return QACheck("pcr_clipping", "WARN",
                       f"{key}: {clipped} strike(s) have PCR > {cap:.0f} "
                       f"(max={max_val:.1f}) — clipped in chart. "
                       "Consider annotating clipped strikes.",
                       f"clipped_count={clipped} max_pcr={max_val:.1f}")
    return QACheck("pcr_clipping", "PASS", f"{key}: no PCR values clipped")


# ── Master run function ───────────────────────────────────────────────────────

def run_qa(
    chain: pd.DataFrame,
    by_str: pd.DataFrame,
    spot: float,
    key: str,
    flip_status: str,
    flip_confidence: str,
    sign_stable: bool,
    sign_unstable_fields: list[str],
    prior_net_gex: Optional[float] = None,
    prior_flip: Optional[float] = None,
) -> QAReport:
    """
    Run all QA checks for one instrument and return a QAReport.
    """
    qa = QAReport()

    qa.add(check_spot_validity(spot, key))
    qa.add(check_chain_not_empty(chain, key))
    qa.add(check_iv_quality(chain, key))
    qa.add(check_oi_sufficiency(chain, key))
    qa.add(check_no_duplicate_contracts(chain, key))
    qa.add(check_front_expiry_present(chain, key))

    if "charmex_$" in chain.columns or "gex_$" in chain.columns:
        qa.add(check_greeks_finite(chain, key))

    qa.add(check_flip_status(flip_status, flip_confidence, key))
    qa.add(check_sign_stability(sign_stable, sign_unstable_fields, key))

    net_gex = float(chain["gex_$"].sum()) if "gex_$" in chain.columns else 0.0
    qa.add(check_gex_jump(net_gex, prior_net_gex, spot, key))
    qa.add(check_flip_jump(
        current_flip=None,  # filled by caller if available
        prior_flip=prior_flip,
        spot=spot, key=key,
    ))

    if not by_str.empty:
        qa.add(check_pcr_axis_clipping(by_str, key))

    qa.finalise()

    for line in qa.summary_lines():
        logger.info(f"[QA/{key}] {line}")

    return qa


# ── Prior state I/O ──────────────────────────────────────────────────────────

def load_prior_state(state_file: str) -> dict:
    """Load prior run state from JSON file. Returns empty dict on failure."""
    import json
    if not os.path.exists(state_file):
        return {}
    try:
        with open(state_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load prior state from {state_file}: {e}")
        return {}


def save_prior_state(state: dict, state_file: str) -> None:
    """Persist current run state for next-run comparison."""
    import json
    try:
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save prior state to {state_file}: {e}")


def build_state_snapshot(results: dict) -> dict:
    """
    Build a serialisable snapshot of this run's key state for next-run comparison.
    """
    snapshot = {"run_timestamp": datetime.now().isoformat(), "instruments": {}}
    for key, res in results.items():
        if res is None:
            continue
        snap = {
            "spot": res.get("spot"),
            "net_gex": res.get("total_net_gex"),
            "flip_level": res.get("flip"),
            "flip_status": res.get("flip_status", "unknown"),
            "front_expiry": res.get("front_expiry"),
            "regime": res.get("regime"),
        }
        snapshot["instruments"][key] = snap
    return snapshot
