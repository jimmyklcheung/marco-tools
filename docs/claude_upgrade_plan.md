# GEX Weekly — Concrete Upgrade Implementation Plan

**Branch:** `claude/gex-weekly-report-system-2xgD9`
**Status as of 2026-03-24:** Analytics pipeline fully wired; template page 1 not redesigned; two runtime bugs in prior-state persistence.

---

## Active Code Path (read order)

```
main.py
  → builder.py::run_weekly_report()
      Step 0  : validation/qa.py::load_prior_state()
      Step 1  : data/fetcher.py (spot, chain, opex)
                analytics/greeks.py (compute_greeks, aggregate_*)
                analytics/scenario.py (compute_scenario, check_sign_sensitivity,
                                       classify_level, compute_books, detect_roll)
                validation/qa.py::run_qa()
      Step 2  : data/fetcher.py::get_vix_data()
      Step 3  : analytics/signals.py::generate_report()
      Step 4  : reports/charts.py (9 charts incl. chart_scenario_gex)
      Step 4.5: reports/commentary.py (build_street_take, build_todays_map,
                                       build_what_changed, format_book_summary)
      Step 5  : multi_data assembly
      Step 6  : reports/pdf_report.py::build_template_vars()
                reports/pdf_report.py::render_report()
                  → reports/templates/report.html.jinja2
      Step 7  : CSV save
      Step 8  : delivery/emailer.py (if enabled)
      Step 9  : validation/qa.py::save_prior_state()  ← BUG
```

---

## What Still Needs Doing

### 1. Runtime bugs in prior-state persistence

**File:** `reports/builder.py` lines 601–618
**File:** `validation/qa.py` lines 401–429

**Bug A — Wrong call signature for `build_state_snapshot`:**

```python
# builder.py line 607 (WRONG — passes keyword args):
snapshot["instruments"][key] = build_state_snapshot(
    key=key, spot=res["spot"], regime=res["regime"],
    flip_level=res.get("primary_flip"), front_expiry=res.get("front_expiry"),
    net_gex=res.get("total_net_gex", 0),
)

# qa.py line 412 (ACTUAL signature — takes a single results dict):
def build_state_snapshot(results: dict) -> dict:
```

**Fix:** Replace `build_state_snapshot` in `qa.py` with a per-instrument helper:

```python
def build_instrument_snapshot(
    key: str, spot: float, regime: str,
    flip_level, front_expiry, net_gex: float,
) -> dict:
    return {
        "spot": spot,
        "net_gex": net_gex,
        "flip_level": flip_level,
        "flip_status": "modeled" if flip_level else "undefined",
        "front_expiry": front_expiry,
        "regime": regime,
    }
```

Then in `builder.py`:
```python
snapshot["instruments"][key] = build_instrument_snapshot(
    key=key, spot=res["spot"], ...
)
```

**Bug B — Reversed argument order for `save_prior_state`:**

```python
# builder.py line 615 (WRONG — state_file first):
save_prior_state(state_file, snapshot)

# qa.py line 401 (ACTUAL — state first):
def save_prior_state(state: dict, state_file: str) -> None:
```

**Fix in builder.py line 615:**
```python
save_prior_state(snapshot, state_file)
```

**Impact:** Until fixed, "What Changed" always shows "No prior run available." Roll detection never fires.

---

### 2. Market-logic correction: above-spot put-wall label

**File:** `reports/pdf_report.py` lines 134–139

```python
# CURRENT (wrong):
else:
    interp = (
        f"Overhead put concentration ({d:+.1f}% above spot) — "
        "deep ITM puts; dealer long-delta hedge unwind risk if spot rallies through"
    )
```

**Correct market logic:** A put wall above spot means dealers are SHORT GAMMA at that level (street sign model). Short gamma above spot = sell-on-rip zone, not "overhead concentration / dealer long-delta." The old label conflates put OI with put wall GEX sign.

**Fix:**
```python
else:
    interp = (
        f"Upside resistance / sell-on-rip supply ({d:+.1f}% above spot) — "
        "short-gamma at this level; dealers sell into rallies through this strike"
    )
```

**Test to update:** `tests/test_pdf_report.py::test_put_wall_far_above_spot_is_overhead`
Change assertion from `"overhead"` to `"resistance"` or `"sell-on-rip"`.

---

### 3. Market-logic correction: pin node must gate on net_gex > 0

**File:** `reports/pdf_report.py` lines 93–99

```python
# CURRENT (wrong — proximity only, no GEX sign check):
is_pin_node = (
    call_wall_1 is not None
    and put_wall_1 is not None
    and abs(call_wall_1 - put_wall_1) <= atm_tolerance
)
```

**Correct logic:** A coincident call+put wall is only a pin if net GEX at that strike is positive (dealers net long gamma → mean-reversion). If net GEX is negative, the node is expansive/ambiguous, not a pin.

**Fix:**
```python
pin_candidate_strike = call_wall_1  # or put_wall_1, same within tolerance
net_gex_at_pin = 0.0
if by_str is not None and not by_str.empty and pin_candidate_strike is not None:
    mask = (by_str["strike"] - pin_candidate_strike).abs() <= atm_tolerance
    if mask.any():
        net_gex_at_pin = float(by_str.loc[mask, "net_gex"].iloc[0])

is_pin_node = (
    call_wall_1 is not None
    and put_wall_1 is not None
    and abs(call_wall_1 - put_wall_1) <= atm_tolerance
    and net_gex_at_pin > 0
)
```

**Test to add:** `tests/test_pdf_report.py::test_dual_sided_negative_gex_is_not_pin`
Verify that when net_gex at the coincident strike is negative, `is_pin_node` is False.

---

### 4. Template page 1 redesign

**File:** `reports/templates/report.html.jinja2` lines 287–386

**Currently:** Regime badge + 8-card metrics grid + summary box + playbook grid.
**Required:** Executive desk-note layout with Street Take, Today's Map, What Changed, and scenario GEX chart on page 1.

**All data is already in `tvars`.** Only the template HTML needs to change.

**New page 1 structure (replace lines 290–386):**

```
┌─────────────────────────────────────────────────────────────┐
│ HEADER: title, date, regime badge, risk badge               │
│ QA STATUS BAR: PASS/WARN/BLOCK + confidence % + warnings    │
├────────────────────────────┬────────────────────────────────┤
│ STREET TAKE                │ KEY METRICS (4 cards)          │
│ (5 bullets from            │  Spot / Net GEX /              │
│  street_take dict)         │  Flip (w/ undefined guard) /   │
│                            │  Dist to Flip                  │
├────────────────────────────┴────────────────────────────────┤
│ SCENARIO GEX CHART (charts.scenario_gex)                    │
├─────────────────────────────────────────────────────────────┤
│ TODAY'S MAP (level_map dict)                                │
│  Magnet: X  |  Slippery: X  |  Supply: X  |  Expiry: X     │
├─────────────────────────────────────────────────────────────┤
│ WHAT CHANGED  |  INVALIDATION (from level_map.invalidation) │
└─────────────────────────────────────────────────────────────┘
```

**Template variables to consume (all already in tvars):**

| Variable | Type | Source |
|---|---|---|
| `street_take.bullets` | list[str] | commentary.py |
| `street_take.headline` | str | commentary.py |
| `level_map.primary_magnet` | str\|None | commentary.py |
| `level_map.downside_slippery_zone` | str\|None | commentary.py |
| `level_map.upside_supply_zone` | str\|None | commentary.py |
| `level_map.invalidation` | str | commentary.py |
| `level_map.dominant_expiry` | str\|None | commentary.py |
| `what_changed.summary` | str | commentary.py |
| `what_changed.roll_detected` | bool | commentary.py |
| `what_changed.roll_note` | str | commentary.py |
| `qa_publish_status` | str | qa.py |
| `qa_confidence` | int | qa.py |
| `qa_confidence_label` | str | qa.py |
| `qa_checks` | list[dict] | qa.py |
| `charts.scenario_gex` | str (b64) | charts.py |
| `flip_status` | str | scenario.py |
| `primary_flip` | float\|None | scenario.py |

**Critical template guard for flip:**
```jinja2
{# Replace bare format(flip_level) call on current line 335 #}
{% if flip_status == "modeled" and primary_flip %}
  {{ "{:,.0f}".format(primary_flip) }}
{% else %}
  Undefined
{% endif %}
```

**New CSS classes to add** (before `</style>`):

```css
.qa-bar { padding: 2mm 4mm; margin-bottom: 3mm; font-size: 8pt; font-weight: bold; border-radius: 2px; }
.qa-pass  { background: #003d00; color: #00c9a7; border: 1px solid #00c9a7; }
.qa-warn  { background: #3d3000; color: #ffd700; border: 1px solid #ffd700; }
.qa-block { background: #3d0000; color: #f76e6e; border: 1px solid #f76e6e; }
.street-take-box { background: #1a1f2e; border-left: 3px solid #5c9eff; padding: 3mm 5mm; margin: 2mm 0; }
.street-take-box .st-line { font-size: 8.5pt; color: #ccc; margin-bottom: 1.5mm; }
.street-take-box .st-line:first-child { color: #e0e0e0; font-weight: bold; font-size: 9pt; }
.todays-map { display: flex; gap: 3mm; margin: 2mm 0; }
.map-cell { flex: 1; background: #1a1f2e; border: 1px solid #2a3040; border-radius: 2px; padding: 2mm 3mm; }
.map-cell .mc-label { font-size: 7pt; color: #888; text-transform: uppercase; }
.map-cell .mc-val { font-size: 10pt; font-weight: bold; color: #e0e0e0; }
.invalidation-box { background: #12161f; border-left: 2px solid #ffd700; padding: 2mm 4mm; font-size: 8pt; color: #ccc; margin: 2mm 0; }
.what-changed-box { background: #12161f; border-left: 2px solid #888; padding: 2mm 4mm; font-size: 8pt; color: #bbb; margin: 2mm 0; }
.roll-note { color: #ffd700; font-weight: bold; }
```

---

### 5. Render scenario key levels on page 2

**File:** `reports/templates/report.html.jinja2` — after existing `key_levels_rows` table (after line 428)

Add a second table consuming `scenario_key_levels` with columns:
Strike | Classification | Confidence | Sign-Sensitive | Dist % | GEX at Strike ($B) | Interpretation

This lets a reader compare the old hardcoded interpretation with the scenario-derived one side by side until the old table is deprecated.

---

### 6. Render books summary on page 3 or a new page

**File:** `reports/templates/report.html.jinja2`

Add a table consuming `books_summary` (list of dicts with keys: label, expiry_filter, regime, net_gex_b, flip, dominant_expiry, dte_range, confidence).

Suggested placement: new section at bottom of page 3 (Expiry & Delta Analysis), below the OPEX box.

---

### 7. New tests to create

**File:** `tests/test_scenario.py` (create)

```python
# Required tests:
test_gex_curve_has_correct_length()         # build_scenario gives 60 points
test_zero_crossing_is_interpolated()        # known synthetic chain → exact flip
test_no_crossing_returns_none_not_fallback()# all-positive GEX → primary_flip is None
test_classify_level_above_spot_negative_gex_is_resistance()
test_classify_level_below_spot_negative_gex_is_accelerator()
test_classify_level_positive_gex_near_spot_is_pin()
test_classify_level_positive_gex_far_from_spot_is_concentration()
test_sign_sensitivity_returns_both_models()
test_compute_books_returns_structural_and_tactical()
test_detect_roll_when_expiry_changes()
test_detect_no_roll_when_expiry_same()
```

**File:** `tests/test_qa.py` (create)

```python
test_block_on_zero_spot()
test_block_on_empty_chain()
test_block_on_all_zero_iv()
test_warn_on_partial_iv_coverage()
test_warn_on_no_flip_in_range()
test_block_on_flip_undefined()
test_warn_on_sign_instability()
test_confidence_decreases_with_warns()
test_confidence_decreases_more_with_blocks()
test_publish_status_pass_when_all_clear()
test_publish_status_warn_when_warns()
test_publish_status_block_when_any_block()
test_prior_state_roundtrip()           # save then load produces same values
```

**File:** `tests/test_pdf_report.py` (modify)

- Update `test_put_wall_far_above_spot_is_overhead`: change assertion to `"resistance"` or `"sell-on-rip"`.
- Add `test_dual_sided_negative_gex_is_not_pin`: verify `is_pin_node` is False when net_gex at strike is negative.

---

## Implementation Order (dependency-safe)

1. **Fix Bug A + Bug B** in `builder.py` and `qa.py` (no downstream deps; safest first)
2. **Fix above-spot put-wall label** in `pdf_report.py` + update test
3. **Fix pin-node net_gex gate** in `pdf_report.py` + add test
4. **Create `tests/test_scenario.py`** and `tests/test_qa.py`
5. **Template page 1 redesign** (HTML/CSS only; all data already available)
6. **Add scenario key levels table** to page 2
7. **Add books summary table** to page 3

Steps 1–4 can be done independently. Step 5 depends on nothing. Steps 6–7 are additive.

---

## Files Changed Per Feature Area

### Scenario-based level classification
- `reports/pdf_report.py` — replace `key_levels_rows` put-wall/pin-node logic with `scenario_key_levels` path
- `reports/templates/report.html.jinja2` — render `scenario_key_levels` table on page 2

### Modeled gamma flip
- `validation/qa.py` — fix `build_instrument_snapshot` signature
- `reports/builder.py` — fix call site + argument order for `save_prior_state`
- `reports/templates/report.html.jinja2` — add `flip_status` guard on flip display

### Forward / instrument metadata
- `core/instruments.py` — already complete; `InstrumentMeta.forward_price()` uses dividend yield
- No further changes needed unless new instruments are added

### Structural / tactical / event books
- `reports/templates/report.html.jinja2` — render `books_summary` table (data already in tvars)
- `config.yaml` — add `event_dte_range` key if event book is needed

### Report redesign (page 1)
- `reports/templates/report.html.jinja2` lines 287–386 — replace with desk-note layout

### Diagnostics / confidence / publish blockers
- `reports/templates/report.html.jinja2` — add QA status bar on page 1
- `reports/builder.py` — optionally abort on BLOCK rather than just logging
