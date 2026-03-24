# GEX Weekly Upgrade — Machine-Readable Checklist

## Legend
- `[ ]` not started
- `[~]` PARTIAL — code exists but not wired or not rendered
- `[x]` complete and tested

---

## BUGS (block prior-state persistence)

- [ ] B1 Fix `build_state_snapshot` signature mismatch in `validation/qa.py` — rename to `build_instrument_snapshot(key, spot, regime, flip_level, front_expiry, net_gex)` and update call site in `reports/builder.py:607`
- [ ] B2 Fix `save_prior_state` argument order in `reports/builder.py:615` — change `save_prior_state(state_file, snapshot)` to `save_prior_state(snapshot, state_file)`

---

## MARKET LOGIC CORRECTIONS

- [ ] C1 Fix above-spot put-wall label in `reports/pdf_report.py:134-139` — change "Overhead put concentration … dealer long-delta hedge unwind" to "Upside resistance / sell-on-rip supply … short-gamma; dealers sell into rallies"
- [ ] C2 Update `tests/test_pdf_report.py::test_put_wall_far_above_spot_is_overhead` — change assertion keyword from `"overhead"` to `"resistance"` or `"sell-on-rip"`
- [ ] C3 Gate `is_pin_node` on `net_gex > 0` at coincident strike in `reports/pdf_report.py:93-99`
- [ ] C4 Add `tests/test_pdf_report.py::test_dual_sided_negative_gex_is_not_pin`

---

## TEMPLATE PAGE 1 REDESIGN

- [ ] T1 Add CSS classes to `reports/templates/report.html.jinja2`: `.qa-bar`, `.qa-pass/.qa-warn/.qa-block`, `.street-take-box`, `.todays-map`, `.map-cell`, `.invalidation-box`, `.what-changed-box`, `.roll-note`
- [ ] T2 Replace page 1 body (lines 290-386) with desk-note layout: QA status bar → Street Take + 4-metric grid → Scenario GEX chart → Today's Map → What Changed + Invalidation
- [ ] T3 Add flip undefined guard: `{% if flip_status == "modeled" and primary_flip %}...{% else %}Undefined{% endif %}` replacing bare `format(flip_level)` call
- [ ] T4 Remove 8-card metrics grid from page 1 (reduce to 4 cards: spot, net_gex, flip, dist_to_flip)
- [ ] T5 Remove summary box and playbook grid from page 1 (move playbook to page 5 or remove)

---

## TEMPLATE: NEW CONTENT SECTIONS

- [~] S1 Render `scenario_key_levels` table on page 2 (data in tvars, not rendered)
- [~] S2 Render `books_summary` table on page 3 (data in tvars, not rendered)
- [~] S3 Render `qa_checks` on page 1 or diagnostics page (data in tvars, not rendered)
- [~] S4 Render `street_take.bullets` on page 1 (data in tvars, not rendered)
- [~] S5 Render `level_map` on page 1 (data in tvars, not rendered)
- [~] S6 Render `what_changed` summary on page 1 (data in tvars, not rendered)
- [x] S7 `charts.scenario_gex` chart generated and saved to disk

---

## TESTS

- [ ] TT1 Create `tests/test_scenario.py` with 11 tests (see plan for list)
- [ ] TT2 Create `tests/test_qa.py` with 12 tests (see plan for list)
- [ ] TT3 Update `tests/test_pdf_report.py` for C2 and C4

---

## ANALYTICS (already complete)

- [x] A1 `analytics/scenario.py` — `compute_scenario`, `classify_level`, `compute_books`, `detect_roll`, `check_sign_sensitivity`
- [x] A2 `analytics/greeks.py` — `aggregate_by_book`
- [x] A3 `validation/qa.py` — 12-check QA pipeline, confidence scoring, PASS/WARN/BLOCK
- [x] A4 `reports/commentary.py` — `build_street_take`, `build_todays_map`, `build_what_changed`, `format_book_summary`
- [x] A5 `core/instruments.py` — typed registry for SPX, SPY, QQQ, IWM, NDX, VIX
- [x] A6 `reports/builder.py` — scenario, QA, commentary, books, roll detection wired
- [x] A7 `reports/pdf_report.py` — extended `build_template_vars` with all new keys
- [x] A8 `reports/charts.py` — `chart_scenario_gex`

---

## KNOWN EXTERNAL-DATA LIMITATIONS (no code fix possible)

- [ ] D1 yfinance IV frequently zero for deep ITM/OTM — affects GEX accuracy (WARN from check_iv_quality)
- [ ] D2 yfinance OI is end-of-day only — no intraday positioning visibility
- [ ] D3 Cross-asset GEX not normalized by notional — SPX vs IWM comparison is raw $B, not comparable
- [ ] D4 Event book requires `event_dte_range` in `config.yaml` — not currently set

---

## PRIORITY ORDER

```
HIGH  : B1, B2  (blocks prior-state persistence; "what changed" always empty)
HIGH  : C1, C2  (wrong market logic displayed to reader)
HIGH  : T1, T2  (page 1 redesign — core deliverable not done)
MED   : C3, C4  (pin logic improvement)
MED   : T3      (flip undefined guard to prevent crash if None propagates)
MED   : TT1, TT2, TT3  (test coverage for new modules)
LOW   : S1, S2, S3  (additional content sections)
LOW   : T4, T5  (cleanup of old page 1 elements)
```
