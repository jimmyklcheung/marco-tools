"""
Report orchestrator: glues data fetching, analytics, charting, and delivery.
"""
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _setup_logging(output_dir: str, verbose: bool = False):
    """Configure logging to console + run.log in monthly folder."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)s %(message)s"

    # Monthly log directory
    monthly_dir = os.path.dirname(output_dir)
    os.makedirs(monthly_dir, exist_ok=True)
    log_path = os.path.join(monthly_dir, "run.log")

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_output_dir(config: dict) -> str:
    """Create archive/{YYYY-MM}/gex_weekly_{YYYY-MM-DD}/ directory."""
    archive_cfg = config.get("archive", {})
    root = archive_cfg.get("root_dir", "./archive")
    folder_fmt = archive_cfg.get("folder_format", "%Y-%m")
    now = datetime.now()
    monthly = now.strftime(folder_fmt)
    date_str = now.strftime("%Y-%m-%d")
    out_dir = os.path.join(root, monthly, f"gex_weekly_{date_str}")
    os.makedirs(os.path.join(out_dir, "charts"), exist_ok=True)
    return out_dir


def _process_instrument(inst: dict, config: dict, prior_state: dict = None) -> Optional[dict]:
    """
    Fetch + compute Greeks for one instrument.
    Includes scenario engine, QA checks, and book decomposition.
    Returns a result dict or None on failure.
    """
    from data.fetcher import get_spot, get_all_chains, get_next_opex
    from analytics.greeks import (
        compute_greeks, aggregate_by_strike, aggregate_by_expiry,
        find_gamma_walls, find_max_pain, opex_expiring_gex,
    )
    from analytics.scenario import (
        compute_scenario, check_sign_sensitivity, classify_level,
        compute_books, detect_roll,
    )
    from validation.qa import run_qa
    from core.instruments import get_or_default

    key = inst["key"]
    ticker = inst["ticker"]
    multiplier = inst.get("multiplier", 100)
    opts = config.get("options", {})
    r = opts.get("risk_free_rate", 0.05)
    max_exp = opts.get("max_expiries", 8)
    spot_range = opts.get("spot_range", 0.10)
    cache_ttl = opts.get("cache_ttl_seconds", 300)

    # Look up instrument metadata (fails loudly for unregistered instruments)
    meta = get_or_default(key, multiplier)

    logger.info(f"Processing {key} ({ticker})")

    spot = get_spot(ticker)
    if spot == 0:
        logger.warning(f"Could not fetch spot for {key} — skipping")
        return None

    logger.info(f"{key} spot = {spot:.2f}")

    chain = get_all_chains(ticker, spot, max_exp, spot_range, cache_ttl)
    if chain.empty:
        logger.warning(f"No chain data for {key} — skipping")
        return None

    # Compute Greeks
    chain = compute_greeks(chain, spot, r, multiplier)

    by_str = aggregate_by_strike(chain)
    by_exp = aggregate_by_expiry(chain)

    # Scenario engine: modeled GEX vs hypothetical spot
    scenario = compute_scenario(chain, spot, r, multiplier, sign_model="street")
    # Sign sensitivity check
    sign_sens = check_sign_sensitivity(chain, spot, r, multiplier)

    # Modeled flip
    primary_flip = scenario.primary_flip
    flip_status = scenario.flip_status
    flip_confidence = scenario.flip_confidence

    # Legacy-compat: also run old cumsum flip for wall computation fallback
    from analytics.greeks import find_gamma_flip
    cumsum_flip = find_gamma_flip(by_str, spot) if not by_str.empty else spot
    # Use modeled flip for display; cumsum as last resort if scenario fails
    flip = primary_flip if primary_flip else cumsum_flip

    walls = find_gamma_walls(by_str) if not by_str.empty else pd.DataFrame()
    max_pain = find_max_pain(chain) if not chain.empty else 0.0

    next_opex = get_next_opex(ticker)
    opex_data = opex_expiring_gex(chain, next_opex) if next_opex else {}

    # Book decomposition: structural / tactical / OPEX
    tactical_dte = config.get("options", {}).get("tactical_max_dte", 5)
    books = compute_books(
        chain=chain, by_str=by_str, spot=spot, r=r, multiplier=multiplier,
        tactical_max_dte=tactical_dte, opex_date=next_opex,
    )

    # Front expiry for roll detection
    front_expiry = str(by_exp["expiry"].iloc[0]) if not by_exp.empty and "expiry" in by_exp.columns else None

    # Roll detection vs prior state
    prior_inst = (prior_state or {}).get(key, {})
    prior_front = prior_inst.get("front_expiry")
    prior_flip_val = prior_inst.get("flip_level")
    roll_status = detect_roll(front_expiry, prior_front, prior_flip_val, flip, spot)
    if roll_status.roll_detected:
        logger.info(f"[{key}] Roll detected: {roll_status.roll_note}")

    # QA
    qa = run_qa(
        chain=chain, by_str=by_str, spot=spot, key=key,
        flip_status=flip_status, flip_confidence=flip_confidence,
        sign_stable=sign_sens.regime_stable and sign_sens.flip_stable,
        sign_unstable_fields=sign_sens.unstable_fields,
        prior_net_gex=prior_inst.get("net_gex"),
        prior_flip=prior_flip_val,
    )
    logger.info(f"[{key}] QA: {qa.publish_status} | Confidence: {qa.confidence}%")

    # Level classification using scenario engine
    stress_scenario = sign_sens.stress
    classified_levels = []
    if not walls.empty:
        for _, row in walls.iterrows():
            lc = classify_level(
                strike=float(row["strike"]),
                wall_type=row["type"],
                scenario=scenario,
                stress_scenario=stress_scenario,
                by_str=by_str,
                spot=spot,
            )
            classified_levels.append(lc)

    # Aggregate totals
    total_net_gex = by_str["net_gex"].sum() if not by_str.empty else 0
    total_call_gex = by_str["call_gex"].sum() if not by_str.empty else 0
    total_put_gex = by_str["put_gex"].sum() if not by_str.empty else 0
    total_vannex = by_str["net_vannex"].sum() if not by_str.empty else 0
    total_charmex = by_str["net_charmex"].sum() if not by_str.empty else 0

    dte_nearest = int(by_exp["dte"].min()) if not by_exp.empty else None

    call_walls = walls[walls["type"] == "call_wall"] if not walls.empty else pd.DataFrame()
    put_walls_df = walls[walls["type"] == "put_wall"] if not walls.empty else pd.DataFrame()
    call_wall_1 = float(call_walls["strike"].iloc[0]) if not call_walls.empty else None
    put_wall_1 = float(put_walls_df["strike"].iloc[0]) if not put_walls_df.empty else None

    regime = scenario.regime_at_spot
    dist_pct = ((spot - flip) / spot * 100) if spot > 0 and flip else 0
    if abs(dist_pct) < 0.01:
        dist_pct = 0.0

    # Hedge flow per 1% move: GEX / spot (normalised sensitivity)
    hedge_flow_per_1pct = total_net_gex / spot / 100 if spot > 0 else 0

    return {
        "key": key,
        "label": inst.get("label", key),
        "ticker": ticker,
        "primary": inst.get("primary", False),
        "spot": spot,
        "chain": chain,
        "by_str": by_str,
        "by_exp": by_exp,
        "flip": flip,
        "flip_status": flip_status,
        "flip_confidence": flip_confidence,
        "primary_flip": primary_flip,
        "scenario": scenario,
        "sign_sensitivity": sign_sens,
        "classified_levels": classified_levels,
        "books": books,
        "roll_status": roll_status,
        "qa": qa,
        "walls": walls,
        "call_walls": call_walls,
        "put_walls": put_walls_df,
        "call_wall_1": call_wall_1,
        "put_wall_1": put_wall_1,
        "max_pain": max_pain,
        "next_opex": next_opex,
        "front_expiry": front_expiry,
        "opex_data": opex_data,
        "total_net_gex": total_net_gex,
        "total_call_gex": total_call_gex,
        "total_put_gex": total_put_gex,
        "total_vannex": total_vannex,
        "total_charmex": total_charmex,
        "dte_nearest": dte_nearest,
        "regime": regime,
        "dist_pct": dist_pct,
        "net_gex_b": total_net_gex / 1e9,
        "hedge_flow_per_1pct": hedge_flow_per_1pct,
        "futures_equivalent": meta.futures_equivalent,
        "futures_multiplier": meta.futures_multiplier,
        "confidence_label": qa.confidence_label,
        "confidence_int": qa.confidence,
    }


def _generate_charts(primary: dict, output_dir: str, report) -> dict:
    """Generate all charts. Returns dict of {name: filepath}."""
    from reports.charts import (
        chart_gex_profile, chart_gex_by_expiry, chart_dex_profile,
        chart_vanna_charm, chart_put_call_ratio, chart_opex_gex,
        chart_macro_signals, chart_signal_gauge, chart_scenario_gex,
        _write_blank_png,
    )

    charts_dir = os.path.join(output_dir, "charts")
    paths = {}

    by_str = primary["by_str"]
    by_exp = primary["by_exp"]
    spot = primary["spot"]
    flip = primary["flip"]
    call_walls = primary["call_walls"]
    put_walls = primary["put_walls"]
    next_opex = primary["next_opex"]
    max_pain = primary["max_pain"]
    chain = primary["chain"]
    scenario = primary.get("scenario")
    classified_levels = primary.get("classified_levels", [])

    def _safe_chart(name, fn, *args, **kwargs):
        path = os.path.join(charts_dir, f"{name}.png")
        try:
            fn(*args, output_path=path, **kwargs)
            paths[name] = path
        except Exception as e:
            logger.warning(f"Chart {name} failed: {e}")
            _write_blank_png(path)
            paths[name] = path

    # Hero chart: modeled GEX vs hypothetical spot (page 1)
    if scenario is not None:
        _safe_chart("scenario_gex", chart_scenario_gex,
                    scenario, spot, classified_levels)

    _safe_chart("gex_profile", chart_gex_profile,
                by_str, spot, flip, call_walls, put_walls)

    _safe_chart("gex_by_expiry", chart_gex_by_expiry,
                by_exp, next_opex)

    _safe_chart("dex_profile", chart_dex_profile,
                by_str, spot, flip)

    _safe_chart("vanna_charm", chart_vanna_charm,
                by_str, spot)

    _safe_chart("put_call_ratio", chart_put_call_ratio,
                by_str, spot)

    # OPEX-specific chart
    opex_by_str = pd.DataFrame()
    if next_opex and not chain.empty:
        from analytics.greeks import aggregate_by_strike
        opex_chain = chain[chain["expiry"] == next_opex].copy()
        if not opex_chain.empty:
            opex_by_str = aggregate_by_strike(opex_chain)

    _safe_chart("opex_gex", chart_opex_gex,
                opex_by_str, spot, max_pain, next_opex)

    _safe_chart("signal_bars", chart_macro_signals, report)

    _safe_chart("gauge", chart_signal_gauge, report.composite_score)

    return paths


def run_weekly_report(
    config_path: str = "config.yaml",
    force: bool = False,
    skip_email: bool = False,
    html_only: bool = False,
) -> str:
    """
    Main orchestrator: fetch data → compute analytics → generate charts
    → render PDF/HTML → optionally email → archive.

    Returns the path to the generated report file.
    """
    config_path = os.path.abspath(config_path)

    # Load config
    try:
        config = _load_config(config_path)
    except Exception as e:
        print(f"FATAL: Cannot load config from {config_path}: {e}")
        raise

    # Setup output dir
    output_dir = _make_output_dir(config)
    _setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("SPX GEX Weekly Report — Starting run")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # ── Step 0: Load prior state for what-changed and QA comparison ──
    from validation.qa import load_prior_state, save_prior_state, build_state_snapshot
    state_file = os.path.join(
        config.get("archive", {}).get("root_dir", "./archive"),
        "prior_state.json"
    )
    prior_state_raw = load_prior_state(state_file)
    prior_instruments = prior_state_raw.get("instruments", {})
    logger.info(f"Prior state: {'loaded' if prior_instruments else 'not available'}")

    # ── Step 1: Process each instrument ──────────────────────────────
    instruments = config.get("instruments", [])
    results = {}
    primary_result = None

    for inst in instruments:
        try:
            res = _process_instrument(inst, config, prior_state=prior_instruments)
            results[inst["key"]] = res
            if res and res.get("primary"):
                primary_result = res
        except Exception as e:
            logger.error(f"Failed processing {inst['key']}: {e}\n{traceback.format_exc()}")
            results[inst["key"]] = None

    # Fallback: use first successful result as primary
    if primary_result is None:
        for res in results.values():
            if res is not None:
                primary_result = res
                logger.warning(f"Using {res['key']} as primary (no primary flag set)")
                break

    if primary_result is None:
        logger.error("No instrument data available — aborting report")
        return ""

    # ── Step 2: Macro data + VIX ──────────────────────────────────────
    vix_level = None
    vix_change_1d = None
    try:
        from data.fetcher import get_vix_data
        vix_info = get_vix_data()
        vix_level = vix_info.get("level")
        vix_change_1d = vix_info.get("change_1d")
        logger.info(f"VIX: {vix_level:.2f} (1d chg: {vix_change_1d:+.2f})" if vix_level else "VIX: unavailable")
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")

    # ── Step 3: Generate macro signal report ─────────────────────────
    from analytics.signals import generate_report
    p = primary_result

    try:
        report = generate_report(
            spot=p["spot"],
            flip_level=p["flip"],
            total_net_gex=p["total_net_gex"],
            total_put_gex=p["total_put_gex"],
            total_call_gex=p["total_call_gex"],
            total_vannex=p["total_vannex"],
            total_charmex=p["total_charmex"],
            opex_gex_net=p["opex_data"].get("net_gex_$M", 0) * 1e6,
            opex_dte=p["opex_data"].get("dte"),
            dte_nearest=p["dte_nearest"],
            put_wall=p["put_wall_1"],
            vix_change_1d=vix_change_1d,
        )
        logger.info(f"Regime: {report.gamma_regime} | Risk: {report.risk_level} | Score: {report.composite_score:+.2f}")
        for sig in report.signals:
            logger.info(f"  {sig.name}: {sig.value} [{sig.score:+.2f}] ({sig.magnitude})")
    except Exception as e:
        logger.error(f"Signal report generation failed: {e}")
        raise

    # ── Step 4: Generate charts ───────────────────────────────────────
    logger.info("Generating charts...")
    try:
        chart_paths = _generate_charts(primary_result, output_dir, report)
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        chart_paths = {}

    # ── Step 4.5: Build commentary objects ───────────────────────────
    from reports.commentary import (
        build_street_take, build_todays_map, build_what_changed,
        format_book_summary, format_cross_asset_row,
    )

    sign_stable = (
        p.get("sign_sensitivity") is not None
        and p["sign_sensitivity"].regime_stable
        and p["sign_sensitivity"].flip_stable
    )
    qa_obj = p.get("qa")
    qa_publish_status = qa_obj.publish_status if qa_obj else "PASS"

    level_map = build_todays_map(
        regime=p["regime"],
        level_classifications=p.get("classified_levels", []),
        spot=p["spot"],
        dominant_expiry=p.get("front_expiry"),
        primary_flip=p.get("primary_flip"),
        flip_status=p.get("flip_status", "undefined"),
        confidence_label=p.get("confidence_label", "MODERATE"),
        confidence_int=p.get("confidence_int", 50),
    )

    street_take = build_street_take(
        regime=p["regime"],
        composite_score=report.composite_score,
        flip_status=p.get("flip_status", "undefined"),
        primary_flip=p.get("primary_flip"),
        spot=p["spot"],
        dist_to_flip_pct=p.get("dist_pct", 0),
        vix=vix_level,
        vix_change_1d=vix_change_1d,
        level_map=level_map,
        confidence=p.get("confidence_label", "MODERATE"),
        confidence_int=p.get("confidence_int", 50),
        sign_stable=sign_stable,
        qa_publish_status=qa_publish_status,
    )

    prior_primary = prior_instruments.get(p["key"], {})
    what_changed = build_what_changed(
        prior_state=prior_primary,
        current_regime=p["regime"],
        current_flip=p.get("primary_flip"),
        current_flip_status=p.get("flip_status", "undefined"),
        current_front_expiry=p.get("front_expiry"),
        spot=p["spot"],
        roll_status=p.get("roll_status"),
    )

    books_summary = format_book_summary(p.get("books", {}))

    # ── Step 5: Multi-instrument summary for template ─────────────────
    multi_data = []
    for inst_cfg in instruments:
        key = inst_cfg["key"]
        res = results.get(key)
        if res is None:
            multi_data.append({
                "key": key,
                "label": inst_cfg.get("label", key),
                "spot": 0,
                "net_gex_b": 0,
                "regime": "DATA_UNAVAILABLE",
                "flip_level": 0,
                "dist_pct": 0,
                "call_wall": None,
                "put_wall": None,
                "vix": vix_level,
                "flip_status": "undefined",
                "confidence_label": "VERY LOW",
                "qa_publish_status": "BLOCK",
                "futures_equivalent": None,
            })
        else:
            # Find nearest magnet and slippery from classified levels
            cl = res.get("classified_levels", [])
            pins = sorted([lc for lc in cl if lc.classification == "pin"], key=lambda x: abs(x.dist_pct))
            accs = sorted([lc for lc in cl if lc.classification == "accelerator"], key=lambda x: abs(x.dist_pct))
            res_qa = res.get("qa")
            multi_data.append({
                "key": key,
                "label": inst_cfg.get("label", key),
                "spot": round(res["spot"], 2),
                "net_gex_b": round(res["net_gex_b"], 3),
                "regime": res["regime"],
                "flip_level": round(res["flip"], 2) if res["flip"] else 0,
                "dist_pct": round(res["dist_pct"], 2),
                "call_wall": res["call_wall_1"],
                "put_wall": res["put_wall_1"],
                "vix": vix_level,
                "flip_status": res.get("flip_status", "undefined"),
                "confidence_label": res.get("confidence_label", "MODERATE"),
                "qa_publish_status": res_qa.publish_status if res_qa else "PASS",
                "nearest_magnet": f"{pins[0].strike:,.0f}" if pins else "none",
                "nearest_slippery": f"{accs[0].strike:,.0f}" if accs else "none",
                "futures_equivalent": res.get("futures_equivalent"),
                "primary_flip": res.get("primary_flip"),
                "classified_levels": res.get("classified_levels", []),
            })

    # ── Step 6: Build template vars + render ─────────────────────────
    from reports.pdf_report import build_template_vars, render_report, render_email_body

    try:
        tvars = build_template_vars(
            spot=p["spot"],
            by_str=p["by_str"],
            by_exp=p["by_exp"],
            chain=p["chain"],
            report=report,
            opex_data=p["opex_data"],
            multi_data=multi_data,
            chart_paths=chart_paths,
            config=config,
            street_take=street_take,
            level_map=level_map,
            what_changed=what_changed,
            qa=qa_obj,
            books_summary=books_summary,
            primary_result=p,
        )
        # Inject VIX into tvars
        tvars["vix"] = vix_level
    except Exception as e:
        logger.error(f"Template var build failed: {e}\n{traceback.format_exc()}")
        raise

    now = datetime.now()
    filename_fmt = config.get("archive", {}).get("filename_format", "gex_weekly_{date}.pdf")
    date_str = now.strftime("%Y-%m-%d")
    pdf_filename = filename_fmt.format(date=date_str)
    output_pdf = os.path.join(output_dir, pdf_filename)

    if html_only:
        # Render HTML only (skip PDF)
        html_path = output_pdf.replace(".pdf", ".html")
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(
            os.path.join(os.path.dirname(__file__), "templates")
        ))
        tmpl = env.get_template("report.html.jinja2")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(tmpl.render(**tvars))
        report_path = html_path
        logger.info(f"HTML-only report saved: {html_path}")
    else:
        report_path = render_report(tvars, output_pdf)

    # ── Step 7: Save CSV ─────────────────────────────────────────────
    if config.get("archive", {}).get("keep_csv", True) and not p["by_str"].empty:
        csv_path = os.path.join(output_dir, "gex_data.csv")
        try:
            p["by_str"].to_csv(csv_path, index=False)
            logger.info(f"CSV saved: {csv_path}")
        except Exception as e:
            logger.warning(f"CSV save failed: {e}")

    # ── Step 8: Email delivery ────────────────────────────────────────
    if not skip_email and config.get("email", {}).get("enabled", False):
        try:
            from delivery.emailer import send_report
            email_cfg = config["email"]
            subj_tmpl = email_cfg.get(
                "subject_template",
                "SPX GEX Weekly | {date} | Regime: {regime} | Risk: {risk_level}"
            )
            subject = subj_tmpl.format(
                date=date_str,
                regime=report.gamma_regime,
                risk_level=report.risk_level,
            )
            html_body = render_email_body(tvars)
            pdf_for_email = report_path if report_path.endswith(".pdf") else None
            send_report(pdf_for_email, subject, html_body, config)
        except Exception as e:
            logger.error(f"Email delivery failed: {e}")

    # ── Step 9: Save prior state for next run ─────────────────────────
    try:
        from validation.qa import build_state_snapshot, save_prior_state
        snapshot = {"instruments": {}}
        for key, res in results.items():
            if res is not None:
                snapshot["instruments"][key] = build_state_snapshot(
                    key=key,
                    spot=res["spot"],
                    regime=res["regime"],
                    flip_level=res.get("primary_flip"),
                    front_expiry=res.get("front_expiry"),
                    net_gex=res.get("total_net_gex", 0),
                )
        save_prior_state(state_file, snapshot)
        logger.info(f"Prior state saved: {state_file}")
    except Exception as e:
        logger.warning(f"Prior state save failed (non-fatal): {e}")

    logger.info("=" * 60)
    logger.info(f"Report ready: {report_path}")
    logger.info("=" * 60)

    return report_path
