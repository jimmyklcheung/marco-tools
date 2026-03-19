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


def _process_instrument(inst: dict, config: dict) -> Optional[dict]:
    """
    Fetch + compute Greeks for one instrument.
    Returns a result dict or None on failure.
    """
    from data.fetcher import get_spot, get_all_chains, get_next_opex
    from analytics.greeks import (
        compute_greeks, aggregate_by_strike, aggregate_by_expiry,
        find_gamma_flip, find_gamma_walls, find_max_pain, opex_expiring_gex,
    )

    key = inst["key"]
    ticker = inst["ticker"]
    multiplier = inst.get("multiplier", 100)
    opts = config.get("options", {})
    r = opts.get("risk_free_rate", 0.05)
    max_exp = opts.get("max_expiries", 8)
    spot_range = opts.get("spot_range", 0.10)
    cache_ttl = opts.get("cache_ttl_seconds", 300)

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

    flip = find_gamma_flip(by_str, spot) if not by_str.empty else spot
    walls = find_gamma_walls(by_str) if not by_str.empty else pd.DataFrame()
    max_pain = find_max_pain(chain) if not chain.empty else 0.0

    next_opex = get_next_opex(ticker)
    opex_data = opex_expiring_gex(chain, next_opex) if next_opex else {}

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

    regime = "LONG_GAMMA" if total_net_gex >= 0 else "SHORT_GAMMA"
    # Compute dist_to_flip from raw (unrounded) values to preserve sign.
    # Invariant: positive => spot is above flip (stable); negative => below flip (dangerous).
    dist_pct = ((spot - flip) / spot * 100) if spot > 0 else 0
    # Guard: clamp near-zero noise — if |dist_pct| < 0.01% treat as zero
    if abs(dist_pct) < 0.01:
        dist_pct = 0.0

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
        "walls": walls,
        "call_walls": call_walls,
        "put_walls": put_walls_df,
        "call_wall_1": call_wall_1,
        "put_wall_1": put_wall_1,
        "max_pain": max_pain,
        "next_opex": next_opex,
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
    }


def _generate_charts(primary: dict, output_dir: str, report) -> dict:
    """Generate all charts. Returns dict of {name: filepath}."""
    from reports.charts import (
        chart_gex_profile, chart_gex_by_expiry, chart_dex_profile,
        chart_vanna_charm, chart_put_call_ratio, chart_opex_gex,
        chart_macro_signals, chart_signal_gauge, _write_blank_png,
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

    def _safe_chart(name, fn, *args, **kwargs):
        path = os.path.join(charts_dir, f"{name}.png")
        try:
            fn(*args, output_path=path, **kwargs)
            paths[name] = path
        except Exception as e:
            logger.warning(f"Chart {name} failed: {e}")
            _write_blank_png(path)
            paths[name] = path

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

    # ── Step 1: Process each instrument ──────────────────────────────
    instruments = config.get("instruments", [])
    results = {}
    primary_result = None

    for inst in instruments:
        try:
            res = _process_instrument(inst, config)
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
            })
        else:
            multi_data.append({
                "key": key,
                "label": inst_cfg.get("label", key),
                "spot": round(res["spot"], 2),
                "net_gex_b": round(res["net_gex_b"], 3),
                "regime": res["regime"],
                "flip_level": round(res["flip"], 2),
                "dist_pct": round(res["dist_pct"], 2),
                "call_wall": res["call_wall_1"],
                "put_wall": res["put_wall_1"],
                "vix": vix_level,
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

    logger.info("=" * 60)
    logger.info(f"Report ready: {report_path}")
    logger.info("=" * 60)

    return report_path
