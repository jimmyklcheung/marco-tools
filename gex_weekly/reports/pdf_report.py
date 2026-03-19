"""
PDF + HTML report renderer.
Uses Jinja2 for templating and WeasyPrint for PDF generation.
"""
import base64
import logging
import os
from typing import Optional

import pandas as pd
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _encode_image(path: Optional[str]) -> str:
    """Base64-encode a PNG file for embedding in HTML. Returns empty string on failure."""
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to encode image {path}: {e}")
        return ""


def _fmt_dollar(value: float, unit: str = "auto") -> str:
    """Format dollar value as $XB or $XM."""
    if unit == "auto":
        if abs(value) >= 1e9:
            return f"${value / 1e9:+.2f}B"
        return f"${value / 1e6:+.0f}M"
    if unit == "B":
        return f"${value / 1e9:+.2f}B"
    return f"${value / 1e6:+.0f}M"


def build_template_vars(
    spot: float,
    by_str: pd.DataFrame,
    by_exp: pd.DataFrame,
    chain: pd.DataFrame,
    report,                  # MacroReport
    opex_data: dict,
    multi_data: list,        # list of dicts, one per instrument
    chart_paths: dict,       # {chart_name: filepath}
    config: dict,
) -> dict:
    """
    Build template variable dict from all analytics outputs.
    """
    cfg_report = config.get("report", {})
    total_net_gex = by_str["net_gex"].sum() if not by_str.empty else 0
    total_call_gex = by_str["call_gex"].sum() if not by_str.empty else 0
    total_put_gex = by_str["put_gex"].sum() if not by_str.empty else 0

    flip = report.flip_level
    dist_to_flip_pct = ((spot - flip) / spot * 100) if spot > 0 else 0

    # Walls
    from analytics.greeks import find_gamma_walls
    walls = find_gamma_walls(by_str) if not by_str.empty else pd.DataFrame()
    call_walls = walls[walls["type"] == "call_wall"] if not walls.empty else pd.DataFrame()
    put_walls = walls[walls["type"] == "put_wall"] if not walls.empty else pd.DataFrame()
    call_wall_1 = float(call_walls["strike"].iloc[0]) if not call_walls.empty else None
    put_wall_1 = float(put_walls["strike"].iloc[0]) if not put_walls.empty else None

    # Max pain
    from analytics.greeks import find_max_pain
    max_pain = find_max_pain(chain) if not chain.empty else None

    # Key levels rows
    key_levels_rows = []
    if not walls.empty:
        for _, row in walls.iterrows():
            d = (float(row["strike"]) - spot) / spot * 100 if spot > 0 else 0
            key_levels_rows.append({
                "strike": float(row["strike"]),
                "type": row["type"].replace("_", " ").title(),
                "gex_m": float(row["gex_$"]) / 1e6,
                "dist_pct": d,
                "interpretation": (
                    "Strong resistance / dealer selling" if row["type"] == "call_wall"
                    else "Support / dealer delta hedge floor"
                ),
            })

    # Playbook
    is_long_gamma = total_net_gex >= 0
    if is_long_gamma:
        playbook = {
            "equities": "Range-trade, sell OTM straddles on strength",
            "vol": "Short VIX / sell vol spikes — dealers will absorb moves",
            "options": "Iron condors, credit spreads, short strangles",
            "sizing": "Normal to increased — stable regime supports risk-on",
        }
    else:
        playbook = {
            "equities": "Reduce net delta, buy downside puts for protection",
            "vol": "Long VIX, buy volatility — trend moves may persist",
            "options": "Long straddles, buy gamma for directional exposure",
            "sizing": "Reduce 30–50% — elevated tail risk in short-gamma regime",
        }

    # OPEX
    opex_gex_expiring_m = opex_data.get("net_gex_$M", 0) or 0
    total_gex_abs = max(abs(total_net_gex) / 1e6, 1)
    opex_pct_of_book = abs(opex_gex_expiring_m) / total_gex_abs * 100

    # VIX from macro signals (passed via report or multi_data)
    vix = None
    for inst in multi_data:
        if inst.get("vix") is not None:
            vix = inst["vix"]
            break

    import datetime
    now = datetime.datetime.now()

    tvars = {
        "title": cfg_report.get("title", "SPX GEX Weekly Report"),
        "author": cfg_report.get("author", "Macro Desk"),
        "report_date": now.strftime("%B %d, %Y"),
        "timestamp": now.strftime("%Y-%m-%d %H:%M ET"),
        "spot": round(spot, 2),
        "net_gex_b": round(total_net_gex / 1e9, 3),
        "flip_level": round(flip, 2),
        "dist_to_flip_pct": round(dist_to_flip_pct, 2),
        "call_wall_1": call_wall_1,
        "put_wall_1": put_wall_1,
        "max_pain": max_pain,
        "vix": vix,
        "regime": report.gamma_regime,
        "risk_level": report.risk_level,
        "summary": report.summary,
        "composite_score": round(report.composite_score, 3),
        "signals": [s.to_dict() for s in report.signals],
        "playbook": playbook,
        "key_levels_rows": key_levels_rows,
        "opex_date": opex_data.get("expiry", ""),
        "opex_dte": opex_data.get("dte", 0),
        "opex_gex_expiring_m": round(opex_gex_expiring_m, 1),
        "opex_pct_of_book": round(opex_pct_of_book, 1),
        "multi_instrument": multi_data,
        "charts": {k: _encode_image(v) for k, v in chart_paths.items()},
    }
    return tvars


def render_report(template_vars: dict, output_path: str) -> str:
    """
    Render HTML from template and convert to PDF via WeasyPrint.
    Returns the output PDF path (or HTML path if PDF fails).
    """
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR))
    template = env.get_template("report.html.jinja2")
    html_content = template.render(**template_vars)

    html_path = output_path.replace(".pdf", ".html")
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML saved: {html_path}")
    except Exception as e:
        logger.error(f"Failed to save HTML: {e}")

    # PDF generation
    try:
        from weasyprint import HTML
        HTML(string=html_content, base_url=_TEMPLATE_DIR).write_pdf(output_path)
        logger.info(f"PDF saved: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"WeasyPrint PDF failed: {e} — HTML report still available at {html_path}")
        return html_path


def render_email_body(template_vars: dict) -> str:
    """Render email HTML body from email template."""
    try:
        env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR))
        template = env.get_template("email.html.jinja2")
        return template.render(**template_vars)
    except Exception as e:
        logger.error(f"Failed to render email template: {e}")
        return f"<p>GEX Weekly Report — {template_vars.get('report_date', '')}. See attached PDF.</p>"
