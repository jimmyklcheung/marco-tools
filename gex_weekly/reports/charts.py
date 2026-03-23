"""
Chart generators: all return filepath to a saved PNG.
Uses Plotly + kaleido for server-side PNG export.
"""
import io
import logging
import os
import struct
import zlib
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_chart_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("charts", {})
    except Exception:
        return {}


def _dims(cfg: dict):
    return cfg.get("width", 1200), cfg.get("height", 500)


def _colors(cfg: dict) -> dict:
    defaults = {
        "call": "#00c9a7",
        "put": "#f76e6e",
        "net": "#5c9eff",
        "spot": "#ffd700",
        "flip": "#ff6b35",
        "wall": "#a8ff78",
    }
    defaults.update(cfg.get("colors", {}))
    return defaults


def _write_blank_png(output_path: str):
    """Write a minimal valid 1×1 white PNG to output_path."""
    def chunk(name: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + name + data
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return c + struct.pack(">I", crc)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw_pixel = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw_pixel)
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")
    with open(output_path, "wb") as f:
        f.write(header + ihdr + idat + iend)


def _save_fig(fig: go.Figure, output_path: str, width: int, height: int):
    """Save figure as PNG, fall back to blank PNG on failure."""
    try:
        fig.write_image(output_path, width=width, height=height, scale=2)
    except Exception as e:
        logger.warning(f"Chart export failed ({output_path}): {e}")
        _write_blank_png(output_path)


# ── Chart functions ───────────────────────────────────────────────────────────

def chart_gex_profile(
    by_str: pd.DataFrame,
    spot: float,
    flip: float,
    call_walls: pd.DataFrame,
    put_walls: pd.DataFrame,
    output_path: str,
) -> str:
    """GEX profile: stacked call/put bars + net line + OI subplot."""
    cfg = _load_chart_config()
    w, h = _dims(cfg)
    col = _colors(cfg)

    try:
        fig = sp.make_subplots(
            rows=2, cols=1,
            row_heights=[0.70, 0.30],
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=["Dealer GEX Profile — All Expiries ($M)", "Open Interest by Strike"],
        )

        strikes = by_str["strike"].values
        call_gex_m = by_str["call_gex"].values / 1e6
        put_gex_m = by_str["put_gex"].values / 1e6
        net_gex_m = by_str["net_gex"].values / 1e6

        fig.add_trace(go.Bar(
            x=strikes, y=call_gex_m, name="Call GEX",
            marker_color=col["call"], opacity=0.8
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=strikes, y=put_gex_m, name="Put GEX",
            marker_color=col["put"], opacity=0.8
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=strikes, y=net_gex_m, name="Net GEX",
            line=dict(color=col["net"], width=2)
        ), row=1, col=1)

        # Vertical lines
        for xval, color, dash, label in [
            (spot, col["spot"], "solid", f"Spot {spot:,.0f}"),
            (flip, col["flip"], "dash", f"Flip {flip:,.0f}"),
        ]:
            fig.add_vline(x=xval, line_color=color, line_dash=dash,
                          annotation_text=label, annotation_font_color=color, row=1, col=1)

        # Call / put walls
        for _, row_w in call_walls.head(1).iterrows():
            fig.add_vline(x=row_w["strike"], line_color=col["wall"], line_dash="dot",
                          annotation_text=f"CW {row_w['strike']:,.0f}", row=1, col=1)
        for _, row_w in put_walls.head(1).iterrows():
            fig.add_vline(x=row_w["strike"], line_color=col["put"], line_dash="dot",
                          annotation_text=f"PW {row_w['strike']:,.0f}", row=1, col=1)

        # Zero-gamma shaded band
        if len(strikes) > 1:
            margin = (strikes[-1] - strikes[0]) * 0.005
            fig.add_vrect(
                x0=flip - margin, x1=flip + margin,
                fillcolor="rgba(255,215,0,0.1)", line_width=0,
                row=1, col=1
            )

        # OI subplot (mirrored bars)
        call_oi = by_str.get("call_oi", pd.Series([0] * len(by_str))).values
        put_oi = by_str.get("put_oi", pd.Series([0] * len(by_str))).values

        fig.add_trace(go.Bar(
            x=strikes, y=call_oi, name="Call OI",
            marker_color=col["call"], opacity=0.7, showlegend=False
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=strikes, y=-put_oi, name="Put OI",
            marker_color=col["put"], opacity=0.7, showlegend=False
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            barmode="relative",
            height=h,
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=60, r=40, t=60, b=40),
        )
        fig.update_yaxes(title_text="GEX ($M)", row=1, col=1)
        fig.update_yaxes(title_text="OI", row=2, col=1)

        _save_fig(fig, output_path, w, h)
    except Exception as e:
        logger.warning(f"chart_gex_profile failed: {e}")
        _write_blank_png(output_path)

    return output_path


def chart_gex_by_expiry(
    by_exp: pd.DataFrame,
    next_opex: Optional[str],
    output_path: str,
) -> str:
    """Horizontal bar chart: net GEX per expiry, coloured by sign."""
    cfg = _load_chart_config()
    w, h = _dims(cfg)
    col = _colors(cfg)

    try:
        net = by_exp["net_gex"].values / 1e6
        expiries = by_exp["expiry"].values
        bar_colors = [col["call"] if v >= 0 else col["put"] for v in net]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=net, y=expiries, orientation="h",
            marker_color=bar_colors,
            text=[f"${v:+.0f}M" for v in net],
            textposition="outside",
            name="Net GEX",
        ))

        if next_opex and next_opex in list(expiries):
            # add_hline doesn't work with categorical string y-axes; use a shape instead
            fig.add_shape(
                type="line",
                x0=0, x1=1, xref="paper",
                y0=next_opex, y1=next_opex, yref="y",
                line=dict(color=col["flip"], dash="dash", width=2),
            )
            fig.add_annotation(
                x=1, xref="paper",
                y=next_opex, yref="y",
                text="OPEX", showarrow=False,
                xanchor="left", font=dict(color=col["flip"]),
            )

        fig.update_layout(
            title="Net GEX by Expiry — Roll-Off Profile ($M)",
            template="plotly_dark",
            xaxis_title="Net GEX ($M)",
            height=h,
            margin=dict(l=120, r=80, t=60, b=40),
        )
        _save_fig(fig, output_path, w, h)
    except Exception as e:
        logger.warning(f"chart_gex_by_expiry failed: {e}")
        _write_blank_png(output_path)

    return output_path


def chart_dex_profile(
    by_str: pd.DataFrame,
    spot: float,
    flip: float,
    output_path: str,
) -> str:
    """DEX profile: stacked call/put DEX bars + net line."""
    cfg = _load_chart_config()
    w, h = _dims(cfg)
    col = _colors(cfg)

    try:
        strikes = by_str["strike"].values
        call_dex_m = by_str["call_dex"].values / 1e6
        put_dex_m = by_str["put_dex"].values / 1e6
        net_dex_m = by_str["net_dex"].values / 1e6

        fig = go.Figure()
        fig.add_trace(go.Bar(x=strikes, y=call_dex_m, name="Call DEX",
                             marker_color=col["call"], opacity=0.8))
        fig.add_trace(go.Bar(x=strikes, y=put_dex_m, name="Put DEX",
                             marker_color=col["put"], opacity=0.8))
        fig.add_trace(go.Scatter(x=strikes, y=net_dex_m, name="Net DEX",
                                 line=dict(color=col["net"], width=2)))

        fig.add_vline(x=spot, line_color=col["spot"], annotation_text=f"Spot {spot:,.0f}",
                      annotation_font_color=col["spot"])
        fig.add_vline(x=flip, line_color=col["flip"], line_dash="dash",
                      annotation_text=f"Flip {flip:,.0f}", annotation_font_color=col["flip"])

        fig.update_layout(
            title="Dealer Delta Exposure (DEX) by Strike ($M)",
            template="plotly_dark",
            barmode="relative",
            yaxis_title="DEX ($M)",
            height=h,
            margin=dict(l=60, r=40, t=60, b=40),
        )
        _save_fig(fig, output_path, w, h)
    except Exception as e:
        logger.warning(f"chart_dex_profile failed: {e}")
        _write_blank_png(output_path)

    return output_path


def chart_vanna_charm(
    by_str: pd.DataFrame,
    spot: float,
    output_path: str,
) -> str:
    """Two-panel: vanna by strike (left) + charm by strike (right)."""
    cfg = _load_chart_config()
    w, h = _dims(cfg)
    col = _colors(cfg)

    try:
        fig = sp.make_subplots(rows=1, cols=2,
                               subplot_titles=["Vanna Exposure ($M)", "Charm Exposure ($M)"])

        strikes = by_str["strike"].values
        vannex_m = by_str["net_vannex"].values / 1e6
        charmex_m = by_str["net_charmex"].values / 1e6

        v_colors = [col["call"] if v >= 0 else col["put"] for v in vannex_m]
        c_colors = [col["call"] if v >= 0 else col["put"] for v in charmex_m]

        fig.add_trace(go.Bar(x=strikes, y=vannex_m, marker_color=v_colors, name="Vanna"),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=strikes, y=charmex_m, marker_color=c_colors, name="Charm"),
                      row=1, col=2)

        for col_idx in [1, 2]:
            fig.add_vline(x=spot, line_color=col["spot"],
                          annotation_text=f"Spot", annotation_font_color=col["spot"],
                          row=1, col=col_idx)

        fig.update_layout(
            title="Vanna & Charm Exposure — Mechanical Flow Drivers",
            template="plotly_dark",
            height=h,
            showlegend=False,
            margin=dict(l=60, r=40, t=60, b=40),
        )
        _save_fig(fig, output_path, w, h)
    except Exception as e:
        logger.warning(f"chart_vanna_charm failed: {e}")
        _write_blank_png(output_path)

    return output_path


def chart_put_call_ratio(
    by_str: pd.DataFrame,
    spot: float,
    output_path: str,
) -> str:
    """Put/Call OI ratio by strike with colour gradient."""
    cfg = _load_chart_config()
    w, h = _dims(cfg)
    col = _colors(cfg)

    try:
        # Keep NaN so the valid-mask filter works; do NOT fillna here.
        strikes = by_str["strike"].values
        pcr = by_str["put_call_oi_ratio"].values  # NaN where call OI < threshold

        # Colour: green < 1, yellow 1-2, red > 2
        def pcr_color(v):
            if v < 1.0:
                return col["call"]
            if v < 2.0:
                return "#ffd700"
            return col["put"]

        # Drop strikes where call OI was too thin to produce a valid ratio
        valid = ~np.isnan(pcr)
        strikes = strikes[valid]
        pcr = pcr[valid]
        bar_colors = [pcr_color(v) for v in pcr]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=strikes, y=pcr, marker_color=bar_colors, name="P/C OI Ratio"))
        fig.add_hline(y=1.0, line_color="white", line_dash="dot",
                      annotation_text="1.0 (balanced)")
        fig.add_hline(y=1.5, line_color="#ffd700", line_dash="dash",
                      annotation_text="1.5 (elevated hedging)")
        fig.add_vline(x=spot, line_color=col["spot"],
                      annotation_text=f"Spot {spot:,.0f}", annotation_font_color=col["spot"])

        fig.update_layout(
            title="Put/Call OI Ratio by Strike",
            template="plotly_dark",
            yaxis_title="Put/Call OI Ratio",
            # Cap y-axis at 5 to prevent visual distortion from extreme outliers.
            # Genuine hedging concentration rarely exceeds 5x; values above are data noise.
            yaxis=dict(range=[0, 5]),
            height=h,
            margin=dict(l=60, r=40, t=60, b=40),
        )
        _save_fig(fig, output_path, w, h)
    except Exception as e:
        logger.warning(f"chart_put_call_ratio failed: {e}")
        _write_blank_png(output_path)

    return output_path


def chart_opex_gex(
    opex_chain_by_str: pd.DataFrame,
    spot: float,
    max_pain: float,
    next_opex: Optional[str],
    output_path: str,
) -> str:
    """GEX profile for the next OPEX expiry only."""
    cfg = _load_chart_config()
    w, h = _dims(cfg)
    col = _colors(cfg)

    try:
        title = f"GEX Profile — {next_opex or 'OPEX'} (OPEX Only)"

        if opex_chain_by_str.empty:
            fig = go.Figure()
            fig.update_layout(title=title, template="plotly_dark",
                              annotations=[dict(text="No OPEX data", x=0.5, y=0.5,
                                               xref="paper", yref="paper")])
            _save_fig(fig, output_path, w, h)
            return output_path

        strikes = opex_chain_by_str["strike"].values
        call_gex_m = opex_chain_by_str["call_gex"].values / 1e6
        put_gex_m = opex_chain_by_str["put_gex"].values / 1e6
        net_gex_m = opex_chain_by_str["net_gex"].values / 1e6

        fig = go.Figure()
        fig.add_trace(go.Bar(x=strikes, y=call_gex_m, name="Call GEX",
                             marker_color=col["call"], opacity=0.8))
        fig.add_trace(go.Bar(x=strikes, y=put_gex_m, name="Put GEX",
                             marker_color=col["put"], opacity=0.8))
        fig.add_trace(go.Scatter(x=strikes, y=net_gex_m, name="Net GEX",
                                 line=dict(color=col["net"], width=2)))

        fig.add_vline(x=spot, line_color=col["spot"],
                      annotation_text=f"Spot {spot:,.0f}", annotation_font_color=col["spot"])
        if max_pain:
            fig.add_vline(x=max_pain, line_color="#ff00ff", line_dash="dot",
                          annotation_text=f"MaxPain {max_pain:,.0f}",
                          annotation_font_color="#ff00ff")

        fig.update_layout(
            title=title, template="plotly_dark",
            barmode="relative", yaxis_title="GEX ($M)",
            height=h, margin=dict(l=60, r=40, t=60, b=40),
        )
        _save_fig(fig, output_path, w, h)
    except Exception as e:
        logger.warning(f"chart_opex_gex failed: {e}")
        _write_blank_png(output_path)

    return output_path


def chart_macro_signals(report, output_path: str) -> str:
    """Horizontal bar chart of signal scores with composite line."""
    cfg = _load_chart_config()
    w, h = _dims(cfg)
    col = _colors(cfg)

    try:
        signals = report.signals
        names = [s.name for s in signals]
        scores = [s.score for s in signals]
        bar_colors = []
        for s in scores:
            if s >= 0.3:
                bar_colors.append(col["call"])
            elif s <= -0.3:
                bar_colors.append(col["put"])
            else:
                bar_colors.append("#ffd700")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scores, y=names, orientation="h",
            marker_color=bar_colors,
            text=[f"{s:+.2f}" for s in scores],
            textposition="outside",
            name="Signal Score",
        ))

        # Composite line
        comp = report.composite_score
        fig.add_vline(x=comp, line_color="white", line_width=3,
                      annotation_text=f"Composite: {comp:+.2f}",
                      annotation_font_color="white", annotation_font_size=14)
        fig.add_vline(x=0, line_color="grey", line_dash="dot")

        fig.update_layout(
            title="Macro Signal Scorecard",
            template="plotly_dark",
            xaxis=dict(range=[-1.2, 1.2], title="Score"),
            height=h,
            margin=dict(l=140, r=80, t=60, b=40),
        )
        _save_fig(fig, output_path, w, h)
    except Exception as e:
        logger.warning(f"chart_macro_signals failed: {e}")
        _write_blank_png(output_path)

    return output_path


def chart_signal_gauge(composite_score: float, output_path: str) -> str:
    """Plotly gauge chart for composite regime score."""
    cfg = _load_chart_config()
    w, h = _dims(cfg)

    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=composite_score,
            number={"suffix": "", "font": {"size": 40}},
            delta={"reference": 0, "increasing": {"color": "#00c9a7"},
                   "decreasing": {"color": "#f76e6e"}},
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": "#5c9eff"},
                "bgcolor": "#1e2530",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [-1, -0.4], "color": "#3d0000"},
                    {"range": [-0.4, 0.4], "color": "#1e2530"},
                    {"range": [0.4, 1], "color": "#003d00"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.75,
                    "value": composite_score,
                },
            },
            title={"text": "Composite Regime Score", "font": {"size": 20}},
        ))

        fig.update_layout(
            template="plotly_dark",
            height=h,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        _save_fig(fig, output_path, w, h)
    except Exception as e:
        logger.warning(f"chart_signal_gauge failed: {e}")
        _write_blank_png(output_path)

    return output_path


def chart_scenario_gex(
    scenario,               # ScenarioResult from analytics/scenario.py
    spot: float,
    level_classifications,  # list[LevelClassification]
    output_path: str,
) -> str:
    """
    Hero chart: modeled net dealer GEX vs hypothetical spot price.

    GEX > 0 (green) = long-gamma zone: dealer flows compress moves.
    GEX < 0 (red)   = short-gamma zone: dealer flows amplify moves.
    Zero-crossing = modeled flip level(s).
    """
    cfg = _load_chart_config()
    w, h = _dims(cfg)
    col = _colors(cfg)

    try:
        grid = scenario.spot_grid
        gex_b = scenario.net_gex_curve / 1e9

        fig = go.Figure()

        # Positive (long-gamma) filled region
        pos_gex = np.where(gex_b > 0, gex_b, 0.0)
        neg_gex = np.where(gex_b < 0, gex_b, 0.0)
        fig.add_trace(go.Scatter(
            x=grid, y=pos_gex, fill="tozeroy",
            fillcolor="rgba(0,201,167,0.15)",
            line=dict(color=col["call"], width=1.5),
            name="Long-Gamma Zone",
        ))
        fig.add_trace(go.Scatter(
            x=grid, y=neg_gex, fill="tozeroy",
            fillcolor="rgba(247,110,110,0.15)",
            line=dict(color=col["put"], width=1.5),
            name="Short-Gamma Zone",
        ))
        fig.add_trace(go.Scatter(
            x=grid, y=gex_b,
            line=dict(color=col["net"], width=2.5),
            name="Net Dealer GEX",
        ))

        # Current spot
        fig.add_vline(
            x=spot, line_color=col["spot"], line_dash="solid", line_width=2,
            annotation_text=f"Spot {spot:,.0f}",
            annotation_font_color=col["spot"],
        )

        # Modeled flip levels
        for i, flip in enumerate(scenario.flip_levels):
            label = f"Flip {flip:,.0f}" if i == 0 else f"Flip₂ {flip:,.0f}"
            fig.add_vline(
                x=flip, line_color=col["flip"], line_dash="dash", line_width=2,
                annotation_text=label, annotation_font_color=col["flip"],
            )

        # Key classified levels
        cls_colors = {
            "pin": "#a8ff78",
            "resistance": col["put"],
            "accelerator": "#ff6b35",
            "support": col["call"],
            "concentration": "#aaa",
            "ambiguous": "#666",
        }
        for lc in (level_classifications or [])[:6]:
            lvl_col = cls_colors.get(lc.classification, "#888")
            fig.add_vline(
                x=lc.strike, line_color=lvl_col, line_dash="dot", line_width=1,
                annotation_text=f"{lc.strike:,.0f}",
                annotation_font_color=lvl_col, annotation_font_size=9,
            )

        fig.add_hline(y=0, line_color="white", line_dash="dot", line_width=1)

        flip_label = (
            scenario.flip_status.replace("_", " ").title()
            if scenario.flip_status != "modeled"
            else f"Flip: {scenario.primary_flip:,.0f}"
        )
        fig.update_layout(
            title=f"Modeled Net Dealer GEX vs Hypothetical Spot ({flip_label})",
            template="plotly_dark",
            xaxis_title="Hypothetical Spot Price",
            yaxis_title="Net Dealer GEX ($B)",
            height=h,
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=70, r=40, t=70, b=50),
        )
        _save_fig(fig, output_path, w, h)
    except Exception as e:
        logger.warning(f"chart_scenario_gex failed: {e}")
        _write_blank_png(output_path)

    return output_path
