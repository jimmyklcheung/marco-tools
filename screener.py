"""
screener.py — Actionable Trading Screen
=========================================
Imports the analysis engine from return_percentile.py and produces:

  1. A composite signal score per instrument, classifying each as:
       MOMENTUM  — high avg %tile + analogues tilt bullish → chase continuation
       REVERSAL  — low avg %tile  + analogues tilt bullish → bet on mean reversion
       NEUTRAL   — no clear edge

  2. Two ranked tables printed to console:
       Top Momentum Chase candidates (sorted by conviction, descending)
       Top Reversal candidates       (sorted by conviction, descending)

  3. Two-panel screen chart (PNG):
       Panel A — Opportunity scatter: avg %tile vs avg hit rate
       Panel B — Conviction ranking bars, split by signal type

  4. Weekly HTML email report sent to REPORT_TO (see configuration below).

Usage
-----
  python screener.py                 # run analysis + screen + email
  python screener.py --no-email      # run + screen, skip email
  python screener.py --screen-only   # use cached all_results if available

Environment variables (or .env file):
  REPORT_EMAIL_USER   Gmail address used as sender
  REPORT_EMAIL_PASS   Gmail App Password (16-char, spaces OK)
  REPORT_TO           Override recipient (default: jimmy.klcheung@gmail.com)
"""

import sys
import os
import base64
import smtplib
import textwrap
import argparse
from email.mime.multipart import MIMEMultipart
from email.mime.text       import MIMEText
from email.mime.base       import MIMEBase
from email                 import encoders
from datetime              import datetime

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.patches   as mpatches

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from return_percentile import run_analysis, TODAY_STR, PTILE_BAND

# ── 1. SCREEN CONFIGURATION ───────────────────────────────────────────────────

SCREEN = {
    "momentum_ptile_min": 65,    # avg %tile must be ≥ this to qualify as MOMENTUM
    "reversal_ptile_max": 35,    # avg %tile must be ≤ this to qualify as REVERSAL
    "min_hit_rate":       54.0,  # avg hit rate must exceed this (edge > coin flip)
    "min_n_obs":          25,    # minimum analogue observations (worst lookback)
    "early_slope_thresh": 20,    # |ptile_2w − ptile_12w| > this = "early/fading"
}

# ── Email ─────────────────────────────────────────────────────────────────────
REPORT_TO   = os.environ.get("REPORT_TO",           "jimmy.klcheung@gmail.com")
EMAIL_USER  = os.environ.get("REPORT_EMAIL_USER",   "")
EMAIL_PASS  = os.environ.get("REPORT_EMAIL_PASS",   "")
SMTP_HOST   = "smtp.gmail.com"
SMTP_PORT   = 587

# Asset-class colours for scatter chart
AC_COLORS = {
    "Equity Indices":  "#1f77b4",
    "US Sectors":      "#2ca02c",
    "Global Sectors":  "#17becf",
    "Equity Factors":  "#9467bd",
    "FX":              "#ff7f0e",
    "Commodities":     "#8c564b",
}


# ── 2. BUILD SCREEN DATAFRAME ─────────────────────────────────────────────────

def build_screen(all_results: dict) -> pd.DataFrame:
    """
    Aggregate (instrument × lookback × forward) results into one row per
    instrument with signal classification and conviction score.

    Key computed columns
    --------------------
    avg_ptile      : mean of 2w/4w/8w/12w current percentiles
    slope          : ptile_2w − ptile_12w
                       > 0  → recent momentum building (early chase signal)
                       < 0  → recent momentum fading  (early reversal signal)
    avg_hit_rate   : mean hit rate across all 12 (lookback × forward) cells
    avg_med_fwd_4w : mean median 4w forward return across all lookbacks
    min_n_obs      : minimum N obs (worst-case confidence)
    signal         : MOMENTUM | REVERSAL | NEUTRAL
    conviction     : composite score used to rank within each signal bucket
    slope_arrow    : ↑↑ / ↑ / → / ↓ / ↓↓  for display
    """
    rows = []

    for ac_label, results in all_results.items():
        for instr, lb_dict in results.items():
            ptiles_by_lb  = {}
            all_hit_rates = []
            fwd4w_meds    = []
            n_obs_vals    = []

            for lb_label, data in lb_dict.items():
                cp = data.get("current_ptile")
                if cp is not None and not np.isnan(cp):
                    ptiles_by_lb[lb_label] = float(cp)

                for fwd_label, stats in data.get("fwd", {}).items():
                    hr = stats.get("hit_rate")
                    if hr is not None and not np.isnan(hr):
                        all_hit_rates.append(float(hr))

                    if fwd_label == "4w fwd":
                        mr = stats.get("med_return")
                        if mr is not None and not np.isnan(mr):
                            fwd4w_meds.append(float(mr))

                    n = stats.get("n_obs", 0)
                    if n and n > 0:
                        n_obs_vals.append(int(n))

            if not ptiles_by_lb:
                continue

            # ── Aggregates ───────────────────────────────────────────────────
            avg_ptile = float(np.mean(list(ptiles_by_lb.values())))
            p2w       = ptiles_by_lb.get("2w",  np.nan)
            p4w       = ptiles_by_lb.get("4w",  np.nan)
            p8w       = ptiles_by_lb.get("8w",  np.nan)
            p12w      = ptiles_by_lb.get("12w", np.nan)

            slope = (p2w - p12w
                     if not (np.isnan(p2w) or np.isnan(p12w))
                     else np.nan)

            avg_hr      = float(np.mean(all_hit_rates)) if all_hit_rates else np.nan
            avg_fwd4w   = float(np.mean(fwd4w_meds))   if fwd4w_meds    else np.nan
            min_n       = int(min(n_obs_vals))          if n_obs_vals    else 0

            # ── Signal classification ─────────────────────────────────────────
            enough = (not np.isnan(avg_hr)) and (min_n >= SCREEN["min_n_obs"])

            if enough and avg_ptile >= SCREEN["momentum_ptile_min"] \
                      and avg_hr    >= SCREEN["min_hit_rate"]:
                signal = "MOMENTUM"
            elif enough and avg_ptile <= SCREEN["reversal_ptile_max"] \
                        and avg_hr    >= SCREEN["min_hit_rate"]:
                signal = "REVERSAL"
            else:
                signal = "NEUTRAL"

            # ── Conviction score ──────────────────────────────────────────────
            # Components (all bounded to prevent outliers dominating):
            #   edge     = avg hit rate above 50 (max ~20 pts)
            #   ret      = expected 4w return, capped at 3% (max ~15 pts)
            #   depth    = log of analogue count (sample reliability)
            #   timing   = bonus if slope confirms an "early" entry
            if signal in ("MOMENTUM", "REVERSAL"):
                edge   = max(0.0, avg_hr - 50.0) * 2.0
                ret    = min(abs(avg_fwd4w) if not np.isnan(avg_fwd4w) else 0.0,
                             3.0) * 5.0
                depth  = np.log1p(min_n) * 2.0
                # "Early" momentum: slope confirms direction
                early_thresh = SCREEN["early_slope_thresh"]
                early = ((signal == "MOMENTUM" and not np.isnan(slope)
                          and slope > early_thresh) or
                         (signal == "REVERSAL"  and not np.isnan(slope)
                          and slope < -early_thresh))
                timing = 10.0 if early else 0.0
                conviction = round(edge + ret + depth + timing, 1)
            else:
                conviction = 0.0

            # ── Slope arrow ───────────────────────────────────────────────────
            if   np.isnan(slope):    arr = "—"
            elif slope >  25:        arr = "↑↑ building"
            elif slope >  10:        arr = "↑"
            elif slope > -10:        arr = "→ confirmed"
            elif slope > -25:        arr = "↓"
            else:                    arr = "↓↓ fading"

            rows.append({
                "Asset Class":    ac_label,
                "Instrument":     instr,
                "Avg %tile":      round(avg_ptile, 1),
                "%tile 2w":       round(p2w,  1) if not np.isnan(p2w)  else np.nan,
                "%tile 4w":       round(p4w,  1) if not np.isnan(p4w)  else np.nan,
                "%tile 8w":       round(p8w,  1) if not np.isnan(p8w)  else np.nan,
                "%tile 12w":      round(p12w, 1) if not np.isnan(p12w) else np.nan,
                "Slope (2w−12w)": round(slope, 1) if not np.isnan(slope) else np.nan,
                "Momentum":       arr,
                "Avg Hit Rate":   round(avg_hr,    1) if not np.isnan(avg_hr)    else np.nan,
                "Avg Fwd 4w (%)": round(avg_fwd4w, 2) if not np.isnan(avg_fwd4w) else np.nan,
                "Min N obs":      min_n,
                "Signal":         signal,
                "Conviction":     conviction,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("Conviction", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


# ── 3. CONSOLE PRINT ──────────────────────────────────────────────────────────

def print_screen(df: pd.DataFrame):
    """Print the two ranked tables to stdout."""
    pd.set_option("display.float_format", "{:.1f}".format)
    pd.set_option("display.max_colwidth", 20)

    date_str = datetime.today().strftime("%d %b %Y")
    sep = "═" * 110

    print(f"\n{sep}")
    print(f"  WEEKLY TRADING SCREEN  |  {date_str}  |  "
          f"Analogue band ±{PTILE_BAND}%tile")
    print(f"  Hit-rate threshold: >{SCREEN['min_hit_rate']}%  |  "
          f"Min analogues: {SCREEN['min_n_obs']}  |  "
          f"Momentum ≥{SCREEN['momentum_ptile_min']}%tile  |  "
          f"Reversal ≤{SCREEN['reversal_ptile_max']}%tile")
    print(sep)

    display_cols = ["Asset Class", "Instrument", "Avg %tile",
                    "Slope (2w−12w)", "Momentum",
                    "Avg Hit Rate", "Avg Fwd 4w (%)", "Min N obs", "Conviction"]

    for signal, label, color_note in [
        ("MOMENTUM", "TOP MOMENTUM CHASE SIGNALS", "high %tile + strong hit rate → chase continuation"),
        ("REVERSAL", "TOP REVERSAL CANDIDATES",    "low %tile  + strong hit rate → bet on mean reversion"),
    ]:
        subset = df[df["Signal"] == signal].copy()
        print(f"\n  ── {label} ({color_note}) ──")
        if subset.empty:
            print("     (none qualify at current thresholds)")
            continue
        print(subset[display_cols].to_string(index=False))

    neutral_count = (df["Signal"] == "NEUTRAL").sum()
    total         = len(df)
    momentum_count = (df["Signal"] == "MOMENTUM").sum()
    reversal_count = (df["Signal"] == "REVERSAL").sum()

    print(f"\n  Summary: {total} instruments  |  "
          f"{momentum_count} MOMENTUM  |  "
          f"{reversal_count} REVERSAL  |  "
          f"{neutral_count} NEUTRAL")
    print(sep)


# ── 4. SCREEN CHARTS ──────────────────────────────────────────────────────────

def make_screen_charts(df: pd.DataFrame, save_path: str = None) -> str:
    """
    Two-panel screen chart:

    Panel A — Opportunity scatter
               x = Avg %tile (0–100)
               y = Avg Hit Rate (%)
               Bubble size ∝ Min N obs
               Colour = asset class
               Quadrant zones: REVERSAL (top-left) / MOMENTUM (top-right)

    Panel B — Conviction ranking bars
               Horizontal bars sorted by conviction score.
               Green = MOMENTUM, Red = REVERSAL.
               Instruments are labelled with their signal type arrow.
    """
    if save_path is None:
        save_path = f"screen_{TODAY_STR}.png"

    # Only show instruments with a real signal in Panel B
    signals = df[df["Signal"].isin(["MOMENTUM", "REVERSAL"])].copy()
    signals.sort_values("Conviction", ascending=True, inplace=True)

    n_signals = len(signals)
    fig_height = max(10, 4 + n_signals * 0.32)
    fig = plt.figure(figsize=(22, fig_height))
    fig.patch.set_facecolor("#f0f0f0")

    date_str = datetime.today().strftime("%d %b %Y")
    fig.suptitle(
        f"Weekly Trading Screen  |  {date_str}  |  "
        f"Analogue band ±{PTILE_BAND}%tile  |  "
        f"Thresholds: momentum ≥{SCREEN['momentum_ptile_min']}  "
        f"reversal ≤{SCREEN['reversal_ptile_max']}  "
        f"hit rate >{SCREEN['min_hit_rate']}%",
        fontsize=12, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    # ── Panel A: Opportunity Scatter ─────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#fafafa")

    # Shaded quadrant zones
    rev_max  = SCREEN["reversal_ptile_max"]
    mom_min  = SCREEN["momentum_ptile_min"]
    hr_min   = SCREEN["min_hit_rate"]

    ax_a.axhspan(hr_min, 100, xmin=0,
                 xmax=rev_max / 100,
                 facecolor="#ffe0e0", alpha=0.45, zorder=0)
    ax_a.axhspan(hr_min, 100, xmin=mom_min / 100,
                 xmax=1.0,
                 facecolor="#e0ffe0", alpha=0.45, zorder=0)
    ax_a.axvline(rev_max, color="#cc4444", lw=1.0, ls="--", alpha=0.6)
    ax_a.axvline(mom_min, color="#228822", lw=1.0, ls="--", alpha=0.6)
    ax_a.axhline(hr_min,  color="#888888", lw=1.0, ls="--", alpha=0.6)

    ax_a.text(rev_max / 2, 98, "REVERSAL\nZONE",
              ha="center", va="top", fontsize=8, color="#993333",
              fontweight="bold", alpha=0.7)
    ax_a.text((mom_min + 100) / 2, 98, "MOMENTUM\nZONE",
              ha="center", va="top", fontsize=8, color="#226622",
              fontweight="bold", alpha=0.7)

    valid = df.dropna(subset=["Avg %tile", "Avg Hit Rate"])
    for ac_label, group in valid.groupby("Asset Class"):
        color = AC_COLORS.get(ac_label, "#999999")
        sizes = (group["Min N obs"].clip(lower=10) / 5).clip(upper=300)
        ax_a.scatter(
            group["Avg %tile"], group["Avg Hit Rate"],
            s=sizes, c=color, alpha=0.80, edgecolors="white",
            linewidths=0.5, zorder=3,
        )
        # Label only instruments with a clear signal
        for _, row in group[group["Signal"] != "NEUTRAL"].iterrows():
            ax_a.annotate(
                row["Instrument"],
                (row["Avg %tile"], row["Avg Hit Rate"]),
                xytext=(4, 3), textcoords="offset points",
                fontsize=6.5, color=color, zorder=4,
            )

    # Legend for asset classes
    legend_handles = [
        mpatches.Patch(color=c, label=ac)
        for ac, c in AC_COLORS.items()
        if ac in valid["Asset Class"].unique()
    ]
    ax_a.legend(handles=legend_handles, fontsize=7,
                loc="lower center", ncol=3, framealpha=0.85)

    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(35, 80)
    ax_a.set_xlabel("Avg Return Percentile  (avg of 2w/4w/8w/12w lookbacks)", fontsize=9)
    ax_a.set_ylabel("Avg Hit Rate  (% of analogues with +ve fwd return)", fontsize=9)
    ax_a.set_title("Opportunity Map — all instruments", fontsize=10, pad=8)
    ax_a.grid(True, alpha=0.3, lw=0.5)

    # ── Panel B: Conviction Ranking Bars ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#fafafa")

    if n_signals == 0:
        ax_b.text(0.5, 0.5, "No signals qualify at current thresholds",
                  ha="center", va="center", fontsize=11, transform=ax_b.transAxes)
    else:
        y_pos    = range(n_signals)
        colors   = ["#2ca02c" if s == "MOMENTUM" else "#d62728"
                    for s in signals["Signal"]]

        bars = ax_b.barh(list(y_pos), signals["Conviction"], color=colors,
                         alpha=0.80, edgecolor="white", linewidth=0.5)

        # Instrument labels (left side)
        ax_b.set_yticks(list(y_pos))
        ax_b.set_yticklabels(
            [f"{row['Instrument']}  [{row['Asset Class']}]"
             for _, row in signals.iterrows()],
            fontsize=7.5,
        )

        # Annotate bars: show avg %tile / hit rate / fwd return
        for bar, (_, row) in zip(bars, signals.iterrows()):
            w    = bar.get_width()
            ptxt = (f"  %tile {row['Avg %tile']:.0f}  |  "
                    f"hit {row['Avg Hit Rate']:.0f}%  |  "
                    f"fwd4w {row['Avg Fwd 4w (%)']:+.1f}%  |  "
                    f"N={row['Min N obs']}  |  "
                    f"{row['Momentum']}")
            ax_b.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                      ptxt, va="center", ha="left", fontsize=6.5,
                      color="#333333")

        # Divider line between REVERSAL and MOMENTUM groups
        rev_idxs = [i for i, s in enumerate(signals["Signal"]) if s == "REVERSAL"]
        mom_idxs = [i for i, s in enumerate(signals["Signal"]) if s == "MOMENTUM"]
        if rev_idxs and mom_idxs:
            boundary = (max(rev_idxs) + min(mom_idxs)) / 2
            ax_b.axhline(boundary, color="#888888", lw=1.5, ls="--")

        # Signal type labels on the bars themselves (group header)
        if mom_idxs:
            mid_mom = np.mean(mom_idxs)
            ax_b.text(-0.5, mid_mom, "MOMENTUM\nCHASE",
                      ha="right", va="center", fontsize=8,
                      color="#2ca02c", fontweight="bold",
                      transform=ax_b.get_yaxis_transform())
        if rev_idxs:
            mid_rev = np.mean(rev_idxs)
            ax_b.text(-0.5, mid_rev, "REVERSAL\nCANDIDATE",
                      ha="right", va="center", fontsize=8,
                      color="#d62728", fontweight="bold",
                      transform=ax_b.get_yaxis_transform())

        max_conv = signals["Conviction"].max()
        ax_b.set_xlim(0, max_conv * 1.9)  # leave room for annotations
        ax_b.set_xlabel("Conviction Score", fontsize=9)

    ax_b.set_title(
        f"Signal Ranking  ({n_signals} actionable signals)  "
        "— sorted by conviction",
        fontsize=10, pad=8,
    )
    ax_b.grid(axis="x", alpha=0.3, lw=0.5)

    fig.text(
        0.5, 0.01,
        f"Source: Yahoo Finance  |  Universe: {len(df)} instruments  |  "
        f"Conviction = f(hit-rate edge, expected return, sample depth, timing)  |  "
        f"↑↑ = slope > {SCREEN['early_slope_thresh']}pts (early momentum)  |  "
        f"↓↓ = slope < −{SCREEN['early_slope_thresh']}pts (early reversal)",
        ha="center", fontsize=7, color="#555555",
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Screen chart saved → {save_path}")
    plt.close(fig)
    return save_path


# ── 5. HTML REPORT ────────────────────────────────────────────────────────────

def _html_table(subset: pd.DataFrame, signal: str) -> str:
    """Render a subset of the screen DataFrame as a styled HTML table."""
    if subset.empty:
        return "<p><i>No instruments qualify at current thresholds.</i></p>"

    header_bg  = "#1a5c1a" if signal == "MOMENTUM" else "#7a1a1a"
    row_colors = ("#e8f5e9", "#ffffff") if signal == "MOMENTUM" else ("#fdecea", "#ffffff")

    cols = ["Instrument", "Asset Class", "Avg %tile", "Slope (2w−12w)",
            "Momentum", "Avg Hit Rate", "Avg Fwd 4w (%)", "Min N obs", "Conviction"]

    th_style = (f"background:{header_bg}; color:#ffffff; padding:6px 10px; "
                "text-align:center; font-size:11px; white-space:nowrap;")
    td_style = "padding:5px 10px; text-align:center; font-size:11px;"

    html = ['<table style="border-collapse:collapse; width:100%; '
            'font-family:Arial,sans-serif; margin-bottom:20px;">']
    html.append("<thead><tr>")
    for c in cols:
        html.append(f'<th style="{th_style}">{c}</th>')
    html.append("</tr></thead><tbody>")

    for i, (_, row) in enumerate(subset.iterrows()):
        bg = row_colors[i % 2]
        html.append(f'<tr style="background:{bg};">')
        for c in cols:
            v = row[c]
            if isinstance(v, float) and not np.isnan(v):
                if c in ("Avg %tile", "%tile 2w", "%tile 12w"):
                    cell = f"{v:.0f}"
                elif c in ("Avg Hit Rate",):
                    cell = f"{v:.1f}%"
                elif c in ("Avg Fwd 4w (%)",):
                    cell = f"{v:+.2f}%"
                elif c in ("Slope (2w−12w)",):
                    col_style = ("color:#1a6f1a;" if v > 0 else
                                 "color:#b21818;" if v < 0 else "")
                    cell = (f'<span style="{col_style}">'
                            f'{"+" if v > 0 else ""}{v:.1f}</span>')
                elif c == "Conviction":
                    cell = f"<b>{v:.1f}</b>"
                else:
                    cell = f"{v:.1f}"
            elif isinstance(v, (int, np.integer)):
                cell = str(v)
            else:
                cell = str(v) if v is not None else "—"
            html.append(f'<td style="{td_style}">{cell}</td>')
        html.append("</tr>")

    html.append("</tbody></table>")
    return "\n".join(html)


def build_html_report(df: pd.DataFrame, chart_path: str) -> str:
    """Build the full HTML email body with embedded chart + ranked tables."""
    date_str = datetime.today().strftime("%A %d %b %Y")

    momentum = df[df["Signal"] == "MOMENTUM"].copy()
    reversal = df[df["Signal"] == "REVERSAL"].copy()

    momentum_count = len(momentum)
    reversal_count = len(reversal)
    total          = len(df)

    # Embed chart as base64 inline image
    try:
        with open(chart_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("ascii")
        chart_html = (
            f'<img src="data:image/png;base64,{img_b64}" '
            f'style="max-width:100%; margin:10px 0;" alt="Screen Chart"/>'
        )
    except FileNotFoundError:
        chart_html = "<p><i>(Chart not available)</i></p>"

    html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <style>
        body {{ font-family: Arial, sans-serif; color: #222; background: #fff;
                max-width: 1000px; margin: auto; padding: 20px; }}
        h1   {{ font-size: 18px; color: #1a1a1a; }}
        h2   {{ font-size: 14px; margin-top: 24px; padding: 6px 10px;
                border-radius: 3px; }}
        .momentum-h  {{ background: #1a5c1a; color: #fff; }}
        .reversal-h  {{ background: #7a1a1a; color: #fff; }}
        .summary     {{ background: #f4f4f4; padding: 10px 14px;
                        border-left: 4px solid #555; margin-bottom: 16px;
                        font-size: 12px; }}
        .footer      {{ font-size: 10px; color: #777; margin-top: 30px;
                        border-top: 1px solid #ddd; padding-top: 10px; }}
      </style>
    </head>
    <body>
      <h1>Weekly Signal Screen &mdash; {date_str}</h1>

      <div class="summary">
        <b>Universe:</b> {total} instruments &nbsp;|&nbsp;
        <b>Analogue band:</b> &plusmn;{PTILE_BAND} %tile &nbsp;|&nbsp;
        <b>Momentum signals:</b> {momentum_count} &nbsp;|&nbsp;
        <b>Reversal signals:</b> {reversal_count}<br/>
        <b>Thresholds:</b>
        Momentum &ge;{SCREEN['momentum_ptile_min']}%tile &nbsp;
        Reversal &le;{SCREEN['reversal_ptile_max']}%tile &nbsp;
        Hit-rate &gt;{SCREEN['min_hit_rate']}% &nbsp;
        Min analogues &ge;{SCREEN['min_n_obs']}
      </div>

      {chart_html}

      <h2 class="momentum-h">
        Top Momentum Chase Signals &mdash;
        high avg %tile &plus; analogues tilt bullish &rarr; chase continuation
      </h2>
      {_html_table(momentum, "MOMENTUM")}

      <h2 class="reversal-h">
        Top Reversal Candidates &mdash;
        low avg %tile &plus; analogues tilt bullish &rarr; bet on mean reversion
      </h2>
      {_html_table(reversal, "REVERSAL")}

      <div class="footer">
        Source: Yahoo Finance &nbsp;|&nbsp;
        History: 30 years &nbsp;|&nbsp;
        Percentile: expanding window (min 252 days, no look-ahead bias) &nbsp;|&nbsp;
        Analogues: dates where historical %tile was within &plusmn;{PTILE_BAND}pts of today&apos;s reading &nbsp;|&nbsp;
        Hit rate: % of analogues followed by a positive 1w/2w/4w return (averaged) &nbsp;|&nbsp;
        Conviction = f(hit-rate edge, expected return, sample depth, timing bonus for early signals) &nbsp;|&nbsp;
        &uarr;&uarr; building = slope &gt;+{SCREEN['early_slope_thresh']}pts &nbsp;
        &darr;&darr; fading = slope &lt;&minus;{SCREEN['early_slope_thresh']}pts
      </div>
    </body>
    </html>
    """).strip()

    return html


# ── 6. EMAIL ──────────────────────────────────────────────────────────────────

def send_email_report(
    html_body:   str,
    attachments: list,
    to_addr:     str,
    from_addr:   str,
    password:    str,
):
    """
    Send an HTML email via Gmail SMTP (TLS on port 587).

    Prerequisites
    -------------
    1. Enable 2-Step Verification on the sender Gmail account.
    2. Create a Gmail App Password at:
           myaccount.google.com/apppasswords
       (select App = Mail, Device = Other)
    3. Set env vars REPORT_EMAIL_USER + REPORT_EMAIL_PASS (see .env.example).
    """
    date_str = datetime.today().strftime("%d %b %Y")
    subject  = f"Weekly Signal Screen — {date_str}"

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"]    = from_addr
    msg["To"]      = to_addr

    # HTML body
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText("Weekly signal screen — please view in an HTML-capable client.",
                        "plain"))
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)

    # Attachments
    for path in attachments:
        if not os.path.exists(path):
            print(f"  Attachment not found, skipping: {path}")
            continue
        with open(path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f'attachment; filename="{os.path.basename(path)}"')
        msg.attach(part)

    print(f"Sending email to {to_addr} …")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(from_addr, password)
            server.send_message(msg)
        print(f"Email sent successfully to {to_addr}")
    except smtplib.SMTPAuthenticationError:
        print("ERROR: Gmail authentication failed.")
        print("  → Make sure you are using a Gmail App Password, not your regular password.")
        print("  → Generate one at: myaccount.google.com/apppasswords")
    except Exception as e:
        print(f"ERROR sending email: {e}")


# ── 7. MAIN ───────────────────────────────────────────────────────────────────

def main(send_email: bool = True):
    """
    Full pipeline:
      1. Run analysis engine (download + percentiles + analogues)
      2. Build screen DataFrame
      3. Print ranked tables to console
      4. Generate screen chart
      5. Export screen CSV
      6. Build HTML report
      7. Send email (if credentials are set)
    """
    print("\n" + "=" * 70)
    print("  WEEKLY SIGNAL SCREEN")
    print("=" * 70)

    # Step 1 — Run analysis
    all_results = run_analysis()
    if not all_results:
        print("No results — check data download.")
        return

    # Step 2 — Build screen
    screen_df = build_screen(all_results)

    # Step 3 — Console output
    print_screen(screen_df)

    # Step 4 — Screen chart
    chart_path = f"screen_{TODAY_STR}.png"
    make_screen_charts(screen_df, save_path=chart_path)

    # Step 5 — Screen CSV
    csv_path = f"screen_{TODAY_STR}.csv"
    screen_df.to_csv(csv_path, index=False)
    print(f"Screen CSV saved → {csv_path}")

    # Step 6 — Precedents CSV
    prec_path = f"precedents_{TODAY_STR}.csv"
    from return_percentile import export_precedents_csv
    export_precedents_csv(all_results, prec_path)

    # Step 7 — Email
    html = build_html_report(screen_df, chart_path)

    if send_email:
        if EMAIL_USER and EMAIL_PASS:
            send_email_report(
                html_body   = html,
                attachments = [chart_path, csv_path],
                to_addr     = REPORT_TO,
                from_addr   = EMAIL_USER,
                password    = EMAIL_PASS,
            )
        else:
            print("\nEmail credentials not configured — report saved locally only.")
            print("Set REPORT_EMAIL_USER and REPORT_EMAIL_PASS in your environment")
            print("or in a .env file (see .env.example).")
    else:
        print("\n(Email skipped — --no-email flag set)")

    # Save HTML for inspection
    html_path = f"screen_{TODAY_STR}.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML report saved → {html_path}")

    return screen_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly Signal Screen")
    parser.add_argument("--no-email", action="store_true",
                        help="Skip sending email, save reports locally only")
    args = parser.parse_args()
    main(send_email=not args.no_email)
