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
    "momentum_ptile_min":  65,   # avg %tile must be ≥ this → MOMENTUM or FADE zone
    "reversal_ptile_max":  35,   # avg %tile must be ≤ this → REVERSAL or BREAKDOWN zone
    "min_hit_rate_long":   54.0, # hit rate must be ABOVE this for a LONG signal
    "max_hit_rate_short":  46.0, # hit rate must be BELOW this for a SHORT signal
    "min_n_obs":           25,   # minimum analogue observations (worst lookback)
    "early_slope_thresh":  20,   # |ptile_2w − ptile_12w| > this = confirmed direction
}

# Signal colours  (LONG = cool, SHORT = warm)
SIG_COLORS = {
    "MOMENTUM":  "#2ca02c",   # green  — LONG, chase strength
    "REVERSAL":  "#1f77b4",   # blue   — LONG, buy the dip
    "FADE":      "#ff7f0e",   # orange — SHORT, fade the exhausted rally
    "BREAKDOWN": "#d62728",   # red    — SHORT, don't catch the falling knife
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
                       > 0  → momentum building  (favours MOMENTUM chase)
                       < 0  → momentum fading    (favours FADE / BREAKDOWN short)
    avg_hit_rate   : mean hit rate across all 12 (lookback × forward) cells
    avg_med_fwd_4w : mean median 4w forward return across all lookbacks
    min_n_obs      : minimum N obs (worst-case confidence)

    Signal matrix (2D: level × direction of historical edge)
    ---------------------------------------------------------
              | hit_rate > min_hit_rate_long | hit_rate < max_hit_rate_short
    ----------|------------------------------|------------------------------
    ptile ≥65 | MOMENTUM  (LONG: chase)      | FADE      (SHORT: fade rally)
    ptile ≤35 | REVERSAL  (LONG: buy dip)    | BREAKDOWN (SHORT: avoid dip)
    other     | NEUTRAL                      | NEUTRAL

    Slope refines timing within each signal:
      MOMENTUM  + slope ↑↑ = early chase (best entry, momentum still building)
      FADE      + slope ↓↓ = early fade  (best entry, strength just turning)
      REVERSAL  + slope ↓↓ = ripe reversal (momentum bottoming)
      BREAKDOWN + slope ↓↓ = active breakdown (keep short)

    conviction : comparable across LONG and SHORT — both use distance from 50%
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
            hi_ptile = avg_ptile >= SCREEN["momentum_ptile_min"]
            lo_ptile = avg_ptile <= SCREEN["reversal_ptile_max"]
            long_edge  = avg_hr >= SCREEN["min_hit_rate_long"]
            short_edge = avg_hr <= SCREEN["max_hit_rate_short"]

            if   enough and hi_ptile and long_edge:   signal = "MOMENTUM"
            elif enough and lo_ptile and long_edge:   signal = "REVERSAL"
            elif enough and hi_ptile and short_edge:  signal = "FADE"
            elif enough and lo_ptile and short_edge:  signal = "BREAKDOWN"
            else:                                     signal = "NEUTRAL"

            # ── Conviction score ──────────────────────────────────────────────
            # Same formula for LONG and SHORT — both measure distance from 50%.
            # This makes conviction scores comparable across all 4 signal types.
            #
            #   edge   = |avg_hit_rate − 50| × 2   (0–20 pts; symmetrical)
            #   ret    = min(|avg_fwd4w|, 3%) × 5   (0–15 pts; magnitude only)
            #   depth  = log(min_n_obs + 1) × 2     (sample reliability)
            #   timing = 10 if slope confirms signal direction (early-entry bonus)
            #
            # Timing bonus:
            #   MOMENTUM  + slope > +thresh  → momentum still building (early chase)
            #   REVERSAL  + slope < −thresh  → still falling but about to turn (ripe)
            #   FADE      + slope < 0        → strength just starting to fade (early)
            #   BREAKDOWN + slope < −thresh  → actively breaking (keep short)
            early_thresh = SCREEN["early_slope_thresh"]
            sl = slope if not np.isnan(slope) else 0.0

            if signal != "NEUTRAL":
                edge  = abs(avg_hr - 50.0) * 2.0
                ret   = min(abs(avg_fwd4w) if not np.isnan(avg_fwd4w) else 0.0,
                            3.0) * 5.0
                depth = np.log1p(min_n) * 2.0

                timing = 10.0 if (
                    (signal == "MOMENTUM"  and sl >  early_thresh) or
                    (signal == "REVERSAL"  and sl < -early_thresh) or
                    (signal == "FADE"      and sl <  0)            or
                    (signal == "BREAKDOWN" and sl < -early_thresh)
                ) else 0.0

                conviction = round(edge + ret + depth + timing, 1)
            else:
                conviction = 0.0

            # ── Slope arrow  (same arrow, interpreted per signal context) ─────
            #   MOMENTUM:  ↑↑ = best entry  |  ↓↓ = late / at risk of turning
            #   REVERSAL:  ↓↓ = ripe        |  ↑↑ = may be recovering already
            #   FADE:      ↓↓ = best entry  |  ↑↑ = premature, still overbought
            #   BREAKDOWN: ↓↓ = best entry  |  ↑↑ = dangerous, may be bouncing
            if   np.isnan(slope):  arr = "—"
            elif slope >  25:      arr = "↑↑ building"
            elif slope >  10:      arr = "↑"
            elif slope > -10:      arr = "→ confirmed"
            elif slope > -25:      arr = "↓"
            else:                  arr = "↓↓ fading"

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
    """Print all four ranked signal tables to stdout."""
    pd.set_option("display.float_format", "{:.1f}".format)
    pd.set_option("display.max_colwidth", 20)

    date_str = datetime.today().strftime("%d %b %Y")
    sep = "═" * 115

    print(f"\n{sep}")
    print(f"  WEEKLY TRADING SCREEN  |  {date_str}  |  Analogue band ±{PTILE_BAND}%tile")
    print(f"  LONG edge: hit-rate >{SCREEN['min_hit_rate_long']}%  |  "
          f"SHORT edge: hit-rate <{SCREEN['max_hit_rate_short']}%  |  "
          f"Min analogues: {SCREEN['min_n_obs']}  |  "
          f"Momentum/Fade zone: %tile ≥{SCREEN['momentum_ptile_min']}  |  "
          f"Reversal/Breakdown zone: %tile ≤{SCREEN['reversal_ptile_max']}")
    print(sep)

    display_cols = ["Asset Class", "Instrument", "Avg %tile",
                    "Slope (2w−12w)", "Momentum",
                    "Avg Hit Rate", "Avg Fwd 4w (%)", "Min N obs", "Conviction"]

    sections = [
        # signal      direction  label                        description
        ("MOMENTUM",  "LONG",  "TOP MOMENTUM CHASE",
         "high %tile + hit>54% → LONG, chase continuation"),
        ("REVERSAL",  "LONG",  "TOP REVERSAL CANDIDATES",
         "low %tile  + hit>54% → LONG, buy the dip (history tilts to bounce)"),
        ("FADE",      "SHORT", "TOP FADE CANDIDATES",
         "high %tile + hit<46% → SHORT, fade exhausted rally (history tilts lower)"),
        ("BREAKDOWN", "SHORT", "TOP BREAKDOWN CANDIDATES",
         "low %tile  + hit<46% → SHORT, don't catch knife (history tilts lower still)"),
    ]

    for signal, direction, label, desc in sections:
        subset = df[df["Signal"] == signal].copy()
        bar = "▲ LONG" if direction == "LONG" else "▼ SHORT"
        print(f"\n  ── {bar}  {label}  ({desc}) ──")
        if subset.empty:
            print("     (none qualify at current thresholds)")
        else:
            print(subset[display_cols].to_string(index=False))

    counts = {s: (df["Signal"] == s).sum() for s in
              ["MOMENTUM", "REVERSAL", "FADE", "BREAKDOWN", "NEUTRAL"]}
    print(f"\n  Summary: {len(df)} instruments  |  "
          f"▲ LONG: {counts['MOMENTUM']} MOMENTUM + {counts['REVERSAL']} REVERSAL  |  "
          f"▼ SHORT: {counts['FADE']} FADE + {counts['BREAKDOWN']} BREAKDOWN  |  "
          f"— {counts['NEUTRAL']} NEUTRAL")
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

    # Panel B shows all 4 signal types, grouped LONG then SHORT
    long_sigs  = df[df["Signal"].isin(["MOMENTUM", "REVERSAL"])].sort_values("Conviction")
    short_sigs = df[df["Signal"].isin(["FADE", "BREAKDOWN"])].sort_values("Conviction")
    signals = pd.concat([short_sigs, long_sigs], ignore_index=True)  # SHORT bottom, LONG top

    n_signals = len(signals)
    fig_height = max(10, 4 + n_signals * 0.32)
    fig = plt.figure(figsize=(22, fig_height))
    fig.patch.set_facecolor("#f0f0f0")

    date_str = datetime.today().strftime("%d %b %Y")
    n_long  = len(long_sigs)
    n_short = len(short_sigs)
    fig.suptitle(
        f"Weekly Trading Screen  |  {date_str}  |  Analogue band ±{PTILE_BAND}%tile  |  "
        f"LONG: %tile≥{SCREEN['momentum_ptile_min']} or ≤{SCREEN['reversal_ptile_max']} + hit>{SCREEN['min_hit_rate_long']}%  "
        f"SHORT: same %tile zones + hit<{SCREEN['max_hit_rate_short']}%  |  "
        f"▲{n_long} long  ▼{n_short} short",
        fontsize=11, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    # ── Panel A: Opportunity Scatter ─────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#fafafa")

    # Shaded quadrant zones — 4 active zones + 2 neutral bands
    rev_max    = SCREEN["reversal_ptile_max"]
    mom_min    = SCREEN["momentum_ptile_min"]
    hr_long    = SCREEN["min_hit_rate_long"]
    hr_short   = SCREEN["max_hit_rate_short"]

    # LONG zones (top band: hit_rate > hr_long)
    ax_a.axhspan(hr_long, 82, xmin=0,           xmax=rev_max / 100,
                 facecolor="#cce5ff", alpha=0.45, zorder=0)   # REVERSAL — blue tint
    ax_a.axhspan(hr_long, 82, xmin=mom_min / 100, xmax=1.0,
                 facecolor="#d4edda", alpha=0.45, zorder=0)   # MOMENTUM — green tint

    # SHORT zones (bottom band: hit_rate < hr_short)
    ax_a.axhspan(28, hr_short, xmin=0,           xmax=rev_max / 100,
                 facecolor="#f8d7da", alpha=0.50, zorder=0)   # BREAKDOWN — red tint
    ax_a.axhspan(28, hr_short, xmin=mom_min / 100, xmax=1.0,
                 facecolor="#ffe8c0", alpha=0.50, zorder=0)   # FADE — orange tint

    # Threshold lines
    ax_a.axvline(rev_max,  color="#444444", lw=1.0, ls="--", alpha=0.5)
    ax_a.axvline(mom_min,  color="#444444", lw=1.0, ls="--", alpha=0.5)
    ax_a.axhline(hr_long,  color="#1f77b4", lw=1.0, ls="--", alpha=0.6)
    ax_a.axhline(hr_short, color="#d62728", lw=1.0, ls="--", alpha=0.6)
    ax_a.axhline(50,       color="#888888", lw=0.6, ls=":",  alpha=0.4)  # coin-flip

    # Zone labels
    ax_a.text(rev_max / 2,        80, "REVERSAL\n▲ LONG",
              ha="center", va="top", fontsize=7.5, color="#1f77b4",
              fontweight="bold", alpha=0.85)
    ax_a.text((mom_min + 100) / 2, 80, "MOMENTUM\n▲ LONG",
              ha="center", va="top", fontsize=7.5, color="#2ca02c",
              fontweight="bold", alpha=0.85)
    ax_a.text(rev_max / 2,        30, "BREAKDOWN\n▼ SHORT",
              ha="center", va="bottom", fontsize=7.5, color="#d62728",
              fontweight="bold", alpha=0.85)
    ax_a.text((mom_min + 100) / 2, 30, "FADE\n▼ SHORT",
              ha="center", va="bottom", fontsize=7.5, color="#ff7f0e",
              fontweight="bold", alpha=0.85)

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
    ax_a.set_ylim(28, 82)
    ax_a.set_xlabel("Avg Return Percentile  (avg of 2w/4w/8w/12w lookbacks)", fontsize=9)
    ax_a.set_ylabel("Avg Hit Rate  (% of analogues with +ve fwd return)", fontsize=9)
    ax_a.set_title("Opportunity Map — 4-quadrant signal framework", fontsize=10, pad=8)
    ax_a.grid(True, alpha=0.3, lw=0.5)

    # ── Panel B: Conviction Ranking Bars ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#fafafa")

    if n_signals == 0:
        ax_b.text(0.5, 0.5, "No signals qualify at current thresholds",
                  ha="center", va="center", fontsize=11, transform=ax_b.transAxes)
    else:
        y_pos  = range(n_signals)
        colors = [SIG_COLORS.get(s, "#999999") for s in signals["Signal"]]

        bars = ax_b.barh(list(y_pos), signals["Conviction"], color=colors,
                         alpha=0.82, edgecolor="white", linewidth=0.5)

        # Instrument labels (left side)
        ax_b.set_yticks(list(y_pos))
        ax_b.set_yticklabels(
            [f"{row['Instrument']}  [{row['Asset Class']}]"
             for _, row in signals.iterrows()],
            fontsize=7.5,
        )

        # Annotate bars: %tile, hit rate, expected return, slope
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

        # Group index sets
        sig_idxs = {s: [i for i, sig in enumerate(signals["Signal"]) if sig == s]
                    for s in ["MOMENTUM", "REVERSAL", "FADE", "BREAKDOWN"]}

        # Divider between LONG (top) and SHORT (bottom) sections
        long_all  = sig_idxs["MOMENTUM"] + sig_idxs["REVERSAL"]
        short_all = sig_idxs["FADE"]     + sig_idxs["BREAKDOWN"]
        if long_all and short_all:
            boundary = (max(short_all) + min(long_all)) / 2
            ax_b.axhline(boundary, color="#444444", lw=2.0, ls="-", alpha=0.4)

        # Group labels
        for sig, label in [("MOMENTUM",  "▲ MOMENTUM\nCHASE"),
                            ("REVERSAL",  "▲ REVERSAL\nCANDIDATE"),
                            ("FADE",      "▼ FADE\nSHORT"),
                            ("BREAKDOWN", "▼ BREAKDOWN\nSHORT")]:
            idxs = sig_idxs[sig]
            if idxs:
                ax_b.text(-0.5, np.mean(idxs), label,
                          ha="right", va="center", fontsize=7.5,
                          color=SIG_COLORS[sig], fontweight="bold",
                          transform=ax_b.get_yaxis_transform())

        max_conv = signals["Conviction"].max() if n_signals else 1
        ax_b.set_xlim(0, max_conv * 1.9)
        ax_b.set_xlabel("Conviction Score  (edge × return magnitude × sample depth × timing)", fontsize=9)

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

    styles = {
        "MOMENTUM":  ("#1a5c1a", "#e8f5e9"),   # dark green header, light green rows
        "REVERSAL":  ("#1a3a6c", "#e8f0fe"),   # dark blue header, light blue rows
        "FADE":      ("#7a3c00", "#fff3e0"),   # dark orange header, light orange rows
        "BREAKDOWN": ("#7a1a1a", "#fdecea"),   # dark red header, light red rows
    }
    header_bg, row_alt = styles.get(signal, ("#333333", "#f5f5f5"))
    row_colors = (row_alt, "#ffffff")

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
    """Build the full HTML email body with embedded chart + 4 ranked tables."""
    date_str = datetime.today().strftime("%A %d %b %Y")

    momentum  = df[df["Signal"] == "MOMENTUM"].copy()
    reversal  = df[df["Signal"] == "REVERSAL"].copy()
    fade      = df[df["Signal"] == "FADE"].copy()
    breakdown = df[df["Signal"] == "BREAKDOWN"].copy()

    total = len(df)

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
        body  {{ font-family: Arial, sans-serif; color: #222; background: #fff;
                 max-width: 1050px; margin: auto; padding: 20px; }}
        h1    {{ font-size: 18px; color: #1a1a1a; }}
        h2    {{ font-size: 13px; margin-top: 22px; padding: 6px 10px;
                 border-radius: 3px; color: #fff; }}
        .h-momentum  {{ background: #1a5c1a; }}
        .h-reversal  {{ background: #1a3a6c; }}
        .h-fade      {{ background: #7a3c00; }}
        .h-breakdown {{ background: #7a1a1a; }}
        .summary  {{ background: #f4f4f4; padding: 10px 14px;
                     border-left: 4px solid #555; margin-bottom: 14px;
                     font-size: 12px; line-height: 1.7; }}
        .section-label {{ font-size: 11px; color: #555; margin-bottom: 4px; }}
        .footer   {{ font-size: 10px; color: #777; margin-top: 30px;
                     border-top: 1px solid #ddd; padding-top: 10px; }}
      </style>
    </head>
    <body>
      <h1>Weekly Signal Screen &mdash; {date_str}</h1>

      <div class="summary">
        <b>Universe:</b> {total} instruments &nbsp;|&nbsp;
        <b>Analogue band:</b> &plusmn;{PTILE_BAND} %tile<br/>
        <b>&#9650; LONG signals:</b>
          {len(momentum)} Momentum + {len(reversal)} Reversal
          &nbsp;(hit-rate &gt;{SCREEN['min_hit_rate_long']}%,
          %tile &ge;{SCREEN['momentum_ptile_min']} or &le;{SCREEN['reversal_ptile_max']})
        &nbsp;|&nbsp;
        <b>&#9660; SHORT signals:</b>
          {len(fade)} Fade + {len(breakdown)} Breakdown
          &nbsp;(hit-rate &lt;{SCREEN['max_hit_rate_short']}%,
          same %tile zones)<br/>
        <b>Min analogues:</b> {SCREEN['min_n_obs']}
        &nbsp;|&nbsp;
        <b>Conviction:</b> edge &times; return &times; depth &times; timing
      </div>

      {chart_html}

      <p class="section-label">&#9650; LONG &mdash; chase or buy the dip</p>

      <h2 class="h-momentum">
        Momentum Chase &mdash;
        high %tile + hit&gt;{SCREEN['min_hit_rate_long']}% &rarr; analogues tilt bullish, chase continuation
      </h2>
      {_html_table(momentum, "MOMENTUM")}

      <h2 class="h-reversal">
        Reversal Candidates &mdash;
        low %tile + hit&gt;{SCREEN['min_hit_rate_long']}% &rarr; history tilts to bounce, buy the dip
      </h2>
      {_html_table(reversal, "REVERSAL")}

      <p class="section-label" style="margin-top:28px;">&#9660; SHORT &mdash; fade strength or avoid the falling knife</p>

      <h2 class="h-fade">
        Fade Candidates &mdash;
        high %tile + hit&lt;{SCREEN['max_hit_rate_short']}% &rarr; analogues tilt bearish, fade the exhausted rally
      </h2>
      {_html_table(fade, "FADE")}

      <h2 class="h-breakdown">
        Breakdown Candidates &mdash;
        low %tile + hit&lt;{SCREEN['max_hit_rate_short']}% &rarr; history keeps falling, don&apos;t catch the knife
      </h2>
      {_html_table(breakdown, "BREAKDOWN")}

      <div class="footer">
        Source: Yahoo Finance &nbsp;|&nbsp; History: 30 years &nbsp;|&nbsp;
        Percentile: expanding window (min 252 days, no look-ahead bias) &nbsp;|&nbsp;
        Analogues: historical dates within &plusmn;{PTILE_BAND}pts of today&apos;s %tile &nbsp;|&nbsp;
        Hit rate: averaged across all (lookback &times; forward) combinations &nbsp;|&nbsp;
        Conviction symmetric: LONG edge = hit&minus;50, SHORT edge = 50&minus;hit &nbsp;|&nbsp;
        &uarr;&uarr; building = slope &gt;+{SCREEN['early_slope_thresh']}pts
        (LONG: early momentum; FADE: premature short) &nbsp;|&nbsp;
        &darr;&darr; fading = slope &lt;&minus;{SCREEN['early_slope_thresh']}pts
        (REVERSAL: ripe; FADE/BREAKDOWN: confirmed direction)
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
