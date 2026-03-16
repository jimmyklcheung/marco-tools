"""
Return %tile Calculator  (v2)
==============================
Pulls daily prices for global equity indices, FX pairs, and commodity
benchmarks from Yahoo Finance (free, no API key required), then:

  1. Calculates rolling 2w / 4w / 8w / 12w returns.
  2. Converts each return to its historical percentile rank (expanding window,
     no look-ahead bias; min 252 days of history before ranking begins).
  3. Finds all past dates where the percentile fell within ±PTILE_BAND of
     today's reading ("historical analogues" or "precedents").
  4. For each set of precedents, computes:
       - Hit rate  : % of analogues followed by a positive forward return
       - Median fwd return : median of those forward returns
  5. Prints the precedent list (date, %tile at that date, fwd returns) to
     console and saves it to a CSV.
  6. Generates per-asset-class heatmap charts summarising current %tile,
     hit rates and median forward returns.

Universe:
  Equities  : S&P 500, Russell 2000, EuroStoxx 50, DAX, CAC 40, FTSE MIB,
              IBEX 35, FTSE 100, Nikkei 225, Hang Seng, CSI 300, TAIEX,
              KOSPI, Nifty 50, Straits Times, TSX Composite
  FX        : EUR, JPY, CHF, GBP, CAD, AUD, NZD, CNH, TWD, KRW, SGD, INR,
              SEK, PLN, CZK, MXN, BRL  (all vs USD)
  Commodities: Gold (proxy: Gold Futures GC=F),
               Energy (proxy: WTI Crude Futures CL=F),
               Industrial Metals (proxy: Copper Futures HG=F)

Dependencies:
  pip install yfinance pandas numpy matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

# Analogue matching: find history dates within ±PTILE_BAND of current %tile
PTILE_BAND    = 3      # tightened to ±3 percentile points

# Rolling look-back windows (in trading days; ≈5 per calendar week)
LOOKBACK_WINDOWS = {
    "2w":  10,
    "4w":  20,
    "8w":  40,
    "12w": 60,
}

# Forward horizons for hit-rate and median-return analysis
FORWARD_WINDOWS = {
    "1w fwd":  5,
    "2w fwd": 10,
    "4w fwd": 20,
}

# Minimum trading-day history before the expanding-window rank is valid
MIN_HISTORY_DAYS = 252   # roughly 1 year

HISTORY_YEARS = 30
START_DATE = (datetime.today() - timedelta(days=365 * HISTORY_YEARS)).strftime("%Y-%m-%d")
END_DATE   = datetime.today().strftime("%Y-%m-%d")
TODAY_STR  = datetime.today().strftime("%Y%m%d")

# ── 2. UNIVERSE DEFINITIONS ───────────────────────────────────────────────────
# Each asset class maps  human-readable label → Yahoo Finance ticker.

ASSET_CLASSES = {

    # ── Equity Indices ────────────────────────────────────────────────────────
    "Equities": {
        "S&P 500":      "^GSPC",
        "Russell 2000": "^RUT",       # US small-cap
        "EuroStoxx 50": "^STOXX50E",
        "DAX":          "^GDAXI",     # Germany
        "CAC 40":       "^FCHI",      # France
        "FTSE MIB":     "FTSEMIB.MI", # Italy
        "IBEX 35":      "^IBEX",      # Spain
        "FTSE 100":     "^FTSE",      # UK
        "Nikkei 225":   "^N225",      # Japan (TOPIX not available free on YF)
        "Hang Seng":    "^HSI",       # Hong Kong
        "CSI 300":      "000300.SS",  # China A-shares
        "TAIEX":        "^TWII",      # Taiwan
        "KOSPI":        "^KS11",      # Korea (composite; KOSPI200 not on YF)
        "Nifty 50":     "^NSEI",      # India
        "Str Times":    "^STI",       # Singapore
        "TSX Comp":     "^GSPTSE",    # Canada (TSX 60 not available free)
    },

    # ── FX Pairs (all quoted as USD per 1 unit of foreign ccy) ───────────────
    # Positive return = foreign currency STRENGTHENED vs USD
    "FX": {
        "EUR/USD":  "EURUSD=X",
        "JPY/USD":  "JPYUSD=X",
        "CHF/USD":  "CHFUSD=X",
        "GBP/USD":  "GBPUSD=X",
        "CAD/USD":  "CADUSD=X",
        "AUD/USD":  "AUDUSD=X",
        "NZD/USD":  "NZDUSD=X",
        "CNH/USD":  "CNHUSD=X",  # offshore RMB
        "TWD/USD":  "TWDUSD=X",
        "KRW/USD":  "KRWUSD=X",
        "SGD/USD":  "SGDUSD=X",
        "INR/USD":  "INRUSD=X",
        "SEK/USD":  "SEKUSD=X",
        "PLN/USD":  "PLNUSD=X",
        "CZK/USD":  "CZKUSD=X",
        "MXN/USD":  "MXNUSD=X",
        "BRL/USD":  "BRLUSD=X",
    },

    # ── Commodity Benchmarks ──────────────────────────────────────────────────
    # Free front-month futures with the longest available history on YF.
    "Commodities": {
        "Gold":       "GC=F",   # COMEX Gold Futures  → Gold Price Index proxy
        "WTI Crude":  "CL=F",   # NYMEX WTI Crude     → Energy Price Index proxy
        "Copper":     "HG=F",   # COMEX Copper Futures → Industrial Metal Index proxy
    },
}


# ── 3. DATA DOWNLOAD ──────────────────────────────────────────────────────────

def download_asset_class(label_ticker: dict, asset_class: str,
                         start: str, end: str) -> pd.DataFrame:
    """
    Bulk-download adjusted closing prices for one asset class from Yahoo Finance.
    Columns are renamed to human-readable labels.
    Any ticker with zero valid rows is silently dropped and reported.
    """
    tickers = list(label_ticker.values())
    labels  = list(label_ticker.keys())

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjust for splits & dividends
        progress=False,
    )

    # yfinance returns a MultiIndex for >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers].copy()
    else:
        # Single ticker: raw has flat columns
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices.columns = labels
    prices.dropna(how="all", inplace=True)

    # Validate and report each instrument
    drop_cols = []
    for col in prices.columns:
        valid = prices[col].dropna()
        if len(valid) < MIN_HISTORY_DAYS * 2:
            print(f"    [{asset_class}] {col:15s}: ⚠ only {len(valid)} rows — skipping")
            drop_cols.append(col)
        else:
            print(f"    [{asset_class}] {col:15s}: {len(valid):5d} days  "
                  f"({valid.index[0].date()} → {valid.index[-1].date()})")

    if drop_cols:
        prices.drop(columns=drop_cols, inplace=True)

    return prices


def download_all(asset_classes: dict, start: str, end: str) -> dict:
    """
    Download prices for every asset class.
    Returns dict  { asset_class_label: DataFrame_of_prices }.
    """
    print(f"\nDownloading price data  ({start} → {end}) …\n")
    all_prices = {}
    for ac_label, label_ticker in asset_classes.items():
        all_prices[ac_label] = download_asset_class(label_ticker, ac_label, start, end)
    return all_prices


# ── 4. RETURN & PERCENTILE CALCULATIONS ──────────────────────────────────────

def calc_rolling_returns(prices: pd.DataFrame, windows: dict) -> dict:
    """
    Compute simple (arithmetic) total return over each look-back window:
        r_t  =  P_t / P_{t-n}  −  1
    Returns dict { window_label: DataFrame_of_returns }.
    """
    return {label: prices.pct_change(periods=n) for label, n in windows.items()}


def calc_percentile_rank(returns: dict, min_periods: int = MIN_HISTORY_DAYS) -> dict:
    """
    Convert each return to its historical percentile rank using an
    *expanding window* (no look-ahead bias):

        rank_t  =  (# past observations ≤ r_t) / (# past observations)  × 100

    min_periods ensures we don't rank until there is enough history.
    """
    return {
        label: ret_df.expanding(min_periods=min_periods).rank(pct=True) * 100
        for label, ret_df in returns.items()
    }


def calc_forward_returns(prices: pd.DataFrame, fwd_windows: dict) -> dict:
    """
    Pre-compute future returns for every forward horizon:
        fwd_r_t  =  P_{t+n} / P_t  −  1

    Last n rows will be NaN (no future data yet).
    """
    return {
        label: prices.shift(-n).div(prices) - 1
        for label, n in fwd_windows.items()
    }


# ── 5. HISTORICAL ANALOGUE ANALYSIS ──────────────────────────────────────────

def analyse_analogues(
    ptile_dict:   dict,
    fwd_ret_dict: dict,
    band:         float = PTILE_BAND,
) -> dict:
    """
    For each (instrument, lookback) pair:

      a) Record the CURRENT (most-recent) percentile rank.
      b) Find every past date where the rank fell within
             [current_ptile − band ,  current_ptile + band]
         These are the "historical analogues" (precedents).
      c) Collect the forward returns at those dates and compute:
             hit_rate    = % of analogues with positive forward return
             med_return  = median of those forward returns

    Data structure returned:
        results[instrument][lookback] = {
            "current_ptile" : float,
            "analogue_dates": pd.DatetimeIndex,        # precedent dates
            "analogue_ptiles": pd.Series,              # %tile on each date
            "fwd": {
                fwd_label: {
                    "hit_rate"   : float (0–100),
                    "med_return" : float (in %, already ×100),
                    "fwd_returns": pd.Series (indexed by analogue date),
                    "n_obs"      : int,
                }
            }
        }
    """
    results = {}
    sample_df = list(ptile_dict.values())[0]
    instruments = sample_df.columns.tolist()

    for instr in instruments:
        results[instr] = {}

        for lb_label, ptile_df in ptile_dict.items():
            # Drop NaNs (early history before min_periods is met)
            series = ptile_df[instr].dropna()
            if series.empty:
                continue

            current_ptile = float(series.iloc[-1])

            # Clamp band to [0, 100]
            lo = max(0.0,   current_ptile - band)
            hi = min(100.0, current_ptile + band)

            # Historical analogue dates (exclude today itself)
            hist = series.iloc[:-1]
            mask = (hist >= lo) & (hist <= hi)
            analogue_dates  = hist[mask].index
            analogue_ptiles = hist[mask]

            # Collect forward returns at each analogue date
            fwd_stats = {}
            for fwd_label, fwd_df in fwd_ret_dict.items():
                fwd_at_analogues = fwd_df[instr].reindex(analogue_dates).dropna()

                if len(fwd_at_analogues) == 0:
                    fwd_stats[fwd_label] = {
                        "hit_rate":    np.nan,
                        "med_return":  np.nan,
                        "fwd_returns": pd.Series(dtype=float),
                        "n_obs":       0,
                    }
                else:
                    fwd_stats[fwd_label] = {
                        "hit_rate":    round((fwd_at_analogues > 0).mean() * 100, 1),
                        "med_return":  round(fwd_at_analogues.median() * 100, 2),
                        "fwd_returns": fwd_at_analogues * 100,  # store in %
                        "n_obs":       len(fwd_at_analogues),
                    }

            results[instr][lb_label] = {
                "current_ptile":  round(current_ptile, 1),
                "analogue_dates": analogue_dates,
                "analogue_ptiles": analogue_ptiles,
                "fwd":            fwd_stats,
            }

    return results


# ── 6. EXPORT PRECEDENTS ─────────────────────────────────────────────────────

def export_precedents_csv(results: dict, path: str):
    """
    Save every analogue observation to a wide CSV:
      Columns: Asset, Lookback, Date, %tile_at_date,
               1w_fwd_ret(%), 2w_fwd_ret(%), 4w_fwd_ret(%)

    Rows are sorted by Asset → Lookback → Date (most recent first).
    """
    rows = []
    fwd_labels = list(FORWARD_WINDOWS.keys())

    for instr, lb_dict in results.items():
        for lb_label, data in lb_dict.items():
            dates   = data["analogue_dates"]
            ptiles  = data["analogue_ptiles"]
            fwd     = data["fwd"]

            # Align all forward return series on the same dates
            fwd_series = {
                fl: fwd[fl]["fwd_returns"].reindex(dates)
                for fl in fwd_labels
                if fl in fwd and len(fwd[fl]["fwd_returns"]) > 0
            }

            for date in dates:
                row = {
                    "Asset":        instr,
                    "Lookback":     lb_label,
                    "Date":         date.date(),
                    "%tile at date": round(float(ptiles[date]), 1),
                }
                for fl in fwd_labels:
                    col = fl.replace(" ", "_") + "(%)"
                    row[col] = round(fwd_series[fl][date], 2) if fl in fwd_series else np.nan
                rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(["Asset", "Lookback", "Date"], ascending=[True, True, False], inplace=True)
    df.to_csv(path, index=False)
    print(f"Precedents CSV saved → {path}  ({len(df):,} rows)")
    return df


def print_precedents(results: dict, n_recent: int = 10):
    """
    Print the most-recent N analogue dates for every (instrument, lookback)
    combination, showing the %tile at that date and subsequent forward returns.
    """
    fwd_labels = list(FORWARD_WINDOWS.keys())
    sep = "─" * 100

    print("\n" + "=" * 100)
    print("  HISTORICAL PRECEDENTS  (dates with similar return percentile, "
          f"within ±{PTILE_BAND} pts of today's reading)")
    print("=" * 100)

    for instr, lb_dict in results.items():
        print(f"\n{'━'*100}")
        print(f"  {instr}")
        print(f"{'━'*100}")

        for lb_label, data in lb_dict.items():
            dates   = data["analogue_dates"]
            ptiles  = data["analogue_ptiles"]
            curr_p  = data["current_ptile"]
            fwd     = data["fwd"]

            n_total = len(dates)
            # Take most recent N
            recent_dates = dates[-n_recent:][::-1]  # newest first

            print(f"\n  Lookback: {lb_label}  |  Current %tile: {curr_p:.1f}  |  "
                  f"Band: [{max(0, curr_p - PTILE_BAND):.1f}, "
                  f"{min(100, curr_p + PTILE_BAND):.1f}]  |  "
                  f"Total analogues: {n_total}")
            print(f"  {sep}")

            if n_total == 0:
                print("    (no analogues found — band may be too narrow for this history length)")
                continue

            # Header
            fwd_header = "  ".join(f"{fl:>10}" for fl in fwd_labels)
            print(f"  {'Date':12s}  {'%tile':>6}  {fwd_header}")
            print(f"  {'-'*12}  {'------':>6}  {'  '.join(['----------'] * len(fwd_labels))}")

            for date in recent_dates:
                ptile_val = ptiles[date]
                fwd_strs = []
                for fl in fwd_labels:
                    if fl in fwd and date in fwd[fl]["fwd_returns"].index:
                        v = fwd[fl]["fwd_returns"][date]
                        fwd_strs.append(f"{v:+10.2f}%")
                    else:
                        fwd_strs.append(f"{'n/a':>10}")
                print(f"  {str(date.date()):12s}  {ptile_val:>6.1f}  {'  '.join(fwd_strs)}")

            if n_total > n_recent:
                print(f"    … and {n_total - n_recent} earlier analogue(s) "
                      f"(see precedents CSV for full list)")

    print("\n" + "=" * 100)


# ── 7. FLATTEN RESULTS TO DATAFRAME ──────────────────────────────────────────

def results_to_dataframe(results: dict) -> pd.DataFrame:
    """
    Flatten the nested results dict into a tidy DataFrame suitable for charting.
    One row per (instrument, lookback, forward_horizon).
    """
    rows = []
    for instr, lb_dict in results.items():
        for lb_label, data in lb_dict.items():
            for fwd_label, stats in data["fwd"].items():
                rows.append({
                    "Index":          instr,
                    "Lookback":       lb_label,
                    "Forward":        fwd_label,
                    "Current %tile":  data["current_ptile"],
                    "Hit Rate (%)":   stats["hit_rate"],
                    "Median Fwd (%)": stats["med_return"],
                    "N obs":          stats["n_obs"],
                })
    return pd.DataFrame(rows)


# ── 8. CHARTING ───────────────────────────────────────────────────────────────

# Shared colour maps used across all charts
CMAP_PTILE = LinearSegmentedColormap.from_list(
    "ptile_bwr", ["#1a6faf", "#d0e4f2", "#ffffff", "#f5c0b0", "#b2182b"]
)  # blue (low %tile) → white (50th) → red (high %tile)

CMAP_HIT = LinearSegmentedColormap.from_list(
    "hit_gwr", ["#b2182b", "#f5c0b0", "#ffffff", "#c7e9c0", "#1a6f1a"]
)  # red (<50% hit rate) → white (50%) → green (>50%)

CMAP_RET = LinearSegmentedColormap.from_list(
    "ret_bwg", ["#b2182b", "#f5c0b0", "#ffffff", "#c7e9c0", "#1a6f1a"]
)  # red (negative return) → white (zero) → green (positive)


def make_asset_class_chart(
    df:          pd.DataFrame,
    asset_class: str,
    save_path:   str,
):
    """
    Generate a 3-panel heatmap chart for one asset class:

      Panel A — Current return percentile
                Rows = instruments, Cols = look-back windows

      Panel B — Hit rate (% positive fwd return)
                One sub-panel per forward horizon (1w / 2w / 4w)

      Panel C — Median forward return (%)
                One sub-panel per forward horizon

    Figure height scales automatically with the number of instruments.
    """
    instruments = df["Index"].unique().tolist()
    lookbacks   = ["2w", "4w", "8w", "12w"]
    forwards    = ["1w fwd", "2w fwd", "4w fwd"]

    n_instr = len(instruments)
    n_lb    = len(lookbacks)
    n_fwd   = len(forwards)

    # Adaptive figure height: ~0.45 in per instrument row + fixed overhead
    row_height   = 0.45
    panel_a_h    = max(1.5, n_instr * row_height)
    panel_b_h    = max(1.5, n_instr * row_height)
    panel_c_h    = max(1.5, n_instr * row_height)
    total_height = panel_a_h + panel_b_h + panel_c_h + 5.0  # +5 for margins/titles

    fig = plt.figure(figsize=(22, total_height))
    fig.patch.set_facecolor("#f8f8f8")

    title_date = datetime.today().strftime("%d %b %Y")
    fig.suptitle(
        f"{asset_class} — Return Percentile & Forward-Return Analysis  "
        f"|  {title_date}  |  Analogue band ±{PTILE_BAND}%tile",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # Height ratios proportional to number of rows in each panel
    gs_outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[panel_a_h, panel_b_h, panel_c_h],
        hspace=0.55, top=0.975, bottom=0.03,
        left=0.10, right=0.97,
    )

    # Helper: build a (n_instr × n_lb) matrix for a given metric
    def build_matrix(metric: str, fwd: str = None):
        mat = np.full((n_instr, n_lb), np.nan)
        for i, instr in enumerate(instruments):
            for j, lb in enumerate(lookbacks):
                q = df[(df["Index"] == instr) & (df["Lookback"] == lb)]
                if fwd:
                    q = q[q["Forward"] == fwd]
                if not q.empty:
                    mat[i, j] = q[metric].iloc[0]
        return mat

    # Helper: annotate heatmap cells with value text
    def annotate(ax, mat, fmt="{:.0f}", threshold=None, abs_max=None):
        for i in range(n_instr):
            for j in range(n_lb):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                # choose white text for extreme values, black otherwise
                if threshold is not None:
                    txt_col = "white" if (v < threshold[0] or v > threshold[1]) else "black"
                elif abs_max is not None:
                    txt_col = "white" if abs(v) > 0.65 * abs_max else "black"
                else:
                    txt_col = "black"
                fsize = max(6, min(9, 100 // n_instr))
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=fsize, color=txt_col)

    # ── Panel A: Current Percentile ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs_outer[0])
    ax_a.set_title(
        "Current Return Percentile  (0 = historically lowest, 100 = historically highest)",
        fontsize=10, pad=6,
    )
    mat_a = build_matrix("Current %tile")
    im_a  = ax_a.imshow(mat_a, cmap=CMAP_PTILE, vmin=0, vmax=100, aspect="auto")
    ax_a.set_xticks(range(n_lb)); ax_a.set_xticklabels(lookbacks, fontsize=9)
    ax_a.set_yticks(range(n_instr)); ax_a.set_yticklabels(instruments, fontsize=8)
    ax_a.set_xlabel("Return look-back window", fontsize=9)
    annotate(ax_a, mat_a, fmt="{:.0f}", threshold=(20, 80))
    cb = plt.colorbar(im_a, ax=ax_a, fraction=0.015, pad=0.01)
    cb.set_label("Percentile", fontsize=8)

    # ── Panel B: Hit Rate ────────────────────────────────────────────────────
    gs_b     = gridspec.GridSpecFromSubplotSpec(1, n_fwd, subplot_spec=gs_outer[1], wspace=0.30)
    ax_b_arr = [fig.add_subplot(gs_b[0, k]) for k in range(n_fwd)]

    for k, fwd in enumerate(forwards):
        ax  = ax_b_arr[k]
        ax.set_title(f"Hit Rate (%)  —  {fwd}", fontsize=9, pad=5)
        mat = build_matrix("Hit Rate (%)", fwd=fwd)
        im  = ax.imshow(mat, cmap=CMAP_HIT, vmin=30, vmax=70, aspect="auto")
        ax.set_xticks(range(n_lb)); ax.set_xticklabels(lookbacks, fontsize=8)
        ax.set_yticks(range(n_instr))
        ax.set_yticklabels(instruments if k == 0 else [""] * n_instr, fontsize=8)
        if k == 0:
            ax.set_ylabel("Instrument", fontsize=8)
        annotate(ax, mat, fmt="{:.0f}%", threshold=(38, 62))
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("%", fontsize=7)

    ax_b_arr[1].set_xlabel(
        f"Positive forward-return frequency  (analogues within ±{PTILE_BAND}%tile of today)",
        fontsize=8,
    )

    # ── Panel C: Median Forward Return ───────────────────────────────────────
    gs_c     = gridspec.GridSpecFromSubplotSpec(1, n_fwd, subplot_spec=gs_outer[2], wspace=0.30)
    ax_c_arr = [fig.add_subplot(gs_c[0, k]) for k in range(n_fwd)]

    # Symmetric colour scale anchored to the 5th/95th pctile of the data
    all_med = df["Median Fwd (%)"].dropna()
    if len(all_med) > 0:
        abs_max = max(abs(all_med.quantile(0.05)), abs(all_med.quantile(0.95)), 0.3)
        abs_max = np.ceil(abs_max * 4) / 4   # round to nearest 0.25
    else:
        abs_max = 2.0

    for k, fwd in enumerate(forwards):
        ax  = ax_c_arr[k]
        ax.set_title(f"Median Fwd Return (%)  —  {fwd}", fontsize=9, pad=5)
        mat = build_matrix("Median Fwd (%)", fwd=fwd)
        im  = ax.imshow(mat, cmap=CMAP_RET, vmin=-abs_max, vmax=abs_max, aspect="auto")
        ax.set_xticks(range(n_lb)); ax.set_xticklabels(lookbacks, fontsize=8)
        ax.set_yticks(range(n_instr))
        ax.set_yticklabels(instruments if k == 0 else [""] * n_instr, fontsize=8)
        if k == 0:
            ax.set_ylabel("Instrument", fontsize=8)
        annotate(ax, mat, fmt="{:+.1f}%", abs_max=abs_max)
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("%", fontsize=7)

    ax_c_arr[1].set_xlabel(
        f"Median return over analogue forward windows  (±{PTILE_BAND}%tile band)",
        fontsize=8,
    )

    # Footnote
    fig.text(
        0.5, 0.005,
        f"Source: Yahoo Finance  |  History: {HISTORY_YEARS} years  |  "
        f"Percentile: expanding window, min {MIN_HISTORY_DAYS} days  |  "
        f"Analogue band: ±{PTILE_BAND} %tile pts",
        ha="center", fontsize=7, color="#666666",
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {save_path}")
    plt.close(fig)


# ── 9. PRINT SUMMARY TABLE ───────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, asset_class: str):
    """Print a compact pivot table summary for one asset class."""
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.float_format", "{:.1f}".format)
    pd.set_option("display.width", 130)

    print(f"\n{'═'*110}")
    print(f"  {asset_class.upper()} — RETURN PERCENTILE SUMMARY")
    print(f"{'═'*110}")

    for instr in df["Index"].unique():
        sub = df[df["Index"] == instr].copy()
        print(f"\n  {'─'*50}  {instr}  {'─'*50}")
        pivot = sub.pivot_table(
            index=["Lookback", "Current %tile"],
            columns="Forward",
            values=["Hit Rate (%)", "Median Fwd (%)", "N obs"],
            aggfunc="first",
        )
        print(pivot.to_string())


# ── 10. MAIN ──────────────────────────────────────────────────────────────────

def main():
    """
    Orchestrate the full pipeline for every asset class:
      1.  Download prices
      2.  Rolling returns → expanding-window percentile ranks
      3.  Forward returns
      4.  Find historical analogues (precedents) within ±PTILE_BAND
      5.  Print precedent lists to console
      6.  Export all precedents to CSV
      7.  Print summary tables
      8.  Generate per-asset-class heatmap charts
    """

    all_results = {}    # { asset_class: results_dict }
    all_df_rows = []    # accumulate rows for combined DataFrame

    for ac_label, label_ticker in ASSET_CLASSES.items():
        print(f"\n{'▶'*3}  {ac_label.upper()}")

        # Step 1 — Download prices
        prices = download_asset_class(label_ticker, ac_label, START_DATE, END_DATE)
        if prices.empty:
            print(f"  No data for {ac_label}, skipping.")
            continue

        # Step 2 — Rolling returns and percentile ranks
        rolling_ret = calc_rolling_returns(prices, LOOKBACK_WINDOWS)
        ptile_ranks = calc_percentile_rank(rolling_ret)

        # Step 3 — Forward returns
        fwd_returns = calc_forward_returns(prices, FORWARD_WINDOWS)

        # Step 4 — Historical analogue analysis (±PTILE_BAND)
        results = analyse_analogues(ptile_ranks, fwd_returns, band=PTILE_BAND)
        all_results[ac_label] = results

        # Flatten to DataFrame for charting/printing
        df_ac = results_to_dataframe(results)
        df_ac.insert(0, "Asset Class", ac_label)
        all_df_rows.append(df_ac)

        # Step 5 — Print precedent dates to console
        print_precedents(results, n_recent=10)

        # Step 7 — Summary table
        print_summary(df_ac, ac_label)

        # Step 8 — Chart
        chart_name = f"return_percentile_{ac_label.lower().replace(' ', '_')}.png"
        make_asset_class_chart(df_ac, ac_label, save_path=chart_name)

    # Step 6 — Export ALL precedents to one CSV (done after loop to merge all)
    print(f"\n{'─'*60}")
    csv_path = f"precedents_{TODAY_STR}.csv"
    prec_dfs = []
    for ac_label, results in all_results.items():
        tmp = export_precedents_csv(results, path=os.devnull)  # suppress per-class print
        tmp.insert(0, "Asset Class", ac_label)
        prec_dfs.append(tmp)

    if prec_dfs:
        combined_prec = pd.concat(prec_dfs, ignore_index=True)
        combined_prec.to_csv(csv_path, index=False)
        print(f"All precedents saved → {csv_path}  ({len(combined_prec):,} rows)")

    # Combine all asset class DataFrames
    if all_df_rows:
        combined_df = pd.concat(all_df_rows, ignore_index=True)
        print(f"\nDone.  {len(combined_df):,} result rows across "
              f"{combined_df['Index'].nunique()} instruments.")
        return combined_df

    return pd.DataFrame()


if __name__ == "__main__":
    df = main()
