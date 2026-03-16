"""
Return %tile Calculator
=======================
Pulls daily prices for major indices, calculates rolling return percentiles,
then maps those percentiles to historical forward returns and hit rates.

Indices covered:
  - S&P 500       (^GSPC)
  - EuroStoxx 50  (^STOXX50E)
  - FTSE 100      (^FTSE)
  - TOPIX         (^TOPX)
  - Hang Seng     (^HSI)
  - TSX 60        (^GSPTSE)   # TSX Composite used as free proxy for TSX 60

Dependencies:
  pip install yfinance pandas numpy matplotlib scipy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

# Ticker symbols on Yahoo Finance
INDICES = {
    "S&P 500":      "^GSPC",
    "EuroStoxx 50": "^STOXX50E",
    "FTSE 100":     "^FTSE",
    "Nikkei 225":   "^N225",    # TOPIX not on YF free tier; Nikkei 225 is best Japan proxy
    "Hang Seng":    "^HSI",
    "TSX Comp":     "^GSPTSE",   # best free proxy; TSX 60 (^TX60) not on YF
}

# Look-back windows for return calculation (in trading days; ~5 days/week)
LOOKBACK_WINDOWS = {
    "2w":  10,   # 2 calendar weeks ≈ 10 trading days
    "4w":  20,
    "8w":  40,
    "12w": 60,
}

# Forward windows for hit-rate / median-return analysis
FORWARD_WINDOWS = {
    "1w fwd":  5,
    "2w fwd": 10,
    "4w fwd": 20,
}

# "Similar percentile" band: ±PTILE_BAND around current percentile
PTILE_BAND = 10   # percentage points, e.g. current 30th → look at 20th–40th

# Data history: 30 years back from today
HISTORY_YEARS = 30
START_DATE = (datetime.today() - timedelta(days=365 * HISTORY_YEARS)).strftime("%Y-%m-%d")
END_DATE   = datetime.today().strftime("%Y-%m-%d")


# ── 2. DATA DOWNLOAD ─────────────────────────────────────────────────────────

def download_prices(indices: dict, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for all indices from Yahoo Finance.
    Returns a DataFrame with one column per index (named by human label).
    """
    print("Downloading price data …")
    tickers = list(indices.values())
    labels  = list(indices.keys())

    # yfinance bulk download
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # use split/dividend-adjusted prices
        progress=False,
    )

    # Extract Close prices; handle single vs multi-ticker output
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices.columns = labels   # rename to human-readable names
    prices.dropna(how="all", inplace=True)

    # Report coverage; warn and drop any column with no data at all
    for col in prices.columns:
        valid = prices[col].dropna()
        if len(valid) == 0:
            print(f"  {col:15s}: WARNING – no data returned; skipping.")
            prices.drop(columns=[col], inplace=True)
        else:
            print(f"  {col:15s}: {len(valid):5d} trading days  "
                  f"({valid.index[0].date()} → {valid.index[-1].date()})")

    return prices


# ── 3. RETURN CALCULATIONS ───────────────────────────────────────────────────

def calc_rolling_returns(prices: pd.DataFrame, windows: dict) -> dict:
    """
    For each look-back window compute simple (arithmetic) total return:
        r_t = (P_t / P_{t-n}) - 1
    Returns a dict keyed by window label, each value a DataFrame of returns.
    """
    returns = {}
    for label, n in windows.items():
        ret = prices.pct_change(periods=n)   # element-wise percentage change
        returns[label] = ret
    return returns


def calc_percentile_rank(returns: dict) -> dict:
    """
    For each date convert each index's rolling return to its historical
    percentile rank using all past observations (expanding window).

    percentile_rank(x) = fraction of past values ≤ x  (0–100 scale)
    """
    ptile = {}
    for label, ret_df in returns.items():
        rank_df = ret_df.expanding(min_periods=252).rank(pct=True) * 100
        ptile[label] = rank_df
    return ptile


# ── 4. FORWARD RETURNS ───────────────────────────────────────────────────────

def calc_forward_returns(prices: pd.DataFrame, fwd_windows: dict) -> dict:
    """
    For each forward horizon compute the future return:
        fwd_r_t = (P_{t+n} / P_t) - 1
    """
    fwd_returns = {}
    for label, n in fwd_windows.items():
        fwd = prices.shift(-n).div(prices) - 1   # shift prices n days into future
        fwd_returns[label] = fwd
    return fwd_returns


# ── 5. HISTORICAL ANALOGUE ANALYSIS ──────────────────────────────────────────

def analyse_analogues(
    ptile_dict: dict,
    fwd_ret_dict: dict,
    band: float = PTILE_BAND,
) -> dict:
    """
    For each (index, lookback window, forward window) triplet:

      1. Take the CURRENT percentile rank for that index & lookback.
      2. Find all historical dates where the percentile was within
         [current_ptile - band, current_ptile + band].
      3. Collect the forward returns on those dates.
      4. Compute:
           - hit_rate  : % of analogues with positive forward return
           - med_return: median forward return across analogues
           - n_obs     : number of analogue observations

    Returns a nested dict:
        results[index][lookback][forward] = {
            "current_ptile": float,
            "hit_rate":      float (0–100),
            "med_return":    float (fraction),
            "n_obs":         int,
        }
    """
    results = {}

    # We use the last lookback window's index for the current date reference
    sample_ptile_df = list(ptile_dict.values())[0]
    indices = sample_ptile_df.columns.tolist()

    for idx in indices:
        results[idx] = {}
        for lb_label, ptile_df in ptile_dict.items():
            results[idx][lb_label] = {}

            series = ptile_df[idx].dropna()
            if series.empty:
                continue

            # Current (most recent) percentile for this index × lookback
            current_ptile = series.iloc[-1]

            lo = max(0,   current_ptile - band)
            hi = min(100, current_ptile + band)

            # Boolean mask: dates where historical percentile was in the band
            # Exclude the last observation itself (it IS the current date)
            historical = series.iloc[:-1]
            in_band = historical[(historical >= lo) & (historical <= hi)].index

            for fwd_label, fwd_df in fwd_ret_dict.items():
                fwd_series = fwd_df[idx]

                # Gather valid forward returns at analogue dates
                analogue_fwd = fwd_series.reindex(in_band).dropna()

                if len(analogue_fwd) == 0:
                    results[idx][lb_label][fwd_label] = {
                        "current_ptile": current_ptile,
                        "hit_rate":      np.nan,
                        "med_return":    np.nan,
                        "n_obs":         0,
                    }
                    continue

                hit_rate   = (analogue_fwd > 0).mean() * 100
                med_return = analogue_fwd.median()

                results[idx][lb_label][fwd_label] = {
                    "current_ptile": round(current_ptile, 1),
                    "hit_rate":      round(hit_rate, 1),
                    "med_return":    round(med_return * 100, 2),  # convert to %
                    "n_obs":         len(analogue_fwd),
                }

    return results


# ── 6. FLATTEN RESULTS TO DATAFRAME ──────────────────────────────────────────

def results_to_dataframe(results: dict) -> pd.DataFrame:
    """
    Flatten the nested results dict into a tidy DataFrame for easy inspection
    and charting.
    """
    rows = []
    for idx, lb_dict in results.items():
        for lb_label, fwd_dict in lb_dict.items():
            for fwd_label, stats in fwd_dict.items():
                rows.append({
                    "Index":         idx,
                    "Lookback":      lb_label,
                    "Forward":       fwd_label,
                    "Current %tile": stats["current_ptile"],
                    "Hit Rate (%)":  stats["hit_rate"],
                    "Median Fwd (%)":stats["med_return"],
                    "N obs":         stats["n_obs"],
                })
    return pd.DataFrame(rows)


# ── 7. CHARTING ───────────────────────────────────────────────────────────────

def make_summary_chart(df: pd.DataFrame, save_path: str = "return_percentile.png"):
    """
    Create a multi-panel summary chart:

    Panel A (top)  : Current return-percentile heatmap
                     Rows = Indices, Columns = Lookback windows
                     Cell colour: blue (low) → white (50th) → red (high)

    Panel B (mid)  : Hit-rate heatmap for each (index × lookback × forward) cell
                     Three sub-grids side by side, one per forward horizon.

    Panel C (bot)  : Median forward return heatmap (same layout as B).
    """

    indices   = df["Index"].unique().tolist()
    lookbacks = ["2w", "4w", "8w", "12w"]
    forwards  = ["1w fwd", "2w fwd", "4w fwd"]

    n_idx = len(indices)
    n_lb  = len(lookbacks)
    n_fwd = len(forwards)

    # ── colour maps ──────────────────────────────────────────────────────────
    # Diverging blue-white-red for percentile (50 = white)
    cmap_ptile = LinearSegmentedColormap.from_list(
        "bwr", ["#1a6faf", "#d0e4f2", "#ffffff", "#f5c0b0", "#b2182b"]
    )
    # Green-white-red for hit rate (50 = white)
    cmap_hit = LinearSegmentedColormap.from_list(
        "gwr", ["#b2182b", "#f5c0b0", "#ffffff", "#c7e9c0", "#1a6f1a"]
    )
    # Blue-white-green for median return (0 = white)
    cmap_ret = LinearSegmentedColormap.from_list(
        "bwg", ["#b2182b", "#f5c0b0", "#ffffff", "#c7e9c0", "#1a6f1a"]
    )

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor("#f8f8f8")

    gs_outer = gridspec.GridSpec(
        3, 1,
        figure=fig,
        hspace=0.45,
        top=0.94, bottom=0.04,
        left=0.08, right=0.97,
    )

    title_date = datetime.today().strftime("%d %b %Y")
    fig.suptitle(
        f"Global Index — Return Percentile & Forward-Return Analysis  |  {title_date}",
        fontsize=15, fontweight="bold", y=0.975,
    )

    # ────────────────────────────────────────────────────────────────────────
    # PANEL A: Current percentile heatmap
    # ────────────────────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs_outer[0])
    ax_a.set_title(
        "Current Return Percentile  (0 = historically low, 100 = historically high)",
        fontsize=11, pad=8,
    )

    # Build matrix: rows = indices, cols = lookbacks
    ptile_matrix = np.full((n_idx, n_lb), np.nan)
    for i, idx in enumerate(indices):
        for j, lb in enumerate(lookbacks):
            sub = df[(df["Index"] == idx) & (df["Lookback"] == lb)]
            if not sub.empty:
                ptile_matrix[i, j] = sub["Current %tile"].iloc[0]

    im_a = ax_a.imshow(
        ptile_matrix,
        cmap=cmap_ptile, vmin=0, vmax=100,
        aspect="auto",
    )
    ax_a.set_xticks(range(n_lb));  ax_a.set_xticklabels(lookbacks, fontsize=10)
    ax_a.set_yticks(range(n_idx)); ax_a.set_yticklabels(indices,   fontsize=10)
    ax_a.set_xlabel("Return Look-back Window", fontsize=10)

    for i in range(n_idx):
        for j in range(n_lb):
            v = ptile_matrix[i, j]
            if not np.isnan(v):
                txt_col = "white" if (v < 20 or v > 80) else "black"
                ax_a.text(j, i, f"{v:.0f}", ha="center", va="center",
                          fontsize=11, fontweight="bold", color=txt_col)

    plt.colorbar(im_a, ax=ax_a, fraction=0.025, pad=0.02, label="Percentile")

    # ────────────────────────────────────────────────────────────────────────
    # PANEL B: Hit-rate heatmap  (one sub-axis per forward horizon)
    # ────────────────────────────────────────────────────────────────────────
    gs_b = gridspec.GridSpecFromSubplotSpec(
        1, n_fwd, subplot_spec=gs_outer[1], wspace=0.35,
    )
    ax_b_list = [fig.add_subplot(gs_b[0, k]) for k in range(n_fwd)]

    for k, fwd in enumerate(forwards):
        ax = ax_b_list[k]
        ax.set_title(f"Hit Rate (%)  —  {fwd}", fontsize=10, pad=6)

        mat = np.full((n_idx, n_lb), np.nan)
        for i, idx in enumerate(indices):
            for j, lb in enumerate(lookbacks):
                sub = df[(df["Index"] == idx) & (df["Lookback"] == lb)
                         & (df["Forward"] == fwd)]
                if not sub.empty:
                    mat[i, j] = sub["Hit Rate (%)"].iloc[0]

        im_b = ax.imshow(mat, cmap=cmap_hit, vmin=30, vmax=70, aspect="auto")
        ax.set_xticks(range(n_lb));  ax.set_xticklabels(lookbacks, fontsize=9)
        ax.set_yticks(range(n_idx)); ax.set_yticklabels(
            indices if k == 0 else [""] * n_idx, fontsize=9
        )
        if k == 0:
            ax.set_ylabel("Index", fontsize=9)

        for i in range(n_idx):
            for j in range(n_lb):
                v = mat[i, j]
                if not np.isnan(v):
                    txt_col = "white" if (v < 38 or v > 62) else "black"
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=9, color=txt_col)

        plt.colorbar(im_b, ax=ax, fraction=0.05, pad=0.03)

    # shared label above panel B
    ax_b_list[1].set_xlabel(
        "Positive forward return frequency based on historical analogues  "
        f"(±{PTILE_BAND}%tile band)",
        fontsize=9,
    )

    # ────────────────────────────────────────────────────────────────────────
    # PANEL C: Median forward return heatmap
    # ────────────────────────────────────────────────────────────────────────
    gs_c = gridspec.GridSpecFromSubplotSpec(
        1, n_fwd, subplot_spec=gs_outer[2], wspace=0.35,
    )
    ax_c_list = [fig.add_subplot(gs_c[0, k]) for k in range(n_fwd)]

    # Dynamic symmetric colour scale
    all_med = df["Median Fwd (%)"].dropna()
    abs_max  = max(abs(all_med.quantile(0.05)), abs(all_med.quantile(0.95)), 0.5)
    abs_max  = np.ceil(abs_max * 2) / 2   # round up to nearest 0.5

    for k, fwd in enumerate(forwards):
        ax = ax_c_list[k]
        ax.set_title(f"Median Fwd Return (%)  —  {fwd}", fontsize=10, pad=6)

        mat = np.full((n_idx, n_lb), np.nan)
        for i, idx in enumerate(indices):
            for j, lb in enumerate(lookbacks):
                sub = df[(df["Index"] == idx) & (df["Lookback"] == lb)
                         & (df["Forward"] == fwd)]
                if not sub.empty:
                    mat[i, j] = sub["Median Fwd (%)"].iloc[0]

        im_c = ax.imshow(
            mat, cmap=cmap_ret,
            vmin=-abs_max, vmax=abs_max,
            aspect="auto",
        )
        ax.set_xticks(range(n_lb));  ax.set_xticklabels(lookbacks, fontsize=9)
        ax.set_yticks(range(n_idx)); ax.set_yticklabels(
            indices if k == 0 else [""] * n_idx, fontsize=9
        )
        if k == 0:
            ax.set_ylabel("Index", fontsize=9)

        for i in range(n_idx):
            for j in range(n_lb):
                v = mat[i, j]
                if not np.isnan(v):
                    txt_col = "white" if abs(v) > 0.6 * abs_max else "black"
                    ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                            fontsize=9, color=txt_col)

        plt.colorbar(im_c, ax=ax, fraction=0.05, pad=0.03)

    ax_c_list[1].set_xlabel(
        "Median forward return of historical analogue dates  "
        f"(±{PTILE_BAND}%tile band)",
        fontsize=9,
    )

    # ── footnote ─────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        f"Source: Yahoo Finance  |  History: {HISTORY_YEARS} years  |  "
        f"Analogue band: current %tile ± {PTILE_BAND} pts  |  "
        "Percentile uses expanding window (min 252 days)",
        ha="center", fontsize=7.5, color="grey",
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {save_path}")
    plt.show()


# ── 8. PRINT SUMMARY TABLE ───────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    """Print a readable summary table to stdout."""
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", "{:.1f}".format)
    pd.set_option("display.width", 120)

    print("\n" + "=" * 90)
    print("  RETURN PERCENTILE SUMMARY")
    print("=" * 90)

    for idx in df["Index"].unique():
        sub = df[df["Index"] == idx].copy()
        print(f"\n{'─'*90}")
        print(f"  {idx}")
        print(f"{'─'*90}")
        pivot = sub.pivot_table(
            index=["Lookback", "Current %tile"],
            columns="Forward",
            values=["Hit Rate (%)", "Median Fwd (%)", "N obs"],
            aggfunc="first",
        )
        print(pivot.to_string())

    print("\n" + "=" * 90)


# ── 9. MAIN ───────────────────────────────────────────────────────────────────

def main():
    # Step 1: Download 30 years of daily closing prices
    prices = download_prices(INDICES, START_DATE, END_DATE)

    # Step 2: Calculate rolling returns for each look-back window
    rolling_returns = calc_rolling_returns(prices, LOOKBACK_WINDOWS)

    # Step 3: Convert returns to historical expanding-window percentile ranks
    ptile_ranks = calc_percentile_rank(rolling_returns)

    # Step 4: Calculate forward returns for each forward horizon
    fwd_returns = calc_forward_returns(prices, FORWARD_WINDOWS)

    # Step 5: Find historical analogues (similar %tile dates) and compute
    #         hit rates and median forward returns
    results = analyse_analogues(ptile_ranks, fwd_returns, band=PTILE_BAND)

    # Step 6: Flatten results to a tidy DataFrame
    df = results_to_dataframe(results)

    # Step 7: Print readable summary to console
    print_summary(df)

    # Step 8: Generate and save the summary chart
    make_summary_chart(df, save_path="return_percentile.png")

    return df


if __name__ == "__main__":
    df = main()
