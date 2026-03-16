"""
Return %tile Calculator — Analysis Engine  (v3)
================================================
Downloads daily prices for a broad multi-asset universe, computes rolling
return percentile ranks, identifies historical analogue dates (precedents),
and records forward-return statistics for each analogue set.

Imported by screener.py for the actionable trading screen.
Can also be run standalone to regenerate the per-asset-class heatmap charts.

Universe (7 asset classes):
  Equity Indices  : S&P 500, Russell 2000, EuroStoxx 50, DAX, CAC 40,
                    FTSE MIB, IBEX 35, FTSE 100, Nikkei 225, Hang Seng,
                    CSI 300, TAIEX, KOSPI, Nifty 50, Straits Times, TSX
  US Sectors      : XLF, BKX, XLE, XLK, XLV, XLI, XLB, XLU, XLY, XLP,
                    XLRE, XLC
  Global Sectors  : EU Banks, EU Energy, EU Healthcare, EU Industrials,
                    Japan Financials, China Tech (KWEB), EM (EEM)
  Equity Factors  : Nasdaq 100, MSCI EAFE Value, US Large Value/Growth,
                    US Momentum, US Quality, US Low Vol, US Small Value,
                    EM ETF, Dividend
  FX (vs USD)     : EUR JPY CHF GBP CAD AUD NZD CNH TWD KRW SGD INR
                    SEK PLN CZK MXN BRL
  Commodities     : Gold, Silver, Platinum, Palladium,
                    WTI Crude, Brent, Nat Gas, Heating Oil, RBOB Gasoline,
                    Copper, Aluminum,
                    Corn, Wheat, Soybeans, Coffee, Sugar, Cotton, Cocoa,
                    Live Cattle
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

PTILE_BAND       = 3       # ±3 %tile points for analogue matching
HISTORY_YEARS    = 30
MIN_HISTORY_DAYS = 252     # expanding window requires ≥1 yr before ranking

START_DATE = (datetime.today() - timedelta(days=365 * HISTORY_YEARS)).strftime("%Y-%m-%d")
END_DATE   = datetime.today().strftime("%Y-%m-%d")
TODAY_STR  = datetime.today().strftime("%Y%m%d")

LOOKBACK_WINDOWS = {"2w": 10, "4w": 20, "8w": 40, "12w": 60}
FORWARD_WINDOWS  = {"1w fwd": 5, "2w fwd": 10, "4w fwd": 20}

# ── 2. UNIVERSE ───────────────────────────────────────────────────────────────

ASSET_CLASSES = {

    # ── Broad Equity Indices ──────────────────────────────────────────────────
    "Equity Indices": {
        "S&P 500":      "^GSPC",
        "Russell 2000": "^RUT",
        "EuroStoxx 50": "^STOXX50E",
        "DAX":          "^GDAXI",
        "CAC 40":       "^FCHI",
        "FTSE MIB":     "FTSEMIB.MI",
        "IBEX 35":      "^IBEX",
        "FTSE 100":     "^FTSE",
        "Nikkei 225":   "^N225",
        "Hang Seng":    "^HSI",
        "CSI 300":      "000300.SS",
        "TAIEX":        "^TWII",
        "KOSPI":        "^KS11",
        "Nifty 50":     "^NSEI",
        "Str Times":    "^STI",
        "TSX Comp":     "^GSPTSE",
    },

    # ── US Sectors (SPDR ETFs; XLF/XLE/XLK/XLV/XLI/XLB/XLU/XLY/XLP ~1998) ─
    "US Sectors": {
        "US Financials":   "XLF",
        "US Banks (KBW)":  "^BKX",   # KBW Bank Index — longer history than XLF
        "US Energy":       "XLE",
        "US Technology":   "XLK",
        "US Healthcare":   "XLV",
        "US Industrials":  "XLI",
        "US Materials":    "XLB",
        "US Utilities":    "XLU",
        "US Cons Discret": "XLY",
        "US Cons Staples": "XLP",
        "US Real Estate":  "XLRE",   # 2015+
        "US Comm Svcs":    "XLC",    # 2018+
    },

    # ── Global Sectors ────────────────────────────────────────────────────────
    # Euro Stoxx ^SX* sub-indices are not freely available on Yahoo Finance.
    # Best free ETF proxies used instead.
    "Global Sectors": {
        "EU Banks":       "EUFN",    # iShares MSCI Europe Financials ETF (2007+)
        "EU Broad":       "EZU",     # iShares MSCI Eurozone ETF (2000+)
        "Japan Broad":    "EWJ",     # iShares MSCI Japan ETF (1996+); covers all sectors
        "Japan Banks":    "8306.T",  # MUFG — Japan's largest bank; best free Japan bank proxy
        "China Tech":     "KWEB",    # KraneShares CSI China Internet ETF (2013+)
        "EM Broad":       "EEM",     # iShares MSCI EM (2003+)
        "Asia ex-JP":     "AAXJ",    # iShares MSCI Asia ex Japan (2008+)
        "LatAm":          "ILF",     # iShares S&P Latin America 40 (2001+)
    },

    # ── Equity Factors / Smart Beta ───────────────────────────────────────────
    "Equity Factors": {
        "Nasdaq 100":      "^NDX",   # Tech-heavy; data back to 1985
        "MSCI EAFE Val":   "EFV",    # iShares MSCI EAFE Value (2005+)
        "US Large Value":  "IVE",    # iShares S&P 500 Value (2000+)
        "US Large Growth": "IVW",    # iShares S&P 500 Growth (2000+)
        "US Momentum":     "MTUM",   # iShares MSCI USA Momentum (2013+)
        "US Quality":      "QUAL",   # iShares MSCI USA Quality (2013+)
        "US Low Vol":      "USMV",   # iShares MSCI USA Min Vol (2011+)
        "US Small Value":  "IWN",    # iShares Russell 2000 Value (2000+)
        "Dividend":        "DVY",    # iShares Dividend ETF (2003+)
        "EAFE Broad":      "EFA",    # iShares MSCI EAFE (2001+)
    },

    # ── FX (all quoted as USD per 1 unit of foreign ccy) ─────────────────────
    # Positive return = foreign currency STRENGTHENED vs USD
    "FX": {
        "EUR/USD":  "EURUSD=X",
        "JPY/USD":  "JPYUSD=X",
        "CHF/USD":  "CHFUSD=X",
        "GBP/USD":  "GBPUSD=X",
        "CAD/USD":  "CADUSD=X",
        "AUD/USD":  "AUDUSD=X",
        "NZD/USD":  "NZDUSD=X",
        "CNH/USD":  "CNHUSD=X",
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

    # ── Commodities (front-month futures; best free long-history on YF) ───────
    "Commodities": {
        # Precious metals
        "Gold":          "GC=F",   # COMEX Gold      — Gold Price Index proxy
        "Silver":        "SI=F",   # COMEX Silver
        "Platinum":      "PL=F",   # NYMEX Platinum
        "Palladium":     "PA=F",   # NYMEX Palladium
        # Energy
        "WTI Crude":     "CL=F",   # NYMEX WTI       — Energy Price Index proxy
        "Brent Crude":   "BZ=F",   # ICE Brent
        "Nat Gas":       "NG=F",   # NYMEX Natural Gas
        "Heating Oil":   "HO=F",   # NYMEX Heating Oil
        "RBOB Gasoline": "RB=F",   # NYMEX RBOB Gasoline
        # Base / industrial metals
        "Copper":        "HG=F",   # COMEX Copper    — Ind. Metal Index proxy
        "Aluminum":      "ALI=F",  # CME Aluminum (gracefully skipped if absent)
        # Grains
        "Corn":          "ZC=F",   # CBOT Corn
        "Wheat":         "ZW=F",   # CBOT Wheat
        "Soybeans":      "ZS=F",   # CBOT Soybeans
        # Softs
        "Coffee":        "KC=F",   # ICE Coffee C
        "Sugar #11":     "SB=F",   # ICE Sugar #11
        "Cotton":        "CT=F",   # ICE Cotton
        "Cocoa":         "CC=F",   # ICE Cocoa
        # Livestock
        "Live Cattle":   "LE=F",   # CME Live Cattle
    },
}


# ── 3. DATA DOWNLOAD ──────────────────────────────────────────────────────────

def download_asset_class(
    label_ticker: dict,
    asset_class:  str,
    start: str,
    end:   str,
    min_rows: int = MIN_HISTORY_DAYS * 2,
) -> pd.DataFrame:
    """
    Bulk-download adjusted closing prices for one asset class.
    Columns renamed to human labels. Tickers with < min_rows of valid data
    are dropped with a warning — this handles futures not yet on Yahoo Finance
    (e.g. ALI=F) without crashing.
    """
    tickers = list(label_ticker.values())
    labels  = list(label_ticker.keys())

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices.columns = labels
    prices.dropna(how="all", inplace=True)

    drop = []
    for col in prices.columns:
        valid = prices[col].dropna()
        if len(valid) < min_rows:
            print(f"    [{asset_class}] {col:18s}: ⚠ only {len(valid)} rows — skipping")
            drop.append(col)
        else:
            print(f"    [{asset_class}] {col:18s}: {len(valid):5d} days  "
                  f"({valid.index[0].date()} → {valid.index[-1].date()})")
    if drop:
        prices.drop(columns=drop, inplace=True)

    return prices


def download_all(start: str = START_DATE, end: str = END_DATE) -> dict:
    """Download prices for every asset class. Returns {ac_label: DataFrame}."""
    print(f"\nDownloading price data  ({start} → {end}) …\n")
    return {
        ac: download_asset_class(lt, ac, start, end)
        for ac, lt in ASSET_CLASSES.items()
    }


# ── 4. RETURN / PERCENTILE / FORWARD CALCULATIONS ────────────────────────────

def calc_rolling_returns(prices: pd.DataFrame, windows: dict) -> dict:
    """Simple total return over each look-back: r_t = P_t/P_{t-n} − 1."""
    return {lbl: prices.pct_change(periods=n) for lbl, n in windows.items()}


def calc_percentile_rank(returns: dict,
                         min_periods: int = MIN_HISTORY_DAYS) -> dict:
    """
    Expanding-window percentile rank (0–100, no look-ahead bias).
    rank_t = fraction of past values ≤ r_t × 100
    """
    return {
        lbl: ret_df.expanding(min_periods=min_periods).rank(pct=True) * 100
        for lbl, ret_df in returns.items()
    }


def calc_forward_returns(prices: pd.DataFrame, fwd_windows: dict) -> dict:
    """Future return: fwd_r_t = P_{t+n}/P_t − 1. Last n rows will be NaN."""
    return {
        lbl: prices.shift(-n).div(prices) - 1
        for lbl, n in fwd_windows.items()
    }


# ── 5. HISTORICAL ANALOGUE ANALYSIS ──────────────────────────────────────────

def analyse_analogues(
    ptile_dict:   dict,
    fwd_ret_dict: dict,
    band:         float = PTILE_BAND,
) -> dict:
    """
    For each (instrument, lookback):
      1. Record the current (most-recent) percentile.
      2. Find every past date with percentile in [current − band, current + band].
      3. Collect forward returns at those dates.
      4. Compute hit_rate, med_return, n_obs.

    Returns nested dict:
      results[instrument][lookback] = {
          "current_ptile"  : float,
          "analogue_dates" : DatetimeIndex,
          "analogue_ptiles": Series,
          "fwd": {
              fwd_label: {
                  "hit_rate"   : float,
                  "med_return" : float  (already in %),
                  "fwd_returns": Series (indexed by analogue date, in %),
                  "n_obs"      : int,
              }
          }
      }
    """
    results     = {}
    instruments = list(ptile_dict.values())[0].columns.tolist()

    for instr in instruments:
        results[instr] = {}

        for lb_label, ptile_df in ptile_dict.items():
            series = ptile_df[instr].dropna()
            if series.empty:
                continue

            current_ptile = float(series.iloc[-1])
            lo = max(0.0,   current_ptile - band)
            hi = min(100.0, current_ptile + band)

            hist           = series.iloc[:-1]
            mask           = (hist >= lo) & (hist <= hi)
            analogue_dates = hist[mask].index
            analogue_ptiles = hist[mask]

            fwd_stats = {}
            for fwd_label, fwd_df in fwd_ret_dict.items():
                at_ana = fwd_df[instr].reindex(analogue_dates).dropna()

                if len(at_ana) == 0:
                    fwd_stats[fwd_label] = {
                        "hit_rate":    np.nan,
                        "med_return":  np.nan,
                        "fwd_returns": pd.Series(dtype=float),
                        "n_obs":       0,
                    }
                else:
                    fwd_stats[fwd_label] = {
                        "hit_rate":    round((at_ana > 0).mean() * 100, 1),
                        "med_return":  round(at_ana.median() * 100, 2),
                        "fwd_returns": at_ana * 100,
                        "n_obs":       len(at_ana),
                    }

            results[instr][lb_label] = {
                "current_ptile":   round(current_ptile, 1),
                "analogue_dates":  analogue_dates,
                "analogue_ptiles": analogue_ptiles,
                "fwd":             fwd_stats,
            }

    return results


# ── 6. MAIN PIPELINE (importable) ─────────────────────────────────────────────

def run_analysis(start: str = START_DATE, end: str = END_DATE) -> dict:
    """
    Run the full data pipeline for every asset class.

    Returns
    -------
    all_results : dict
        { asset_class_label: analyse_analogues() output }
    """
    all_prices  = download_all(start, end)
    all_results = {}

    for ac_label, prices in all_prices.items():
        if prices.empty:
            continue
        rolling_ret = calc_rolling_returns(prices, LOOKBACK_WINDOWS)
        ptile_ranks = calc_percentile_rank(rolling_ret)
        fwd_returns = calc_forward_returns(prices, FORWARD_WINDOWS)
        all_results[ac_label] = analyse_analogues(ptile_ranks, fwd_returns)

    return all_results


# ── 7. FLATTEN RESULTS TO DATAFRAME ──────────────────────────────────────────

def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Flatten one asset class's results dict to a tidy DataFrame."""
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


# ── 8. PRECEDENTS EXPORT ──────────────────────────────────────────────────────

def export_precedents_csv(all_results: dict, path: str):
    """Save every analogue observation across all asset classes to CSV."""
    fwd_labels = list(FORWARD_WINDOWS.keys())
    rows = []

    for ac_label, results in all_results.items():
        for instr, lb_dict in results.items():
            for lb_label, data in lb_dict.items():
                dates   = data["analogue_dates"]
                ptiles  = data["analogue_ptiles"]

                fwd_series = {
                    fl: data["fwd"][fl]["fwd_returns"].reindex(dates)
                    for fl in fwd_labels
                }

                for date in dates:
                    row = {
                        "Asset Class": ac_label,
                        "Asset":       instr,
                        "Lookback":    lb_label,
                        "Date":        date.date(),
                        "%tile":       round(float(ptiles[date]), 1),
                    }
                    for fl in fwd_labels:
                        col = fl.replace(" ", "_") + "(%)"
                        v = fwd_series[fl].get(date, np.nan)
                        row[col] = round(float(v), 2) if not np.isnan(float(v if v is not None else np.nan)) else np.nan
                    rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(["Asset Class", "Asset", "Lookback", "Date"],
                   ascending=[True, True, True, False], inplace=True)
    df.to_csv(path, index=False)
    print(f"Precedents saved → {path}  ({len(df):,} rows)")
    return df


# ── 9. PER-ASSET-CLASS HEATMAP CHARTS ─────────────────────────────────────────

CMAP_PTILE = LinearSegmentedColormap.from_list(
    "ptile", ["#1a6faf", "#d0e4f2", "#ffffff", "#f5c0b0", "#b2182b"])
CMAP_HIT = LinearSegmentedColormap.from_list(
    "hit",   ["#b2182b", "#f5c0b0", "#ffffff", "#c7e9c0", "#1a6f1a"])
CMAP_RET = LinearSegmentedColormap.from_list(
    "ret",   ["#b2182b", "#f5c0b0", "#ffffff", "#c7e9c0", "#1a6f1a"])


def make_asset_class_chart(df: pd.DataFrame, asset_class: str, save_path: str):
    """3-panel heatmap (current %tile / hit rate / median fwd return)."""
    instruments = df["Index"].unique().tolist()
    lookbacks   = ["2w", "4w", "8w", "12w"]
    forwards    = ["1w fwd", "2w fwd", "4w fwd"]
    n_instr, n_lb, n_fwd = len(instruments), len(lookbacks), len(forwards)

    rh = max(1.5, n_instr * 0.42)
    fig = plt.figure(figsize=(22, rh * 3 + 5.0))
    fig.patch.set_facecolor("#f8f8f8")

    fig.suptitle(
        f"{asset_class} — Return Percentile & Forward-Return Analysis  "
        f"|  {datetime.today().strftime('%d %b %Y')}  |  Analogue band ±{PTILE_BAND}%tile",
        fontsize=13, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[rh, rh, rh],
                           hspace=0.55, top=0.975, bottom=0.03,
                           left=0.12, right=0.97)

    def build(metric, fwd=None):
        mat = np.full((n_instr, n_lb), np.nan)
        for i, ins in enumerate(instruments):
            for j, lb in enumerate(lookbacks):
                q = df[(df["Index"] == ins) & (df["Lookback"] == lb)]
                if fwd:
                    q = q[q["Forward"] == fwd]
                if not q.empty:
                    mat[i, j] = q[metric].iloc[0]
        return mat

    def ann(ax, mat, fmt, threshold=None, abs_max=None):
        fs = max(6, min(9, 100 // n_instr))
        for i in range(n_instr):
            for j in range(n_lb):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                if threshold:
                    tc = "white" if (v < threshold[0] or v > threshold[1]) else "black"
                elif abs_max:
                    tc = "white" if abs(v) > 0.65 * abs_max else "black"
                else:
                    tc = "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=fs, color=tc)

    # Panel A — current percentile
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_title("Current Return Percentile", fontsize=10, pad=6)
    mat_a = build("Current %tile")
    im_a  = ax_a.imshow(mat_a, cmap=CMAP_PTILE, vmin=0, vmax=100, aspect="auto")
    ax_a.set_xticks(range(n_lb)); ax_a.set_xticklabels(lookbacks, fontsize=9)
    ax_a.set_yticks(range(n_instr)); ax_a.set_yticklabels(instruments, fontsize=8)
    ann(ax_a, mat_a, "{:.0f}", threshold=(20, 80))
    plt.colorbar(im_a, ax=ax_a, fraction=0.015, pad=0.01, label="Percentile")

    # Panels B/C — hit rate / median fwd return
    for row_idx, (metric, cmap, vmin, vmax, fmt, thr) in enumerate([
        ("Hit Rate (%)",   CMAP_HIT, 30, 70,  "{:.0f}%", (38, 62)),
        ("Median Fwd (%)", CMAP_RET, None, None, "{:+.1f}%", None),
    ], start=1):
        gs_sub = gridspec.GridSpecFromSubplotSpec(1, n_fwd, subplot_spec=gs[row_idx],
                                                  wspace=0.30)
        all_vals = df[metric].dropna()
        if metric == "Median Fwd (%)":
            abs_max = max(abs(all_vals.quantile(0.05)), abs(all_vals.quantile(0.95)), 0.3) if len(all_vals) else 2.0
            abs_max = np.ceil(abs_max * 4) / 4
            vmin, vmax = -abs_max, abs_max
        else:
            abs_max = None

        for k, fwd in enumerate(forwards):
            ax = fig.add_subplot(gs_sub[0, k])
            title_prefix = "Hit Rate (%)" if metric == "Hit Rate (%)" else "Median Fwd Ret (%)"
            ax.set_title(f"{title_prefix}  —  {fwd}", fontsize=9, pad=5)
            mat = build(metric, fwd=fwd)
            im  = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(n_lb)); ax.set_xticklabels(lookbacks, fontsize=8)
            ax.set_yticks(range(n_instr))
            ax.set_yticklabels(instruments if k == 0 else [""] * n_instr, fontsize=8)
            if k == 0:
                ax.set_ylabel("Instrument", fontsize=8)
            ann(ax, mat, fmt, threshold=thr, abs_max=abs_max)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.text(0.5, 0.005,
             f"Source: Yahoo Finance  |  {HISTORY_YEARS}yr history  |  "
             f"Expanding-window percentile (min {MIN_HISTORY_DAYS}d)  |  "
             f"Analogues: ±{PTILE_BAND}%tile",
             ha="center", fontsize=7, color="#666666")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {save_path}")
    plt.close(fig)


# ── 10. STANDALONE MAIN ───────────────────────────────────────────────────────

def main():
    """
    Standalone run: download, analyse, save per-asset-class charts and
    precedents CSV.  For the actionable screen + email use run_weekly.py.
    """
    all_results = run_analysis()

    for ac_label, results in all_results.items():
        df_ac = results_to_dataframe(results)
        df_ac.insert(0, "Asset Class", ac_label)
        chart_name = ("return_percentile_"
                      + ac_label.lower().replace(" ", "_").replace("/", "") + ".png")
        make_asset_class_chart(df_ac, ac_label, save_path=chart_name)

    prec_path = f"precedents_{TODAY_STR}.csv"
    export_precedents_csv(all_results, prec_path)
    print(f"\nDone.")
    return all_results


if __name__ == "__main__":
    main()
