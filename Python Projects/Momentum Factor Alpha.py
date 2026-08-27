import argparse
import warnings
from datetime import datetime
import io

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ── 1. Universe Definition ───────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """
    Scrape S&P 500 constituents from Wikipedia.
    Falls back to a broad 106-stock large-cap list if network scraping is blocked.
    """
    FALLBACK = [
        'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','BRK-B','AVGO','JPM',
        'LLY','UNH','V','XOM','MA','JNJ','COST','PG','ORCL','HD','WMT','ABBV',
        'BAC','KO','MRK','CVX','NFLX','AMD','CRM','TMO','PEP','ADBE','WFC','ACN',
        'LIN','CSCO','MCD','ABT','DHR','TXN','PM','CAT','NEE','NKE','IBM','INTC',
        'UNP','RTX','AMGN','HON','VZ','LOW','QCOM','SPGI','GS','AMAT','ELV','MDT',
        'PLD','MS','SCHW','SYK','BMY','BLK','AXP','TJX','ISRG','GILD','ADI','C',
        'MMC','DE','EOG','MO','CI','USB','SO','REGN','PYPL','SLB','ZTS','CME',
        'DUK','CL','ITW','GE','ICE','NOC','BSX','ETN','AON','HUM','MU','MCO',
        'TGT','FCX','APD','PSA','HCA','FDX','MMM','PNC','ECL','KLAC','NSC','LRCX'
    ]
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
        tables = pd.read_html(io.StringIO(html))
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"[Universe] {len(tickers)} S&P 500 tickers loaded from Wikipedia.")
        return tickers
    except Exception as e:
        print(f"[Universe] Wikipedia fallback triggered ({e}). Processing broad 106 large-cap fallback baseline.")
        return FALLBACK


# ── 2. Data Sourcing ──────────────────────────────────────────────────────────

def download_data(tickers: list[str], start: str, end: str, batch: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """Download matrix of adjusted daily closing prices and market index baseline."""
    print(f"[Data] Accessing universe tracking: {start} → {end} …")
    
    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        raw = yf.download(chunk, start=start, end=end, auto_adjust=True, progress=False, threads=True)
        
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
            close.columns = chunk
        frames.append(close)
    
    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    
    # Prune elements missing more than 20% of trading history
    prices = prices.loc[:, prices.notna().mean() >= 0.80]
    
    # Secure Benchmark Index Data
    bench_raw = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    bench_prices = bench_raw["Close"].squeeze()
    if isinstance(bench_prices, pd.DataFrame):
        bench_prices = bench_prices.iloc[:, 0]
    bench_prices.index = pd.to_datetime(bench_prices.index)
    
    print(f"[Data] Structural Matrix Built: {prices.shape[0]} days × {prices.shape[1]} stocks.")
    return prices, bench_prices


# ── 3. Factor Math & Signals ──────────────────────────────────────────────────

def compute_momentum(prices: pd.DataFrame, lookback_months: int = 12, skip_months: int = 1) -> pd.DataFrame:
    """12-1 Month Fama-French Momentum Signal Engine (skips nearest 21 days)."""
    return prices.shift(skip_months * 21) / prices.shift(lookback_months * 21) - 1


def compute_information_coefficient(momentum: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Computes monthly Spearman Rank Information Coefficient (IC) tracking forward 21-day returns."""
    fwd_returns = prices.pct_change(21).shift(-21)
    rebal_dates = momentum.resample("ME").last().index
    rebal_dates = rebal_dates[rebal_dates.isin(momentum.index)]
    
    ic_series = {}
    for date in rebal_dates:
        m_scores = momentum.loc[date].dropna()
        f_rets = fwd_returns.loc[date].dropna()
        common = m_scores.index.intersection(f_rets.index)
        
        if len(common) > 15:
            rho, _ = spearmanr(m_scores.loc[common], f_rets.loc[common])
            ic_series[date] = rho
            
    return pd.Series(ic_series).rename("Information_Coefficient")


def compute_horizon_decay(momentum: pd.DataFrame, prices: pd.DataFrame, start_date: str) -> pd.Series:
    """Simulates performance decay profiles across alternative multi-month holding lockups."""
    horizons = range(1, 7)
    horizon_returns = {}
    p_sub = prices.loc[start_date:]
    returns_daily = p_sub.pct_change()
    
    rebal_dates = momentum.resample("ME").last().index
    rebal_dates = rebal_dates[rebal_dates.isin(momentum.index)]
    
    for h in horizons:
        weights_h = pd.DataFrame(0.0, index=p_sub.index, columns=prices.columns)
        target_dates = rebal_dates[::h]
        
        for date in target_dates:
            if date not in momentum.index: continue
            scores = momentum.loc[date].dropna()
            if len(scores) < 10: continue
            
            n_target = max(1, round(len(scores) * 10 / 100))
            top_stocks = scores.nlargest(n_target).index
            
            idx = p_sub.index.get_indexer([date])[0]
            if idx == -1: continue
            end_idx = min(idx + (h * 21), len(p_sub))
            
            date_labels = p_sub.index[idx:end_idx]
            weights_h.loc[date_labels, top_stocks] = 1.0 / len(top_stocks)
                
        # FIX: Fill missing asset returns with 0 to prevent total portfolio return calculations from failing to NaN
        clean_returns_daily = returns_daily.fillna(0.0)
        p_ret = (weights_h.shift(1) * clean_returns_daily).sum(axis=1)
        
        if (weights_h.sum(axis=1) == 0).all():
            ann_ret = 0.0
        else:
            clean_rets = p_ret.loc[start_date:]
            cum_prod = (1 + clean_rets).prod()
            ann_ret = (cum_prod) ** (252 / len(clean_rets)) - 1 if len(clean_rets) else 0
            
        horizon_returns[f"{h}M Holder"] = ann_ret
        
    return pd.Series(horizon_returns)


def build_decile_weights(momentum: pd.DataFrame, decile_pct: int = 10, top: bool = True) -> pd.DataFrame:
    """
    Generates daily rebalanced weight frames isolating a factor score decile.
    top=True selects the highest-momentum decile (D10); top=False selects the lowest (D1).
    """
    rebal_dates = momentum.resample("ME").last().index
    rebal_dates = rebal_dates[rebal_dates.isin(momentum.index)]
    weights_on_rebal = []

    for date in rebal_dates:
        scores = momentum.loc[date].dropna()
        if len(scores) < 10: continue
        n_target = max(1, round(len(scores) * decile_pct / 100))
        selected_stocks = scores.nlargest(n_target).index if top else scores.nsmallest(n_target).index
        selected = pd.Series(0.0, index=momentum.columns)
        selected[selected_stocks] = 1.0 / len(selected_stocks)
        weights_on_rebal.append(pd.Series(selected, name=date))

    return pd.DataFrame(weights_on_rebal).reindex(momentum.index).ffill().fillna(0.0)


def compute_rolling_ir(strat_ret: pd.Series, bench_ret: pd.Series, window: int = 252) -> pd.Series:
    """Rolling annualized Information Ratio: rolling mean active return / rolling std of active return * sqrt(252)."""
    active = (strat_ret - bench_ret).dropna()
    rolling_mean = active.rolling(window).mean() * 252
    rolling_std = active.rolling(window).std() * np.sqrt(252)
    return (rolling_mean / rolling_std).rename(f"Rolling_{window}D_IR")


# ── 4. Visualization Dashboard ───────────────────────────────────────

def plot_alpha_dashboard(
    port_ret: pd.Series, bench_ret: pd.Series, 
    ic_series: pd.Series, decay_series: pd.Series, 
    spread_ret: pd.Series, rolling_ir: pd.Series,
    ir_val: float, tracking_error: float
) -> None:
    
    r = port_ret.dropna()
    b = bench_ret.loc[r.index].dropna()
    
    common_idx = r.index.intersection(b.index)
    r = r.loc[common_idx]
    b = b.loc[common_idx]
    
    cum_p = (1 + r).cumprod()
    cum_b = (1 + b).cumprod()
    dd_p = cum_p / cum_p.cummax() - 1
    
    # Color Profiles 
    BG, PANEL, BORDER = "#0d1117", "#161b22", "#30363d"
    STRAT_COLOR, BENCH_COLOR, ACCENT = "#3fb950", "#58a6ff", "#f0883e"
    RED, MUTED, TEXT = "#f85149", "#8b949e", "#e6edf3"

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": BORDER,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": TEXT, "grid.color": BORDER, "grid.linewidth": 0.5,
        "font.family": "monospace", "font.size": 8.5,
    })

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    fig.suptitle("Momentum Alpha Attribution Framework & Predictive Metric Dash", fontsize=14, fontweight="bold", y=0.985)

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 8], hspace=0.20)
    stat_gs = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0], wspace=0.3)
    chart_gs = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[1], hspace=0.40, wspace=0.22)

    def stat_card(ax, title, main_val, sub_val="", color=TEXT):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values(): s.set_edgecolor(BORDER)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.72, title, transform=ax.transAxes, ha="center", fontsize=8, color=MUTED, fontweight="bold")
        ax.text(0.5, 0.40, main_val, transform=ax.transAxes, ha="center", fontsize=14, fontweight="bold", color=color)
        if sub_val:
            ax.text(0.5, 0.15, sub_val, transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED)

    # Compile Summary Data 
    tot_ret = cum_p.iloc[-1] - 1 if len(cum_p) else 0
    b_tot_ret = cum_b.iloc[-1] - 1 if len(cum_b) else 0
    ann_ret = (1 + tot_ret) ** (252 / len(r)) - 1
    b_ann_ret = (1 + b_tot_ret) ** (252 / len(b)) - 1
    
    stat_card(fig.add_subplot(stat_gs[0]), "Strategy Annualized Return", f"{ann_ret:.1%}", f"S&P 500: {b_ann_ret:.1%}", STRAT_COLOR)
    stat_card(fig.add_subplot(stat_gs[1]), "Information Ratio (IR)", f"{ir_val:.2f}", f"Active Risk: {tracking_error:.1%}", STRAT_COLOR)
    stat_card(fig.add_subplot(stat_gs[2]), "Mean Information Coeff (IC)", f"{ic_series.mean():.3f}", f"T-Stat: {ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series))):.2f}" if len(ic_series) else "N/A", ACCENT)
    stat_card(fig.add_subplot(stat_gs[3]), "Max Portfolio Drawdown", f"{dd_p.min():.1%}", f"Index Max DD: {(cum_b / cum_b.cummax() - 1).min():.1%}", RED)
    stat_card(fig.add_subplot(stat_gs[4]), "IC Positive Months", f"{(ic_series > 0).mean():.1%}" if len(ic_series) else "0.0%", f"Total Rebalances: {len(ic_series)}", TEXT)

    # Panel A: Performance Growth Chart (log scale — 45yr compounding is unreadable linear)
    ax1 = fig.add_subplot(chart_gs[0, 0])
    ax1.plot(cum_p.index, cum_p.values, color=STRAT_COLOR, linewidth=1.5, label="Momentum D10 Portfolio")
    ax1.plot(cum_b.index, cum_b.values, color=BENCH_COLOR, linewidth=1.1, linestyle="--", label="S&P 500 Baseline")
    ax1.set_yscale("log")
    ax1.set_title("Strategy vs S&P 500 Baseline Accumulation Profile (log scale, $1 → $N)", color=TEXT, fontsize=9, pad=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}x" if v >= 1 else f"{v:.2f}x"))
    ax1.grid(True, axis="y", alpha=0.3, which="both")
    ax1.legend(facecolor=PANEL, edgecolor=BORDER, loc="upper left")

    # Panel B: Rank Factor IC Realizations
    ax2 = fig.add_subplot(chart_gs[0, 1])
    if len(ic_series):
        ax2.bar(ic_series.index, ic_series.values, color=[STRAT_COLOR if x >= 0 else RED for x in ic_series.values], width=18, alpha=0.7)
        ax2.axhline(ic_series.mean(), color=ACCENT, linestyle="-.", linewidth=1, label=f"Mean IC ({ic_series.mean():.3f})")
    ax2.set_title("Cross-Sectional Spearman Rank Information Coefficient (IC)", color=TEXT, fontsize=9, pad=8)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(facecolor=PANEL, edgecolor=BORDER)

    # Panel C: Alpha Holding Horizon Decay Function
    ax3 = fig.add_subplot(chart_gs[1, 0])
    decay_pct = decay_series * 100
    bars = ax3.bar(decay_pct.index, decay_pct.values, color=BENCH_COLOR, alpha=0.7, width=0.45, edgecolor=BORDER)
    ax3.axhline(b_ann_ret * 100, color=RED, linestyle=":", label="Passive S&P 500 Floor")
    ax3.set_title("Alpha Structural Decay across Extended Rebalance Lock-up Horizons", color=TEXT, fontsize=9, pad=8)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend(facecolor=PANEL, edgecolor=BORDER)
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=7.5, color=MUTED)

    # Panel D: Rolling 12M Information Ratio (Edge Stability Diagnostic)
    ax4 = fig.add_subplot(chart_gs[1, 1])
    rir = rolling_ir.dropna()
    if len(rir):
        # Fill positive in green, negative in red for visual asymmetry
        ax4.fill_between(rir.index, rir.values, 0, where=(rir.values >= 0), 
                         color=STRAT_COLOR, alpha=0.35, interpolate=True)
        ax4.fill_between(rir.index, rir.values, 0, where=(rir.values < 0), 
                         color=RED, alpha=0.35, interpolate=True)
        ax4.plot(rir.index, rir.values, color=TEXT, linewidth=1.0)
        ax4.axhline(0, color=MUTED, linewidth=0.8, linestyle="-")
        ax4.axhline(rir.mean(), color=ACCENT, linestyle="-.", linewidth=1, 
                    label=f"Mean ({rir.mean():.2f})")
        ax4.axhline(0.5, color=BENCH_COLOR, linestyle=":", linewidth=0.8, alpha=0.6,
                    label="IR = 0.5 threshold")
    ax4.set_title("Rolling 12M Information Ratio — Edge Stability Across Regimes", color=TEXT, fontsize=9, pad=8)
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.legend(facecolor=PANEL, edgecolor=BORDER, loc="upper left", fontsize=7.5)

    # Panel E: Top vs Bottom Decile Spread (Signal Quality Diagnostic) — full width, log scale
    ax5 = fig.add_subplot(chart_gs[2, :])
    spread_clean = spread_ret.dropna()
    if len(spread_clean):
        cum_spread = (1 + spread_clean).cumprod()
        cum_p_aligned = (1 + r.loc[r.index.intersection(spread_clean.index)]).cumprod()
        
        ax5.plot(cum_spread.index, cum_spread.values, color=ACCENT, linewidth=1.5, 
                 label="D10 − D1 Long/Short Spread (pure factor)")
        ax5.plot(cum_p_aligned.index, cum_p_aligned.values, color=STRAT_COLOR, linewidth=1.0, 
                 linestyle="--", alpha=0.7, label="D10 Long-Only (for reference)")
        ax5.axhline(1.0, color=MUTED, linewidth=0.8, linestyle="-")
        ax5.set_yscale("log")
        
        # Annualized spread return for context
        spread_ann = (cum_spread.iloc[-1]) ** (252 / len(spread_clean)) - 1
        spread_sharpe = (spread_clean.mean() * 252) / (spread_clean.std() * np.sqrt(252)) if spread_clean.std() else 0
        
        ax5.text(0.35, 0.95, 
                 f"Spread Ann. Return: {spread_ann:+.2%}   |   Spread Sharpe: {spread_sharpe:.2f}",
                 transform=ax5.transAxes, fontsize=9, color=TEXT, fontweight="bold",
                 verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL, edgecolor=BORDER))
    
    ax5.set_title("Top vs Bottom Decile Spread (D10 − D1) — Cross-Sectional Signal Purity Test (log scale)", 
                  color=TEXT, fontsize=9, pad=8)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.1f}x" if v < 10 else f"{v:,.0f}x"))
    ax5.grid(True, axis="y", alpha=0.3, which="both")
    ax5.legend(facecolor=PANEL, edgecolor=BORDER, loc="upper left")

    plt.tight_layout(rect=[0, 0.02, 1, 0.965])
    plt.show()


# ── 5. Backtest Runner ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Advanced Strategic Momentum Evaluation Engine")
    parser.add_argument("--start", default="1980-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    data_start = (pd.to_datetime(args.start) - pd.DateOffset(months=13)).strftime("%Y-%m-%d")

    tickers = get_sp500_tickers()
    prices, bench_prices = download_data(tickers, start=data_start, end=args.end)

    print("[Signal] Generating 12-1 Cross-Sectional Factor Engine Signals…")
    momentum = compute_momentum(prices)
    
    # Build both top and bottom decile weights for spread analysis
    weights_top = build_decile_weights(momentum, top=True).loc[args.start : args.end]
    weights_bot = build_decile_weights(momentum, top=False).loc[args.start : args.end]
    
    p_prices = prices.loc[args.start : args.end]
    daily_rets = p_prices.pct_change().fillna(0.0)
    
    strat_ret = (weights_top.shift(1) * daily_rets).sum(axis=1).rename("Momentum_D10")
    bot_ret = (weights_bot.shift(1) * daily_rets).sum(axis=1).rename("Momentum_D1")
    spread_ret = (strat_ret - bot_ret).rename("D10_minus_D1")
    
    bench_ret = bench_prices.loc[args.start : args.end].pct_change().rename("SP500_Index")

    print("[Attribution] Computing Predictive Information Coefficients (IC)…")
    ic_series = compute_information_coefficient(momentum.loc[args.start : args.end], prices)
    
    print("[Attribution] Running Staggered Rebalance Horizons Simulation (Alpha Decay Evaluation)…")
    decay_series = compute_horizon_decay(momentum, prices, args.start)

    # Active Matrix Statistical Reductions
    ann = 252
    s_ret_clean = strat_ret.dropna()
    b_ret_clean = bench_ret.loc[s_ret_clean.index].dropna()
    
    common_dates = s_ret_clean.index.intersection(b_ret_clean.index)
    s_ret_clean = s_ret_clean.loc[common_dates]
    b_ret_clean = b_ret_clean.loc[common_dates]

    active_ret = s_ret_clean - b_ret_clean
    tracking_error = active_ret.std() * np.sqrt(ann)
    
    p_tot = (1 + s_ret_clean).prod() - 1
    b_tot = (1 + b_ret_clean).prod() - 1
    p_ann = (1 + p_tot) ** (ann / len(s_ret_clean)) - 1
    b_ann = (1 + b_tot) ** (ann / len(b_ret_clean)) - 1
    
    ir_val = (p_ann - b_ann) / tracking_error if tracking_error else np.nan

    print("[Attribution] Computing Rolling 12M Information Ratio (Edge Stability)…")
    rolling_ir = compute_rolling_ir(s_ret_clean, b_ret_clean, window=252)

    # D10-D1 spread summary
    spread_clean = spread_ret.dropna().loc[common_dates]
    spread_ann_ret = (1 + spread_clean).prod() ** (ann / len(spread_clean)) - 1 if len(spread_clean) else 0
    spread_sharpe = (spread_clean.mean() * ann) / (spread_clean.std() * np.sqrt(ann)) if spread_clean.std() else 0

    print("\n── Strategic Performance Summary ────────────────")
    print(f"Annualized Excess Alpha:    {p_ann - b_ann:+.2%}")
    print(f"Tracking Error Risk:       {tracking_error:.2%}")
    print(f"Information Ratio (IR):     {ir_val:.2f}")
    print(f"Mean Rank Factor IC:        {ic_series.mean():.4f}" if len(ic_series) else "Mean Rank Factor IC: N/A")
    print(f"D10−D1 Spread Ann Return:   {spread_ann_ret:+.2%}")
    print(f"D10−D1 Spread Sharpe:       {spread_sharpe:.2f}")
    print(f"Rolling IR — Mean / Min:    {rolling_ir.mean():.2f} / {rolling_ir.min():.2f}")

    plot_alpha_dashboard(s_ret_clean, b_ret_clean, ic_series, decay_series, 
                         spread_ret, rolling_ir, ir_val, tracking_error)


if __name__ == "__main__":
    main()