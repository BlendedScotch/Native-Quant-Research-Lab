# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative trading research repository focused on systematic equity factor strategies and options market analysis. Implements backtesting engines, signal generation, and visualization dashboards with emphasis on momentum factors, volatility analysis, and risk management.

## Directory Structure

- **Python Projects/** – Standalone strategy scripts (momentum factors, IV surfaces, volatility premium analysis, data ingestion)
- **Jupyter Notebooks/** – Research notebooks (factor validation, pairs trading, option portfolio construction, volatility regime analysis)
- **Research Papers/** – Academic references (AQR momentum, volatility risk premium studies)
- **Claude/** – Claude AI conversation artifacts

## Architecture & Key Patterns

### Strategy Pipeline
Python scripts follow a consistent flow:
1. Universe definition (S&P 500 via Wikipedia scrape → fallback 106-stock curated list if network fails)
2. Data download (yfinance, batched)
3. Signal computation (momentum, volatility, alpha)
4. Portfolio construction (percentile ranking, weighting, rebalancing)
5. Performance metrics (Sharpe, Calmar, max drawdown, rolling 63-day Sharpe)
6. Dashboard visualization (matplotlib dark theme or Plotly interactive)

`Momentum Factor.py` is the canonical reference implementation of this full pipeline.

### Factor Signals
- **Momentum:** 12-month lookback, 1-month skip (avoids short-term reversal)
- **Information Coefficient:** Spearman rank correlation of signal vs 21-day forward returns
- **Cross-sectional ranking:** Percentile-based (top decile = top 10%)

### Options Visualization
`scipy.interpolate.griddata` interpolates 2D IV/OI surfaces:
- **Matplotlib 3D** – static publication plots
- **Plotly 3D** – interactive with synchronized call/put color scales and ATM path overlay

### Configuration
- argparse for date ranges, decile selection, rebalance frequency (`"ME"`, `"QE"`, `"YE"`)
- Hardcoded config vars (`MIN_VOLUME`, `MONEYNESS_RANGE`, `ticker`) at top of each script

## Running Scripts

```bash
# Momentum factor (default date range)
python "Python Projects/Momentum Factor.py"

# Custom range and decile
python "Python Projects/Momentum Factor.py" --start 2020-01-01 --end 2024-12-31 --decile 10

# SPX realized vs implied volatility
python "Python Projects/SPX Realized vs Implied Volatility.py"

# Interactive IV surface (Plotly)
python "Python Projects/Interactive IV Surface.py"
```

Notebooks use `parser.parse_args(args=[])` to bypass sys.argv conflicts in Jupyter — run cells sequentially.

## Dependencies

No `requirements.txt`. Core imports: `yfinance`, `pandas`, `numpy`, `matplotlib`, `plotly`, `scipy.stats`, `scipy.interpolate`.

## Code Conventions

- **Theme:** GitHub dark (#0d1117 bg, #e6edf3 text, #3fb950 green, #f85149 red), monospace font
- **Logging:** Print statements with `[Section]` prefix (`[Data]`, `[Signal]`, `[Portfolio]`)
- **Error handling:** try/except with fallback logic (e.g., Wikipedia scrape → hardcoded list)
- **Docstrings:** Triple-quoted with numbered section headers matching the pipeline steps

## Key Files

| File | Purpose |
|------|---------|
| `Momentum Factor.py` | Gold standard — full 7-step pipeline with metric dashboards |
| `Momentum Factor Alpha.py` | Extended version with IC (signal predictiveness) |
| `Interactive IV Surface.py` | Plotly 3D IV surface template for interactive viz |
| `Sandbox.py`, `Sandbox2.py` | Throwaway exploration |

## Performance Metrics

All strategies report: Total Return, Annualized Return (252 days/year), Annualized Volatility (daily std × √252), Sharpe, Max Drawdown, Calmar (Ann Return / |Max DD|), Win Rate, Rolling 63-day Sharpe.
