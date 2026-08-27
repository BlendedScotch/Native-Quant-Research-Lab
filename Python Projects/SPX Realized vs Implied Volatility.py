import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_volatility_analysis(years=3):
    # 1. Download S&P 500 (^GSPC) and VIX (^VIX) data
    tickers = ["^GSPC", "^VIX"]
    # We pull extra data to account for the initial 21-day window calculation
    data = yf.download(tickers, period=f"{years+1}y")['Close']
    
    # 2. Calculate Realized Volatility (RV)
    # Log returns: ln(Price_t / Price_{t-1})
    log_returns = np.log(data['^GSPC'] / data['^GSPC'].shift(1))
    
    # Annualized Realized Volatility: StdDev * Sqrt(252 trading days) * 100
    # A 21-day window is used to match the VIX 30-day (one month) forecast
    realized_vol = log_returns.rolling(window=21).std() * np.sqrt(252) * 100
    
    # 3. Prepare Dataframe
    df = pd.DataFrame({
        'Realized_Vol': realized_vol,
        'Implied_Vol': data['^VIX']
    }).dropna()
    
    # Filter to the requested timeframe
    df = df.tail(252 * years)
    
    # 4. Plotting
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['Implied_Vol'], label='Implied Vol (VIX)', color='#1f77b4', lw=1.5)
    plt.plot(df.index, df['Realized_Vol'], label='Realized Vol (21d)', color='#ff7f0e', lw=1.5)
    
    # Zone 1: Volatility Risk Premium (VIX > Realized)
    # This is the "normal" state where insurance is expensive
    plt.fill_between(df.index, df['Implied_Vol'], df['Realized_Vol'], 
                     where=(df['Implied_Vol'] >= df['Realized_Vol']),
                     interpolate=True, color='green', alpha=0.15, label='Risk Premium (VIX > RV)')
    
    # Zone 2: Volatility Discount (VIX < Realized)
    # Occurs when the market is caught off guard by high realized movement
    plt.fill_between(df.index, df['Implied_Vol'], df['Realized_Vol'], 
                     where=(df['Implied_Vol'] < df['Realized_Vol']),
                     interpolate=True, color='red', alpha=0.15, label='Discount (VIX < RV)')
    
    plt.title('S&P 500: Implied vs. Realized Volatility', fontsize=16, fontweight='bold')
    plt.ylabel('Annualized Volatility (%)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_volatility_analysis(years=3)