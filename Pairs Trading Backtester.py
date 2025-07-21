"""
Pairs Trading Backtester (Multi-Level Column Fixed)
--------------------------------------------------
A robust Python script to simulate and visualize a pairs trading strategy on two stocks.
Handles yfinance multi-level column structure correctly.
Author: [Your Name]
Date: 2025-07-22
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_data(stock1, stock2, start_date, end_date):
    print(f"Downloading data for {stock1} and {stock2}...")
    
    # Download data for both stocks with auto_adjust=False to avoid warning
    data1 = yf.download(stock1, start=start_date, end=end_date, auto_adjust=False)
    data2 = yf.download(stock2, start=start_date, end=end_date, auto_adjust=False)
    
    print(f"Columns for {stock1}: {list(data1.columns)}")
    print(f"Columns for {stock2}: {list(data2.columns)}")
    
    if data1.empty or data2.empty:
        print("One or both stocks returned empty data")
        return None
    
    # Extract closing prices from multi-level columns
    try:
        # For multi-level columns like ('Close', 'AAPL')
        if isinstance(data1.columns, pd.MultiIndex):
            close1 = data1[('Close', stock1)]
        else:
            close1 = data1['Close']
            
        if isinstance(data2.columns, pd.MultiIndex):
            close2 = data2[('Close', stock2)]
        else:
            close2 = data2['Close']
        
        # Create DataFrame with proper index alignment
        df = pd.DataFrame(index=close1.index)
        df[stock1] = close1
        df[stock2] = close2
        
        # Drop any rows with missing data
        df = df.dropna()
        
        if df.empty:
            print("No overlapping data found between the two stocks")
            return None
            
        print(f"Successfully fetched {len(df)} days of data")
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def calculate_zscore(df, lookback=30):
    """Calculate rolling z-score of price ratio"""
    df = df.copy()
    stock1, stock2 = df.columns[0], df.columns[1]
    
    df['Ratio'] = df[stock1] / df[stock2]
    df['Ratio_Mean'] = df['Ratio'].rolling(window=lookback).mean()
    df['Ratio_Std'] = df['Ratio'].rolling(window=lookback).std()
    df['Zscore'] = (df['Ratio'] - df['Ratio_Mean']) / df['Ratio_Std']
    
    return df

def generate_signals(df, entry_threshold=1.0, exit_threshold=0.0):
    """Generate trading signals based on z-score"""
    df = df.copy()
    df['Signal'] = 0
    
    # Long spread if z < -entry, short spread if z > entry
    df.loc[df['Zscore'] < -entry_threshold, 'Signal'] = 1      # Long Stock1, Short Stock2
    df.loc[df['Zscore'] > entry_threshold, 'Signal'] = -1      # Short Stock1, Long Stock2
    # Exit if |z| < exit threshold
    df.loc[df['Zscore'].abs() < exit_threshold, 'Signal'] = 0

    # Forward fill to simulate holding position
    df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0)
    return df

def calculate_returns(df):
    """Calculate strategy returns"""
    df = df.copy()
    stock1, stock2 = df.columns[0], df.columns[1]
    
    # Spread return: long stock1, short stock2
    df['Spread_Return'] = df[stock1].pct_change() - df[stock2].pct_change()
    # Strategy return: position * spread return (with 1-day lag for position)
    df['Strategy_Return'] = df['Position'].shift(1) * df['Spread_Return']
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    return df

def plot_results(df, stock1, stock2):
    """Plot comprehensive results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Price chart
    axes[0].plot(df.index, df[stock1], label=stock1, color='blue', alpha=0.8)
    axes[0].plot(df.index, df[stock2], label=stock2, color='red', alpha=0.8)
    axes[0].set_ylabel("Price ($)")
    axes[0].set_title(f"Stock Prices: {stock1} vs {stock2}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Z-Score & thresholds
    axes[1].plot(df.index, df['Zscore'], color='purple', label='Z-Score')
    axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='±1.0 Threshold')
    axes[1].axhline(-1.0, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_ylabel("Z-Score")
    axes[1].set_title("Rolling Z-Score of Price Ratio (30-day)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Equity curve
    axes[2].plot(df.index, df['Cumulative_Return'], color='green', linewidth=2, label='Strategy')
    axes[2].axhline(1.0, color='black', linestyle='-', alpha=0.5)
    axes[2].set_ylabel("Cumulative Return")
    axes[2].set_xlabel("Date")
    axes[2].set_title("Strategy Performance")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pairs_trading_sample_output.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_performance_summary(df, stock1, stock2):
    """Print strategy performance metrics"""
    total_return = df['Cumulative_Return'].iloc[-1] - 1
    n_trades = int((df['Signal'].diff().abs() == 2).sum())
    
    # Calculate some basic metrics
    returns = df['Strategy_Return'].dropna()
    if len(returns) > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_return = df['Cumulative_Return'].max() - 1
        min_return = df['Cumulative_Return'].min() - 1
        
        print("\n" + "="*50)
        print(f"PAIRS TRADING STRATEGY RESULTS")
        print("="*50)
        print(f"Stocks:           {stock1} & {stock2}")
        print(f"Period:           {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Total Return:     {total_return:.2%}")
        print(f"Number of Trades: {n_trades}")
        print(f"Sharpe Ratio:     {sharpe_ratio:.2f}")
        print(f"Max Return:       {max_return:.2%}")
        print(f"Max Drawdown:     {min_return:.2%}")
        print("="*50)
    else:
        print(f"No trades executed during the period.")

def main():
    # Configuration
    stock1 = "AAPL"       
    stock2 = "MSFT"       
    end_date = datetime.today()
    start_date = end_date - timedelta(days=2*365)  # 2 years of data
    
    print(f"Starting Pairs Trading Backtest...")
    print(f"Stocks: {stock1} & {stock2}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Fetch and process data
    df = fetch_data(stock1, stock2, start_date, end_date)
    
    if df is None:
        print("Failed to fetch data. Please check your tickers and try again.")
        return
    
    # Run the strategy
    df = calculate_zscore(df, lookback=30)
    df = generate_signals(df, entry_threshold=1.0, exit_threshold=0.0)
    df = calculate_returns(df)
    
    # Display results
    print_performance_summary(df, stock1, stock2)
    plot_results(df, stock1, stock2)
    
    print(f"\nChart saved as 'pairs_trading_sample_output.png'")

if __name__ == "__main__":
    main()
