import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from src.database import engine

def load_data(table_name):
    print(f"Loading data from {table_name}...")
    query = f"SELECT * FROM {table_name} ORDER BY date ASC"
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def plot_price_and_volatility(df, ticker_name):
    plt.figure(figsize=(14, 7))
    
    # 1. Price Plot
    plt.subplot(2, 1, 1)
    plt.plot(df['close'], label='Close Price', color='blue')
    plt.title(f'{ticker_name} Price History')
    plt.legend()
    plt.grid(True)
    
    # 2. Daily Returns (Volatility)
    # We calculate percent change to see how volatile the market is
    df['returns'] = df['close'].pct_change()
    
    plt.subplot(2, 1, 2)
    plt.plot(df['returns'], label='Daily Returns', color='orange', alpha=0.7)
    plt.title(f'{ticker_name} Daily Volatility')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def check_stationarity(series):
    """
    Performs Augmented Dickey-Fuller test to check if data has a trend.
    Model needs Stationary data (no trend) to learn patterns effectively.
    """
    result = adfuller(series.dropna())
    print("\n--- Augmented Dickey-Fuller Test Results ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    
    if result[1] <= 0.05:
        print("✅ Result: Data is Stationary (Good for inference)")
    else:
        print("❌ Result: Data is Non-Stationary (Trend present - needs processing)")

if __name__ == "__main__":
    # Analyze Bitcoin
    ticker = "raw_btc_usd"
    df = load_data(ticker)
    
    print(df.head())
    
    # Check if the price series is stationary
    check_stationarity(df['close'])
    
    # Visualize
    plot_price_and_volatility(df, "Bitcoin")