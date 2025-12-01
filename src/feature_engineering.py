import pandas as pd
import numpy as np
from sqlalchemy import text
from src.database import engine

def calculate_rsi(data, window=14):
    """Relative Strength Index (Momentum Indicator)"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data):
    """Moving Average Convergence Divergence (Trend Indicator)"""
    # Short-term EMA (Fast)
    short_ema = data['close'].ewm(span=12, adjust=False).mean()
    # Long-term EMA (Slow)
    long_ema = data['close'].ewm(span=26, adjust=False).mean()
    
    # MACD Line
    macd = short_ema - long_ema
    # Signal Line
    signal = macd.ewm(span=9, adjust=False).mean()
    
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    """Bollinger Bands (Volatility Indicator)"""
    sma = data['close'].rolling(window=window).mean()
    std = data['close'].rolling(window=window).std()
    
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    
    return upper_band, lower_band

def process_data(ticker):
    print(f"⚙️ Processing data for {ticker}...")
    
    # 1. Load Raw Data
    raw_table = f"raw_{ticker.lower().replace('-', '_')}"
    query = f"SELECT * FROM {raw_table} ORDER BY date ASC"
    df = pd.read_sql(query, engine)
    
    # 2. Add Technical Indicators
    df['rsi'] = calculate_rsi(df)
    df['macd'], df['macd_signal'] = calculate_macd(df)
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df)
    
    # --- 3. STATIONARITY TRANSFORMS (The Fix) ---
    
    # Price -> Log Return (The Speed of price)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volume -> Log Return (The Speed of volume)
    # Add 1e-9 to avoid log(0) errors
    df['volume_log_return'] = np.log((df['volume'] + 1e-9) / (df['volume'].shift(1) + 1e-9))
    
    # Bollinger Bands -> Position (0 to 1 scale)
    # (Price - Lower) / (Upper - Lower)
    # > 1.0 means broke top band, < 0.0 means broke bottom band
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-9)
    
    # MACD -> Normalized (MACD / Price)
    # Because MACD grows as price grows, we divide by close to keep it relative
    df['macd_norm'] = df['macd'] / df['close']
    
    # NEW: 7-Day Rolling Momentum
    # This tells the model: "Is the price generally higher than last week?"
    df['momentum_7d'] = df['log_return'].rolling(window=7).mean()
    
    # Target (Next day's return)
    df['target_next_return'] = df['log_return'].shift(-1)
    
    # 4. Clean NaN
    df.dropna(inplace=True)
    
    # 5. Save
    feature_table = f"features_{ticker.lower().replace('-', '_')}"
    df.to_sql(feature_table, engine, index=False, if_exists='replace')
    
    print(f"✅ Saved {len(df)} rows. New features: bb_position, volume_log_return, macd_norm, momentum_7d.")

if __name__ == "__main__":
    process_data("BTC-USD")
    # process_data("ETH-USD")