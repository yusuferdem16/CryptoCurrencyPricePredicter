import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pmdarima as pm
from src.data_processing import load_data
from sklearn.metrics import mean_absolute_error
from src.database import engine

def train_sarimax(ticker="BTC-USD"):
    print(f"📊 Starting SARIMAX Training for {ticker}...")
    
    # 1. Load Data
    # We use the raw feature dataframe, not the sequence-processed one
    # SARIMAX doesn't need 'sequences' like LSTM, it needs a 2D matrix
    df = load_data(ticker)
    
    # 2. Prepare Data
    # Target: 'target_next_return'
    # Exogenous Features (The "X" in SARIMAX): RSI, Momentum, etc.
    feature_cols = ['volume_log_return', 'rsi', 'bb_position', 'macd_norm', 'momentum_7d']
    target_col = 'target_next_return'
    
    # Train/Test Split (Time-based, same 80/20 ratio)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    y_train = train_df[target_col]
    X_train = train_df[feature_cols]
    
    y_test = test_df[target_col]
    X_test = test_df[feature_cols]
    
    print(f"   Train Size: {len(y_train)} days")
    print(f"   Test Size: {len(y_test)} days")
    
    # 3. Auto-ARIMA Search (The "Magic")
    # It tries different combinations of p, d, q to minimize AIC (error metric)
    print("\n🔍 Searching for best ARIMA parameters (this takes time)...")
    model = pm.auto_arima(
        y=y_train,
        X=X_train,
        start_p=1, start_q=1,
        max_p=3, max_q=3, # Limit complexity to prevent overfitting
        d=0,              # Data is already stationary (log returns), so d=0
        seasonal=False,   # Crypto generally doesn't have weekly seasonality like retail sales
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True        # Prints progress
    )
    
    print(f"\n✅ Best Model Found: {model.order}")
    print(model.summary())
    
    # 4. Evaluate on Test Set
    # We must predict loop-wise or use the built-in predict with X_test
    print("\n📉 Generating Forecasts...")
    preds = model.predict(n_periods=len(y_test), X=X_test)
    
    # 5. Convert Log Returns back to Price for MAE calculation
    # We need the "base price" (Price of the day BEFORE prediction)
    # The test_df contains the 'close' price of "Today". 
    # Since y_test is "Target Return (Tomorrow)", the base price is "Close (Today)"
    base_prices = test_df['close'].values
    
    # Formula: Price_Tomorrow = Price_Today * exp(Predicted_Return)
    predicted_prices = base_prices * np.exp(preds)
    
    # Actual Price Tomorrow?
    # We don't have it in the dataframe row directly (we shifted).
    # We can reconstruct it: Actual_Tomorrow = Price_Today * exp(Actual_Return)
    actual_prices = base_prices * np.exp(y_test.values)
    
    mae = mean_absolute_error(actual_prices, predicted_prices)
    
    print(f"\n✨ SARIMAX RESULTS ✨")
    print(f"SARIMAX MAE: ${mae:.2f}")
    
    # 6. Save Model
    # We use joblib because pmdarima wraps statsmodels, which pickle handles well

 
    model_path = f"models/{ticker.lower()}_sarimax.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")

    # 7. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices[-100:], label='Actual Price', color='blue')
    plt.plot(predicted_prices[-100:], label='SARIMAX Forecast', color='green', linestyle='--')
    plt.title(f'SARIMAX Model: Actual vs Predicted (MAE: ${mae:.2f})')
    plt.legend()
    # plt.show()

if __name__ == "__main__":
    train_sarimax("BTC-USD")