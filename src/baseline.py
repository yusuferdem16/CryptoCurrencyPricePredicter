import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_processing import load_data

def run_baseline(ticker):
    print(f"📊 Running Baseline Evaluation for {ticker}...")
    
    # Load raw data (we don't need scaling for baseline)
    df = load_data(ticker)
    
    # Split Data (Simple 80/20 split)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    # --- MODEL 1: Naive Forecast ---
    # Prediction: Tomorrow's price = Today's 'close' price
    # In our dataframe, 'close' is today. 'target_next_close' is tomorrow.
    test_df['pred_naive'] = test_df['close']
    
    # Calculate Error
    mae = mean_absolute_error(test_df['target_next_close'], test_df['pred_naive'])
    rmse = np.sqrt(mean_squared_error(test_df['target_next_close'], test_df['pred_naive']))
    
    print("\n--- 🐢 Baseline Results (Naive Forecast) ---")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Interpretation: On average, guessing 'same price' is off by ${mae:.2f}.")
    print("--------------------------------------------\n")
    
    return mae

if __name__ == "__main__":
    btc_mae = run_baseline("BTC-USD")