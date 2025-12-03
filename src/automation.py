import schedule
import time
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sqlalchemy import text
from tensorflow.keras.models import load_model
from src.feature_engineering import process_data  # <--- NEW IMPORT

# Import our existing pipelines
from src.database import engine
from src.ingestion import fetch_and_store
from src.data_processing import get_processed_data, load_data
from src.train import train_model as train_lstm
from src.sarimax_pipeline import train_sarimax

TICKER = "BTC-USD"

def update_accuracy_metrics():
    """
    Step 2: Check past predictions and fill in the 'Actual', 'MAE', and 'MAPE'.
    """
    print("🔍 Verifying past predictions...")
    
    with engine.connect() as conn:
        # Get pending predictions
        query = text(f"""
            SELECT id, predicted_date, predicted_price 
            FROM predictions 
            WHERE ticker = '{TICKER}' AND actual_price IS NULL
        """)
        pending = pd.read_sql(query, conn)
        
        if pending.empty:
            print("   No pending predictions to verify.")
            return

        # Get latest data
        df_history = load_data(TICKER)
        
        if 'date' in df_history.columns:
            df_history['date'] = pd.to_datetime(df_history['date'])
        else:
            df_history = df_history.reset_index()
            df_history['date'] = pd.to_datetime(df_history['date'])

        for _, row in pending.iterrows():
            pred_date = pd.to_datetime(row['predicted_date']).date()
            
            # Find the match
            match = df_history[df_history['date'].dt.date == pred_date]
            
            if not match.empty:
                actual_price = match['close'].values[0]
                
                # --- CALCULATE METRICS ---
                error_mae = abs(actual_price - row['predicted_price'])
                error_mape = (error_mae / actual_price) * 100  # <--- NEW CALCULATION
                
                # Update DB with BOTH metrics
                update_query = text(f"""
                    UPDATE predictions 
                    SET actual_price = {actual_price}, 
                        mae = {error_mae}, 
                        mape = {error_mape}
                    WHERE id = {row['id']}
                """)
                conn.execute(update_query)
                conn.commit()
                print(f"   ✅ Verified {pred_date}: Actual=${actual_price:.2f}, MAE=${error_mae:.2f}, MAPE={error_mape:.2f}%")

def generate_daily_forecast():
    """
    Step 4: Make a NEW prediction for tomorrow and save it.
    """
    print("🔮 Generating new forecast for tomorrow...")
    
    # --- Load Models ---
    lstm_path = f"models/{TICKER.lower()}_gru_v4.keras"
    sarimax_path = f"models/{TICKER.lower()}_sarimax.pkl"
    
    if not os.path.exists(lstm_path) or not os.path.exists(sarimax_path):
        print("   ⚠️ Models not found. Skipping forecast.")
        return

    lstm_model = load_model(lstm_path)
    sarimax_model = joblib.load(sarimax_path)

    # --- Prepare Data ---
    # 1. LSTM Input (Last 30 days sequences)
    data_lstm = get_processed_data(TICKER, seq_length=30)
    X_input = data_lstm['X_test'][-1:] 
    
    # 2. SARIMAX Input (Last row features)
    df_raw = load_data(TICKER)
    feature_cols = ['volume_log_return', 'rsi', 'bb_position', 'macd_norm', 'momentum_7d']
    X_sarimax = df_raw.iloc[-1:][feature_cols]

    # --- Predict ---
    # LSTM
    pred_scaled = lstm_model.predict(X_input)
    target_scaler = data_lstm['target_scaler']
    lstm_log_return = target_scaler.inverse_transform(pred_scaled)[0][0]
    
    # SARIMAX
    sarimax_log_return = sarimax_model.predict(n_periods=1, X=X_sarimax).iloc[0]

    # Convert to Price
    current_price = df_raw['close'].iloc[-1]
    price_lstm = float(current_price * np.exp(lstm_log_return))
    price_sarimax = float(current_price * np.exp(sarimax_log_return))

    # --- Save to DB ---
    # FIX: Use the date column to find "Tomorrow"
    # Ensure date is parsed correctly
    last_date_ts = pd.to_datetime(df_raw['date'].iloc[-1])
    tomorrow = (last_date_ts + timedelta(days=1)).date()
    
    with engine.connect() as conn:
        # Insert LSTM Forecast
        conn.execute(text(f"""
            INSERT INTO predictions (timestamp, ticker, model_version, predicted_date, predicted_price)
            VALUES (NOW(), '{TICKER}', 'LSTM_BiDir_v4', '{tomorrow}', {price_lstm})
        """))
        
        # Insert SARIMAX Forecast
        conn.execute(text(f"""
            INSERT INTO predictions (timestamp, ticker, model_version, predicted_date, predicted_price)
            VALUES (NOW(), '{TICKER}', 'SARIMAX_v1', '{tomorrow}', {price_sarimax})
        """))
        conn.commit()
    
    print(f"   💾 Saved forecasts for {tomorrow}: LSTM=${price_lstm:.2f}, SARIMAX=${price_sarimax:.2f}")

def daily_job():
    print(f"\n⏰ Waking up! Starting daily cycle for {datetime.now().date()}...")
    
    # 1. Ingest New Data
    fetch_and_store(TICKER)
    
    # 1.5. Calculate Features (CRITICAL STEP)
    print("⚙️ Updating Technical Indicators...")
    process_data(TICKER)  # <--- THIS WAS MISSING
    
    # 2. Verify Yesterday's Prediction
    update_accuracy_metrics()
    
    # 3. Retrain Models
    print("🏋️ Retraining Models...")
    try:
        train_lstm(TICKER)
        train_sarimax(TICKER)
    except Exception as e:
        print(f"   ⚠️ Retraining failed: {e}")
    
    # 4. Predict Tomorrow
    generate_daily_forecast()
    
    print("💤 Cycle complete. Going back to sleep...")

if __name__ == "__main__":
    # GitHub Actions will run this script once and then exit.
    # We don't need schedule loops or "smart checks" here anymore 
    # because the CRON trigger in YAML handles the timing perfectly.
    
    print("🤖 GitHub Action Triggered: Starting Daily Cycle...")
    daily_job()
    print("✅ Daily Cycle Finished. Exiting.")