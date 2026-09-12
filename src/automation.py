import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from src.feature_engineering import process_data

# Import our existing pipelines
from src.storage import load_predictions, save_predictions
from src.ingestion import fetch_and_store
from src.data_processing import get_processed_data, load_data

TICKER = "BTC-USD"


def update_accuracy_metrics():
    """
    Check past predictions and fill in the 'Actual', 'MAE', and 'MAPE'.
    """
    print("🔍 Verifying past predictions...")

    records = load_predictions()
    pending = [r for r in records if r["ticker"] == TICKER and r.get("actual_price") is None]

    if not pending:
        print("   No pending predictions to verify.")
        return

    # Get latest data
    df_history = load_data(TICKER)

    if 'date' in df_history.columns:
        df_history['date'] = pd.to_datetime(df_history['date'])
    else:
        df_history = df_history.reset_index()
        df_history['date'] = pd.to_datetime(df_history['date'])

    changed = False
    for row in pending:
        pred_date = pd.to_datetime(row['predicted_date']).date()

        # Find the match
        match = df_history[df_history['date'].dt.date == pred_date]

        if not match.empty:
            actual_price = float(match['close'].values[0])

            # --- CALCULATE METRICS ---
            error_mae = abs(actual_price - row['predicted_price'])
            error_mape = (error_mae / actual_price) * 100

            row['actual_price'] = actual_price
            row['mae'] = error_mae
            row['mape'] = error_mape
            changed = True
            print(f"   ✅ Verified {pred_date}: Actual=${actual_price:.2f}, MAE=${error_mae:.2f}, MAPE={error_mape:.2f}%")

    if changed:
        save_predictions(records)


def generate_daily_forecast():
    """
    Make a NEW prediction for tomorrow and save it.
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

    # --- Save Forecast (Idempotent) ---
    last_date_ts = pd.to_datetime(df_raw['date'].iloc[-1])
    tomorrow = (last_date_ts + timedelta(days=1)).date().isoformat()

    records = load_predictions()
    # Drop any existing prediction for this ticker/date so we overwrite with the fresh one
    records = [r for r in records if not (r['ticker'] == TICKER and r['predicted_date'] == tomorrow)]

    now = datetime.utcnow().isoformat()
    records.append({
        "timestamp": now,
        "ticker": TICKER,
        "model_version": "LSTM_BiDir_v4",
        "predicted_date": tomorrow,
        "predicted_price": price_lstm,
        "actual_price": None,
        "mae": None,
        "mape": None,
    })
    records.append({
        "timestamp": now,
        "ticker": TICKER,
        "model_version": "SARIMAX_v1",
        "predicted_date": tomorrow,
        "predicted_price": price_sarimax,
        "actual_price": None,
        "mae": None,
        "mape": None,
    })

    save_predictions(records)
    print(f"   💾 Saved forecasts for {tomorrow} (Overwrote previous if existed).")


def daily_job():
    """
    Runs once a day: fresh data in, verify yesterday's guess, forecast tomorrow.
    Model retraining is intentionally NOT part of this job - see src/retrain.py,
    which runs on its own weekly schedule so the daily run stays fast and cheap.
    """
    print(f"\n⏰ Waking up! Starting daily cycle for {datetime.now().date()}...")

    # 1. Ingest New Data
    fetch_and_store(TICKER)

    # 1.5. Calculate Features
    print("⚙️ Updating Technical Indicators...")
    process_data(TICKER)

    # 2. Verify Yesterday's Prediction
    update_accuracy_metrics()

    # 3. Predict Tomorrow
    generate_daily_forecast()

    print("💤 Cycle complete. Going back to sleep...")


if __name__ == "__main__":
    # GitHub Actions runs this script once and then exits - the cron trigger
    # in the workflow YAML handles the timing.
    print("🤖 GitHub Action Triggered: Starting Daily Cycle...")
    daily_job()
    print("✅ Daily Cycle Finished. Exiting.")
