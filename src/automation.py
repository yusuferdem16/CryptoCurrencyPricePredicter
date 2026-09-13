import pandas as pd
from datetime import datetime
from src.feature_engineering import process_data

# Import our existing pipelines
from src.storage import load_predictions, save_predictions
from src.ingestion import fetch_and_store
from src.data_processing import load_data
from src.forecasting import generate_forecasts

TICKER = "BTC-USD"
HORIZON_DAYS = 7  # matches the weekly retrain cadence - see src/forecasting.py


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


def daily_job():
    """
    Runs once a day: fresh data in, verify past guesses, top up the forecast
    horizon if needed. Model retraining is intentionally NOT part of this
    job - see src/retrain.py, which runs on its own weekly schedule and is
    also what actually generates each week's forecasts. Calling
    generate_forecasts() here is a safety net (first-ever run, or a missed
    retrain) - most days it's a no-op since the horizon is already filled.
    """
    print(f"\n⏰ Waking up! Starting daily cycle for {datetime.now().date()}...")

    # 1. Ingest New Data
    fetch_and_store(TICKER)

    # 1.5. Calculate Features
    print("⚙️ Updating Technical Indicators...")
    process_data(TICKER)

    # 2. Verify Past Predictions
    update_accuracy_metrics()

    # 3. Top Up the Forecast Horizon
    generate_forecasts(TICKER, horizon_days=HORIZON_DAYS)

    print("💤 Cycle complete. Going back to sleep...")


if __name__ == "__main__":
    # GitHub Actions runs this script once and then exits - the cron trigger
    # in the workflow YAML handles the timing.
    print("🤖 GitHub Action Triggered: Starting Daily Cycle...")
    daily_job()
    print("✅ Daily Cycle Finished. Exiting.")
