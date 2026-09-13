from src.ingestion import fetch_and_store
from src.feature_engineering import process_data
from src.train import train_model as train_lstm
from src.sarimax_pipeline import train_sarimax
from src.forecasting import generate_forecasts

TICKER = "BTC-USD"
HORIZON_DAYS = 7  # one retrain cycle - see src/forecasting.py


def weekly_retrain():
    """
    Runs on a weekly schedule (see .github/workflows/weekly_retrain.yml).
    Refreshes the data first so both models retrain on the latest history,
    retrains and overwrites the model artifacts in models/, then seeds
    forecasts for the whole week until the next retrain. force=True here
    because the daily job's safety-net call may have already filled part of
    this horizon with the outgoing (pre-retrain) model earlier that same
    day - the fresh model should always win for any date that hasn't
    resolved yet.
    """
    print(f"\n🏋️ Starting weekly retrain for {TICKER}...")

    print("📥 Refreshing data before retrain...")
    fetch_and_store(TICKER)
    process_data(TICKER)

    print("🧠 Retraining Bi-LSTM...")
    train_lstm(TICKER)

    print("📊 Retraining SARIMAX...")
    train_sarimax(TICKER)

    generate_forecasts(TICKER, horizon_days=HORIZON_DAYS, force=True)

    print("✅ Weekly retrain complete.")


if __name__ == "__main__":
    weekly_retrain()
