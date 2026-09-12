from src.ingestion import fetch_and_store
from src.feature_engineering import process_data
from src.train import train_model as train_lstm
from src.sarimax_pipeline import train_sarimax

TICKER = "BTC-USD"


def weekly_retrain():
    """
    Runs on a weekly schedule (see .github/workflows/weekly_retrain.yml).
    Refreshes the data first so both models retrain on the latest history,
    then retrains and overwrites the model artifacts in models/.
    """
    print(f"\n🏋️ Starting weekly retrain for {TICKER}...")

    print("📥 Refreshing data before retrain...")
    fetch_and_store(TICKER)
    process_data(TICKER)

    print("🧠 Retraining Bi-LSTM...")
    train_lstm(TICKER)

    print("📊 Retraining SARIMAX...")
    train_sarimax(TICKER)

    print("✅ Weekly retrain complete.")


if __name__ == "__main__":
    weekly_retrain()
