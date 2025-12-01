import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import threading
import schedule
import time
import os
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from src.data_processing import get_processed_data, load_data
from src.automation import daily_job

# --- 1. SCHEDULER LOGIC ---
def run_scheduler():
    """Runs the schedule loop in a background thread."""
    while True:
        schedule.run_pending()
        time.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Schedule the job and launch the thread
    schedule.every().day.at("08:00").do(daily_job)
    print("⏰ Scheduler started within API...")
    
    # Run loop in a separate thread so it doesn't block the API
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
    
    yield
    # Shutdown logic (if needed)

# --- 2. API SETUP ---
app = FastAPI(title="Crypto Forecaster API", version="2.0", lifespan=lifespan)

class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    pred_lstm: float
    pred_sarimax: float
    lstm_direction: str
    sarimax_direction: str

# --- 3. ENDPOINTS ---

@app.get("/health", include_in_schema=False)
@app.head("/health", include_in_schema=False)
async def health():
    return {"status": "healthy", "scheduler": "running"}


@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict(ticker: str):
    try:
        # Load Models
        lstm_path = f"models/{ticker.lower()}_gru_v4.keras"
        sarimax_path = f"models/{ticker.lower()}_sarimax.pkl"
        
        if not os.path.exists(lstm_path) or not os.path.exists(sarimax_path):
            raise HTTPException(status_code=404, detail="Models not found. Train them first!")
            
        lstm_model = load_model(lstm_path)
        sarimax_model = joblib.load(sarimax_path)

        # LSTM Prediction
        seq_length = 30
        data_lstm = get_processed_data(ticker, seq_length=seq_length)
        X_input = data_lstm['X_test'][-1:] 
        
        pred_scaled = lstm_model.predict(X_input)
        target_scaler = data_lstm['target_scaler']
        lstm_log_return = target_scaler.inverse_transform(pred_scaled)[0][0]
        
        # SARIMAX Prediction
        df_raw = load_data(ticker)
        feature_cols = ['volume_log_return', 'rsi', 'bb_position', 'macd_norm', 'momentum_7d']
        X_sarimax = df_raw.iloc[-1:][feature_cols]
        
        # Use .iloc[0] to safely get the value
        sarimax_log_return = sarimax_model.predict(n_periods=1, X=X_sarimax).iloc[0]
        
        # Convert to Prices
        current_price = df_raw['close'].iloc[-1]
        price_lstm = current_price * np.exp(lstm_log_return)
        price_sarimax = current_price * np.exp(sarimax_log_return)
        
        return {
            "ticker": ticker,
            "current_price": float(current_price),
            "pred_lstm": float(price_lstm),
            "pred_sarimax": float(price_sarimax),
            "lstm_direction": "UP" if price_lstm > current_price else "DOWN",
            "sarimax_direction": "UP" if price_sarimax > current_price else "DOWN"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))