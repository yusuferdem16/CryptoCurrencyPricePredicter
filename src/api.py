import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from src.data_processing import get_processed_data, load_data
import asyncio
from contextlib import asynccontextmanager
import threading
import schedule
import time
from src.automation import daily_job  # Import your job logic

# 1. Define the Scheduler Loop
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

# 2. Start Scheduler on API Startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the job
    schedule.every().day.at("08:00").do(daily_job)
    print("⏰ Scheduler started within API...")
    
    # Run loop in a separate thread so it doesn't block the API
    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
    
    yield
    # (Cleanup code would go here if needed)

# 3. Attach lifespan to App
app = FastAPI(title="Crypto Forecaster API", version="2.0", lifespan=lifespan)
app = FastAPI(title="Crypto Forecaster API", version="2.0")

class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    pred_lstm: float
    pred_sarimax: float
    lstm_direction: str
    sarimax_direction: str

@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict(ticker: str):
    try:
        # --- 1. Load Models ---
        lstm_path = f"models/{ticker.lower()}_gru_v4.keras"
        sarimax_path = f"models/{ticker.lower()}_sarimax.pkl"
        
        if not os.path.exists(lstm_path) or not os.path.exists(sarimax_path):
            raise HTTPException(status_code=404, detail="Models not found. Train them first!")
            
        lstm_model = load_model(lstm_path)
        sarimax_model = joblib.load(sarimax_path)

        # --- 2. LSTM Prediction ---
        # Get processed sequences
        seq_length = 30
        data_lstm = get_processed_data(ticker, seq_length=seq_length)
        X_input = data_lstm['X_test'][-1:] # Last 30 days
        
        # Predict Log Return
        pred_scaled = lstm_model.predict(X_input)
        target_scaler = data_lstm['target_scaler']
        lstm_log_return = target_scaler.inverse_transform(pred_scaled)[0][0]
        
        # --- 3. SARIMAX Prediction ---
        # Get raw features (Last row of data)
        df_raw = load_data(ticker)
        
        # Prepare exogenous features for SARIMAX
        # Must match training cols: ['volume_log_return', 'rsi', 'bb_position', 'macd_norm', 'momentum_7d']
        feature_cols = ['volume_log_return', 'rsi', 'bb_position', 'macd_norm', 'momentum_7d']
        X_sarimax = df_raw.iloc[-1:][feature_cols]
        
        # Predict Log Return
        sarimax_log_return = sarimax_model.predict(n_periods=1, X=X_sarimax).iloc[0]
        
        # --- 4. Convert to Prices ---
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
    
@app.get("/health")
def health_check():
    return {"status": "healthy"}