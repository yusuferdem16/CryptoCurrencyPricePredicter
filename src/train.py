import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.data_processing import get_processed_data
from src.models.gru_model import build_gru_model

def train_model(ticker="BTC-USD"):
    print(f"🚀 Starting SHARPSHOOTER training pipeline for {ticker}...")
    
    # TWEAK 1: Shorten Lookback Window
    seq_length = 30
    
    data = get_processed_data(ticker, seq_length=seq_length)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Build Model
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    
    # TWEAK 2: Override with MAE Loss
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mae', metrics=['mae'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    print(f"\n🧠 Training with Sequence Length: {seq_length} days...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,  # TWEAK 3: Larger batch size
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print("\n📊 Converting Predictions back to Price...")
    
    pred_returns_scaled = model.predict(X_test)
    
    target_scaler = data['target_scaler']
    pred_returns = target_scaler.inverse_transform(pred_returns_scaled)
    actual_returns = target_scaler.inverse_transform(y_test)
    
    test_df = data['test_data_original']
    
    # Correct slicing using the variable seq_length
    base_prices = test_df['close'].values[seq_length-1 : -1]
    
    n_samples = len(pred_returns)
    base_prices = base_prices[:n_samples].reshape(-1, 1)
    
    predicted_prices = base_prices * np.exp(pred_returns)
    actual_prices = base_prices * np.exp(actual_returns)
    
    mae = np.mean(np.abs(predicted_prices - actual_prices))
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices[-100:], label='Actual Price', color='blue')
    plt.plot(predicted_prices[-100:], label='Predicted Price', color='red', linestyle='--')
    plt.title(f'Sharpshooter Model (Seq={seq_length}): Actual vs Predicted (MAE: ${mae:.2f})')
    plt.legend()
    plt.show()

    print(f"\n✨ FINAL RESULTS ✨")
    print(f"Baseline MAE to beat: $1595.53")
    print(f"Model MAE: ${mae:.2f}")
    
    # Force save regardless of result
    print(f"💾 Saving model to models/{ticker.lower()}_gru_v4.keras...")
    model.save(f"models/{ticker.lower()}_gru_v4.keras")

if __name__ == "__main__":
    train_model("BTC-USD")