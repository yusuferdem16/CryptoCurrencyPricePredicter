import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.database import engine

def load_data(ticker):
    table_name = f"features_{ticker.lower().replace('-', '_')}"
    query = f"SELECT * FROM {table_name} ORDER BY date ASC"
    df = pd.read_sql(query, engine)
    df.dropna(inplace=True)
    return df

def create_sequences(data, seq_length=60):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        # Predict the NEXT step's return (index + seq_length)
        y = data[i + seq_length] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_processed_data(ticker, seq_length=60, split_ratio=0.8):
    df = load_data(ticker)

    # 1. Create the Target: Next Day's Log Return
    # We want to predict what the log_return WILL BE tomorrow.
    # Current 'log_return' column is Today vs Yesterday. 
    # So we shift it backward by 1 to align "Today's Features" with "Tomorrow's Return"
    df['target_return'] = df['log_return'].shift(-1)
    df.dropna(inplace=True)

    # 2. Split (Time-based)
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # 3. Feature Selection
    # We train on everything, including recent price/vol/returns
    # 3. Feature Selection
    # REMOVED: 'close', 'volume', 'bb_upper', 'bb_lower', 'macd'
    # ADDED: Relative versions only
    feature_cols = [
        'log_return', 
        'volume_log_return', 
        'rsi', 
        'bb_position', 
        'macd_norm',
        'momentum_7d'  # <--- NEW
    ]
    # 4. Scaling (StandardScaler is better for Returns than MinMax)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # Target is already a small float (e.g., 0.02), but scaling helps convergence
    target_scaler = StandardScaler()
    # We need to reshape to (N, 1) for scaler
    y_train_raw = train_df[['target_return']].values
    y_test_raw = test_df[['target_return']].values
    
    train_target_scaled = target_scaler.fit_transform(y_train_raw)
    test_target_scaled = target_scaler.transform(y_test_raw)

    # 5. Sequence Creation
    X_train, _ = create_sequences(train_scaled, seq_length)
    X_test, _ = create_sequences(test_scaled, seq_length)
    
    # Align Targets
    # Sequence i ends at row (i + seq_len - 1). 
    # We want to predict target at row (i + seq_len).
    # create_sequences loop handles index alignment, so we just take the y-arrays directly
    _, y_train_seq = create_sequences(train_target_scaled, seq_length)
    _, y_test_seq = create_sequences(test_target_scaled, seq_length)

    return {
        'X_train': X_train,
        'y_train': y_train_seq,
        'X_test': X_test,
        'y_test': y_test_seq,
        'scaler': scaler,
        'target_scaler': target_scaler,
        'test_data_original': test_df, # Crucial for reconstructing price later
        'feature_cols': feature_cols
    }