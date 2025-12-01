from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_gru_model(input_shape):
    # We keep the function name 'build_gru_model' so train.py doesn't break
    model = Sequential([
        Input(shape=input_shape),
        
        # Layer 1: Bi-Directional LSTM
        # Reads data forwards and backwards. L2 regularizer stops it from memorizing noise.
        Bidirectional(LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.001))),
        Dropout(0.3),
        
        # Layer 2: Standard LSTM
        LSTM(units=32, return_sequences=False),
        Dropout(0.3),
        
        # Output
        Dense(units=1)
    ])
    
    # Slower learning rate (0.0001) for the complex model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    
    return model