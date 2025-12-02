import sys
import os

# Add the root directory to sys.path so Python can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import requests
import os
import pandas as pd
import plotly.graph_objects as go
from src.database import engine

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Crypto Sharpshooter", layout="wide")
st.title("⚡️ Crypto Sharpshooter: Model Arena")

# Create Tabs
tab1, tab2 = st.tabs(["🔮 Live Forecast", "📈 Accuracy Tracker"])

ticker = st.sidebar.selectbox("Select Asset", ["BTC-USD"])

# --- TAB 1: The Live Forecast (Existing Code) ---
with tab1:
    if st.button("Run Forecast Models", key="forecast_btn"):
        with st.spinner("Competing Models are Calculating..."):
            try:
                response = requests.get(f"{API_URL}/predict/{ticker}")
                if response.status_code == 200:
                    data = response.json()
                    curr = data['current_price']
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"${curr:,.2f}")
                    
                    delta_lstm = data['pred_lstm'] - curr
                    col2.metric("LSTM (Deep Learning)", f"${data['pred_lstm']:,.2f}", f"{delta_lstm:+.2f}")
                    
                    delta_sari = data['pred_sarimax'] - curr
                    col3.metric("SARIMAX (Statistical)", f"${data['pred_sarimax']:,.2f}", f"{delta_sari:+.2f}")
                    
                    # Visualization (Keep existing plot code here)
                    st.subheader("Forecast Visualization")
                    query = f"SELECT date, close FROM raw_{ticker.lower().replace('-', '_')} ORDER BY date DESC LIMIT 60"
                    history_df = pd.read_sql(query, engine).sort_values('date')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=history_df['date'], y=history_df['close'], name='History', line=dict(color='gray')))
                    
                    last_date = pd.to_datetime(history_df['date'].iloc[-1])
                    next_date = last_date + pd.Timedelta(days=1)
                    
                    fig.add_trace(go.Scatter(x=[last_date, next_date], y=[curr, data['pred_lstm']], mode='lines+markers', name='LSTM', line=dict(color='#FF4B4B', width=3)))
                    fig.add_trace(go.Scatter(x=[last_date, next_date], y=[curr, data['pred_sarimax']], mode='lines+markers', name='SARIMAX', line=dict(color='#1E88E5', width=3, dash='dot')))
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# --- TAB 2: Accuracy Tracker (NEW!) ---
with tab2:
    st.header("🔍 Model Performance Ledger")
    
    # 1. Fetch Data (Now including mape)
    query = f"""
        SELECT predicted_date, model_version, predicted_price, actual_price, mae, mape 
        FROM predictions 
        WHERE ticker = '{ticker}' 
        ORDER BY predicted_date DESC
    """
    df_preds = pd.read_sql(query, engine)
    
    if not df_preds.empty:
        df_preds['predicted_date'] = pd.to_datetime(df_preds['predicted_date']).dt.date
        
        # Ensure numeric types
        cols = ['actual_price', 'mae', 'mape']
        for c in cols:
            df_preds[c] = pd.to_numeric(df_preds[c])

        # Custom formatters
        def fmt_price(x): return f"${x:,.2f}" if pd.notnull(x) else "⏳ Pending"
        def fmt_mape(x): return f"{x:.2f}%" if pd.notnull(x) else "—"

        # Apply Styling
        st.dataframe(
            df_preds.style.format({
                "predicted_price": "${:,.2f}",
                "actual_price": fmt_price,
                "mae": fmt_price,
                "mape": fmt_mape
            }).background_gradient(subset=['mape'], cmap='RdYlGn_r', vmin=0, vmax=5), # Green if <5% error
            use_container_width=True
        )
        
        # 2. Scoreboard
        st.subheader("🏆 Overall Scoreboard")
        df_finished = df_preds.dropna(subset=['mape'])
        
        if not df_finished.empty:
            avg_metrics = df_finished.groupby('model_version')[['mae', 'mape']].mean().reset_index()
            
            col_a, col_b = st.columns(2)
            
            # LSTM Stats
            lstm_row = avg_metrics[avg_metrics['model_version'].str.contains('LSTM')]
            if not lstm_row.empty:
                mae = lstm_row['mae'].values[0]
                mape = lstm_row['mape'].values[0]
                col_a.metric("LSTM Average Error", f"${mae:,.2f}", f"MAPE: {mape:.2f}%", delta_color="inverse")
                
            # SARIMAX Stats
            sari_row = avg_metrics[avg_metrics['model_version'].str.contains('SARIMAX')]
            if not sari_row.empty:
                mae = sari_row['mae'].values[0]
                mape = sari_row['mape'].values[0]
                col_b.metric("SARIMAX Average Error", f"${mae:,.2f}", f"MAPE: {mape:.2f}%", delta_color="inverse")
        else:
            st.info("Waiting for data validation to calculate scores.")
            
    else:
        st.info("No predictions yet.")