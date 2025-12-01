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
    st.markdown("This table tracks every prediction made by the automated system and verifies it against reality the next day.")
    
    # 1. Fetch Data
    query = f"""
        SELECT predicted_date, model_version, predicted_price, actual_price, mae 
        FROM predictions 
        WHERE ticker = '{ticker}' 
        ORDER BY predicted_date DESC
    """
    df_preds = pd.read_sql(query, engine)
    
    if not df_preds.empty:
        # Format date
        df_preds['predicted_date'] = pd.to_datetime(df_preds['predicted_date']).dt.date
        
        # FIX 1: Ensure columns are numeric (converts SQL NULL to Pandas NaN)
        df_preds['actual_price'] = pd.to_numeric(df_preds['actual_price'])
        df_preds['mae'] = pd.to_numeric(df_preds['mae'])

        # FIX 2: Custom formatter that handles missing (NaN) values safely
        def safe_format(val):
            return f"${val:,.2f}" if pd.notnull(val) else "⏳ Pending"

        # Apply Styling with the safe formatter
        st.dataframe(
            df_preds.style.format({
                "predicted_price": "${:,.2f}",
                "actual_price": safe_format,
                "mae": safe_format
            }).background_gradient(subset=['mae'], cmap='RdYlGn_r', vmin=0, vmax=2000), 
            use_container_width=True
        )
        
        # 2. Scoreboard
        st.subheader("🏆 Overall Scoreboard")
        
        # Filter out "Pending" rows before calculating averages
        df_finished = df_preds.dropna(subset=['mae'])
        
        if not df_finished.empty:
            avg_mae = df_finished.groupby('model_version')['mae'].mean().reset_index()
            
            col_a, col_b = st.columns(2)
            
            # Safe extraction of values
            lstm_rows = avg_mae[avg_mae['model_version'].str.contains('LSTM')]
            if not lstm_rows.empty:
                col_a.metric("LSTM Average MAE", f"${lstm_rows['mae'].values[0]:,.2f}")
                
            sari_rows = avg_mae[avg_mae['model_version'].str.contains('SARIMAX')]
            if not sari_rows.empty:
                col_b.metric("SARIMAX Average MAE", f"${sari_rows['mae'].values[0]:,.2f}")
        else:
            st.info("Waiting for tomorrow's close price to calculate first accuracy scores.")
            
    else:
        st.info("No automated predictions recorded yet. Run the 'daily_job' script to start tracking.")