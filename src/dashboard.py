import sys
import os

# Add the root directory to sys.path so Python can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.storage import read_table, load_predictions

st.set_page_config(page_title="Crypto Sharpshooter", layout="wide")
st.title("⚡️ Crypto Sharpshooter: Model Arena")

# Create Tabs
tab1, tab2 = st.tabs(["🔮 Live Forecast", "📈 Accuracy Tracker"])

ticker = st.sidebar.selectbox("Select Asset", ["BTC-USD"])
raw_table = f"raw_{ticker.lower().replace('-', '_')}"

history_df = read_table(raw_table)
predictions = [r for r in load_predictions() if r["ticker"] == ticker]

if history_df is None or history_df.empty:
    st.info(
        "No data yet. The daily GitHub Actions job populates data/ on its first run - "
        "check back after it has run once."
    )
else:
    history_df = history_df.sort_values("date")
    current_price = float(history_df["close"].iloc[-1])
    last_date = pd.to_datetime(history_df["date"].iloc[-1])

    # Most recent forecast (highest predicted_date) per model
    latest_by_model = {}
    for rec in predictions:
        model = rec["model_version"]
        if model not in latest_by_model or rec["predicted_date"] > latest_by_model[model]["predicted_date"]:
            latest_by_model[model] = rec

    lstm_pred = next((r for m, r in latest_by_model.items() if "LSTM" in m), None)
    sarimax_pred = next((r for m, r in latest_by_model.items() if "SARIMAX" in m), None)

    # --- TAB 1: The Live Forecast ---
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:,.2f}")

        if lstm_pred:
            delta_lstm = lstm_pred["predicted_price"] - current_price
            col2.metric("LSTM (Deep Learning)", f"${lstm_pred['predicted_price']:,.2f}", f"{delta_lstm:+.2f}")
        else:
            col2.metric("LSTM (Deep Learning)", "—")

        if sarimax_pred:
            delta_sari = sarimax_pred["predicted_price"] - current_price
            col3.metric("SARIMAX (Statistical)", f"${sarimax_pred['predicted_price']:,.2f}", f"{delta_sari:+.2f}")
        else:
            col3.metric("SARIMAX (Statistical)", "—")

        if lstm_pred and sarimax_pred:
            st.caption(
                f"Forecast generated {lstm_pred['timestamp']} UTC for {lstm_pred['predicted_date']}. "
                "Refreshed automatically once a day by GitHub Actions."
            )

            st.subheader("Forecast Visualization")
            recent_history = history_df.tail(60)

            next_date = pd.to_datetime(lstm_pred["predicted_date"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent_history["date"], y=recent_history["close"],
                name="History", line=dict(color="gray"),
            ))
            fig.add_trace(go.Scatter(
                x=[last_date, next_date], y=[current_price, lstm_pred["predicted_price"]],
                mode="lines+markers", name="LSTM", line=dict(color="#FF4B4B", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=[last_date, next_date], y=[current_price, sarimax_pred["predicted_price"]],
                mode="lines+markers", name="SARIMAX", line=dict(color="#1E88E5", width=3, dash="dot"),
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No forecast saved yet. It will appear after the next daily run.")

    # --- TAB 2: Accuracy Tracker ---
    with tab2:
        st.header("🔍 Model Performance Ledger")

        if predictions:
            df_preds = pd.DataFrame(predictions).sort_values("predicted_date", ascending=False)
            for col in ["actual_price", "mae", "mape"]:
                df_preds[col] = pd.to_numeric(df_preds[col], errors="coerce")

            def fmt_price(x):
                return f"${x:,.2f}" if pd.notnull(x) else "⏳ Pending"

            def fmt_mape(x):
                return f"{x:.2f}%" if pd.notnull(x) else "—"

            mape_numeric = df_preds["mape"]
            display_df = df_preds[["predicted_date", "model_version", "predicted_price", "actual_price", "mae", "mape"]].copy()
            display_df["predicted_price"] = display_df["predicted_price"].map(fmt_price)
            display_df["actual_price"] = display_df["actual_price"].map(fmt_price)
            display_df["mae"] = display_df["mae"].map(fmt_price)
            display_df["mape"] = display_df["mape"].map(fmt_mape)

            # Color the mape column by the original numeric value, manually - avoids
            # relying on Styler.format()'s na_rep and background_gradient's NaN handling,
            # neither of which this Streamlit/pandas combo passes through st.dataframe()
            # correctly (na_rep silently dropped, NaN gmap cells rendered solid black).
            cmap = matplotlib.colormaps["RdYlGn_r"]

            def mape_color(_col):
                colors = []
                for val in mape_numeric:
                    if pd.isnull(val):
                        colors.append("")
                    else:
                        r, g, b, _ = cmap(min(max(val / 5, 0), 1))
                        colors.append(f"background-color: rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})")
                return colors

            st.dataframe(
                display_df.style.apply(mape_color, subset=["mape"]),
                use_container_width=True,
            )

            st.subheader("🏆 Overall Scoreboard")
            df_finished = df_preds.dropna(subset=["mape"])

            if not df_finished.empty:
                avg_metrics = df_finished.groupby("model_version")[["mae", "mape"]].mean().reset_index()

                col_a, col_b = st.columns(2)

                lstm_row = avg_metrics[avg_metrics["model_version"].str.contains("LSTM")]
                if not lstm_row.empty:
                    mae = lstm_row["mae"].values[0]
                    mape = lstm_row["mape"].values[0]
                    col_a.metric("LSTM Average Error", f"${mae:,.2f}", f"MAPE: {mape:.2f}%", delta_color="inverse")

                sari_row = avg_metrics[avg_metrics["model_version"].str.contains("SARIMAX")]
                if not sari_row.empty:
                    mae = sari_row["mae"].values[0]
                    mape = sari_row["mape"].values[0]
                    col_b.metric("SARIMAX Average Error", f"${mae:,.2f}", f"MAPE: {mape:.2f}%", delta_color="inverse")
            else:
                st.info("Waiting for data validation to calculate scores.")
        else:
            st.info("No predictions yet.")
