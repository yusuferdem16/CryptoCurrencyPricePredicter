import sys
import os

# Add the root directory to sys.path so Python can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.storage import read_table, load_predictions

REPO_URL = "https://github.com/yusuferdem16/CryptoCurrencyPricePredicter"

st.set_page_config(page_title="Crypto Sharpshooter", page_icon="⚡", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚡ Crypto Sharpshooter")
    st.caption(
        "A live model arena: **Bi-Directional LSTM** vs. **SARIMAX**, forecasting "
        "Bitcoin's next close every day, fully unattended."
    )
    ticker = st.selectbox("Asset", ["BTC-USD"])
    st.divider()
    st.markdown(
        f"[📂 View source on GitHub]({REPO_URL})  \n"
        "**Author:** Abdullah Yusuf Erdem"
    )

# --- Header ---
st.title("⚡ Crypto Sharpshooter: Model Arena")
st.caption(
    "**Bi-Directional LSTM** (deep learning) vs. **SARIMAX** (statistical) compete daily to forecast "
    "Bitcoin's next close. Ingestion, retraining, and inference all run unattended on GitHub Actions - "
    "no server to keep alive, no database, no manual steps."
)

tab1, tab2 = st.tabs(["🔮 Live Forecast", "📈 Accuracy Tracker"])

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
        st.subheader("Tomorrow's Forecast")

        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.metric("💰 Current Price", f"${current_price:,.2f}")

        with col2:
            with st.container(border=True):
                if lstm_pred:
                    delta_lstm = lstm_pred["predicted_price"] - current_price
                    st.metric("🧠 LSTM (Deep Learning)", f"${lstm_pred['predicted_price']:,.2f}", f"{delta_lstm:+.2f}")
                else:
                    st.metric("🧠 LSTM (Deep Learning)", "—")

        with col3:
            with st.container(border=True):
                if sarimax_pred:
                    delta_sari = sarimax_pred["predicted_price"] - current_price
                    st.metric("📐 SARIMAX (Statistical)", f"${sarimax_pred['predicted_price']:,.2f}", f"{delta_sari:+.2f}")
                else:
                    st.metric("📐 SARIMAX (Statistical)", "—")

        if lstm_pred and sarimax_pred:
            st.caption(
                f"Forecast generated {lstm_pred['timestamp']} UTC for {lstm_pred['predicted_date']}. "
                "Refreshed automatically once a day by GitHub Actions."
            )

            st.divider()
            st.subheader("📉 Forecast Visualization")
            recent_history = history_df.tail(60)

            next_date = pd.to_datetime(lstm_pred["predicted_date"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent_history["date"], y=recent_history["close"],
                name="History", line=dict(color="#94A3B8", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=[last_date, next_date], y=[current_price, lstm_pred["predicted_price"]],
                mode="lines+markers", name="LSTM", line=dict(color="#FF4B4B", width=3),
            ))
            fig.add_trace(go.Scatter(
                x=[last_date, next_date], y=[current_price, sarimax_pred["predicted_price"]],
                mode="lines+markers", name="SARIMAX", line=dict(color="#2563EB", width=3, dash="dot"),
            ))
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=10, r=10, t=10, b=10),
                height=420,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(title="Price (USD)", tickprefix="$", separatethousands=True),
                xaxis=dict(title=None),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No forecast saved yet. It will appear after the next daily run.")

    # --- TAB 2: Accuracy Tracker ---
    with tab2:
        st.subheader("🔍 Model Performance Ledger")
        st.caption("Every forecast either model has made, verified against the actual close once it's known.")

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
            display_df.columns = ["Date", "Model", "Predicted", "Actual", "MAE", "MAPE"]
            display_df["Predicted"] = display_df["Predicted"].map(fmt_price)
            display_df["Actual"] = display_df["Actual"].map(fmt_price)
            display_df["MAE"] = display_df["MAE"].map(fmt_price)
            display_df["MAPE"] = display_df["MAPE"].map(fmt_mape)

            # Color the MAPE column by the original numeric value, manually - avoids
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
                display_df.style.apply(mape_color, subset=["MAPE"]),
                use_container_width=True,
                hide_index=True,
            )

            st.divider()
            st.subheader("🏆 Overall Scoreboard")
            df_finished = df_preds.dropna(subset=["mape"])

            if not df_finished.empty:
                avg_metrics = df_finished.groupby("model_version")[["mae", "mape"]].mean().reset_index()

                col_a, col_b = st.columns(2)

                lstm_row = avg_metrics[avg_metrics["model_version"].str.contains("LSTM")]
                if not lstm_row.empty:
                    mae = lstm_row["mae"].values[0]
                    mape = lstm_row["mape"].values[0]
                    with col_a:
                        with st.container(border=True):
                            st.metric("🧠 LSTM Average Error", f"${mae:,.2f}", f"MAPE: {mape:.2f}%", delta_color="inverse")

                sari_row = avg_metrics[avg_metrics["model_version"].str.contains("SARIMAX")]
                if not sari_row.empty:
                    mae = sari_row["mae"].values[0]
                    mape = sari_row["mape"].values[0]
                    with col_b:
                        with st.container(border=True):
                            st.metric("📐 SARIMAX Average Error", f"${mae:,.2f}", f"MAPE: {mape:.2f}%", delta_color="inverse")
            else:
                st.info("Waiting for data validation to calculate scores.")
        else:
            st.info("No predictions yet.")

st.divider()
st.caption(f"[Crypto Sharpshooter]({REPO_URL}) · Bi-LSTM vs. SARIMAX on daily BTC-USD · Built by Abdullah Yusuf Erdem")
