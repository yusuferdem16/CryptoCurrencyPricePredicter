# 📈 Crypto Price Predictor  
### End-to-End Machine Learning System for Daily Bitcoin Price Forecasting

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Automation-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A full-stack, production-grade ML system that forecasts Bitcoin prices using a **live model arena** where **Bi-Directional LSTM** and **SARIMAX** compete daily.  
The system automates data ingestion, feature engineering, training, evaluation, storage, and prediction serving.

👉 **Live Demo:** *https://btcforecaster.streamlit.app/*  
👉 **Author:** *Abdullah Yusuf Erdem*

---

## 📸 Dashboard Preview  
- **Model Arena:** Real-time comparison of LSTM vs SARIMAX  
- **Daily MAE Audit:** Tracks prediction accuracy  
- **Technical Indicator View:** Log returns, RSI, Bollinger metrics  

(Add your screenshots here.)

---

## 🧠 Project Overview

This project goes *far beyond* a Jupyter Notebook.  
It is a **fully automated, serverless MLOps system** that runs indefinitely for free with zero manual steps:

- Ingests daily real-time Bitcoin price data via GitHub Actions  
- Engineers stationarity-friendly features  
- Produces next-day price forecasts every day  
- Retrains LSTM + SARIMAX models on a weekly cadence  
- Tracks and evaluates prediction accuracy over time  
- Commits every result straight back to the repo as the system of record  
- Displays forecasts and metrics on a Streamlit dashboard that auto-redeploys on every update  

Perfect for showcasing **Data Engineering**, **ML Engineering**, and **MLOps** skills.

---

## ⭐ Key Features

### 🔀 Dual-Engine Forecasting  
Two competing forecasting models:
- **Bi-Directional LSTM (Deep Learning)**
- **SARIMAX (Statistical Time Series)**

The system identifies the best daily signal.

### 🤖 Automated MLOps Pipeline
**Daily** (GitHub Actions cron):
1. Ingest new data
2. Validate yesterday's prediction (MAE / MAPE)
3. Generate tomorrow's forecast

**Weekly** (separate GitHub Actions cron):
4. Retrain both models on the latest history
5. Overwrite the model artifacts

### 🗄️ File-Based Data Store
- No database to host, pay for, or lose network access to
- Price history and predictions live as version-controlled CSV/JSON files under `data/`
- Every GitHub Actions run commits its output straight back to the repo, which is also what triggers the dashboard to auto-redeploy with fresh numbers

### 📐 Stationarity Engineering  
Mitigates ML’s extrapolation issues via:
- Log Returns  
- RSI  
- Bollinger Band Position  
- Lag Features  
- Scaling & sequence generation  

### 🧩 Architecture  
- **Automation:** GitHub Actions (daily forecast + weekly retrain, both scheduled cron jobs)  
- **Frontend:** Streamlit (reads committed files directly - no backend to run)  
- **Training:** LSTM + SARIMAX  
- **Storage:** Versioned CSV/JSON files in the repo (`data/`, `models/`)  

---

## 🏗️ System Architecture  

```mermaid
flowchart LR
    A[Yahoo Finance API] -->|Daily Ingest| B[data/*.csv in repo]
    B -->|Load| C[Bi-LSTM + SARIMAX Inference]
    C -->|Forecast| D[data/predictions.json]
    D -->|git commit + push| E[GitHub repo]
    E -->|weekly cron| F{Retrain Pipeline}
    F -->|Train LSTM| G[Bi-Directional LSTM]
    F -->|Train SARIMAX| H[SARIMAX]
    G & H -->|git commit + push| I[models/*.keras + *.pkl]
    E -->|auto-redeploy on push| J[Streamlit Dashboard]
````

Nothing in this pipeline needs a network-reachable server: GitHub Actions runs on GitHub's own infrastructure (full internet access, unlike a laptop behind NAT/Docker), writes its output as files, and pushes them to the repo. Streamlit Community Cloud auto-redeploys on every push, so the dashboard always reflects the latest committed data with zero manual steps and zero database to keep alive.

---

## 📊 Results & Findings

### **Backtests & Live Forward-Testing**

| Model        | Type                | MAE         | Insight                                                              |
| ------------ | ------------------- | ----------- | -------------------------------------------------------------------- |
| **SARIMAX**  | Statistical         | **~$1,608** | ⭐ **Winner** — crypto daily prices are mean-reverting and efficient. |
| **Bi-LSTM**  | Deep Learning       | ~$1,639     | Good at trend direction but struggled with volatility.               |
| **Baseline** | Naive (Random Walk) | ~$1,595     | Hard to beat a random-walk baseline in crypto.                       |

### 🧠 Conclusion

Daily crypto forecasting has low signal-to-noise.
Deep Learning finds patterns, but **statistical baselines remain strong competitors** on daily data.

---

## 🛠️ Installation & Setup

### **Prerequisites**

* Python 3.10+

No database and no Docker required - everything reads and writes plain files under `data/` and `models/`.

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/CryptoPricePredictor.git
cd CryptoPricePredictor
```

### 2️⃣ Install Dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the Pipeline Locally

```bash
python -m src.automation   # daily job: ingest, verify, forecast
python -m src.retrain      # weekly job: retrain both models
```

Both commands are idempotent and safe to re-run; they read/write the CSV/JSON files under `data/` and the model artifacts under `models/`.

---

## 🚀 Usage

### ▶️ Start the Dashboard

```bash
streamlit run src/dashboard.py
```

Access: **[http://localhost:8501](http://localhost:8501)**

The dashboard reads `data/*.csv` and `data/predictions.json` directly - it does not call any backend or database.

---

## ⚙️ Automation (Production)

Two GitHub Actions workflows drive the live deployment, no manual steps required:

| Workflow | Schedule | What it does |
| --- | --- | --- |
| `.github/workflows/daily_prediction.yml` | Daily, 05:00 UTC | Ingest latest price → verify yesterday's forecast → generate tomorrow's forecast → commit `data/` |
| `.github/workflows/weekly_retrain.yml` | Weekly, Sunday 06:00 UTC | Retrain LSTM + SARIMAX on latest data → commit `models/` and `data/` |

Both workflows use the default `GITHUB_TOKEN` (with `contents: write` permission declared in the workflow) to push their results back to the repo - **no secrets need to be configured**. Every push to the repo also triggers Streamlit Community Cloud to auto-redeploy the dashboard with the latest data.

---

## 📂 Project Structure

```
CryptoPricePredictor/
├── data/                     # Price history + predictions (committed by CI)
│   ├── raw_btc_usd.csv
│   ├── features_btc_usd.csv
│   └── predictions.json
├── models/                   # Saved .keras and .pkl artifacts (committed by CI)
├── src/
│   ├── automation.py         # Daily job: ingest, verify, forecast
│   ├── retrain.py            # Weekly job: retrain both models
│   ├── dashboard.py          # Streamlit UI (reads data/ directly)
│   ├── data_processing.py    # Scaling + sequence generation
│   ├── storage.py            # File-based data store (CSV/JSON)
│   ├── feature_engineering.py # RSI, MACD, Bollinger
│   ├── ingestion.py          # Yahoo data fetcher
│   ├── sarimax_pipeline.py   # SARIMAX trainer
│   ├── train.py              # LSTM trainer
│   └── models/                # Model definitions
├── .github/workflows/
│   ├── daily_prediction.yml
│   └── weekly_retrain.yml
└── requirements.txt
```

---

## 🛡️ License

This project is licensed under the **MIT License**.

---

## 👤 Author

**A. Yusuf Erdem**
Final Year Capstone Project | ML Engineering | MLOps | Data Science

Feel free to reach out for discussion, collaboration, or feedback!
