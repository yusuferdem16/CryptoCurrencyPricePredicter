# 📈 Crypto Price Predictor  
### End-to-End Machine Learning System for Daily Bitcoin Price Forecasting

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

A full-stack, production-grade ML system that forecasts Bitcoin prices using a **live model arena** where **Bi-Directional LSTM** and **SARIMAX** compete daily.  
The system automates data ingestion, feature engineering, training, evaluation, storage, and prediction serving.

👉 **Live Demo:** *Add your Streamlit link here*  
👉 **Author:** *Erdem — Final Year Capstone Project / Portfolio Piece*

---

## 📸 Dashboard Preview  
- **Model Arena:** Real-time comparison of LSTM vs SARIMAX  
- **Daily MAE Audit:** Tracks prediction accuracy  
- **Technical Indicator View:** Log returns, RSI, Bollinger metrics  

(Add your screenshots here.)

---

## 🧠 Project Overview

This project goes *far beyond* a Jupyter Notebook.  
It is a **fully automated MLOps system** that:

- Ingests daily real-time Bitcoin price data  
- Engineers stationarity-friendly features  
- Retrains LSTM + SARIMAX models  
- Evaluates their performance  
- Stores model artifacts  
- Produces next-day price forecasts  
- Serves them via a REST API  
- Displays forecasts and metrics on a Streamlit dashboard  

Perfect for showcasing **Data Engineering**, **ML Engineering**, and **Full-Stack ML** skills.

---

## ⭐ Key Features

### 🔀 Dual-Engine Forecasting  
Two competing forecasting models:
- **Bi-Directional LSTM (Deep Learning)**
- **SARIMAX (Statistical Time Series)**

The system identifies the best daily signal.

### 🤖 Automated MLOps Pipeline  
Daily scheduler:
1. Ingest new data  
2. Validate yesterday’s prediction (MAE)  
3. Retrain models  
4. Update model registry  
5. Generate new forecast  

### 🗄️ Robust Data Infrastructure  
- Dockerized **PostgreSQL** database  
- No CSV headaches  
- Persistent, queryable time-series storage  

### 📐 Stationarity Engineering  
Mitigates ML’s extrapolation issues via:
- Log Returns  
- RSI  
- Bollinger Band Position  
- Lag Features  
- Scaling & sequence generation  

### 🧩 Full-Stack Architecture  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Training:** LSTM + SARIMAX  
- **Storage:** PostgreSQL  
- **Orchestration:** Python (cron-like automation)  

---

## 🏗️ System Architecture  

```mermaid
flowchart LR
    A[Yahoo Finance API] -->|Daily Ingest| B(PostgreSQL DB)
    B -->|Fetch| C{Training Pipeline}
    C -->|Train LSTM| D[Bi-Directional LSTM]
    C -->|Train SARIMAX| E[SARIMAX]
    D & E -->|Save Artifacts| F[Model Registry]
    F -->|Load| G[FastAPI Backend]
    G -->|Serve JSON| H[Streamlit Dashboard]
````

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

* Docker Desktop
* Python 3.10+

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/CryptoPricePredictor.git
cd CryptoPricePredictor
```

### 2️⃣ Start PostgreSQL (Docker)

```bash
docker-compose up -d
```

### 3️⃣ Install Dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4️⃣ Initialize the Database

```bash
python -m src.init_db
```

### 5️⃣ Run the Automated Pipeline

```bash
python -m src.automation
```

---

## 🚀 Usage

### ▶️ Start the API (Backend)

```bash
uvicorn src.api:app --reload
```

Access: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

### ▶️ Start the Dashboard (Frontend)

```bash
streamlit run src/dashboard.py
```

Access: **[http://localhost:8501](http://localhost:8501)**

---

## 📂 Project Structure

```
CryptoPricePredictor/
├── models/                  # Saved .keras and .pkl artifacts
├── src/
│   ├── api.py               # FastAPI backend
│   ├── automation.py        # Daily MLOps scheduler
│   ├── dashboard.py         # Streamlit UI
│   ├── data_processing.py   # Scaling + sequence generation
│   ├── database.py          # DB connection
│   ├── feature_engineering.py # RSI, MACD, Bollinger
│   ├── ingestion.py         # Yahoo data fetcher
│   ├── sarimax_pipeline.py  # SARIMAX trainer
│   ├── train.py             # LSTM trainer
│   └── models/              # Model definitions
├── docker-compose.yml
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
