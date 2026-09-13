"""
Multi-day forecast generation.

Models retrain weekly (see src/retrain.py), so there is no benefit to
re-predicting from scratch every day with the same weights - the model's
view of the world doesn't change until the next retrain. Instead, right
after each retrain we predict the whole span of days until the next one in
a single pass, and the daily job (src/automation.py) just verifies those
predictions against real closes as they arrive. generate_forecasts() only
fills in dates that don't already have a prediction for a given model, so
calling it again on an ordinary day (no retrain since) is a safe no-op.

Neither model natively predicts more than one day ahead, so a multi-day
forecast means recursively chaining single-day predictions: each day's
predicted return feeds the next day's input. Exogenous/indicator features
beyond day 1 are held at their last known real value - a standard
simplifying assumption when the future values of those inputs are
themselves unknown. Error compounds across the horizon like any recursive
forecast; that's expected, and is exactly why each retrain re-anchors the
whole horizon to fresh real data rather than trying to predict further and
further from stale weights.
"""
import os

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

from src.data_processing import get_processed_data, load_data
from src.storage import load_predictions, save_predictions

LSTM_FEATURE_COLS = ['log_return', 'volume_log_return', 'rsi', 'bb_position', 'macd_norm', 'momentum_7d']
SARIMAX_FEATURE_COLS = ['volume_log_return', 'rsi', 'bb_position', 'macd_norm', 'momentum_7d']


def _target_dates(records, ticker, model_version, candidate_dates, force):
    """
    Which candidate dates need a fresh prediction. Normally: any date that
    doesn't have one yet. With force=True (used right after a retrain): any
    date that isn't already resolved (has a real actual_price) - a fresh
    model should always supersede an existing but still-unresolved forecast
    for a future date, even one already written by that day's safety-net
    call in automation.daily_job() before this retrain ran.
    """
    if force:
        resolved = {
            r['predicted_date'] for r in records
            if r['ticker'] == ticker and r['model_version'] == model_version
            and r.get('actual_price') is not None
        }
        return [d for d in candidate_dates if d not in resolved]

    existing = {
        r['predicted_date'] for r in records
        if r['ticker'] == ticker and r['model_version'] == model_version
    }
    return [d for d in candidate_dates if d not in existing]


def _forecast_lstm_path(lstm_model, scaler, target_scaler, window, base_close, dates):
    """
    Recursive rollout: predict one day, fold the predicted return into the
    window (other indicators held at their last real value), repeat.
    window: DataFrame of the most recent `seq_length` real days, columns =
    LSTM_FEATURE_COLS (kept as a DataFrame, not a bare array, so it matches
    the column-named DataFrame the scaler was originally fit on).
    Returns {date: predicted_price} for every date in `dates`, in order.
    """
    window = window.copy()
    last_static = window.iloc[-1].copy()

    prices = {}
    current_close = base_close
    for date in dates:
        scaled_window = scaler.transform(window)
        X_input = scaled_window.reshape(1, scaled_window.shape[0], scaled_window.shape[1])
        pred_scaled = lstm_model.predict(X_input, verbose=0)
        pred_return = target_scaler.inverse_transform(pred_scaled)[0][0]

        current_close = float(current_close * np.exp(pred_return))
        prices[date] = current_close

        new_row = last_static.copy()
        new_row['log_return'] = pred_return
        window = pd.concat([window.iloc[1:], new_row.to_frame().T], ignore_index=True)

    return prices


def _forecast_sarimax_path(sarimax_model, exog_row, base_close, dates):
    """
    One batch call covering the whole horizon - pmdarima's own internal
    recursion handles the multi-step chaining correctly; calling .predict()
    repeatedly with n_periods=1 would NOT (it wouldn't advance the model's
    internal state between calls). Exogenous values held constant at their
    last known real value across the horizon.
    """
    X_future = pd.DataFrame(
        np.tile(exog_row, (len(dates), 1)),
        columns=SARIMAX_FEATURE_COLS,
    )
    returns = sarimax_model.predict(n_periods=len(dates), X=X_future)
    returns = returns.to_numpy() if hasattr(returns, "to_numpy") else np.asarray(returns)
    prices = base_close * np.exp(np.cumsum(returns))
    return dict(zip(dates, prices.tolist()))


def generate_forecasts(ticker, horizon_days=7, force=False):
    """
    Predicts forward from the latest known real day out to `horizon_days`.

    force=False (the daily safety-net call): only fills genuine gaps - dates
    that don't have a prediction yet for a given model. A no-op on an
    ordinary day once the horizon is filled.

    force=True (called right after a retrain): regenerates every
    not-yet-resolved date in the horizon with the fresh model, even if a
    prediction already exists for it - see _target_dates() for why that
    matters on retrain day specifically.
    """
    print(f"🔮 Generating forecasts through the next {horizon_days} days...")

    lstm_path = f"models/{ticker.lower()}_gru_v4.keras"
    sarimax_path = f"models/{ticker.lower()}_sarimax.pkl"
    if not os.path.exists(lstm_path) or not os.path.exists(sarimax_path):
        print("   ⚠️ Models not found. Skipping forecast.")
        return

    df_features = load_data(ticker)
    base_date = pd.to_datetime(df_features['date'].iloc[-1])
    base_close = float(df_features['close'].iloc[-1])
    candidate_dates = [
        (base_date + timedelta(days=i)).date().isoformat()
        for i in range(1, horizon_days + 1)
    ]

    records = load_predictions()
    missing_lstm = set(_target_dates(records, ticker, "LSTM_BiDir_v4", candidate_dates, force))
    missing_sarimax = set(_target_dates(records, ticker, "SARIMAX_v1", candidate_dates, force))

    if not missing_lstm and not missing_sarimax:
        print(f"   Already have forecasts through {candidate_dates[-1]}. Nothing to do.")
        return

    if force:
        # Drop existing-but-unresolved entries we're about to replace, so we
        # don't end up with duplicate rows for the same model/date.
        replace_keys = {("LSTM_BiDir_v4", d) for d in missing_lstm} | {("SARIMAX_v1", d) for d in missing_sarimax}
        records = [
            r for r in records
            if not (r['ticker'] == ticker and (r['model_version'], r['predicted_date']) in replace_keys)
        ]

    now = datetime.utcnow().isoformat()
    changed = False

    if missing_lstm:
        lstm_model = load_model(lstm_path)
        processed = get_processed_data(ticker, seq_length=30)
        window = df_features[LSTM_FEATURE_COLS].iloc[-30:].astype(float).reset_index(drop=True)
        lstm_prices = _forecast_lstm_path(
            lstm_model, processed['scaler'], processed['target_scaler'], window, base_close, candidate_dates
        )
        for date in candidate_dates:
            if date in missing_lstm:
                records.append({
                    "timestamp": now, "ticker": ticker, "model_version": "LSTM_BiDir_v4",
                    "predicted_date": date, "predicted_price": lstm_prices[date],
                    "actual_price": None, "mae": None, "mape": None,
                })
                changed = True
        print(f"   💾 LSTM: added forecasts for {sorted(missing_lstm)}")

    if missing_sarimax:
        sarimax_model = joblib.load(sarimax_path)
        exog_row = df_features[SARIMAX_FEATURE_COLS].iloc[-1].to_numpy(dtype=float)
        sarimax_prices = _forecast_sarimax_path(sarimax_model, exog_row, base_close, candidate_dates)
        for date in candidate_dates:
            if date in missing_sarimax:
                records.append({
                    "timestamp": now, "ticker": ticker, "model_version": "SARIMAX_v1",
                    "predicted_date": date, "predicted_price": sarimax_prices[date],
                    "actual_price": None, "mae": None, "mape": None,
                })
                changed = True
        print(f"   💾 SARIMAX: added forecasts for {sorted(missing_sarimax)}")

    if changed:
        save_predictions(records)
