"""
File-based data store.

Replaces the old Postgres "data lake" with plain CSV/JSON files committed to
the repo under data/. There is no server to reach, so GitHub Actions and the
Streamlit dashboard can both read/write the same source of truth without any
network-reachable database or connection secrets.
"""
import json
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.json")


def _table_filename(name):
    return os.path.join(DATA_DIR, f"{name}.csv")


def read_table(name):
    """Returns a DataFrame for the given table name, or None if it doesn't exist yet."""
    path = _table_filename(name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def write_table(df, name):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(_table_filename(name), index=False)


def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        return []
    with open(PREDICTIONS_PATH) as f:
        return json.load(f)


def save_predictions(records):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)
