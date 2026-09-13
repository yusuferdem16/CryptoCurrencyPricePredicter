import yfinance as yf
import pandas as pd
from datetime import timedelta
from src.storage import read_table, write_table


def _table_name(ticker):
    return f"raw_{ticker.lower().replace('-', '_')}"


def get_latest_date(ticker):
    """Finds the last date we have stored for this ticker."""
    df = read_table(_table_name(ticker))
    if df is None or df.empty:
        return None
    return df["date"].max()


def fetch_and_store(ticker):
    table_name = _table_name(ticker)

    # 1. Check existing data
    last_date = get_latest_date(ticker)

    if last_date:
        print(f"📉 Found data for {ticker} up to {last_date.date()}. Appending new data...")
        # A narrow start date very close to today routes yfinance through a
        # crumb/cookie-authenticated request path that has proven unreliable
        # in CI (ImpersonateError from curl_cffi, download silently empty).
        # A wider trailing window uses the plain historical endpoint instead,
        # which works; drop_duplicates() below makes the overlap harmless.
        start_date = (last_date - timedelta(days=10)).strftime('%Y-%m-%d')
    else:
        print(f"🆕 No data found for {ticker}. Fetching full history...")
        start_date = "2020-01-01"  # Default start

    # 2. Download Data
    try:
        df_new = yf.download(ticker, start=start_date, interval="1d", progress=False)

        if df_new.empty:
            print(f"⚠️ No new data available for {ticker}.")
            return

        # 3. Clean Data
        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = df_new.columns.get_level_values(0)

        df_new.reset_index(inplace=True)
        df_new.columns = [c.lower() for c in df_new.columns]
        df_new["date"] = pd.to_datetime(df_new["date"])

        # 4. Merge with existing history (dedupe in case of overlap/reruns)
        existing = read_table(table_name)
        prior_count = len(existing) if existing is not None else 0
        combined = pd.concat([existing, df_new], ignore_index=True) if existing is not None else df_new
        combined = combined.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)

        # 5. Store Data
        write_table(combined, table_name)
        print(f"✅ Added {len(combined) - prior_count} new rows to {table_name} ({len(combined)} rows total).")

    except Exception as e:
        print(f"❌ Error updating {ticker}: {e}")


if __name__ == "__main__":
    fetch_and_store("BTC-USD")
    fetch_and_store("ETH-USD")
