import yfinance as yf
import pandas as pd
from sqlalchemy import text
from datetime import timedelta
from src.database import engine

def get_latest_date(table_name):
    """Finds the last date we have in the database"""
    with engine.connect() as conn:
        # Check if table exists first
        exists = conn.execute(text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            );
        """)).scalar()
        
        if not exists:
            return None

        # Get max date
        result = conn.execute(text(f"SELECT MAX(date) FROM {table_name}")).scalar()
        return pd.to_datetime(result) if result else None

def fetch_and_store(ticker):
    # Sanitize table name
    table_name = f"raw_{ticker.lower().replace('-', '_')}"
    
    # 1. Check existing data
    last_date = get_latest_date(table_name)
    
    if last_date:
        print(f"📉 Found data for {ticker} up to {last_date.date()}. Appending new data...")
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        action = 'append'
    else:
        print(f"🆕 No data found for {ticker}. Fetching full history...")
        start_date = "2020-01-01" # Default start
        action = 'replace'

    # 2. Download Data
    try:
        # We use today's date as end to capture everything
        df = yf.download(ticker, start=start_date, interval="1d", progress=False)
        
        if df.empty:
            print(f"⚠️ No new data available for {ticker}.")
            return

        # 3. Clean Data
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.reset_index(inplace=True)
        df.columns = [c.lower() for c in df.columns]

        # 4. Store Data
        df.to_sql(table_name, engine, index=False, if_exists=action)
        print(f"✅ Added {len(df)} new rows to {table_name}.")
        
    except Exception as e:
        print(f"❌ Error updating {ticker}: {e}")

if __name__ == "__main__":
    fetch_and_store("BTC-USD")
    fetch_and_store("ETH-USD")