import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.storage import read_table, write_table

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def _table_name(ticker):
    return f"raw_{ticker.lower().replace('-', '_')}"


def _binance_symbol(ticker):
    # "BTC-USD" -> "BTCUSDT": Binance has no native USD spot market, USDT
    # (a USD-pegged stablecoin) is the standard proxy everyone trades against.
    base = ticker.split('-')[0]
    return f"{base}USDT"


def get_latest_date(ticker):
    """Finds the last date we have stored for this ticker."""
    df = read_table(_table_name(ticker))
    if df is None or df.empty:
        return None
    return df["date"].max()


def _fetch_klines(symbol, start_ms, end_ms):
    """Fetches daily candles from Binance's public REST API, paginating past its 1000-candle cap."""
    rows = []
    cursor = start_ms
    while True:
        resp = requests.get(BINANCE_KLINES_URL, params={
            "symbol": symbol,
            "interval": "1d",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cursor = batch[-1][0] + 1  # next candle's open time, in ms
    return rows


def fetch_and_store(ticker):
    """
    Pulls daily OHLCV candles from Binance's public API (no key required).
    Switched from yfinance after its Yahoo Finance scraping consistently
    failed from GitHub Actions runners with an ImpersonateError - the crumb
    fetch behind it appears to be rate-limited/blocked for shared CI IP
    ranges, which silently produced empty downloads and stalled the pipeline.
    """
    table_name = _table_name(ticker)
    symbol = _binance_symbol(ticker)

    last_date = get_latest_date(ticker)

    if last_date:
        print(f"📉 Found data for {ticker} up to {last_date.date()}. Appending new data...")
        start_dt = last_date.to_pydatetime().replace(tzinfo=timezone.utc) - timedelta(days=10)
    else:
        print(f"🆕 No data found for {ticker}. Fetching full history...")
        start_dt = datetime(2020, 1, 1, tzinfo=timezone.utc)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    try:
        klines = _fetch_klines(symbol, start_ms, end_ms)

        if not klines:
            print(f"⚠️ No new data available for {ticker}.")
            return

        df_new = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df_new["date"] = pd.to_datetime(df_new["open_time"], unit="ms").dt.normalize()
        for col in ["open", "high", "low", "close", "volume"]:
            df_new[col] = df_new[col].astype(float)
        df_new = df_new[["date", "open", "high", "low", "close", "volume"]]

        # Drop today's still-forming candle - Binance's daily candle for "today"
        # is incomplete until the day closes at 00:00 UTC, so treating it as a
        # final close would corrupt the most recent row.
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        df_new = df_new[df_new["date"] < today]

        if df_new.empty:
            print(f"⚠️ No new data available for {ticker}.")
            return

        # Merge with existing history (dedupe in case of overlap/reruns)
        existing = read_table(table_name)
        prior_count = len(existing) if existing is not None else 0
        combined = pd.concat([existing, df_new], ignore_index=True) if existing is not None else df_new
        combined = combined.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)

        write_table(combined, table_name)
        print(f"✅ Added {len(combined) - prior_count} new rows to {table_name} ({len(combined)} rows total).")

    except Exception as e:
        print(f"❌ Error updating {ticker}: {e}")


if __name__ == "__main__":
    fetch_and_store("BTC-USD")
    fetch_and_store("ETH-USD")
