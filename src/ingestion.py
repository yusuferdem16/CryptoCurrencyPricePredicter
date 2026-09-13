import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.storage import read_table, write_table

COINGECKO_RANGE_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart/range"

COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
}


def _table_name(ticker):
    return f"raw_{ticker.lower().replace('-', '_')}"


def _coingecko_id(ticker):
    base = ticker.split('-')[0].upper()
    return COINGECKO_IDS.get(base, base.lower())


def get_latest_date(ticker):
    """Finds the last date we have stored for this ticker."""
    df = read_table(_table_name(ticker))
    if df is None or df.empty:
        return None
    return df["date"].max()


def _fetch_market_chart(coin_id, from_ts, to_ts):
    resp = requests.get(
        COINGECKO_RANGE_URL.format(id=coin_id),
        params={"vs_currency": "usd", "from": from_ts, "to": to_ts},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_and_store(ticker):
    """
    Pulls daily close/volume from CoinGecko's public API (no key required).

    Two prior data sources both failed from GitHub Actions runners: Yahoo
    Finance (via yfinance) rejected every crumb/cookie handshake with an
    ImpersonateError, and Binance's public API returned HTTP 451 (Unavailable
    For Legal Reasons) - almost certainly its US-persons geo-restriction
    applied to GitHub's Azure-hosted runner IPs. CoinGecko is a global
    aggregator with no such regulatory geo-fencing and no bot-detection
    dance to fail.
    """
    table_name = _table_name(ticker)
    coin_id = _coingecko_id(ticker)

    last_date = get_latest_date(ticker)

    if last_date:
        print(f"📉 Found data for {ticker} up to {last_date.date()}. Appending new data...")
        start_dt = last_date.to_pydatetime().replace(tzinfo=timezone.utc) - timedelta(days=10)
    else:
        print(f"🆕 No data found for {ticker}. Fetching full history...")
        start_dt = datetime(2020, 1, 1, tzinfo=timezone.utc)

    from_ts = int(start_dt.timestamp())
    to_ts = int(datetime.now(timezone.utc).timestamp())

    try:
        payload = _fetch_market_chart(coin_id, from_ts, to_ts)
        prices = payload.get("prices", [])
        volumes = payload.get("total_volumes", [])

        if not prices:
            print(f"⚠️ No new data available for {ticker}.")
            return

        price_df = pd.DataFrame(prices, columns=["ts", "close"])
        volume_df = pd.DataFrame(volumes, columns=["ts", "volume"])
        price_df["date"] = pd.to_datetime(price_df["ts"], unit="ms").dt.normalize()
        volume_df["date"] = pd.to_datetime(volume_df["ts"], unit="ms").dt.normalize()

        # CoinGecko returns hourly (or finer) granularity for short ranges -
        # collapse to one row per UTC day using that day's last observation.
        daily_close = price_df.groupby("date")["close"].last()
        daily_volume = volume_df.groupby("date")["volume"].last()
        df_new = pd.concat([daily_close, daily_volume], axis=1).reset_index()

        # Drop today's still-forming day - its "last observation so far" is
        # not a final close and would corrupt the most recent row.
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        df_new = df_new[df_new["date"] < today]

        if df_new.empty:
            print(f"⚠️ No new data available for {ticker}.")
            return

        # Merge with existing history (dedupe in case of overlap/reruns).
        # Drop any leftover open/high/low columns from the old yfinance-era
        # schema - nothing downstream uses them, and CoinGecko doesn't
        # provide them, so keep the schema consistently date/close/volume.
        existing = read_table(table_name)
        if existing is not None:
            existing = existing.drop(columns=["open", "high", "low"], errors="ignore")
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
