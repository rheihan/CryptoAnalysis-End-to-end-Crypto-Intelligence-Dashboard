import os
from datetime import datetime

import pandas as pd  # kalau belum ada ya udah kepake di project
from src.data_collection import fetch_market_chart
from src.preprocessing import add_basic_features, drop_na_for_model

# List coin ID sesuai API (CoinGecko style)
COINS = ["bitcoin", "ethereum", "solana"]   # BTC, ETH, SOL
DAYS = 180  # berapa hari ke belakang (boleh lu ubah, misal 30 / 365)

def save_raw_coin(coin_id: str, days: int = DAYS):
    df = fetch_market_chart(coin_id=coin_id, days=days)
    df = add_basic_features(df)
    df = drop_na_for_model(df)

    os.makedirs("data/raw", exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_name = f"{coin_id}_{days}d_{ts}.parquet"
    file_path = os.path.join("data", "raw", file_name)

    df.to_parquet(file_path)
    print(f"[OK] Saved {coin_id} → {file_path} (rows: {len(df)})")


def main():
    for cid in COINS:
        save_raw_coin(cid, DAYS)


if __name__ == "__main__":
    main()
