from datetime import timedelta

# Konfigurasi dasar project

SUPPORTED_COINS = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana"
}

DEFAULT_VS_CURRENCY = "usd"
DEFAULT_DAYS = 365  # data 1 tahun ke belakang

ROLLING_WINDOW_VOL = 30  # hari untuk hitung volatility
MA_SHORT = 50
MA_LONG = 200

TRADING_DAYS_PER_YEAR = 365  # crypto jalan 24/7
