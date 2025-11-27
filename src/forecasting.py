import numpy as np

# Patch buat kompatibilitas NumPy 2.x (aman juga kalau masih 1.26.x)
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "float_"):
    np.float_ = np.float64

import pandas as pd

# Prophet kita import, tapi nanti kalau error di runtime kita fallback
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    Prophet = None
    HAS_PROPHET = False

import pmdarima as pm


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert index-based df to Prophet format (ds, y).
    Assumes index name = 'timestamp'.
    """
    prophet_df = df[["price"]].reset_index().rename(
        columns={"timestamp": "ds", "price": "y"}
    )
    return prophet_df


def train_prophet(df: pd.DataFrame, periods: int = 30):
    """
    Coba train Prophet. Kalau gagal (issue stan_backend / build tools),
    fallback ke naive forecast (harga terakhir / moving average).
    Return: (model, forecast_df)
    forecast_df minimal punya kolom: ds, yhat
    """
    prophet_df = prepare_prophet_df(df)

    # --- Coba Prophet dulu ---
    if HAS_PROPHET:
        try:
            model = Prophet()
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            # Prophet pakai kolom 'ds' & 'yhat'
            return model, forecast
        except Exception as e:
            # Di sini biasanya kena masalah stan_backend di Windows
            print("[WARN] Prophet gagal dipakai, fallback ke naive forecast:", e)

    # --- Fallback: naive forecast (tanpa Prophet) ---
    last_date = prophet_df["ds"].max()
    last_price = prophet_df["y"].iloc[-1]

    # Generate tanggal ke depan
    future_dates = pd.date_range(last_date, periods=periods + 1, freq="D")[1:]

    # Di sini kita pakai simple forecast: harga tetap = last_price
    # Bisa lu ganti pakai moving average kalau mau lebih halus
    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": [last_price] * len(future_dates)
    })

    # Biar plot di Streamlit tetap full, kita gabungkan history + future
    history_part = prophet_df.rename(columns={"y": "yhat"})[["ds", "yhat"]]
    full_forecast = pd.concat([history_part, forecast], ignore_index=True)

    dummy_model = None
    return dummy_model, full_forecast


def train_arima(df: pd.DataFrame, forecast_steps: int = 30):
    """
    Train ARIMA (auto_arima) menggunakan harga.
    Return: model ARIMA + numpy array forecast.
    """
    series = df["price"].astype("float64")
    model = pm.auto_arima(
        series,
        seasonal=False,
        trace=False,
        error_action="ignore",
        suppress_warnings=True
    )
    forecast = model.predict(n_periods=forecast_steps)
    return model, forecast
