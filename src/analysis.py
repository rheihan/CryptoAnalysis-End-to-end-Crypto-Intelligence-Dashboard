import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_correlation(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    Hitung korelasi antar kolom (default: price, volume, return, volatility).
    """
    if cols is None:
        cols = ["price", "volume", "return", "volatility"]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()
    return corr


def plot_price_volume(df: pd.DataFrame):
    """
    (Opsional) Plot static price & volume dengan matplotlib.
    Tidak dipakai di Streamlit (kita pakai Plotly), tapi bisa dipakai di notebook.
    """
    fig, ax1 = plt.subplots()

    color1 = "tab:blue"
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price", color=color1)
    ax1.plot(df.index, df["price"], color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Volume", color=color2)
    ax2.bar(df.index, df["volume"], alpha=0.3)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame):
    """
    (Opsional) Heatmap korelasi pakai seaborn.
    """
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig
