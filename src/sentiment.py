import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def add_sentiment_scores(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Tambah kolom 'sentiment' ke dataframe berisi teks (tweet/news).
    Sentiment dihitung pakai VADER (compound score: -1 s/d 1).
    """
    df = df.copy()
    analyzer = SentimentIntensityAnalyzer()

    def _score(text: str) -> float:
        if not isinstance(text, str):
            text = str(text)
        scores = analyzer.polarity_scores(text)
        return scores["compound"]

    df["sentiment"] = df[text_col].apply(_score)
    return df


def aggregate_sentiment_by_time(
    df: pd.DataFrame,
    time_col: str = "created_at",
    sentiment_col: str = "sentiment",
    freq: str = "D",
) -> pd.DataFrame:
    """
    Aggregate sentiment by time (default daily).
    Output: DataFrame index datetime, kolom 'sentiment'.
    """
    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col])
    tmp.set_index(time_col, inplace=True)
    agg = tmp[sentiment_col].resample(freq).mean().dropna()
    return agg.to_frame("sentiment")


def align_sentiment_with_market(
    market_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align sentiment timeseries dengan market dataframe (price, return, volatility).
    Market & sentiment di-resample jadi daily supaya sinkron.

    Output: dataframe gabungan dengan kolom:
    price, volume, return, volatility, sentiment, dll (kalau ada).
    """
    # Resample market ke daily (ambil close = last of day)
    mkt = market_df.copy()
    mkt_daily = mkt.resample("D").last()

    # Pastikan sentiment index datetime & daily
    sent = sentiment_df.copy()
    if not isinstance(sent.index, pd.DatetimeIndex):
        sent.index = pd.to_datetime(sent.index)
    sent_daily = sent.resample("D").mean()

    joined = mkt_daily.join(sent_daily, how="inner")
    return joined
