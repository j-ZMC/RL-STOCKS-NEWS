"""Build a timestamped 15-minute sentiment series from news predictions."""

from __future__ import annotations

from typing import Final

import pandas as pd


INTERVAL: Final[str] = "15min"
DEFAULT_PREVIOUS_WINDOW_WEIGHT: Final[float] = 0.8
LABEL_TO_VALUE: Final[dict[str, float]] = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


def _get_publication_timestamps(
    news: pd.DataFrame,
    timestamp_column: str,
    date_column: str,
    time_column: str,
) -> pd.Series:
    if timestamp_column in news.columns:
        raw_timestamps = news[timestamp_column]
    elif "published_at" in news.columns:
        raw_timestamps = news["published_at"]
    elif date_column in news.columns and time_column in news.columns:
        raw_timestamps = (
            news[date_column].astype(str).str.strip()
            + " "
            + news[time_column].astype(str).str.strip()
        )
    else:
        raise ValueError(
            "News must contain a timestamp, published_at, or both date and hour_minute columns"
        )

    try:
        return pd.to_datetime(raw_timestamps, errors="raise", utc=True)
    except (TypeError, ValueError) as error:
        raise ValueError("News publication timestamps must be valid datetimes") from error


def _get_sentiment_values(news: pd.DataFrame, sentiment_column: str) -> pd.Series:
    if sentiment_column in news.columns:
        values = pd.to_numeric(news[sentiment_column], errors="raise")
    elif {"score_positive", "score_negative"}.issubset(news.columns):
        values = news["score_positive"] - news["score_negative"]
    elif "sentiment" in news.columns:
        values = news["sentiment"].map(LABEL_TO_VALUE)
        if values.isna().any():
            raise ValueError("sentiment labels must be positive, neutral, or negative")
    else:
        raise ValueError(
            "News must contain sentiment_value, FinBERT scores, or sentiment labels"
        )

    if values.isna().any():
        raise ValueError("News sentiment values cannot be missing")

    return values.astype(float)


def aggregate_sentiment_15m(
    news: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    sentiment_column: str = "sentiment_value",
    date_column: str = "day",
    time_column: str = "hour_minute",
    previous_window_weight: float = DEFAULT_PREVIOUS_WINDOW_WEIGHT,
) -> pd.DataFrame:
    """Aggregate news sentiment into a complete, timestamped 15-minute series.

    ``sentiment_mean`` is the arithmetic mean for news published in the window.
    ``sentiment`` forward-fills empty windows from the immediately previous window.
    ``sentiment_influenced`` is the model-ready signal: each window combines its
    own carried value with the previous effective window. With the default weight,
    the previous window contributes 80% to the subsequent one.

    The first output window is the first window containing news, so a leading gap
    never requires an undefined previous value. Timestamps are normalized to UTC;
    timezone-naive source timestamps are interpreted as UTC.
    """
    if news.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "news_count",
                "sentiment_mean",
                "sentiment",
                "sentiment_influenced",
            ]
        )

    if not 0.0 <= previous_window_weight <= 1.0:
        raise ValueError("previous_window_weight must be between 0 and 1")

    working = pd.DataFrame(
        {
            "timestamp": _get_publication_timestamps(
                news, timestamp_column, date_column, time_column
            ),
            "sentiment_value": _get_sentiment_values(news, sentiment_column),
        }
    )
    working["interval_start"] = working["timestamp"].dt.floor(INTERVAL)

    means = working.groupby("interval_start")["sentiment_value"].mean()
    counts = working.groupby("interval_start").size()
    interval_index = pd.date_range(
        start=means.index.min(),
        end=means.index.max(),
        freq=INTERVAL,
        tz="UTC",
    )

    result = pd.DataFrame(index=interval_index)
    result.index.name = "timestamp"
    result["news_count"] = counts.reindex(interval_index, fill_value=0).astype(int)
    result["sentiment_mean"] = means.reindex(interval_index)
    result["sentiment"] = result["sentiment_mean"].ffill()

    influenced = []
    previous_effective = None
    for carried_value in result["sentiment"]:
        if previous_effective is None:
            effective_value = carried_value
        else:
            effective_value = (
                previous_window_weight * previous_effective
                + (1.0 - previous_window_weight) * carried_value
            )
        influenced.append(effective_value)
        previous_effective = effective_value

    result["sentiment_influenced"] = influenced
    return result.reset_index()