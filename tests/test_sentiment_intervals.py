import pandas as pd

from finBERT_model.sentiment_intervals import aggregate_sentiment_15m


def test_aggregates_fills_gaps_and_carries_strong_previous_influence():
    news = pd.DataFrame(
        {
            "timestamp": [
                "2026-08-30T10:00:00Z",
                "2026-08-30T10:00:00Z",
                "2026-08-30T10:30:00Z",
            ],
            "sentiment_value": [1.0, -1.0, 1.0],
        }
    )

    result = aggregate_sentiment_15m(news)

    assert result["timestamp"].tolist() == list(
        pd.date_range("2026-08-30T10:00:00Z", periods=3, freq="15min")
    )
    assert result["news_count"].tolist() == [2, 0, 1]
    assert result["sentiment_mean"].iloc[0] == 0.0
    assert pd.isna(result["sentiment_mean"].iloc[1])
    assert result["sentiment_mean"].iloc[2] == 1.0
    assert result["sentiment"].tolist() == [0.0, 0.0, 1.0]
    assert result["sentiment_influenced"].tolist() == [0.0, 0.0, 0.2]


def test_accepts_existing_day_and_hour_minute_columns_and_finbert_scores():
    news = pd.DataFrame(
        {
            "day": ["2026-08-30", "2026-08-30"],
            "hour_minute": ["10:00", "10:15"],
            "score_positive": [0.8, 0.1],
            "score_negative": [0.1, 0.1],
        }
    )

    result = aggregate_sentiment_15m(news)

    assert all(abs(actual - expected) < 1e-9 for actual, expected in zip(
        result["sentiment_mean"].tolist(), [0.7, 0.0]
    ))
    assert all(abs(actual - expected) < 1e-9 for actual, expected in zip(
        result["sentiment_influenced"].tolist(), [0.7, 0.56]
    ))