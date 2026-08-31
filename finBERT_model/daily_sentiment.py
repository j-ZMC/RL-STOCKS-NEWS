"""Daily sentiment aggregation for every sub-industry in the taxonomy."""

from __future__ import annotations

from typing import Any

import pandas as pd


LABEL_TO_VALUE = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


def taxonomy_subindustries(taxonomy: dict[str, Any]) -> pd.DataFrame:
    """Return every taxonomy sub-industry, including ones with no news."""
    rows = []
    if "industrias" in taxonomy:
        taxonomy_rows = (
            (industry["id"], subindustry)
            for industry in taxonomy["industrias"]
            for subindustry in industry.get("subindustrias", [])
        )
    else:
        taxonomy_rows = (
            (category.get("sector", "sin_sector"), {
                "id": category["id"],
                "nombre": category["categoria"],
            })
            for category in taxonomy.get("categorias", [])
        )

    for industry_id, subindustry in taxonomy_rows:
        if "id" not in subindustry or "nombre" not in subindustry:
            raise ValueError("Each taxonomy sub-industry must have an id and nombre")
        
        rows.append(
            {
                "industria_id": industry_id,
                "subindustria_id": subindustry["id"],
                "subindustria_nombre": subindustry["nombre"],
            }
        )

    if not rows:
        raise ValueError("The taxonomy does not contain any sub-industries")

    result = pd.DataFrame(rows)
    if result["subindustria_id"].duplicated().any():
        raise ValueError("The taxonomy contains duplicate sub-industry IDs")
    return result


def _sentiment_values(news: pd.DataFrame) -> pd.Series:
    if "sentiment_value" in news.columns:
        values = pd.to_numeric(news["sentiment_value"], errors="raise")
    elif {"score_positive", "score_negative"}.issubset(news.columns):
        values = news["score_positive"] - news["score_negative"]
    elif "sentiment" in news.columns:
        values = news["sentiment"].map(LABEL_TO_VALUE)
        if values.isna().any():
            raise ValueError("sentiment labels must be positive, neutral, or negative")
    else:
        raise ValueError("News must contain FinBERT scores, sentiment_value, or sentiment")

    if values.isna().any():
        raise ValueError("Sentiment values cannot be missing")
    return values.astype(float)


def _news_dates(news: pd.DataFrame) -> pd.Series:
    if "timestamp" in news.columns:
        raw_dates = news["timestamp"]
    elif "published_at" in news.columns:
        raw_dates = news["published_at"]
    elif {"day", "hour_minute"}.issubset(news.columns):
        raw_dates = news["day"].astype(str) + " " + news["hour_minute"].astype(str)
    elif "date" in news.columns:
        raw_dates = news["date"]
    else:
        raise ValueError("News must contain timestamp, published_at, day, or date")

    try:
        return pd.to_datetime(raw_dates, errors="raise", utc=True).dt.normalize().dt.tz_localize(None)
    except (TypeError, ValueError) as error:
        raise ValueError("News publication dates must be valid datetimes") from error


def aggregate_daily_sentiment(
    news: pd.DataFrame,
    taxonomy: dict[str, Any],
    *,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Calculate daily sentiment and forward-fill each sub-industry independently.

    ``sentiment_mean`` is the mean of all article sentiments assigned to the
    sub-industry on that date. ``sentiment`` is the value consumed downstream:
    missing dates after a sub-industry's first observation use its previous day.
    Leading dates with no previous observation remain null rather than inventing
    a sentiment value.
    """
    subindustries = taxonomy_subindustries(taxonomy)
    first_date = pd.Timestamp(start_date)
    last_date = pd.Timestamp(end_date)
    if first_date > last_date:
        raise ValueError("start_date must not be after end_date")

    full_dates = pd.date_range(first_date, last_date, freq="D")
    full_index = pd.MultiIndex.from_product(
        [full_dates, subindustries["subindustria_id"]],
        names=["date", "subindustria_id"],
    )

    if news.empty:
        result = pd.DataFrame(index=full_index).reset_index()
        result["news_count"] = 0
        result["sentiment_mean"] = float("nan")
        result["sentiment"] = float("nan")
    else:
        working = pd.DataFrame(
            {
                "date": _news_dates(news),
                "subindustria_id": news["subindustria_id"],
                "sentiment_value": _sentiment_values(news),
            }
        )
        working = working[
            (working["date"] >= first_date) & (working["date"] <= last_date)
        ]

        means = working.groupby(["date", "subindustria_id"])["sentiment_value"].mean()
        counts = working.groupby(["date", "subindustria_id"]).size()
        result = pd.DataFrame(index=full_index)
        result["news_count"] = counts.reindex(full_index, fill_value=0).astype(int)
        result["sentiment_mean"] = means.reindex(full_index)
        result = result.reset_index()
        result["sentiment"] = result.groupby("subindustria_id")["sentiment_mean"].ffill()

    return result.merge(subindustries, on="subindustria_id", how="left").sort_values(
        ["date", "subindustria_id"]
    ).reset_index(drop=True)