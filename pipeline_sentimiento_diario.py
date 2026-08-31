"""End-to-end news classification and daily sub-industry sentiment pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from finBERT_model.daily_sentiment import aggregate_daily_sentiment, taxonomy_subindustries


DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_TAXONOMY = "GICS_Taxonomy_Data/optimized_taxonomy.json"


def fetch_news(start_date: str, end_date: str, output_path: str, keywords: Optional[list[str]] = None) -> None:
    """Fetch the requested GDELT range into the raw news CSV."""
    from Internet_scraper.URL_news_bot import BatchCSVWriter, collect_gdelt_backfill

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = BatchCSVWriter(output_path)
    collect_gdelt_backfill(writer, start_date, end_date, keywords=keywords)
    writer.flush()


def enrich_news(
    input_path: str,
    output_path: str,
    max_workers: int = 8,
    timeout: int = 10,
    checkpoint_every: int = 1000,
    fetch_urls: bool = True,
) -> None:
    """Extract full article text from URLs in the raw news CSV."""
    from Internet_scraper.URL_TO_TEXT_Working import extract_csv_text

    extract_csv_text(
        input_path,
        output_path,
        max_workers=max_workers,
        timeout=timeout,
        checkpoint_every=checkpoint_every,
        fetch_urls=fetch_urls,
    )


def prepare_news(
    input_path: str,
    output_path: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> None:
    """Create one classifier text field while retaining exact publication time."""
    news = pd.read_csv(input_path, low_memory=False)
    if {"day", "hour_minute"}.issubset(news.columns):
        raw_timestamp = news["day"].astype(str) + " " + news["hour_minute"].astype(str)
        news["published_at"] = pd.to_datetime(raw_timestamp, errors="raise", utc=True)
    elif "published_at" not in news.columns and "timestamp" not in news.columns:
        raise ValueError("News input must contain day/hour_minute or published_at/timestamp")

    timestamp_column = "published_at" if "published_at" in news.columns else "timestamp"
    timestamps = pd.to_datetime(news[timestamp_column], errors="raise", utc=True)
    first_timestamp = pd.Timestamp(start_date, tz="UTC")
    last_timestamp = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    news = news[(timestamps >= first_timestamp) & (timestamps < last_timestamp)].copy()

    title = news.get("title", news.get("headline", pd.Series("", index=news.index))).fillna("")
    summary = news.get("news_summary", pd.Series("", index=news.index)).fillna("")
    fallback_text = (title.astype(str) + ". " + summary.astype(str)).str.strip()
    article_text = news.get("article_text", pd.Series("", index=news.index)).fillna("").astype(str)
    source = news.get("article_text_source", pd.Series("web", index=news.index)).fillna("web")
    url_text = (article_text + ". " + summary.astype(str)).str.strip()
    classifier_text = article_text.where(source.ne("url"), url_text)
    news["headline"] = classifier_text.where(classifier_text.str.strip().ne(""), fallback_text)
    news["article_id"] = range(len(news))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    news.to_csv(output_path, index=False, encoding="utf-8")


def classify_news(
    input_path: str,
    output_path: str,
    taxonomy_path: str,
    confidence_threshold: float = 0.55,
    batch_size: int = 128,
    device: str = "auto",
) -> int:
    """Filter stock-relevant articles and retain every qualifying sub-industry."""
    from Internet_scraper.noticias_classifier_faiss import CGICNewsClassifierFAISS
    from finBERT_model.finBERT import resolver_dispositivo

    classifier = CGICNewsClassifierFAISS(
        confidence_threshold=confidence_threshold,
        device=resolver_dispositivo(device),
    )
    classifier.cargar_taxonomia(taxonomy_path)
    top_k = len(classifier.subindustrias_mapping)
    if top_k == 0:
        raise ValueError("The taxonomy index contains no sub-industries")

    _, classifications = classifier.procesar_csv_batch(
        input_path,
        output_path,
        batch_size=batch_size,
        columna_titular="headline",
        top_k=top_k,
    )
    return classifications


def score_with_finbert(
    input_path: str,
    output_path: str,
    batch_size: int = 32,
    device: str = "auto",
) -> None:
    """Score each article once; duplicate assignments share its article sentiment."""
    from finBERT_model.finBERT import analizar_sentimiento, cargar_modelo_finbert

    news = pd.read_csv(input_path, low_memory=False)
    article_text = news[["article_id", "headline"]].drop_duplicates("article_id")
    tokenizer, model, device = cargar_modelo_finbert(device=device)
    scored_rows = []

    for start in range(0, len(article_text), batch_size):
        batch = article_text.iloc[start:start + batch_size]
        predictions = analizar_sentimiento(
            batch["headline"].fillna("").tolist(), tokenizer, model, device
        )
        for (_, row), prediction in zip(batch.iterrows(), predictions):
            scored_rows.append({
                "article_id": row["article_id"],
                "sentiment": prediction["sentiment"],
                "score_positive": prediction["score_positive"],
                "score_negative": prediction["score_negative"],
                "score_neutral": prediction["score_neutral"],
                "sentiment_value": prediction["score_positive"] - prediction["score_negative"],
            })

    scored = news.merge(pd.DataFrame(scored_rows), on="article_id", how="left", validate="many_to_one")
    output_columns = list(news.columns) + [
        "sentiment", "score_positive", "score_negative", "score_neutral", "sentiment_value"
    ]
    scored[output_columns].to_csv(output_path, index=False, encoding="utf-8")


def aggregate_sentiment(
    input_path: str,
    output_path: str,
    taxonomy_path: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> int:
    """Write one row for every day and taxonomy sub-industry."""
    with open(taxonomy_path, "r", encoding="utf-8") as taxonomy_file:
        taxonomy = json.load(taxonomy_file)

    scored_news = pd.read_csv(input_path, low_memory=False)
    daily = aggregate_daily_sentiment(
        scored_news,
        taxonomy,
        start_date=start_date,
        end_date=end_date,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False, encoding="utf-8")
    return len(taxonomy_subindustries(taxonomy))


def run_pipeline(args: argparse.Namespace) -> None:
    raw_path = args.raw_news
    prepared_path = args.prepared_news
    classified_path = args.classified_news
    scored_path = args.scored_news

    if not args.skip_fetch:
        keywords = args.keywords.split(",") if args.keywords else None
        fetch_news(args.start_date, args.end_date, raw_path, keywords)
    if not args.skip_enrich:
        enrich_news(
            raw_path,
            args.extracted_news,
            max_workers=args.extraction_workers,
            timeout=args.extraction_timeout,
            checkpoint_every=args.extraction_checkpoint_every,
            fetch_urls=args.extraction_mode == "web",
        )
        preparation_input = args.extracted_news
    else:
        preparation_input = args.extracted_news if Path(args.extracted_news).exists() else raw_path

    prepare_news(preparation_input, prepared_path, args.start_date, args.end_date)
    classify_news(
        prepared_path,
        classified_path,
        args.taxonomy,
        confidence_threshold=args.threshold,
        batch_size=args.classification_batch_size,
        device=args.device,
    )
    score_with_finbert(
        classified_path,
        scored_path,
        batch_size=args.sentiment_batch_size,
        device=args.device,
    )
    total_subindustries = aggregate_sentiment(
        scored_path,
        args.daily_sentiment,
        args.taxonomy,
        args.start_date,
        args.end_date,
    )
    print(f"Daily sentiment written for {total_subindustries} taxonomy sub-industries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taxonomy",
        default=DEFAULT_TAXONOMY,
        help=f"Taxonomy JSON used by FAISS (default: {DEFAULT_TAXONOMY})",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--raw-news", default="data/noticias_2022_2024_raw.csv")
    parser.add_argument("--extracted-news", default="data/noticias_2022_2024_extracted.csv")
    parser.add_argument("--prepared-news", default="data/noticias_2022_2024_prepared.csv")
    parser.add_argument("--classified-news", default="data/noticias_2022_2024_classified.csv")
    parser.add_argument("--scored-news", default="data/noticias_2022_2024_finbert.csv")
    parser.add_argument("--daily-sentiment", default="data/sentimiento_diario_subindustrias.csv")
    parser.add_argument("--keywords", help="Optional comma-separated prefilter for GDELT")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--classification-batch-size", type=int, default=128)
    parser.add_argument("--sentiment-batch-size", type=int, default=32)
    parser.add_argument("--extraction-workers", type=int, default=8)
    parser.add_argument("--extraction-timeout", type=int, default=10)
    parser.add_argument("--extraction-checkpoint-every", type=int, default=1000)
    parser.add_argument(
        "--extraction-mode",
        choices=["url", "web"],
        default="url",
        help="Use URL text locally, or request each publisher page",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Model device: auto selects CUDA when available",
    )
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-enrich", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())