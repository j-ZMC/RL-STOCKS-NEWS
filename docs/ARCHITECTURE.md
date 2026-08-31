# Architecture

## Ownership

| Area | Canonical entry point | Responsibility |
| --- | --- | --- |
| Orchestration | `pipeline_sentimiento_diario.py` | Coordinates the complete daily news workflow. |
| News ingestion | `Internet_scraper/URL_news_bot.py` | GDELT/RSS collection, enrichment, and date-range handling. |
| Taxonomy classification | `Internet_scraper/noticias_classifier_faiss.py` | FAISS search and multi-label sub-industry assignment. |
| Sentiment inference | `finBERT_model/finBERT.py` | Lazy FinBERT loading, device selection, and article scoring. |
| Daily aggregation | `finBERT_model/daily_sentiment.py` | Full date/sub-industry grid and per-sub-industry forward fill. |
| Taxonomy preparation | `Internet_scraper/preparar_taxonomia.py` | Converts source taxonomy data into optimized category files. |
| Tests | `tests/` | Lightweight aggregation and interval behavior checks. |

## Taxonomy

The canonical file is `GICS_Taxonomy_Data/optimized_taxonomy.json`. It is a flat `categorias` file with 153 entries. The FAISS classifier and daily aggregation layer adapt this format to their internal nested representation.

## Alternate and Legacy Modules

These files remain available because they have different behavior or historical users:

- `Internet_scraper/Updated_News.py` and `Internet_scraper/URL_TEXT_noticas.py`: alternate ingestion implementations.
- `Internet_scraper/noticias_classifier_produccion.py`: more configurable production classifier.
- `Internet_scraper/URL_TO_TEXT_Working.py`: standalone URL extraction utility.
- `Internet_scraper/discriminatory_outlet.py`: outlet filtering utility.
- `Company_clasification_model/`: separate company classification workflow.
- `S&P500_stock_data/`: separate market-data workflow.
- `XHTML-10K/`: separate SEC filing workflow.

Do not delete or merge these modules solely because their names overlap. Compare schemas and run a fixture through both implementations first.

## Data Lifecycle

```text
GDELT/RSS
  -> data/noticias_2022_2024_raw.csv
  -> data/noticias_2022_2024_prepared.csv
  -> data/noticias_2022_2024_classified.csv
  -> data/noticias_2022_2024_finbert.csv
  -> data/sentimiento_diario_subindustrias.csv
```

Local model weights, downloaded filings, databases, CSV datasets, virtual environments, and Python caches are runtime artifacts rather than source code.