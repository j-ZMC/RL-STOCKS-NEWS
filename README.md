# Financial News and Market Data Pipeline

This project collects financial news, filters and categorizes articles with a GICS taxonomy and FAISS, scores article sentiment with FinBERT, and produces daily sentiment for every taxonomy sub-industry. It also contains separate market-data and SEC filing utilities.

## Start Here

The active end-to-end news workflow is:

```powershell
& ".\venv\Scripts\python.exe" pipeline_sentimiento_diario.py --device auto
```

The default taxonomy is `GICS_Taxonomy_Data/optimized_taxonomy.json`. It currently contains 153 categories. The pipeline derives the category count from the taxonomy, so it does not hardcode 153 or 163.

Install dependencies with:

```powershell
& ".\venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Use `--device auto` to select CUDA when available. Use `--device cpu` to force CPU execution.

## Repository Map

```text
pipeline_sentimiento_diario.py   Active end-to-end news pipeline
Internet_scraper/                News collection, enrichment, filtering, and FAISS classification
finBERT_model/                   FinBERT inference and sentiment aggregation
GICS_Taxonomy_Data/              Canonical GICS taxonomy files
modelo_gics_local/               Local Sentence Transformers model assets
Company_clasification_model/     Standalone company classification utilities
S&P500_stock_data/               Market data and technical indicators
XHTML-10K/                      SEC filing download and cleaning utilities
data/                            Raw, intermediate, and generated pipeline files
tests/                           Automated aggregation tests
docs/                            Architecture and workflow documentation
```

## Active News Workflow

1. Fetch GDELT articles from `2022-01-01` through `2024-12-31`.
2. Extract full visible article text from the URL stored in `title`.
3. Preserve the publication timestamp and build classifier text from `article_text`.
4. Use FAISS against every taxonomy category and retain all labels over the confidence threshold.
5. Run FinBERT once per unique article, even when an article has multiple category labels.
6. Calculate the daily mean per sub-industry.
7. Create a complete date/sub-industry grid and forward-fill missing days independently for each sub-industry.

Generated files are written under `data/`:

```text
noticias_2022_2024_raw.csv
noticias_2022_2024_extracted.csv
noticias_2022_2024_prepared.csv
noticias_2022_2024_classified.csv
noticias_2022_2024_finbert.csv
sentimiento_diario_subindustrias.csv
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md): responsibilities, active modules, and legacy alternatives.
- [Pipeline guide](docs/PIPELINE.md): commands, options, inputs, outputs, and troubleshooting.
- [Existing command notes](docs.md): original operational commands retained for compatibility.

## Validation

Run the lightweight tests without downloading data or model weights:

```powershell
& ".\venv\Scripts\python.exe" -m unittest discover -s tests -v
```

The full pipeline downloads a multi-year GDELT range and may require substantial disk space, network time, RAM, and model cache space.