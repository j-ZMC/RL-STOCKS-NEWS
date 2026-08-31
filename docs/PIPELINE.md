# Daily News Pipeline

## Standard Run

From the project root:

```powershell
& ".\venv\Scripts\python.exe" pipeline_sentimiento_diario.py --device auto
```

Defaults:

- Start date: `2022-01-01`
- End date: `2024-12-31` inclusive
- Taxonomy: `GICS_Taxonomy_Data/optimized_taxonomy.json`
- Classification threshold: `0.55`
- Device: `auto`

## Useful Options

```powershell
# Force CPU
& ".\venv\Scripts\python.exe" pipeline_sentimiento_diario.py --device cpu

# Request CUDA and fall back to CPU if it is unavailable
& ".\venv\Scripts\python.exe" pipeline_sentimiento_diario.py --device cuda

# Use a small pre-filter before FAISS to reduce the GDELT volume
& ".\venv\Scripts\python.exe" pipeline_sentimiento_diario.py --keywords "Tesla,Apple,Nvidia"

# Reuse completed URL extraction and run only preparation/models
& ".\venv\Scripts\python.exe" pipeline_sentimiento_diario.py --skip-fetch --skip-enrich
```

## Processing Rules

- Article publication time is preserved in `published_at`.
- `URL_TO_TEXT_Working.py` writes full extracted page text to `article_text` while preserving the raw URL in `title`.
- FAISS and FinBERT both consume the prepared `headline` field, which is sourced from `article_text` with a title/summary fallback.
- FAISS can assign multiple taxonomy sub-industries to one article.
- FinBERT sentiment is represented as `score_positive - score_negative`.
- `sentiment_mean` is the arithmetic daily mean for a sub-industry.
- `sentiment` is the downstream value; missing dates after the first observation carry forward that sub-industry's previous value.
- A sub-industry with no observations has null sentiment until its first observation; no artificial starting value is invented.

## Validation

```powershell
& ".\venv\Scripts\python.exe" -m unittest discover -s tests -v
& ".\venv\Scripts\python.exe" -m py_compile pipeline_sentimiento_diario.py finBERT_model\daily_sentiment.py
```

The complete run is intentionally separate from validation because GDELT collection, article extraction, FAISS encoding, and FinBERT inference can take significant time and storage.