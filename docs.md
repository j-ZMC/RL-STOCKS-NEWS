cd Internet_scraper

python URL_news_bot.py --filter --filter-input URL_noticias_empresas.csv --filter-output filtrado.csv --keywords Tesla,Apple,Nvidia,Amazon,Meta

python URL_news_bot.py --backfill --start-date 2022-01-01 --end-date 2023-12-31

# Pipeline completo: GDELT 2022-01-01 a 2024-12-31, FAISS multi-etiqueta,
# FinBERT y sentimiento diario por cada sub-industria de la taxonomía.
cd ..
python pipeline_sentimiento_diario.py --device auto

# Prueba de la agregación diaria sin descargar noticias ni cargar modelos.
python -m unittest discover -s tests -p "test_daily_sentiment.py"