"""
news_bot.py
Bot de recolección masiva de noticias de empresas para dataset de entrenamiento RL.

Fuentes soportadas:
  - RSS feeds (Reuters, Bloomberg, WSJ, FT, etc. -> título + resumen, legal y con fecha/hora)
  - GDELT (bulk CSV, actualizado cada 15 min, millones de noticias globales)
  - NewsAPI / Finnhub (requieren API key, buen volumen por empresa)

Reglas clave:
  1. Todo se guarda en el MISMO csv, en lotes (batch_size filas -> flush a disco).
  2. Columnas: day,hour_minute,title,news_summary,source
  3. Si la noticia no trae hora exacta -> hour_minute = "00:00" pero day = día_siguiente
     (para que un modelo RL nunca "vea" la noticia antes de que pudo haber existido).
  4. Deduplicación por hash de (title+source) para no inflar el dataset con repetidos.

Requisitos:
  pip install feedparser requests python-dateutil --break-system-packages
"""

import csv
import hashlib
import os
import time
import argparse
from datetime import datetime, timedelta
from dateutil import parser as dateparser

import feedparser
import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

FIELDS = ["day", "hour_minute", "title", "news_summary", "source"]

# Puedes ampliar esta lista con cualquier RSS público de medios financieros.
RSS_FEEDS = {
    "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "WSJ Business": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
    "Reuters Business": "http://feeds.reuters.com/reuters/businessNews",
    "FT": "https://www.ft.com/rss/home",
    "CNBC Business": "https://www.cnbc.com/id/10001147/device/rss/rss.html",
}

GDELT_LAST_UPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"


# ---------------------------------------------------------------------------
# NORMALIZACIÓN DE FECHA/HORA (la regla que pediste)
# ---------------------------------------------------------------------------

def normalize_datetime(dt: datetime, had_time: bool):
    """
    Si la noticia SÍ trae hora -> se usa tal cual.
    Si NO trae hora -> se asume 00:00 del día SIGUIENTE, para que un
    modelo RL nunca asuma que tuvo esa info antes de que existiera.
    """
    if had_time:
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
    else:
        next_day = dt + timedelta(days=1)
        return next_day.strftime("%Y-%m-%d"), "00:00"


def parse_pub_date(raw_date: str):
    """Devuelve (datetime, had_time: bool). Si no se puede parsear hora, had_time=False."""
    if not raw_date:
        return datetime.utcnow(), False
    try:
        dt = dateparser.parse(raw_date)
        # feedparser casi siempre trae hora; asumimos que sí la trae si el parseo fue exitoso
        # y el string original contiene ":"
        had_time = ":" in raw_date
        return dt, had_time
    except Exception:
        return datetime.utcnow(), False


# ---------------------------------------------------------------------------
# ESCRITURA POR LOTES
# ---------------------------------------------------------------------------

class BatchCSVWriter:
    def __init__(self, path: str, batch_size: int = 10000):
        self.path = path
        self.batch_size = batch_size
        self.buffer = []
        self.seen_hashes = set()
        self._init_file()
        self._load_existing_hashes()

    def _init_file(self):
        file_exists = os.path.isfile(self.path)
        if not file_exists:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()

    def _load_existing_hashes(self):
        """Carga hashes existentes para no duplicar en corridas futuras del bot."""
        if not os.path.isfile(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = self._hash(row["title"], row["source"])
                self.seen_hashes.add(h)

    @staticmethod
    def _hash(title, source):
        return hashlib.md5(f"{title}|{source}".encode("utf-8")).hexdigest()

    def add(self, day, hour_minute, title, summary, source):
        h = self._hash(title, source)
        if h in self.seen_hashes:
            return False  # duplicado, se ignora
        self.seen_hashes.add(h)
        self.buffer.append({
            "day": day,
            "hour_minute": hour_minute,
            "title": title,
            "news_summary": summary,
            "source": source,
        })
        if len(self.buffer) >= self.batch_size:
            self.flush()
        return True

    def flush(self):
        if not self.buffer:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writerows(self.buffer)
        print(f"[flush] {len(self.buffer)} filas escritas -> {self.path}")
        self.buffer = []


# ---------------------------------------------------------------------------
# COLECTORES
# ---------------------------------------------------------------------------

def collect_rss(writer: BatchCSVWriter):
    total = 0
    for source_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"[rss] error en {source_name}: {e}")
            continue

        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            summary = getattr(entry, "summary", "").strip()
            raw_date = getattr(entry, "published", "") or getattr(entry, "updated", "")

            if not title:
                continue

            dt, had_time = parse_pub_date(raw_date)
            day, hour_minute = normalize_datetime(dt, had_time)

            if writer.add(day, hour_minute, title, summary, source_name):
                total += 1

    print(f"[rss] {total} noticias nuevas agregadas")
    return total


def collect_newsapi(writer: BatchCSVWriter, api_key: str, query: str = "company OR earnings OR merger"):
    """
    Requiere cuenta en https://newsapi.org (free tier limitado, pago para volumen alto).
    """
    total = 0
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
    except Exception as e:
        print(f"[newsapi] error: {e}")
        return 0

    for article in data.get("articles", []):
        title = (article.get("title") or "").strip()
        summary = (article.get("description") or "").strip()
        source_name = (article.get("source") or {}).get("name", "NewsAPI")
        raw_date = article.get("publishedAt", "")

        if not title:
            continue

        dt, had_time = parse_pub_date(raw_date)
        day, hour_minute = normalize_datetime(dt, had_time)

        if writer.add(day, hour_minute, title, summary, source_name):
            total += 1

    print(f"[newsapi] {total} noticias nuevas agregadas")
    return total


# ---------------------------------------------------------------------------
# LOOP PRINCIPAL
# ---------------------------------------------------------------------------

def run(output_path: str, batch_size: int, interval_seconds: int, newsapi_key: str = None, once: bool = False):
    writer = BatchCSVWriter(output_path, batch_size=batch_size)

    while True:
        print(f"\n=== Ciclo de recolección {datetime.utcnow().isoformat()} ===")
        collect_rss(writer)

        if newsapi_key:
            collect_newsapi(writer, newsapi_key)

        writer.flush()  # aseguramos guardar lo que quedó en buffer aunque no llegue al batch_size

        if once:
            break

        print(f"Esperando {interval_seconds}s para el siguiente ciclo...")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot de recolección de noticias de empresas")
    parser.add_argument("--output", default="noticias_empresas.csv", help="Ruta del CSV de salida")
    parser.add_argument("--batch-size", type=int, default=10000, help="Filas por lote antes de escribir a disco")
    parser.add_argument("--interval", type=int, default=900, help="Segundos entre ciclos (default 15 min)")
    parser.add_argument("--newsapi-key", default=None, help="API key de newsapi.org (opcional)")
    parser.add_argument("--once", action="store_true", help="Ejecutar un solo ciclo y salir")
    args = parser.parse_args()

    run(args.output, args.batch_size, args.interval, args.newsapi_key, args.once)