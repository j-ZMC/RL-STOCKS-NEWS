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

# Índice maestro de GDELT: lista TODOS los archivos históricos disponibles (15-min desde 2015)
GDELT_MASTER_FILELIST = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"


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


def collect_gdelt_backfill(writer: BatchCSVWriter, start_date: str, end_date: str,
                            keywords=None, max_files: int = None):
    """
    Descarga histórico masivo de GDELT (GKG - Global Knowledge Graph) entre dos fechas.
    Esto es lo que te da VOLUMEN real (millones de filas), a diferencia del RSS.

    start_date / end_date: "YYYY-MM-DD"
    keywords: lista opcional de palabras para filtrar por empresa/tema (case-insensitive).
              Si es None, trae TODO (cuidado: puede ser gigante).
    max_files: limita cuántos archivos de 15-min procesar (útil para pruebas).
    """
    import io
    import zipfile

    print("[gdelt] descargando índice maestro de archivos...")
    try:
        resp = requests.get(GDELT_MASTER_FILELIST, timeout=60)
        lines = resp.text.splitlines()
    except Exception as e:
        print(f"[gdelt] error descargando índice: {e}")
        return 0

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    # Filtramos solo archivos ".gkg.csv.zip" (contienen título/resumen/tema) dentro del rango de fechas
    candidate_files = []
    for line in lines:
        parts = line.strip().split(" ")
        if len(parts) < 3:
            continue
        url = parts[-1]
        if not url.endswith(".gkg.csv.zip"):
            continue
        # el nombre del archivo empieza con YYYYMMDDHHMMSS
        fname = url.split("/")[-1]
        try:
            file_dt = datetime.strptime(fname[:14], "%Y%m%d%H%M%S")
        except Exception:
            continue
        if start_dt <= file_dt < end_dt:
            candidate_files.append((file_dt, url))

    candidate_files.sort()
    if max_files:
        candidate_files = candidate_files[:max_files]

    print(f"[gdelt] {len(candidate_files)} archivos a procesar entre {start_date} y {end_date}")

    total = 0
    for i, (file_dt, url) in enumerate(candidate_files):
        try:
            r = requests.get(url, timeout=30)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                content = f.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[gdelt] error en {url}: {e}")
            continue

        # GKG es tab-separated. Columnas relevantes: DATE(1), SourceCommonName(4 aprox), DocumentIdentifier(5),
        # V2Themes, V2Tone, ... el título real no viene directo en GKG, pero DocumentIdentifier es la URL
        # y V2Themes/Organizations sirven para filtrar por empresa. Usamos la URL + tono como proxy de "title".
        for row in content.split("\n"):
            cols = row.split("\t")
            if len(cols) < 5:
                continue
            gkg_date = cols[1]  # formato YYYYMMDDHHMMSS
            source_url = cols[4] if len(cols) > 4 else ""
            organizations = cols[10] if len(cols) > 10 else ""

            if keywords and not any(k.lower() in row.lower() for k in keywords):
                continue
            if not source_url:
                continue

            try:
                dt = datetime.strptime(gkg_date, "%Y%m%d%H%M%S")
                had_time = True
            except Exception:
                dt, had_time = datetime.utcnow(), False

            day, hour_minute = normalize_datetime(dt, had_time)
            # GDELT GKG no da título limpio; usamos la URL como "title" y organizations como summary.
            # Para título/resumen legibles de verdad, cruza esta URL después con newspaper3k o similar.
            title = source_url
            summary = organizations

            if writer.add(day, hour_minute, title, summary, "GDELT"):
                total += 1

        if (i + 1) % 20 == 0:
            print(f"[gdelt] procesados {i+1}/{len(candidate_files)} archivos, {total} filas nuevas hasta ahora")
            writer.flush()

    print(f"[gdelt] backfill terminado: {total} noticias nuevas agregadas")
    return total


def gdelt_dry_run(start_date: str, end_date: str, keywords=None, sample_files: int = 20):
    """
    Estima el volumen de un backfill de GDELT SIN descargar todo:
      1. Cuenta cuántos archivos .gkg.csv.zip caen en el rango de fechas (esto es gratis, solo lee el índice).
      2. Descarga una MUESTRA pequeña (sample_files) para estimar tamaño promedio por archivo
         y tasa de coincidencia con tus keywords.
      3. Extrapola: tamaño total de descarga, tiempo estimado, y noticias estimadas que pasarían el filtro.
    """
    import io
    import zipfile

    print("[dry-run] descargando índice maestro de archivos...")
    try:
        resp = requests.get(GDELT_MASTER_FILELIST, timeout=60)
        lines = resp.text.splitlines()
    except Exception as e:
        print(f"[dry-run] error descargando índice: {e}")
        return

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    candidate_files = []
    for line in lines:
        parts = line.strip().split(" ")
        if len(parts) < 3:
            continue
        url = parts[-1]
        if not url.endswith(".gkg.csv.zip"):
            continue
        fname = url.split("/")[-1]
        try:
            file_dt = datetime.strptime(fname[:14], "%Y%m%d%H%M%S")
        except Exception:
            continue
        if start_dt <= file_dt < end_dt:
            # el índice maestro también trae el tamaño en bytes como primer campo
            try:
                size_bytes = int(parts[0])
            except Exception:
                size_bytes = None
            candidate_files.append((file_dt, url, size_bytes))

    candidate_files.sort()
    n_total = len(candidate_files)

    if n_total == 0:
        print("[dry-run] no se encontraron archivos en ese rango de fechas.")
        return

    # Tamaño total comprimido, si el índice trae el dato de tamaño
    known_sizes = [s for (_, _, s) in candidate_files if s]
    if known_sizes:
        avg_size = sum(known_sizes) / len(known_sizes)
        total_compressed_bytes = avg_size * n_total
    else:
        total_compressed_bytes = None

    print(f"[dry-run] {n_total} archivos de 15-min en el rango {start_date} -> {end_date}")
    if total_compressed_bytes:
        print(f"[dry-run] tamaño total comprimido estimado: {total_compressed_bytes / 1e9:.2f} GB")

    # Muestra real para estimar tasa de coincidencia con keywords y tamaño descomprimido
    sample = candidate_files[:: max(1, n_total // sample_files)][:sample_files]
    print(f"[dry-run] descargando muestra de {len(sample)} archivos para estimar coincidencias...")

    matched_rows = 0
    total_rows = 0
    compressed_sample_bytes = 0
    uncompressed_sample_bytes = 0

    for file_dt, url, size_bytes in sample:
        try:
            r = requests.get(url, timeout=30)
            compressed_sample_bytes += len(r.content)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                content = f.read().decode("utf-8", errors="ignore")
            uncompressed_sample_bytes += len(content.encode("utf-8"))
        except Exception as e:
            print(f"[dry-run] error en {url}: {e}")
            continue

        for row in content.split("\n"):
            if not row.strip():
                continue
            total_rows += 1
            if keywords and any(k.lower() in row.lower() for k in keywords):
                matched_rows += 1
            elif not keywords:
                matched_rows += 1

    if not sample or total_rows == 0:
        print("[dry-run] no se pudo muestrear contenido (revisa conexión).")
        return

    match_rate = matched_rows / total_rows
    avg_compressed_per_file = compressed_sample_bytes / len(sample)
    avg_uncompressed_per_file = uncompressed_sample_bytes / len(sample)
    avg_rows_per_file = total_rows / len(sample)

    est_total_compressed_gb = (avg_compressed_per_file * n_total) / 1e9
    est_total_uncompressed_gb = (avg_uncompressed_per_file * n_total) / 1e9
    est_total_matched_rows = int(avg_rows_per_file * n_total * match_rate)
    est_csv_output_mb = (est_total_matched_rows * 400) / 1e6  # ~400 bytes/fila promedio final

    print("\n========== ESTIMACIÓN ==========")
    print(f"Archivos a procesar:              {n_total:,}")
    print(f"Descarga total estimada (zip):     {est_total_compressed_gb:,.2f} GB")
    print(f"Descomprimido temporal estimado:   {est_total_uncompressed_gb:,.2f} GB")
    print(f"Tasa de coincidencia con keywords: {match_rate*100:.2f}%")
    print(f"Noticias estimadas que pasan filtro: {est_total_matched_rows:,}")
    print(f"Tamaño CSV final estimado:          {est_csv_output_mb:,.1f} MB")
    print(f"Tiempo estimado (a ~0.5s/archivo):  {n_total * 0.5 / 3600:.1f} horas")
    print("=================================\n")


# ---------------------------------------------------------------------------
# ENRIQUECIMIENTO: título/resumen real a partir de las URLs guardadas por GDELT
# ---------------------------------------------------------------------------

def enrich_gdelt_rows(csv_path: str, checkpoint_every: int = 200, request_timeout: int = 10,
                       max_rows: int = None, sleep_seconds: float = 0.0):
    """
    Recorre el CSV y, para las filas con source == "GDELT" (donde title es una URL cruda),
    visita la URL y extrae título/resumen real con trafilatura. Sobrescribe el CSV en el mismo
    lugar, con checkpoints periódicos para no perder avance si se corta a la mitad.

    Requiere: pip install trafilatura --break-system-packages
    """
    import trafilatura

    if not os.path.isfile(csv_path):
        print(f"[enrich] no existe el archivo: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    pending_idx = [i for i, r in enumerate(rows) if r["source"] == "GDELT" and r["title"].startswith("http")]

    if max_rows:
        pending_idx = pending_idx[:max_rows]

    print(f"[enrich] {total} filas totales, {len(pending_idx)} pendientes de enriquecer")

    processed = 0
    failed = 0

    for count, i in enumerate(pending_idx, start=1):
        url = rows[i]["title"]
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                failed += 1
                continue

            extracted_title = None
            metadata = trafilatura.extract_metadata(downloaded)
            if metadata and metadata.title:
                extracted_title = metadata.title.strip()

            extracted_text = trafilatura.extract(downloaded, include_comments=False, favor_recall=False)

            if extracted_title:
                rows[i]["title"] = extracted_title
            if extracted_text:
                # resumen: primeros ~500 caracteres del cuerpo extraído
                rows[i]["news_summary"] = extracted_text.strip().replace("\n", " ")[:500]

            # conservamos la URL original en source para trazabilidad
            rows[i]["source"] = f"GDELT|{url}"
            processed += 1

        except Exception as e:
            failed += 1
            print(f"[enrich] error en {url}: {e}")

        if sleep_seconds:
            time.sleep(sleep_seconds)

        if count % checkpoint_every == 0:
            _write_all_rows(csv_path, rows)
            print(f"[enrich] checkpoint: {count}/{len(pending_idx)} procesadas "
                  f"({processed} ok, {failed} fallidas) -> guardado en {csv_path}")

    _write_all_rows(csv_path, rows)
    print(f"[enrich] terminado: {processed} enriquecidas, {failed} fallidas de {len(pending_idx)}")


def _write_all_rows(csv_path: str, rows: list):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# FILTRADO: quedarnos solo con las filas relevantes a tus empresas/temas
# ---------------------------------------------------------------------------

def filter_csv_by_keywords(input_path: str, output_path: str, keywords: list,
                            match_in_url: bool = True, match_in_summary: bool = True):
    """
    Lee un CSV ya descargado (crudo, de GDELT o de RSS) y escribe uno nuevo
    solo con las filas donde alguno de los keywords aparece en la URL/title
    y/o en news_summary (organizations/themes de GDELT).

    Útil para reducir un backfill masivo (millones de filas de TODO el mundo)
    a solo lo que te interesa, ANTES de gastar tiempo enriqueciendo con --enrich.
    """
    if not os.path.isfile(input_path):
        print(f"[filter] no existe el archivo: {input_path}")
        return

    keywords_lower = [k.lower().strip() for k in keywords if k.strip()]
    if not keywords_lower:
        print("[filter] no se dieron keywords, no se filtra nada.")
        return

    total = 0
    kept = 0

    with open(input_path, "r", encoding="utf-8", newline="") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        writer.writeheader()

        for row in reader:
            total += 1
            haystacks = []
            if match_in_url:
                haystacks.append(row.get("title", ""))
            if match_in_summary:
                haystacks.append(row.get("news_summary", ""))

            blob = " ".join(haystacks).lower()

            if any(k in blob for k in keywords_lower):
                writer.writerow(row)
                kept += 1

            if total % 100000 == 0:
                print(f"[filter] {total:,} filas revisadas, {kept:,} coinciden hasta ahora")

    print(f"[filter] terminado: {kept:,}/{total:,} filas coinciden con {keywords} -> {output_path}")


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
    parser.add_argument("--output", default="URL_noticias_empresas.csv", help="Ruta del CSV de salida")
    parser.add_argument("--batch-size", type=int, default=10000, help="Filas por lote antes de escribir a disco")
    parser.add_argument("--interval", type=int, default=900, help="Segundos entre ciclos (default 15 min)")
    parser.add_argument("--newsapi-key", default=None, help="API key de newsapi.org (opcional)")
    parser.add_argument("--once", action="store_true", help="Ejecutar un solo ciclo y salir")

    # Modo backfill histórico (para volumen masivo, esto es lo que quieres para "millones de noticias")
    parser.add_argument("--backfill", action="store_true", help="Descargar histórico de GDELT en vez de correr el loop en vivo")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD (requerido con --backfill)")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD (requerido con --backfill)")
    parser.add_argument("--keywords", default=None, help="Palabras separadas por coma para filtrar por empresa, ej: Tesla,Apple,Nvidia")
    parser.add_argument("--max-files", type=int, default=None, help="Límite de archivos de 15-min a procesar (para pruebas)")

    # Modo dry-run: estima tamaño/volumen SIN descargar todo
    parser.add_argument("--dry-run", action="store_true", help="Estimar tamaño/volumen del backfill sin descargarlo completo")
    parser.add_argument("--sample-files", type=int, default=20, help="Cuántos archivos muestrear para el dry-run")

    # Modo enrich: rellena title/news_summary reales a partir de las URLs de GDELT
    parser.add_argument("--enrich", action="store_true", help="Enriquecer filas de GDELT con título/resumen real (visita cada URL)")
    parser.add_argument("--enrich-max-rows", type=int, default=None, help="Límite de filas a enriquecer (para pruebas)")
    parser.add_argument("--enrich-sleep", type=float, default=0.0, help="Segundos de espera entre requests (para no saturar sitios)")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="Cada cuántas filas guardar progreso")

    # Modo filter: reduce un CSV crudo a solo las filas relevantes a tus keywords
    parser.add_argument("--filter", action="store_true", help="Filtrar un CSV existente por keywords, sin re-descargar nada")
    parser.add_argument("--filter-input", default=None, help="CSV de entrada a filtrar (ej. el crudo de GDELT)")
    parser.add_argument("--filter-output", default=None, help="CSV de salida ya filtrado")

    args = parser.parse_args()

    if args.filter:
        if not args.filter_input or not args.filter_output or not args.keywords:
            parser.error("--filter requiere --filter-input, --filter-output y --keywords")
        kw = args.keywords.split(",")
        filter_csv_by_keywords(args.filter_input, args.filter_output, kw)
    elif args.enrich:
        enrich_gdelt_rows(args.output, checkpoint_every=args.checkpoint_every,
                           max_rows=args.enrich_max_rows, sleep_seconds=args.enrich_sleep)
    elif args.dry_run:
        if not args.start_date or not args.end_date:
            parser.error("--dry-run requiere --start-date y --end-date")
        kw = args.keywords.split(",") if args.keywords else None
        gdelt_dry_run(args.start_date, args.end_date, keywords=kw, sample_files=args.sample_files)
    elif args.backfill:
        if not args.start_date or not args.end_date:
            parser.error("--backfill requiere --start-date y --end-date")
        writer = BatchCSVWriter(args.output, batch_size=args.batch_size)
        kw = args.keywords.split(",") if args.keywords else None
        collect_gdelt_backfill(writer, args.start_date, args.end_date, keywords=kw, max_files=args.max_files)
        writer.flush()
    else:
        run(args.output, args.batch_size, args.interval, args.newsapi_key, args.once)