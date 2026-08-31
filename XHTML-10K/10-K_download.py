from pathlib import Path
from sec_edgar_downloader import Downloader
import html2text

def downloader(ticket):
    save_path = "C:/Users/jesus/OneDrive/Escritorio/Ingest_data/Ollama_3.1_8B"

    dl = Downloader("AnalisisFinanciero", "jesuszahidmorenocalderon@gmail.com", save_path)
    
    # IMPORTANTE: download_details=True fuerza a descargar los archivos .htm individuales
    dl.get("10-K", ticket, limit=1, download_details=True)

    filings_dir = Path(save_path) / "sec-edgar-filings" / ticket / "10-K"

    # Buscar los archivos .htm (filtrando los reportes secundarios)
    html_files = [f for f in filings_dir.rglob("*.htm") if not f.name.startswith("R")]

    if not html_files:
        print("No se encontraron archivos .htm. Revisa la carpeta descargada.")
        return

    # El archivo más grande en bytes suele ser el HTML del 10-K principal
    primary_html = max(html_files, key=lambda f: f.stat().st_size)
    print(f"HTML encontrado: {primary_html.name}")

    # Leer HTML y convertir a Markdown
    with open(primary_html, "r", encoding="utf-8") as f:
        html_content = f.read()

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0

    markdown_text = h.handle(html_content)

    md_path = primary_html.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"¡Markdown generado con éxito en:\n{md_path}")

downloader("AAPL")