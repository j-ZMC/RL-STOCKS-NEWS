import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
import re
from urllib.parse import unquote


def extract_text(target_url, timeout=10, log_errors=True):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # Petición HTTP a la página web
        response = requests.get(target_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Analizar el HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Eliminar elementos no visibles (código JS, estilos CSS, etc.)
        for element in soup(["script", "style", "header", "footer", "nav", "noscript"]):
            element.decompose()
            
        # Extraer el texto separando los bloques por espacios en blanco
        text = soup.get_text(separator=' ', strip=True)
        
        return text
        
    except requests.exceptions.RequestException as error:
        if log_errors:
            print(f"Error accessing {target_url}: {error}")
        return ""


def text_from_url(target_url):
    """Return a readable title from a URL when the publisher is unavailable."""
    parsed_url = urlparse(target_url)
    slug = unquote(parsed_url.path).rstrip("/").split("/")[-1]
    slug = re.sub(r"\.[a-z0-9]{2,5}$", "", slug, flags=re.IGNORECASE)
    slug = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "",
        slug,
        flags=re.IGNORECASE,
    )
    slug = re.sub(r"[-_]+", " ", slug)
    return re.sub(r"\s+", " ", slug).strip(" -")


def _extract_url_text(url_timeout):
    url, timeout, fetch_url = url_timeout
    if fetch_url:
        text = extract_text(url, timeout=timeout, log_errors=False)
        if text:
            return url, text, "web"
    return url, text_from_url(url), "url"


def extract_csv_text(
    input_path,
    output_path,
    max_workers=8,
    timeout=10,
    checkpoint_every=1000,
    max_rows=None,
    fetch_urls=True,
):
    """Extract article text from URL values in the CSV ``title`` column."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    temporary_path = f"{output_path}.tmp"
    processed = 0
    failed = 0

    with open(input_path, "r", encoding="utf-8", newline="") as input_file, \
            open(temporary_path, "w", encoding="utf-8", newline="") as output_file, \
            ThreadPoolExecutor(max_workers=max_workers) as executor:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames or "title" not in reader.fieldnames:
            raise ValueError("The CSV must contain a title column")

        fieldnames = list(reader.fieldnames)
        if "article_text" not in fieldnames:
            fieldnames.append("article_text")
        if "article_text_source" not in fieldnames:
            fieldnames.append("article_text_source")
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        while max_rows is None or processed < max_rows:
            batch = []
            for _ in range(min(checkpoint_every, max_rows - processed) if max_rows else checkpoint_every):
                try:
                    batch.append(next(reader))
                except StopIteration:
                    break
            if not batch:
                break

            urls = {
                row["title"]
                for row in batch
                if row.get("title", "").startswith(("http://", "https://"))
            }
            extracted = {
                item[0]: item[1:]
                for item in executor.map(
                    _extract_url_text,
                    ((url, timeout, fetch_urls) for url in urls),
                )
            }

            for row in batch:
                url = row.get("title", "")
                row["article_text"], row["article_text_source"] = extracted.get(
                    url, ("", "none")
                )
                if url.startswith(("http://", "https://")) and not row["article_text"]:
                    failed += 1
                writer.writerow(row)

            processed += len(batch)
            output_file.flush()
            print(f"[extract] {processed} filas procesadas, {failed} sin texto")

    os.replace(temporary_path, output_path)
    print(f"[extract] terminado: {processed} filas -> {output_path}")
    return processed, failed

def extract_urls(target_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # Fetch web page content
        response = requests.get(target_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        discovered_urls = set()
        
        # Locate all anchor tags with href attributes
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            # Convert relative paths into absolute URLs
            absolute_url = urljoin(target_url, href)
            
            # Clean and filter to ensure it is a valid HTTP(S) link
            parsed_url = urlparse(absolute_url)
            if parsed_url.scheme in ['http', 'https']:
                discovered_urls.add(absolute_url)
                
        return sorted(list(discovered_urls))
        
    except requests.exceptions.RequestException as error:
        print(f"Error accessing {target_url}: {error}")
        return []

# Example execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract article text from a CSV title URL column")
    parser.add_argument("--input-csv")
    parser.add_argument("--output-csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    if args.input_csv and args.output_csv:
        extract_csv_text(
            args.input_csv,
            args.output_csv,
            max_workers=args.workers,
            timeout=args.timeout,
            checkpoint_every=args.checkpoint_every,
            max_rows=args.max_rows,
        )
    else:
        target = "https://mexiconewsdaily.com/mexicolife/atole-beverage-of-champions/"
        links = extract_urls(target)
        text = extract_text(target)

        print(f"\nFound {len(links)} unique URLs:")
        for link in links:
            print(link)

        print(f"\nExtracted text ({len(text)} characters):")
        print(text)