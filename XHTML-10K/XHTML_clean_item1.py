import re
from pathlib import Path
from bs4 import BeautifulSoup

def extraer_item1_limpio(html_path):
    # 1. Leer el HTML descargado
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 2. Parsear con BeautifulSoup para eliminar etiquetas no visibles y scripts
    soup = BeautifulSoup(html_content, "html.parser")

    # Eliminar bloques ocultos o de metadatos de XBRL (como <div style="display:none">)
    for hidden_tag in soup.find_all(style=re.compile(r"display:\s*none", re.IGNORECASE)):
        hidden_tag.decompose()

    # Extraer todo el texto plano manteniendo espacios
    text = soup.get_text(separator=" ")

    # Normalizar espacios múltiples y saltos de línea
    text_clean = re.sub(r"\s+", " ", text)

    # 3. Buscar el inicio del Item 1 y el inicio del Item 1A (Risk Factors)
    # Patrón para localizar 'Item 1. Business' ignorando mayúsculas/minúsculas
    pattern_item1 = re.compile(r"ITEM\s+1\.\s+BUSINESS", re.IGNORECASE)
    pattern_item1a = re.compile(r"ITEM\s+1A\.\s+RISK\s+FACTORS", re.IGNORECASE)

    matches_1 = list(pattern_item1.finditer(text_clean))
    matches_1a = list(pattern_item1a.finditer(text_clean))

    if not matches_1 or not matches_1a:
        print("No se encontraron los delimitadores estándar del Item 1.")
        return None

    # Tomar la ÚLTIMA coincidencia del Item 1 (para saltarse el índice/Table of Contents)
    start_pos = matches_1[-1].start()
    
    # Tomar la primera coincidencia del Item 1A que ocurra DESPUÉS del Item 1
    end_pos = None
    for m in matches_1a:
        if m.start() > start_pos:
            end_pos = m.start()
            break

    if not end_pos:
        print("No se pudo delimitar el final del Item 1.")
        return None

    # 4. Recortar el Item 1
    item_1_text = text_clean[start_pos:end_pos].strip()
    return item_1_text

# Ejemplo de uso apuntando al archivo primary-document.html descargado
html_file = Path("C:/Users/jesus/OneDrive/Escritorio/Ingest_data/Ollama_3.1_8B/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/primary-document.html")

item1_texto = extraer_item1_limpio(html_file)

if item1_texto:
    # Guardar la sección limpia lista para Llama
    output_path = html_file.parent / "item_1_clean.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(item1_texto)
    
    print(f"¡Item 1 extraído con éxito! Palabras aproximadas: {len(item1_texto.split())}")