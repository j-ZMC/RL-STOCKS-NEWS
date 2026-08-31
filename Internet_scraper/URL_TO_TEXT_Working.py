import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def extract_text(target_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # Petición HTTP a la página web
        response = requests.get(target_url, headers=headers, timeout=10)
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
        print(f"Error accessing {target_url}: {error}")
        return ""

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
    target = "https://mexiconewsdaily.com/mexicolife/atole-beverage-of-champions/"  # Replace with your target URL
    links = extract_urls(target)
    text = extract_text(target)
    
    print(f"\nFound {len(links)} unique URLs:")
    for link in links:
        print(link)
    
    print(f"\nExtracted text ({len(text)} characters):")
    print(text)