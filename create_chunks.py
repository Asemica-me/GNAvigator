import os
from dotenv import load_dotenv
import asyncio
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import httpx
from urllib.parse import urljoin
import time
import hashlib
import spacy
from mistralai import Mistral, UserMessage
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import InferenceClient, login
from transformers import pipeline

# Load the Italian spaCy model
nlp = spacy.load("it_core_news_sm")

load_dotenv()
# api_key = os.getenv("MISTRAL_API_KEY")
# llm = os.getenv("MODEL")
# client = Mistral(api_key=api_key)


api_key = os.getenv("HF_TOKEN")
login(token=api_key)
llm = "mistralai/Mistral-7B-Instruct-v0.2"
client = InferenceClient(model=llm, token=api_key)

# Global variables
SITEMAP_PATH = 'GNA_sitemap.xml'
BASE_DOMAIN = 'https://gna.cultura.gov.it'

OUTPUT_FOLDER = "data"
OUTPUT_FILENAME = "chunks_memory.json"

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)

def generate_chunk_id(source: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{source}-{chunk_index}".encode()).hexdigest()

def get_urls_from_sitemap_file(file_path: str) -> list:
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = []
        for url in root.findall('.//ns:url', namespaces):
            loc = url.find('ns:loc', namespaces)
            if loc is not None:
                urls.append(loc.text)
        print(f"Found {len(urls)} URLs in sitemap.")
        return urls
    except Exception as e:
        print(f"Error reading the sitemap: {e}")
        return []

async def fetch_and_process_page(client: httpx.AsyncClient, url: str, base_domain: str) -> dict:
    """
    Asynchronously fetch and process content (text, tables, images) from a wiki page.
    Separates text, tables, and image URLs.
    """
    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        content = soup.find('div', {'id': 'mw-content-text'})
        if not content:
            return {'url': url, 'text': "", 'tables': [], 'image_urls': [], 'title': "No Title"}

        # Extract text
        text_elements = content.find_all(['p', 'ul', 'ol', 'h2', 'h3', 'h4', 'h5', 'h6'])
        text = "\n\n".join(t.get_text(strip=True) for t in text_elements if t.get_text(strip=True))

        # Extract tables
        tables_html = [str(table) for table in content.find_all('table', class_='wikitable')]

        # Extract image URLs
        image_urls = set()
        for a_tag in content.find_all('a', class_='image'):
            href = a_tag.get('href')
            if href:
                image_urls.add(urljoin(base_domain, href))
        for img_tag in content.find_all('img'):
            src = img_tag.get('src')
            if src:
                image_urls.add(urljoin(base_domain, src))

        title_tag = soup.find('h1', id='firstHeading')
        title = title_tag.text if title_tag else "No Title"

        return {
            'url': url,
            'title': title,
            'text': text,
            'tables': tables_html,
            'image_urls': list(image_urls)
        }
    except httpx.RequestError as e:
        print(f"HTTP request error for {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

async def create_semantic_chunks(page_data: dict, chunk_size: int = 512, chunk_overlap: int = 100):
    """
    Creates context-aware chunks with metadata from the page data.
    """
    url = page_data['url']
    title = page_data['title']
    chunks = []
    chunk_index = 0

    # Chunk text content
    if page_data['text']:
        paragraphs = [p for p in page_data['text'].split("\n\n") if p.strip()]
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= 512:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                
                    doc = nlp(current_chunk)
                    chunks.append({
                        'chunk_id': hashlib.sha256(f"{url}-{chunk_index}".encode()).hexdigest(),
                        'source': url,
                        'content_type': 'text',
                        'title': title,
                        'keywords': [token.text for token in doc if token.pos_ in ("NOUN", "ADJ")],
                        'entities': [(ent.text, ent.label_) for ent in doc.ents],
                        'content': current_chunk
                    })
                    chunk_index += 1
                    current_chunk = paragraph[-100:] + "\n\n" + paragraph if len(paragraph) > 100 else paragraph

        # Add final chunk
        if current_chunk:
            doc = nlp(current_chunk)
            chunks.append({
                'chunk_id': hashlib.sha256(f"{url}-{chunk_index}".encode()).hexdigest(),
                'source': url,
                'content_type': 'text',
                'title': title,
                'keywords': [token.text for token in doc if token.pos_ in ("NOUN", "ADJ")],
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'content': current_chunk
            })

    # Chunk tables (each table as a separate chunk)
    for table_html in page_data['tables']:
        chunk_id = generate_chunk_id(url, chunk_index)
        chunks.append({
            'chunk_id': chunk_id,
            'source': url,
            'content_type': 'table',
            'title': title,
            'keywords': ['tabella'],
            'questions': [],
            'entities': [],
            'content': table_html
        })
        chunk_index += 1

    # "Chunk" images (each image URL as a separate chunk)
    for img_url in page_data['image_urls']:
        chunk_id = generate_chunk_id(url, chunk_index)
        chunks.append({
            'chunk_id': chunk_id,
            'source': url,
            'content_type': 'image',
            'title': title,
            'keywords': ['immagine'],
            'questions': [],
            'entities': [],
            'content': img_url
        })
        chunk_index += 1

    return chunks

async def crawl_and_chunk(sitemap_path: str, base_domain: str):
    """
    Crawls the website and creates semantic chunks with metadata, saving them to a file.
    """
    urls = get_urls_from_sitemap_file(sitemap_path)
    all_chunks = []
    
    async with httpx.AsyncClient() as client:
        for url in urls:
            page_data = await fetch_and_process_page(client, url, base_domain)
            if page_data:
                page_chunks = await create_semantic_chunks(page_data)
                all_chunks.extend(page_chunks)
            await asyncio.sleep(1.1)  # Respectful crawling delay

    print(f"Generated a total of {len(all_chunks)} chunks.")

    # Save the chunks to a JSON file
    with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    print(f"\nChunks saved to: {OUTPUT_PATH}")
    return all_chunks

if __name__ == "__main__":
    all_chunks_data = asyncio.run(crawl_and_chunk(SITEMAP_PATH, BASE_DOMAIN))
    # You can now work with the list of chunk dictionaries