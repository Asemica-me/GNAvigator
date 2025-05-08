import os
from dotenv import load_dotenv
import asyncio
from bs4 import BeautifulSoup, NavigableString
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
from tqdm.asyncio import tqdm

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
    Asynchronously fetch and process content from a web page, extracting text
    in a hierarchical structure based on HTML.
    """
    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        content = soup.find('div', {'id': 'mw-content-text'})  # Or the main content div
        if not content:
            return {'url': url, 'title': "No Title", 'content': ""}

        title_tag = soup.find('h1', id='firstHeading')
        title = title_tag.text if title_tag else "No Title"

        # Recursive function to extract text with HTML structure
        def extract_text_with_structure(element):
            text_list = []
            for child in element.descendants: # Changed from recursiveChildGenerator to descendants
                if isinstance(child, NavigableString):
                    text = child.strip()
                    if text:
                        text_list.append({'type': 'text', 'content': text})
                elif child.name in ['h2', 'h3', 'h4', 'h5', 'h6']:
                    text = child.get_text(strip=True)
                    if text:
                        text_list.append({'type': child.name, 'content': text})
                elif child.name == 'p':
                    text = child.get_text(strip=True)
                    if text:
                         text_list.append({'type': 'paragraph', 'content': text})
                elif child.name == 'table' and 'wikitable' in child.get('class', []):
                    #convert table to string
                    text_list.append({'type': 'table', 'content': str(child)})
                elif child.name == 'img':
                    src = child.get('src')
                    if src:
                        image_url = urljoin(base_domain, src)
                        text_list.append({'type': 'image', 'content': image_url})
            return text_list

        page_content = extract_text_with_structure(content)

        return {
            'url': url,
            'title': title,
            'content': page_content,  # Hierarchical list of text, tables, and images
        }
    except httpx.RequestError as e:
        print(f"HTTP request error for {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

async def create_semantic_chunks(page_data: dict, chunk_size: int = 512, chunk_overlap: int = 100):
    """
    Creates context-aware chunks with metadata from the page data, handling
    the hierarchical content structure.
    """
    url = page_data['url']
    title = page_data['title']
    chunks = []
    chunk_index = 0
    
    def create_text_chunk(text_content, chunk_type="text"):
        """Helper function to create a text chunk."""
        nonlocal chunk_index
        doc = nlp(text_content)
        unique_entities = list(set([(ent.text, ent.label_) for ent in doc.ents]))
        chunk = {
            'chunk_id': generate_chunk_id(url, chunk_index),
            'source': url,
            'content_type': chunk_type,
            'questions': [],
            'title': title,
            'keywords': [token.text for token in doc if token.pos_ in ("NOUN", "ADJ")],
            'entities': unique_entities,
            'content': text_content,
        }
        chunk_index += 1
        return chunk

    def process_content_list(content_list):
        """Recursively processes the content list."""
        nonlocal chunk_index
        for item in content_list:
            if item['type'] in ['h2', 'h3', 'h4', 'h5', 'h6', 'paragraph', 'text']:
                text = item['content']
                sentences = list(nlp(text).sents)
                current_chunk = ""
                previous_sentence = ""  # To store the previous sentence

                for sent in sentences:
                    sent_text = sent.text_with_ws
                    # Include previous sentence for context, if it exists and isn't too long
                    contextual_text = (previous_sentence + " " + sent_text).strip() if previous_sentence else sent_text

                    if len(current_chunk) + len(contextual_text) <= chunk_size:
                        current_chunk += contextual_text
                    else:
                        if current_chunk:
                            chunks.append(create_text_chunk(current_chunk, item['type']))
                        current_chunk = sent_text

                    previous_sentence = sent_text  # Update for the next iteration

                if current_chunk:
                    chunks.append(create_text_chunk(current_chunk, item['type']))

            elif item['type'] == 'table':
                chunk_id = generate_chunk_id(url, chunk_index)
                chunks.append({
                    'chunk_id': chunk_id,
                    'source': url,
                    'content_type': 'table',
                    'title': title,
                    'keywords': ['tabella'],
                    'questions': [],
                    'entities': [],
                    'content': item['content'],
                })
                chunk_index += 1
            elif item['type'] == 'image':
                chunk_id = generate_chunk_id(url, chunk_index)
                chunks.append({
                    'chunk_id': chunk_id,
                    'source': url,
                    'content_type': 'image',
                    'title': title,
                    'keywords': ['immagine'],
                    'questions': [],
                    'entities': [],
                    'content': item['content'],
                })
                chunk_index += 1

    # Process the top-level content list
    process_content_list(page_data['content'])
    return chunks

    def process_content_list(content_list):
        """Recursively processes the content list."""
        nonlocal chunk_index
        for item in content_list:
            if item['type'] in ['h2', 'h3', 'h4', 'h5', 'h6', 'paragraph', 'text']:
                text = item['content']
                sentences = list(nlp(text).sents)
                current_chunk = ""
                for sent in sentences:
                    sent_text = sent.text_with_ws
                    if len(current_chunk) + len(sent_text) <= chunk_size:
                        current_chunk += sent_text
                    else:
                        if current_chunk:
                            chunks.append(create_text_chunk(current_chunk, item['type'])) # Pass the type
                        current_chunk = sent_text
                if current_chunk:
                    chunks.append(create_text_chunk(current_chunk, item['type'])) # Pass the type
            elif item['type'] == 'table':
                chunk_id = generate_chunk_id(url, chunk_index)
                chunks.append({
                    'chunk_id': chunk_id,
                    'source': url,
                    'content_type': 'table',
                    'title': title,
                    'keywords': ['tabella'],
                    'questions': [],
                    'entities': [],
                    'content': item['content'],
                })
                chunk_index += 1
            elif item['type'] == 'image':
                chunk_id = generate_chunk_id(url, chunk_index)
                chunks.append({
                    'chunk_id': chunk_id,
                    'source': url,
                    'content_type': 'image',
                    'title': title,
                    'keywords': ['immagine'],
                    'questions': [],
                    'entities': [],
                    'content': item['content'],
                })
                chunk_index += 1

    # Process the top-level content list
    process_content_list(page_data['content'])
    return chunks



async def crawl_and_chunk(sitemap_path: str, base_domain: str):
    """
    Crawls the website and creates semantic chunks with metadata, saving them to a file.
    """
    urls = get_urls_from_sitemap_file(sitemap_path)
    all_chunks = []

    async with httpx.AsyncClient() as client:
        for url in tqdm(urls, desc="Processing URLs", unit="URL"):
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
