import os
from dotenv import load_dotenv
import pandas as pd
import asyncio
from bs4 import BeautifulSoup, NavigableString, Comment
import xml.etree.ElementTree as ET
import httpx
from urllib.parse import urljoin
import hashlib
import spacy
import json
from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords
from tqdm.asyncio import tqdm
from tesseract import *

# --- Configuration ---
load_dotenv()
SITEMAP_PATH = 'GNA_sitemap.xml'
BASE_DOMAIN = 'https://gna.cultura.gov.it'
OUTPUT_FOLDER = "data"
OUTPUT_FILENAME = "chunks_memory.json"
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Initialize Models and Resources ---
nlp = spacy.load("it_core_news_lg")
kw_model = KeyBERT()
nltk.download('stopwords', quiet=True)
italian_stopwords = stopwords.words('italian')
custom_stopwords = ["così", "torna", "su"]
italian_stopwords.extend([word.lower() for word in custom_stopwords])
italian_stopwords = list(set([token.text.lower() for doc in nlp.pipe(italian_stopwords) for token in doc]))

# --- Utility Functions ---
def generate_chunk_id(source: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{source}-{chunk_index}".encode()).hexdigest()

def extract_keywords_and_entities(text: str):
    """Extracts keywords and entities from text using KeyBERT and spaCy."""
    doc = nlp(text)
    entities = list(set([(ent.text, ent.label_) for ent in doc.ents]))
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words=italian_stopwords,
        top_n=10,
        use_mmr=True,
        diversity=0.5,
    )
    keyword_list = [kw[0] for kw in keywords] if keywords else []
    return keyword_list, entities

def save_chunks_to_json(chunks: list, output_path: str):
    """Saves a list of chunks to a JSON file."""
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    print(f"\nChunks saved to: {output_path}")

def get_urls_from_sitemap_file(file_path: str) -> list:
    """Extracts URLs from an XML sitemap file."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for url in root.findall('.//ns:url', namespaces) if (loc := url.find('ns:loc', namespaces)) is not None]
        print(f"Found {len(urls)} URLs in sitemap.")
        return urls
    except Exception as e:
        print(f"Error reading the sitemap: {e}")
        return []

def extract_text_with_structure(soup: BeautifulSoup):
    """Recursively extracts text from HTML with structural information."""
    text_list = []

    def process_element(elem, headers_path):
        if isinstance(elem, Comment) or elem.name == 'table':
            return

        if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            header_text = elem.get_text(strip=True)
            if header_text:
                new_headers_path = headers_path + [header_text]
                text_list.append({'type': elem.name, 'content': header_text, 'headers_context': list(new_headers_path[:-1])})
            return

        elif elem.name == 'p':
            p_text = elem.get_text(strip=True, separator=' ')
            if p_text:
                text_list.append({'type': 'paragraph', 'content': p_text, 'headers_context': list(headers_path)})
            return

        for child in elem.contents:
            if isinstance(child, NavigableString):
                text = child.strip()
                if text:
                    text_list.append({'type': 'text', 'content': text, 'headers_context': list(headers_path)})
            else:
                process_element(child, headers_path)

    content_div = soup.find('div', {'id': 'mw-content-text'})
    if content_div:
        process_element(content_div, [])
    return text_list

async def fetch_page_content(client: httpx.AsyncClient, url: str) -> BeautifulSoup | None:
    """Asynchronously fetches the content of a web page and returns a BeautifulSoup object."""
    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        return soup
    except httpx.RequestError as e:
        print(f"HTTP request error for {url}: {e}")
        return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_page_data(soup: BeautifulSoup, url: str, base_domain: str) -> dict:
    """Extracts structured content (text, images, tables) from a BeautifulSoup object."""
    if not soup:
        return {'url': url, 'title': "No Title", 'content': [], 'images': [], 'tables': []}

    content_div = soup.find('div', {'id': 'mw-content-text'})
    if not content_div:
        return {'url': url, 'title': "No Title", 'content': [], 'images': [], 'tables': []}

    title_tag = soup.find('h1', id='firstHeading')
    title = title_tag.text if title_tag else "No Title"

    images = []
    for img_tag in content_div.find_all('img'):
        if 'src' in img_tag.attrs:
            img_src = img_tag['src']
            img_url = urljoin(base_domain, img_src) if not img_src.startswith('http') else img_src
            alt_text = img_tag.get('alt', None)
            images.append({'url': img_url, 'alt': alt_text})

    tables_data = []
    for table in content_div.find_all('table', class_='wikitable'):
        try:
            dfs = pd.read_html(str(table))
            if dfs:
                df = dfs[0].fillna('')
                headers = [th.get_text(strip=True) for th in table.find('tr').find_all('th')] if table.find('tr') else []
                tables_data.append({'markdown': df.to_markdown(index=False), 'headers': headers})
            else:
                print(f"No data found in table on {url}")
        except ValueError:
            print(f"Could not parse table on {url}")
        except Exception as e:
            print(f"Error converting table to markdown on {url}: {e}")

    text_content = extract_text_with_structure(soup)

    return {
        'url': url,
        'title': title,
        'content': text_content,
        'images': images,
        'tables': tables_data
    }

async def create_semantic_chunks(page_data: dict, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Creates context-aware chunks from structured page content."""
    url = page_data['url']
    title = page_data['title']
    final_chunks = []
    buffer = []
    current_length = 0
    sentence_lengths = []
    context_for_buffer = []
    chunk_index = 0

    def create_chunk(text_content, headers_context=None, content_type="text"):
        nonlocal chunk_index
        keywords, entities = extract_keywords_and_entities(text_content)
        chunk = {
            'chunk_id': generate_chunk_id(url, chunk_index),
            'source': url,
            'content_type': content_type,
            'questions': [],
            'title': title,
            'headers_context': headers_context.copy() if headers_context else [],
            'keywords': keywords,
            'entities': entities,
            'content': text_content,
        }
        chunk_index += 1
        return chunk

    all_page_segments = []
    current_headers = []
    for item in page_data['content']:
        if item['type'].startswith('h'):
            header_level = int(item['type'][1:])
            current_headers = current_headers[:header_level - 1] + [item['content']]
            all_page_segments.append({'type': item['type'], 'content': item['content'], 'headers_context': list(current_headers[:-1])})
        elif item['type'] in ['paragraph', 'text']:
            all_page_segments.append({'type': 'paragraph', 'content': item['content'], 'headers_context': list(current_headers)})

    for item in all_page_segments:
        if item['type'].startswith('h'):
            if buffer:
                final_chunks.append(create_chunk(" ".join(buffer), headers_context=list(context_for_buffer)))
                buffer = []
                current_length = 0
                sentence_lengths = []
                context_for_buffer = []

            final_chunks.append(create_chunk(item['content'], headers_context=item['headers_context'], content_type=item['type']))
            header_level = int(item['type'][1:])
            current_headers = current_headers[:header_level - 1] + [item['content']]
            context_for_buffer = list(current_headers[:-1])

        elif item['type'] == 'paragraph':
            sentences = [sent.text_with_ws for sent in nlp(item['content']).sents]
            for sentence in sentences:
                sent_len = len(sentence)
                if current_length + sent_len > chunk_size and buffer:
                    final_chunks.append(create_chunk(" ".join(buffer), headers_context=list(context_for_buffer)))
                    overlap_buffer = buffer[max(0, len(buffer) - sum([1 for length in reversed(sentence_lengths) if sum(reversed(sentence_lengths[:sentence_lengths.index(length) + 1])) < chunk_overlap])):]
                    buffer = list(overlap_buffer)
                    sentence_lengths = [len(s) for s in buffer]
                    current_length = sum(sentence_lengths)

                buffer.append(sentence)
                sentence_lengths.append(sent_len)
                current_length += sent_len
                context_for_buffer = item['headers_context']

    if buffer:
        final_chunks.append(create_chunk(" ".join(buffer), headers_context=list(context_for_buffer)))

    return final_chunks

async def process_page(client: httpx.AsyncClient, url: str, base_domain: str) -> list:
    """Fetches, processes, and chunks a single web page."""
    soup = await fetch_page_content(client, url)
    if not soup:
        return []

    page_data = extract_page_data(soup, url, base_domain)
    all_chunks = []

    # Process tables
    for i, table_info in enumerate(page_data.get('tables', [])):
        table_markdown = table_info['markdown']
        keywords, entities = extract_keywords_and_entities(table_markdown)
        chunk_id = generate_chunk_id(f"{url}-table", i)
        table_chunk = {
            'chunk_id': chunk_id,
            'source': url,
            'content_type': 'table',
            'questions': [],
            'title': page_data['title'],
            'headers_context': [],
            'keywords': keywords,
            'entities': entities,
            'content': table_markdown,
            'metadata': {'source_type': 'table', 'headers': table_info.get('headers', [])}
        }
        all_chunks.append(table_chunk)

    # Process text content if no tables
    if not page_data['tables']:
        text_chunks = await create_semantic_chunks(page_data)
        all_chunks.extend(text_chunks)

    # Process images
    for img_data in page_data.get('images', []):
        img_url = img_data['url']
        alt_text = img_data.get('alt', None)
        ocr_text = extract_text_from_image(img_url)
        if ocr_text:
            keywords, entities = extract_keywords_and_entities(ocr_text)
            chunk_id = generate_chunk_id(img_url, 0)
            image_chunk = {
                'chunk_id': chunk_id,
                'source': urljoin(base_domain, img_url),
                'content_type': 'image_ocr',
                'questions': [],
                'title': page_data['title'],
                'headers_context': [],
                'keywords': keywords,
                'entities': entities,
                'content': ocr_text,
                'metadata': {
                    'source_type': 'image',
                    'alt_text': alt_text,
                    'image_url': img_url
                }
            }
            all_chunks.append(image_chunk)

    return all_chunks

async def crawl_and_chunk(sitemap_path: str, base_domain: str):
    """Crawls a website from a sitemap, processes each page, and chunks the content."""
    urls = get_urls_from_sitemap_file(sitemap_path)
    all_chunks = []

    async with httpx.AsyncClient() as client:
        for url in tqdm(urls, desc="Processing URLs", unit="URL"):
            page_chunks = await process_page(client, url, base_domain)
            all_chunks.extend(page_chunks)
            await asyncio.sleep(1.1)

    print(f"Generated a total of {len(all_chunks)} chunks.")
    save_chunks_to_json(all_chunks, OUTPUT_PATH)
    return all_chunks

if __name__ == "__main__":
    all_chunks_data = asyncio.run(crawl_and_chunk(SITEMAP_PATH, BASE_DOMAIN))