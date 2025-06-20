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
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from tqdm.asyncio import tqdm
import time
from ocr_tesseract import *

# --- Configuration ---
load_dotenv()
SITEMAP_PATH = os.path.join("sitemap", "GNA__sitemap.xml")
BASE_DOMAIN = 'https://gna.cultura.gov.it'
OUTPUT_FOLDER = "data"
OUTPUT_FILENAME = "chunks_memory.json"
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
CONCURRENCY_LIMIT = 3  # Max concurrent requests
OCR_CONCURRENCY_LIMIT = 2  # Max concurrent OCR operations
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Initialize Models and Resources ---
try:
    nlp = spacy.load("it_core_news_md")
except OSError:
    import spacy.cli
    spacy.cli.download("it_core_news_md")
    nlp = spacy.load("it_core_news_md")
    
kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")
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
    """Saves the list of chunks to a JSON file"""
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)
    print(f"\nSaved {len(chunks)} chunks to: {output_path}")

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
        print(f"Error reading sitemap: {e}")
        return []

async def fetch_page_content(client: httpx.AsyncClient, url: str) -> BeautifulSoup | None:
    """Asynchronously fetches page content with retry mechanism"""
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            return soup
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < MAX_RETRIES - 1:
                retry_wait = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"Attempt {attempt+1}/{MAX_RETRIES} failed for {url}. Retrying in {retry_wait}s. Error: {e}")
                await asyncio.sleep(retry_wait)
            else:
                print(f"Failed to fetch {url} after {MAX_RETRIES} attempts. Error: {e}")
                return None
        except Exception as e:
            print(f"Unexpected error with {url}: {e}")
            return None
    return None

def extract_structured_content(soup: BeautifulSoup, base_domain: str):
    """Extracts content with structural context in document order"""
    content_div = soup.find('div', {'id': 'mw-content-text'})
    if not content_div:
        return []
    
    structured_content = []
    current_headers = [""] * 6  # Track headers from h1 to h6

    def process_element(elem):
        nonlocal current_headers
        
        if isinstance(elem, Comment):
            return
            
        # Handle headers
        if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(elem.name[1])
            header_text = elem.get_text(strip=True, separator=' ')
            if header_text:
                # Update header tracking
                current_headers[level-1] = header_text
                # Reset lower-level headers
                for i in range(level, 6):
                    current_headers[i] = ""
                    
                structured_content.append({
                    'type': 'header',
                    'level': level,
                    'content': header_text,
                    'context': [h for h in current_headers[:level] if h]
                })
                # Skip processing children of headers to avoid duplication
                return
                
        # Handle paragraphs
        elif elem.name == 'p':
            p_text = elem.get_text(strip=True, separator=' ')
            if p_text:
                structured_content.append({
                    'type': 'paragraph',
                    'content': p_text,
                    'context': [h for h in current_headers if h]
                })
                
        # Handle tables - improved detection
        elif elem.name == 'table':
            table_classes = elem.get('class', [])
            # Handle both standard MediaWiki tables and sortable tables
            if 'wikitable' in table_classes or 'sortable' in table_classes:
                try:
                    dfs = pd.read_html(str(elem))
                    if dfs:
                        df = dfs[0].fillna('')
                        markdown_table = df.to_markdown(index=False)
                        structured_content.append({
                            'type': 'table',
                            'content': markdown_table,
                            'context': [h for h in current_headers if h]
                        })
                        # Skip processing table children since we've handled it
                        return
                except Exception as e:
                    print(f"Table processing error: {e}")
            # If not a recognized table, process as generic element
                    
        # Handle images
        elif elem.name == 'img':
            if 'src' in elem.attrs:
                img_src = elem['src']
                img_url = urljoin(base_domain, img_src) if not img_src.startswith('http') else img_src
                alt_text = elem.get('alt', '')
                structured_content.append({
                    'type': 'image',
                    'content': {'url': img_url, 'alt': alt_text},
                    'context': [h for h in current_headers if h]
                })
                return
                
        # Handle lists
        elif elem.name in ['ul', 'ol']:
            list_items = []
            for li in elem.find_all('li', recursive=False):
                item_text = li.get_text(strip=True, separator=' ')
                if item_text:
                    list_items.append(f"- {item_text}")
            
            if list_items:
                list_content = "\n".join(list_items)
                structured_content.append({
                    'type': 'list',
                    'content': list_content,
                    'context': [h for h in current_headers if h]
                })
                return
                
        # Recursively process child elements
        if hasattr(elem, 'children'):
            for child in elem.children:
                if isinstance(child, NavigableString):
                    text = child.strip()
                    if text:
                        structured_content.append({
                            'type': 'text',
                            'content': text,
                            'context': [h for h in current_headers if h]
                        })
                elif child and child.name:
                    process_element(child)
                
    process_element(content_div)
    return structured_content

def extract_page_data(soup: BeautifulSoup, url: str, base_domain: str) -> dict:
    """Extracts structured content with context preservation"""
    if not soup:
        return {'url': url, 'title': "No Title", 'structured_content': []}

    title_tag = soup.find('h1', id='firstHeading')
    title = title_tag.text if title_tag else "No Title"

    return {
        'url': url,
        'title': title,
        'structured_content': extract_structured_content(soup, base_domain)
    }

def create_semantic_chunks(page_data: dict) -> list:
    url = page_data['url']
    title = page_data['title']
    final_chunks = []
    text_buffer = []
    current_context = []
    char_count = 0
    chunk_index = 0

    def create_chunk(content: str, context: list, content_type: str = "text"):
        nonlocal chunk_index
        keywords, entities = extract_keywords_and_entities(content)
        return {
            'chunk_id': generate_chunk_id(url, chunk_index),
            'source': url,
            'content_type': content_type,
            'title': title,
            'headers_context': context,
            'keywords': keywords,
            'entities': entities,
            'content': content,
            'chunk_index': chunk_index
        }

    for item in page_data['structured_content']:
        item_type = item['type']
        context = item['context']
        content = item['content']

        # Handle headers (update context but don't create chunks)
        if item_type == 'header':
            # Only update context if it's a new header
            if not current_context or content != current_context[-1]:
                current_context = context + [content]
            continue

        # Handle tables as separate chunks
        elif item_type == 'table':
            # Flush any text buffer first
            if text_buffer:
                chunk = create_chunk(" ".join(text_buffer), current_context)
                final_chunks.append(chunk)
                chunk_index += 1
                text_buffer = []
                char_count = 0
                
            # Create table chunk
            chunk = create_chunk(content, context, 'table')
            final_chunks.append(chunk)
            chunk_index += 1
            continue

        # Handle lists as separate chunks
        elif item_type == 'list':
            # Flush text buffer
            if text_buffer:
                chunk = create_chunk(" ".join(text_buffer), current_context)
                final_chunks.append(chunk)
                chunk_index += 1
                text_buffer = []
                char_count = 0
                
            # Create list chunk
            chunk = create_chunk(content, context, 'list')
            final_chunks.append(chunk)
            chunk_index += 1
            continue

        # Handle images (requires OCR processing later)
        elif item_type == 'image':
            # Flush text buffer
            if text_buffer:
                chunk = create_chunk(" ".join(text_buffer), current_context)
                final_chunks.append(chunk)
                chunk_index += 1
                text_buffer = []
                char_count = 0
                
            # Store for later OCR processing
            final_chunks.append({
                'type': 'image_reference',
                'image_data': content,
                'context': context
            })
            continue

        # Process text content (paragraphs and raw text)
        if item_type in ['paragraph', 'text']:
            # Handle long sentences
            sentences = nltk.sent_tokenize(content, language='italian')
            for sentence in sentences:
                # Split oversized sentences
                if len(sentence) > CHUNK_SIZE:
                    parts = [sentence[i:i+CHUNK_SIZE] for i in range(0, len(sentence), CHUNK_SIZE)]
                    for part in parts:
                        if char_count + len(part) > CHUNK_SIZE and text_buffer:
                            chunk = create_chunk(" ".join(text_buffer), current_context)
                            final_chunks.append(chunk)
                            chunk_index += 1
                            
                            # Maintain overlap
                            overlap_start = max(0, len(text_buffer) - max(1, int(len(text_buffer) * CHUNK_OVERLAP/CHUNK_SIZE)))
                            text_buffer = text_buffer[overlap_start:]
                            char_count = sum(len(s) for s in text_buffer)
                            
                        text_buffer.append(part)
                        char_count += len(part)
                else:
                    if char_count + len(sentence) > CHUNK_SIZE and text_buffer:
                        chunk = create_chunk(" ".join(text_buffer), current_context)
                        final_chunks.append(chunk)
                        chunk_index += 1
                        
                        # Maintain overlap
                        overlap_start = max(0, len(text_buffer) - max(1, int(len(text_buffer) * CHUNK_OVERLAP/CHUNK_SIZE)))
                        text_buffer = text_buffer[overlap_start:]
                        char_count = sum(len(s) for s in text_buffer)
                        
                    text_buffer.append(sentence)
                    char_count += len(sentence)

    # Process remaining text buffer
    if text_buffer:
        chunk = create_chunk(" ".join(text_buffer), current_context)
        final_chunks.append(chunk)
        
    return final_chunks

async def process_page(client: httpx.AsyncClient, url: str, base_domain: str, ocr_semaphore: asyncio.Semaphore) -> list:
    """Processes a page and returns chunks with OCR handling"""
    soup = await fetch_page_content(client, url)
    if not soup:
        return []

    page_data = extract_page_data(soup, url, base_domain)
    chunks = await asyncio.to_thread(create_semantic_chunks, page_data)
    final_chunks = []

    # Process images with OCR
    for chunk in chunks:
        if isinstance(chunk, dict) and chunk.get('type') == 'image_reference':
            img_data = chunk['image_data']
            ocr_text = None
            
            try:
                # Limit OCR concurrency
                async with ocr_semaphore:
                    ocr_text = await asyncio.to_thread(
                        extract_text_from_image, 
                        img_data['url']
                    )
            except Exception as e:
                print(f"OCR extraction failed for {img_data['url']}: {e}")
            
            # Fallback to alt text if OCR fails
            if not ocr_text:
                ocr_text = f"ALT_TEXT: {img_data.get('alt', '')}"
            
            # Offload keyword extraction
            try:
                keywords, entities = await asyncio.to_thread(
                    extract_keywords_and_entities, 
                    ocr_text
                )
            except Exception as e:
                print(f"Keyword extraction failed for OCR text: {e}")
                keywords, entities = [], []
                
            image_chunk = {
                'chunk_id': generate_chunk_id(img_data['url'], 0),
                'source': url,
                'content_type': 'image_ocr',
                'title': page_data['title'],
                'headers_context': chunk['context'],
                'keywords': keywords,
                'entities': entities,
                'content': ocr_text,
                'metadata': {
                    'alt_text': img_data.get('alt'),
                    'image_url': img_data['url']
                }
            }
            final_chunks.append(image_chunk)
        else:
            final_chunks.append(chunk)

    return final_chunks

async def crawl_and_chunk(sitemap_path: str, base_domain: str):
    """Main crawling and chunking function with robust error handling"""
    urls = get_urls_from_sitemap_file(sitemap_path)
    all_chunks = []
    failed_urls = []
    success_count = 0

    # Create client with custom headers to mimic browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive"
    }
    
    async with httpx.AsyncClient(
        headers=headers,
        timeout=30.0,
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=CONCURRENCY_LIMIT)
    ) as client:
        
        # Process with concurrency control
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        ocr_semaphore = asyncio.Semaphore(OCR_CONCURRENCY_LIMIT)
        
        async def process_with_semaphore(url):
            async with semaphore:
                try:
                    chunks = await process_page(client, url, base_domain, ocr_semaphore)
                    return url, chunks, None
                except Exception as e:
                    return url, [], str(e)
        
        tasks = [process_with_semaphore(url) for url in urls]
        
        # Process with progress tracking
        for future in tqdm.as_completed(tasks, total=len(urls), desc="Processing Pages"):
            url, page_chunks, error = await future
            
            if error:
                print(f"\nError processing {url}: {error}")
                failed_urls.append(url)
            else:
                all_chunks.extend(page_chunks)
                success_count += 1
                print(f"\nProcessed {url} successfully. Chunks: {len(page_chunks)}")
            
            # Add delay between requests to avoid overwhelming server
            await asyncio.sleep(2.0)

    # Generate report
    print(f"\nProcessing complete:")
    print(f"- Total URLs: {len(urls)}")
    print(f"- Successfully processed: {success_count}")
    print(f"- Failed URLs: {len(failed_urls)}")
    
    if failed_urls:
        print("\nFailed URLs:")
        for url in failed_urls:
            print(f"  - {url}")
            
        # Save failed URLs for retry
        with open(os.path.join(OUTPUT_FOLDER, "failed_urls.txt"), "w") as f:
            f.write("\n".join(failed_urls))
        print(f"Saved failed URLs to: {os.path.join(OUTPUT_FOLDER, 'failed_urls.txt')}")

    # Save results
    save_chunks_to_json(all_chunks, OUTPUT_PATH)
    return all_chunks

if __name__ == "__main__":
    start_time = time.time()
    print("Starting crawling and chunking process...")
    asyncio.run(crawl_and_chunk(SITEMAP_PATH, BASE_DOMAIN))
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")