import os
from dotenv import load_dotenv
import asyncio
from bs4 import BeautifulSoup, NavigableString, Comment
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
import subprocess
import tempfile
from PIL import Image
from io import BytesIO
from tesseract import *

# Load the Italian spaCy model
nlp = spacy.load("it_core_news_lg")

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

        # Remove all HTML comments from the parsed content
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()  # Remove the comment from the tree

        content = soup.find('div', {'id': 'mw-content-text'})  # Or the main content div
        if not content:
            return {'url': url, 'title': "No Title", 'content': "", 'images': []}

        title_tag = soup.find('h1', id='firstHeading')
        title = title_tag.text if title_tag else "No Title"

        # Extract image URLs
        images = []
        for img_tag in content.find_all('img'):
            if 'src' in img_tag.attrs:
                img_src = img_tag['src']
                # Make sure the URL is absolute
                if not img_src.startswith('http'):
                    img_src = urljoin(base_domain, img_src)
                alt_text = img_tag.get('alt', None)
                images.append({'url': img_src, 'alt': alt_text})

        # Recursive function to extract text with HTML structure
        def extract_text_with_structure(element):
            text_list = []
            current_paragraph = []
            
            for child in element.descendants:
                if isinstance(child, Comment):
                    continue
                if isinstance(child, NavigableString):
                    text = child.strip()
                    if text:
                        current_paragraph.append(text)
                elif child.name in ['h2', 'h3', 'h4', 'h5', 'h6']:
                    if current_paragraph:
                        text_list.append({
                            'type': 'paragraph',
                            'content': ' '.join(current_paragraph)
                        })
                        current_paragraph = []
                    text_list.append({
                        'type': child.name,
                        'content': child.get_text(strip=True)
                    })
                elif child.name == 'p':
                    if current_paragraph:
                        text_list.append({
                            'type': 'paragraph',
                            'content': ' '.join(current_paragraph)
                        })
                        current_paragraph = []
            
            # Add any remaining text
            if current_paragraph:
                text_list.append({
                    'type': 'paragraph',
                    'content': ' '.join(current_paragraph)
                })
            
            return text_list

        page_content = extract_text_with_structure(content)

        return {
            'url': url,
            'title': title,
            'content': page_content,  # Hierarchical list of text, tables, and images
            'images': images
        }
    except httpx.RequestError as e:
        print(f"HTTP request error for {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None
    
from keybert import KeyBERT
kw_model = KeyBERT()
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
# Load Italian stopwords
italian_stopwords = stopwords.words('italian')

# Add custom stopwords PROPERLY TOKENIZED
custom_stopwords = ["così", "torna", "su"]  # Split multi-word terms
italian_stopwords += [word.lower() for word in custom_stopwords]

# Preprocess with spaCy for better matching
italian_stopwords = list(set([
    token.text.lower() 
    for doc in nlp.pipe(italian_stopwords) 
    for token in doc
]))

async def create_semantic_chunks(page_data: dict, chunk_size: int = 1024, chunk_overlap: int = 256):
    """
    Creates context-aware chunks with complete sentences and structural context.
    """
    url = page_data['url']
    title = page_data['title']
    chunks = []
    chunk_index = 0
    current_context = []  # Track hierarchical headers for context

    def create_text_chunk(text_content, chunk_type="text", headers_context=None):
        nonlocal chunk_index
        
        doc = nlp(text_content)
        unique_entities = list(set([(ent.text, ent.label_) for ent in doc.ents]))
        
        # KeyBERT keyword extraction
        keywords = kw_model.extract_keywords(
            text_content,
            keyphrase_ngram_range=(1, 2),  # Allow single words or phrases   
            stop_words=italian_stopwords,  # Use Italian stopwords  
            top_n=10,
            use_mmr=True,
            diversity=0.5,                 # Adjust diversity for more varied keywords
        )
        
        # Extract keyword strings (discard scores)
        keyword_list = [kw[0] for kw in keywords] if keywords else []
        
        chunk = {
            'chunk_id': generate_chunk_id(url, chunk_index),
            'source': url,
            'content_type': chunk_type,
            'questions': [],
            'title': title,
            'headers_context': headers_context.copy() if headers_context else [],
            'keywords': keyword_list,  # Replaced spaCy logic with KeyBERT keywords
            'entities': unique_entities,
            'content': text_content,
        }
        chunk_index += 1
        return chunk

    def process_content_list(content_list, current_headers=None):
        nonlocal chunk_index, current_context
        current_headers = current_headers or []
        buffer = []
        current_length = 0
        sentence_lengths = []  # Track lengths of sentences in buffer
        
        for item in content_list:
            # Update header context
            if item['type'].startswith('h'):
                header_level = int(item['type'][1:])
                current_headers = current_headers[:header_level-2] + [item['content']]
                current_context = current_headers
            
            if item['type'] in ['paragraph', 'text']:
                # Process text content with sentence-aware splitting
                sentences = [sent.text_with_ws for sent in nlp(item['content']).sents]
                
                for sentence in sentences:
                    sent_len = len(sentence)
                    
                    if current_length + sent_len > chunk_size:
                        # Create chunk from buffer
                        if buffer:
                            chunk_text = " ".join(buffer)
                            chunks.append(create_text_chunk(
                                chunk_text, 
                                chunk_type="text",
                                headers_context=current_context
                            ))
                            
                            # Calculate overlap using actual sentence lengths
                            overlap_total = 0
                            overlap_index = len(buffer) - 1
                            
                            # Work backwards to find overlap sentences
                            while overlap_index >= 0 and overlap_total < chunk_overlap:
                                overlap_total += sentence_lengths[overlap_index]
                                overlap_index -= 1
                            
                            # Keep overlapping sentences
                            overlap_start = max(0, overlap_index + 1)
                            buffer = buffer[overlap_start:]
                            sentence_lengths = sentence_lengths[overlap_start:]
                            current_length = sum(sentence_lengths)
                    
                    buffer.append(sentence)
                    sentence_lengths.append(sent_len)
                    current_length += sent_len

        # Add remaining content
        if buffer:
            chunk_text = " ".join(buffer)
            chunks.append(create_text_chunk(
                chunk_text, 
                chunk_type="text",
                headers_context=current_context
            ))

    # Process the content with header tracking
    process_content_list(page_data['content'])
    return chunks


async def crawl_and_chunk(sitemap_path: str, base_domain: str):
    urls = get_urls_from_sitemap_file(sitemap_path)
    all_chunks = []

    async with httpx.AsyncClient() as client:
        for url in tqdm(urls, desc="Processing URLs", unit="URL"):
            page_data = await fetch_and_process_page(client, url, base_domain)
            if page_data:
                # Process text content into chunks
                text_chunks = await create_semantic_chunks(page_data)
                all_chunks.extend(text_chunks)

                # Process images using the extract_text_from_image function
                for img_data in page_data.get('images', []):
                    img_url = img_data['url']
                    alt_text = img_data.get('alt', None)
                    ocr_text = extract_text_from_image(img_url)
                    if ocr_text:
                        chunk_id = generate_chunk_id(img_url, 0)
                        image_chunk = {
                            'chunk_id': chunk_id,
                            'source': img_url,
                            'content_type': 'image_ocr',
                            'questions': [],
                            'title': page_data['title'],
                            'headers_context': [],
                            'keywords': [],
                            'entities': [], 
                            'content': ocr_text,
                            'metadata': {
                                'source_type': 'image',
                                'alt_text': alt_text,
                                'image_url': img_url
                            }
                        }
                        all_chunks.append(image_chunk)

            await asyncio.sleep(1.1)

    print(f"Generated a total of {len(all_chunks)} chunks (including image OCR).")

    # Save the chunks to a JSON file
    with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    print(f"\nChunks saved to: {OUTPUT_PATH}")
    return all_chunks


if __name__ == "__main__":
    all_chunks_data = asyncio.run(crawl_and_chunk(SITEMAP_PATH, BASE_DOMAIN))