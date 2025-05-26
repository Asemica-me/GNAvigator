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
from huggingface_hub import InferenceClient, login
from transformers import pipeline
from tqdm.asyncio import tqdm
from tesseract import *

# Load the Italian spaCy model
nlp = spacy.load("it_core_news_lg")

load_dotenv()

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
            return {'url': url, 'title': "No Title", 'content': [], 'images': [], 'tables': []}

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

        # Extract tables
        tables_data = []
        for table in content.find_all('table', class_='wikitable'):
            headers = []
            first_row = table.find('tr')
            if first_row:
                header_tags = first_row.find_all('th')
                headers = [th.get_text(strip=True) for th in header_tags]

            try:
                dfs = pd.read_html(str(table))
                if dfs:
                    df = dfs[0]
                    df = df.fillna('') # Fill NaN values with empty strings
                    markdown_table = df.to_markdown(index=False)
                    tables_data.append({'markdown': markdown_table, 'headers': headers})
                else:
                    print(f"No data found in table on {url}")
            except ValueError:
                print(f"Could not parse table on {url}")
            except Exception as e:
                print(f"Error converting table to markdown on {url}: {e}")

        # Recursive function to extract text with HTML structure
        def extract_text_with_structure(element):
            text_list = []
            current_paragraph = []
            current_headers_on_path = []

            def process_element(elem, headers_path):
                nonlocal current_paragraph
                if isinstance(elem, Comment) or elem.name == 'table':
                    return
                if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if current_paragraph:
                        text_list.append({'type': 'paragraph', 'content': ' '.join(current_paragraph)})
                        current_paragraph = []
                    header_text = elem.get_text(strip=True)
                    if header_text:
                        new_headers_path = headers_path + [header_text]
                        text_list.append({'type': elem.name, 'content': header_text, 'headers_context': list(new_headers_path[:-1])}) # Store context before this header
                    return  # Skip processing children of headers
                
                elif elem.name == 'p':
                    p_text = elem.get_text(strip=True, separator=' ')
                    if p_text:
                        text_list.append({'type': 'paragraph', 'content': p_text, 'headers_context': list(headers_path)})
                    return # Skip processing children of paragraphs
                
                # Process other elements
                for child in elem.contents:
                    if isinstance(child, NavigableString):
                        text = child.strip()
                        if text:
                            current_paragraph.append(text)
                    else:
                        process_element(child, headers_path)

            process_element(element, [])

            # Add any remaining text in the buffer
            if current_paragraph:
                text_list.append({'type': 'paragraph', 'content': ' '.join(current_paragraph), 'headers_context': []})

            return text_list

        page_content = []
        if not tables_data:
            page_content = extract_text_with_structure(content)

        return {
            'url': url,
            'title': title,
            'content': page_content,
            'images': images,
            'tables': tables_data
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

    all_page_chunks = []
    current_headers = []

    for item in page_data['content']:
        if item['type'].startswith('h'):
            header_level = int(item['type'][1:])
            current_headers = current_headers[:header_level - 1] + [item['content']]
            # We might want to create a chunk for the header itself with its context
            all_page_chunks.append({
                'type': item['type'],
                'content': item['content'],
                'headers_context': list(current_headers[:-1]) # Context before this header
            })
        elif item['type'] == 'paragraph':
            all_page_chunks.append({
                'type': 'paragraph',
                'content': item['content'],
                'headers_context': list(current_headers)
            })

    final_chunks = []
    buffer = []
    current_length = 0
    sentence_lengths = []
    context_for_buffer = []

    for item in all_page_chunks:
        if item['type'].startswith('h'):
            # Create a chunk for the preceding buffer
            if buffer:
                chunk_text = " ".join(buffer)
                final_chunks.append(create_text_chunk(chunk_text, headers_context=list(context_for_buffer)))
                buffer = []
                current_length = 0
                sentence_lengths = []
                context_for_buffer = [] # Reset context for new buffer

            # Create a chunk for the header
            final_chunks.append(create_text_chunk(item['content'], headers_context=item['headers_context']))
            # Update current headers for subsequent content
            header_level = int(item['type'][1:])
            if header_level <= len(current_headers):
                current_headers = current_headers[:header_level - 1] + [item['content']]
            else:
                current_headers.append(item['content'])
            context_for_buffer = list(current_headers[:-1]) # Context after the header

        elif item['type'] == 'paragraph':
            sentences = [sent.text_with_ws for sent in nlp(item['content']).sents]
            for sentence in sentences:
                sent_len = len(sentence)
                if current_length + sent_len > chunk_size and buffer:
                    # Create a chunk
                    chunk_text = " ".join(buffer)
                    final_chunks.append(create_text_chunk(chunk_text, headers_context=list(context_for_buffer)))

                    # Prepare for overlap
                    overlap_buffer = []
                    overlap_length = 0
                    overlap_indices = []
                    for i in range(len(buffer) - 1, -1, -1):
                        overlap_length += sentence_lengths[i]
                        overlap_buffer.insert(0, buffer[i])
                        overlap_indices.insert(0, i)
                        if overlap_length > chunk_overlap:
                            break
                    buffer = list(overlap_buffer)
                    sentence_lengths = [sentence_lengths[i] for i in overlap_indices]
                    current_length = sum(sentence_lengths)
                    # Keep the current context
                buffer.append(sentence)
                sentence_lengths.append(sent_len)
                current_length += sent_len
                context_for_buffer = item['headers_context'] # Use the paragraph's header context

    # Add any remaining buffer
    if buffer:
        chunk_text = " ".join(buffer)
        final_chunks.append(create_text_chunk(chunk_text, headers_context=list(context_for_buffer)))

    return final_chunks

async def extract_keywords_and_entities(text: str):
    """
    Extracts keywords and entities from the given text.
    """
    doc = nlp(text)
    unique_entities = list(set([(ent.text, ent.label_) for ent in doc.ents]))
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words=italian_stopwords,
        top_n=10,
        use_mmr=True,
        diversity=0.5,
    )
    keyword_list = [kw[0] for kw in keywords] if keywords else []
    return keyword_list, unique_entities

async def crawl_and_chunk(sitemap_path: str, base_domain: str):
    urls = get_urls_from_sitemap_file(sitemap_path)
    all_chunks = []

    async with httpx.AsyncClient() as client:
        for url in tqdm(urls, desc="Processing URLs", unit="URL"):
            page_data = await fetch_and_process_page(client, url, base_domain)
            if page_data:
                # Process tables
                for i, table_info in enumerate(page_data.get('tables', [])):
                    table_markdown = table_info['markdown']
                    keywords, entities = await extract_keywords_and_entities(table_markdown)
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

                # Process text content into chunks (only if no tables were found)
                if not page_data['tables']:
                    text_chunks = await create_semantic_chunks(page_data)
                    for chunk in text_chunks:
                        keywords, entities = await extract_keywords_and_entities(chunk['content'])
                        chunk['keywords'] = keywords
                        chunk['entities'] = entities
                        all_chunks.append(chunk)

                # Process images
                for img_data in page_data.get('images', []):
                    img_url = img_data['url']
                    alt_text = img_data.get('alt', None)
                    ocr_text = extract_text_from_image(img_url)
                    if ocr_text:
                        keywords, entities = await extract_keywords_and_entities(ocr_text)
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

            await asyncio.sleep(1.1)

    print(f"Generated a total of {len(all_chunks)} chunks.")

    # Save the chunks to a JSON file
    with open(OUTPUT_PATH, "w", encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    print(f"\nChunks saved to: {OUTPUT_PATH}")
    return all_chunks


if __name__ == "__main__":
    all_chunks_data = asyncio.run(crawl_and_chunk(SITEMAP_PATH, BASE_DOMAIN))