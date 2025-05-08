import asyncio
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import httpx
from urllib.parse import urljoin
import time

# Global variables
GLOBAL_SITEMAP_PATH = 'Gna_sitemap.xml'  # Replace with the actual path to your sitemap
GLOBAL_BASE_DOMAIN = 'https://gna.cultura.gov.it'

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
    """
    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        content = soup.find('div', {'id': 'mw-content-text'})
        if not content:
            return {'url': url, 'text': None, 'tables': [], 'image_urls': []}
        text = content.get_text(separator='\n', strip=True) if content else ""
        tables = [str(table) for table in content.find_all('table', class_='wikitable')]
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
        title = title_tag.text if title_tag else "No title"
        description_tag = soup.find('meta', {'name': 'description'})
        description = description_tag['content'] if description_tag and 'content' in description_tag.attrs else ""
        return {
            'url': url,
            'title': title,
            'description': description,
            'text': text,
            'tables': tables,
            'image_urls': list(image_urls),
            'last_updated': time.time()
        }
    except httpx.RequestError as e:
        print(f"HTTP request error for {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

async def crawl_website(sitemap_path: str, base_domain: str):
    """
    Crawls the website based on the sitemap, fetches content, and respects the rate limit.
    """
    urls = get_urls_from_sitemap_file(sitemap_path)
    all_page_data = []
    async with httpx.AsyncClient() as client:
        for url in urls:
            page_data = await fetch_and_process_page(client, url, base_domain)
            if page_data:
                all_page_data.append(page_data)
            await asyncio.sleep(1.1) # Respect mistral 1 request per second limit
    return all_page_data

async def main_crawl():
    """Main function to orchestrate the crawling using global variables."""
    print("Starting the crawling process...")
    start_time = time.time()
    crawled_data = await crawl_website(GLOBAL_SITEMAP_PATH, GLOBAL_BASE_DOMAIN)
    end_time = time.time()
    print(f"Crawling completed in {end_time - start_time:.2f} seconds.")
    print(f"Successfully crawled {len(crawled_data)} pages.")

    # For examples print the title and number of images for each page:
    # for page in crawled_data:
    #     print(f"\nURL: {page['url']}")
    #     print(f"Title: {page['title']}")
    #     if page['image_urls']:
    #         print(f"Number of images: {len(page['image_urls'])}")
    #     if page['tables']:
    #         print(f"Number of tables: {len(page['tables'])}")
    #     if page['text']:
    #         print(f"First 50 characters of text: {page['text'][:50]}...")

    return crawled_data 
    # Optionally return the data for further use

if __name__ == "__main__":
    crawled_output = asyncio.run(main_crawl())
    # You can work with the crawled_output here if needed
    # print("\nFirst crawled item:", crawled_output[0] if crawled_output else "No items crawled")