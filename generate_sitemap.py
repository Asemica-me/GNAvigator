import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime
from collections import deque

# Configuration
base_domain = "https://gna.cultura.gov.it"
start_url = "https://gna.cultura.gov.it/wiki/index.php/Pagina_principale"
output_dir = "sitemap"
output_file = os.path.join(output_dir, "GNA__sitemap.xml")
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
delay = 1.0  # seconds between requests
max_depth = 10
max_pages = 200
excluded_namespaces = ['Special:', 'User:', 'Talk:', 'File:', 'MediaWiki:', 'Template:', 'Help:', 'Category:', 'Aiuto:']

os.makedirs(output_dir, exist_ok=True)

def normalize_url(url):
    """Normalize URL by removing fragments and query parameters"""
    parsed = urlparse(url)
    # Remove query and fragment
    clean = parsed._replace(query="", fragment="")
    return urlunparse(clean)

def is_valid_wiki_url(url):
    """Check if URL is a valid wiki content URL"""
    # Must be within the target domain
    if not url.startswith(base_domain):
        return False
    
    # Must be a wiki page
    if "/wiki/index.php/" not in url:
        return False
    
    # Check for excluded namespaces
    for ns in excluded_namespaces:
        if ns in url:
            return False
    
    # Extract page title from URL
    page_title = url.split("/wiki/index.php/")[-1].split("#")[0]
    
    # Exclude pages with special characters
    if ':' in page_title or '?' in page_title:
        return False
    
    return True

def fetch_page(url):
    """Fetch page content with polite crawling practices"""
    headers = {"User-Agent": user_agent}
    try:
        time.sleep(delay)  # Be polite to the server
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check if it's HTML content
        if 'text/html' not in response.headers.get('Content-Type', ''):
            return None
        
        return response.content
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return None

def extract_links(html, base_url):
    """Extract all valid links from HTML content"""
    if not html:
        return set()
    
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    
    # Focus on the main content area if available
    content_div = soup.find('div', {'id': 'mw-content-text'}) or soup
    
    for link in content_div.find_all('a', href=True):
        href = link['href']
        absolute_url = urljoin(base_url, href)
        normalized_url = normalize_url(absolute_url)
        
        if is_valid_wiki_url(normalized_url):
            links.add(normalized_url)
    
    return links

def generate_xml_sitemap(urls):
    """Generate XML sitemap from URL list"""
    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
    
    for url in urls:
        xml.append('  <url>')
        xml.append(f'    <loc>{url}</loc>')
        xml.append(f'    <lastmod>{datetime.now().strftime("%Y-%m-%d")}</lastmod>')
        xml.append('    <changefreq>weekly</changefreq>')
        xml.append(f'    <priority>{1.0 if url == start_url else 0.8}</priority>')
        xml.append('  </url>')
    
    xml.append('</urlset>')
    return '\n'.join(xml)

def crawl_site():
    """Crawl the site using BFS to discover important URLs"""
    queue = deque([(start_url, 0)])
    discovered = {normalize_url(start_url)}
    sitemap_urls = set()
    
    while queue and len(sitemap_urls) < max_pages:
        url, depth = queue.popleft()
        print(f"Crawling: {url} (depth {depth})")
        
        # Fetch page content
        html = fetch_page(url)
        if html is not None:
            sitemap_urls.add(url)
            
            # Extract links if we haven't reached max depth
            if depth < max_depth:
                new_links = extract_links(html, url)
                for link in new_links:
                    if link not in discovered:
                        discovered.add(link)
                        queue.append((link, depth + 1))
    
    return sorted(sitemap_urls)

if __name__ == "__main__":
    print(f"Starting crawl of {base_domain}")
    print(f"Max depth: {max_depth}, Max pages: {max_pages}")
    
    sitemap_urls = crawl_site()
    
    if not sitemap_urls:
        print("No URLs found. Exiting.")
        exit(1)
        
    print(f"Found {len(sitemap_urls)} URLs for sitemap")
    
    print("Generating XML sitemap...")
    sitemap_xml = generate_xml_sitemap(sitemap_urls)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(sitemap_xml)
    
    print(f"Sitemap generated successfully at: {output_file}")
    print(f"Total URLs included: {len(sitemap_urls)}")