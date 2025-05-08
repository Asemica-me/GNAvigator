import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import warnings
import time
from tqdm import tqdm
from Pbar import pbar

def generate_sitemap(start_url, output_file="sitemap_GNA.xml"):
    visited = set()
    to_visit = [start_url]
    domain = urlparse(start_url).netloc
    start_time = time.time()

    pbar = tqdm(desc="Crawling URLs", unit="URL", dynamic_ncols=True)

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Determine content type
                content_type = response.headers.get("Content-Type", "").lower()
                is_xml = "xml" in content_type

                # Choose parser based on content type
                if is_xml:
                    soup = BeautifulSoup(response.text, "xml")  # Requires lxml
                    # Extract XML links (e.g., <loc> in sitemaps)
                    links = [loc.text for loc in soup.find_all("loc")]
                else:
                    soup = BeautifulSoup(response.text, "html.parser")
                    # Extract HTML links
                    links = [a.get("href") for a in soup.find_all("a")]

                # Process links
                for link in links:
                    if not link:
                        continue  # Skip empty links
                    full_url = urljoin(url, link)
                    parsed = urlparse(full_url)
                    if parsed.netloc == domain and full_url not in visited:
                        to_visit.append(full_url)

                visited.add(url)
                pbar.update(1)
                pbar.set_postfix({"Current": url[:30] + "..." if len(url) > 30 else url})
                tqdm.write(f" Crawled: {url}")

        except Exception as e:
            print(f"Error crawling {url}: {e}")

    # Generate XML sitemap
    with open(output_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        for url in visited:
            f.write(f'  <url>\n    <loc>{url}</loc>\n  </url>\n')
        f.write('</urlset>')
    
    total_time = time.time() - start_time
    print(f"\nSitemap saved to {output_file}")
    print(f"Total URLs crawled: {len(visited)}")
    print(f"Total execution time: {total_time:.2f} seconds")

# Example usage:
generate_sitemap("https://gna.cultura.gov.it/wiki/index.php/Pagina_principale")