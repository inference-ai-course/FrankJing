import os
import requests
import feedparser
from bs4 import BeautifulSoup

# Parameters
SEARCH_QUERY = "cat:cs.CL"   # Category: Computation and Language
MAX_RESULTS = 50             # Number of papers to fetch
SAVE_DIR = "./data"      # Folder to save PDFs

# Ensure output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Query URL
base_url = "http://export.arxiv.org/api/query?"
query = f"search_query={SEARCH_QUERY}&start=0&max_results={MAX_RESULTS}"

print("Fetching metadata from arXiv...")
feed = feedparser.parse(base_url + query)

print(f"Found {len(feed.entries)} papers.")

for i, entry in enumerate(feed.entries, 1):
    abs_url = None
    for link in entry.links:
        if link.rel == "alternate":
            abs_url = link.href   # abstract page URL
    
    if abs_url:
        try:
            print(f"[{i}] Parsing abs page for: {entry.title[:70]}...")
            abs_page = requests.get(abs_url)
            soup = BeautifulSoup(abs_page.text, "html.parser")

            # Find the first link that looks like a PDF
            pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
            if pdf_meta and pdf_meta.get("content"):
                pdf_url = pdf_meta["content"]            
                # Download PDF
                response = requests.get(pdf_url)
                filename = os.path.join(SAVE_DIR, f"paper_{i}.pdf")
                with open(filename, "wb") as f:
                    f.write(response.content)

                print(f"   ✅ Saved as {filename}")
            else:
                print(f"   ❌ No PDF link found on {abs_url}")
        except Exception as e:
            print(f"   ❌ Failed to fetch {entry.title[:50]}: {e}")

print("✅ Done! PDFs saved in:", SAVE_DIR)