import os
import json
import feedparser

# Parameters
SEARCH_QUERY = "all:LLM+reason"  # Search papers related to LLM reasoning
MAX_RESULTS = 5       
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "data")                             # Number of papers to fetch
SAVE_FILE = os.path.join(SAVE_DIR, "llm_reasoning_papers.json")

# Ensure output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Query URL (arXiv API)
base_url = "http://export.arxiv.org/api/query?"
query = f"search_query={SEARCH_QUERY}&sortBy=submittedDate&sortOrder=descending&start=0&max_results={MAX_RESULTS}"

print("Fetching metadata from arXiv...")
feed = feedparser.parse(base_url + query)

print(f"Found {len(feed.entries)} papers.")

papers = []
for i, entry in enumerate(feed.entries, 1):
    abs_url = None
    for link in entry.links:
        if link.rel == "alternate":
            abs_url = link.href  # abstract page URL

    paper_info = {
        "title": entry.title,
        "link": abs_url,
        "abstract": entry.summary.replace("\n", " ").strip()
    }
    papers.append(paper_info)

    print(f"[{i}] {entry.title[:80]}")

# Save results into JSON
with open(SAVE_FILE, "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print(f"\n✅ Done! Saved {len(papers)} papers to {SAVE_FILE}")
