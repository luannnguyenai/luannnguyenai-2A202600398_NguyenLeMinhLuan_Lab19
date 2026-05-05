"""
Scrape 10 related Wikipedia pages to create a multi-document corpus for GraphRAG evaluation.
"""

import os
import time
import requests
import logging
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup

# Config
URLS = [
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/Google_DeepMind",
    "https://en.wikipedia.org/wiki/Sam_Altman",
    "https://en.wikipedia.org/wiki/Demis_Hassabis",
    "https://en.wikipedia.org/wiki/Microsoft",
    "https://en.wikipedia.org/wiki/Anthropic",
    "https://en.wikipedia.org/wiki/Nvidia",
    "https://en.wikipedia.org/wiki/Satya_Nadella",
    "https://en.wikipedia.org/wiki/Sundar_Pichai",
    "https://en.wikipedia.org/wiki/Elon_Musk",
]

ROOT = Path(__file__).resolve().parent.parent
WIKI_DATA_DIR = ROOT / "data" / "wiki"
CORPUS_PATH = ROOT / "data" / "wiki_corpus.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def scrape_page(url: str) -> list[str]:
    """Scrape paragraphs from a Wikipedia URL."""
    logger.info("Scraping: %s", url)
    headers = {"User-Agent": "lab19-bot/1.0"}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        logger.error("Failed to fetch %s: Status %d", url, res.status_code)
        res.raise_for_status()
    
    soup = BeautifulSoup(res.text, "html.parser")
    # Wikipedia paragraphs
    paragraphs = [p.get_text(strip=True) for p in soup.select("p") if p.get_text(strip=True)]
    
    # Filter: min length 40, truncate to first 30
    filtered = []
    for p in paragraphs:
        if len(p) >= 40:
            filtered.append(p)
        if len(filtered) >= 30:
            break
            
    return filtered

def main():
    WIKI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_texts = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_texts.append(f"Scrape Timestamp: {timestamp}\n")

    for url in URLS:
        slug = url.split("/")[-1].lower()
        try:
            paragraphs = scrape_page(url)
            
            # Save individual file
            file_path = WIKI_DATA_DIR / f"{slug}.txt"
            file_path.write_text("\n\n".join(paragraphs), encoding="utf-8")
            
            # Add to corpus buffer
            all_texts.append(f"=== {slug} ===")
            all_texts.extend(paragraphs)
            all_texts.append("") # Blank line between page blocks
            
            logger.info("Captured %d paragraphs for %s", len(paragraphs), slug)
            time.sleep(1) # Polite scraping
        except Exception as e:
            logger.error("Error scraping %s: %s", url, e)
            raise

    # Write consolidated corpus
    CORPUS_PATH.write_text("\n\n".join(all_texts), encoding="utf-8")
    logger.info("Final wiki corpus saved to %s", CORPUS_PATH)
    
    # Confirm stats
    size_kb = CORPUS_PATH.stat().st_size / 1024
    para_count = len(CORPUS_PATH.read_text(encoding="utf-8").split("\n\n"))
    print(f"\nScraping Success!")
    print(f"File: {CORPUS_PATH}")
    print(f"Size: {size_kb:.2f} KB")
    print(f"Total Blocks (Paragraphs + Headers): {para_count}")

if __name__ == "__main__":
    main()
