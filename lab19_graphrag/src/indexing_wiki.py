"""
Wikipedia Indexing: CORPUS -> triples_wiki.json.
Strips citation markers [1], [citation needed] before indexing.
"""

import json
import logging
import re
from pathlib import Path
from src.indexing import run_indexing, TRIPLES_PATH as ORIG_TRIPLES_PATH, CORPUS_PATH as ORIG_CORPUS_PATH

# Config
ROOT = Path(__file__).resolve().parent.parent
WIKI_CORPUS_PATH = ROOT / "data" / "wiki_corpus.txt"
WIKI_TRIPLES_PATH = ROOT / "outputs" / "triples_wiki.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def clean_wiki_text(text: str) -> str:
    """Strip Wikipedia citation markers like [1], [citation needed], [2][3]."""
    # Remove [12], [citation needed], etc.
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text)
    # Remove multiple citations like [1][2]
    text = re.sub(r"\[\d+\]\[\d+\]", "", text)
    return text

def main():
    # Monkeypatch indexing paths
    import src.indexing
    src.indexing.CORPUS_PATH = WIKI_CORPUS_PATH
    src.indexing.TRIPLES_PATH = WIKI_TRIPLES_PATH
    
    # Wrap extract_triples_from_chunk to clean text
    original_extract = src.indexing.extract_triples_from_chunk
    
    def wrapped_extract(text, chunk_id, client):
        cleaned = clean_wiki_text(text)
        return original_extract(cleaned, chunk_id, client)
    
    src.indexing.extract_triples_from_chunk = wrapped_extract
    
    # Run pipeline
    logger.info("Starting Wikipedia indexing stage: indexing_wiki")
    # Change stage name in TRACKER calls inside indexing.py
    # This is tricky without modifying indexing.py, but indexing.py uses stage="indexing"
    # We'll just let it run and note the tokens.
    
    triples = run_indexing()
    
    # Verify
    unique_triples = len(triples)
    logger.info("Wikipedia indexing complete. Found %d unique triples.", unique_triples)
    
    # Check for canonical nodes
    nodes = set()
    for t in triples:
        nodes.add(t["subject"])
        nodes.add(t["object"])
        
    canonical_checks = ["Sam Altman", "Microsoft", "Demis Hassabis", "DeepMind"]
    for check in canonical_checks:
        found = any(check in n for n in nodes)
        logger.info("Canonical node check [%s]: %s", check, "Found" if found else "MISSING")

if __name__ == "__main__":
    main()
