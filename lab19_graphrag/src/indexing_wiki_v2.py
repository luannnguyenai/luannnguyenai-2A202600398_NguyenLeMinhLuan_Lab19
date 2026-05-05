"""
Fast Wikipedia Indexing: Parallel LLM extraction with checkpointing.
"""

import json
import logging
import re
import concurrent.futures
from pathlib import Path
from src.utils.llm_client import get_client
from src.indexing import extract_triples_from_chunk, CanonicalMap

# Config
ROOT = Path(__file__).resolve().parent.parent
WIKI_CORPUS_PATH = ROOT / "data" / "wiki_corpus.txt"
WIKI_TRIPLES_PATH = ROOT / "outputs" / "triples_wiki.json"
PARTIAL_PATH = ROOT / "outputs" / "triples_wiki_partial.json"
CONCURRENCY = 10  # Process 10 chunks at once

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def clean_wiki_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text)
    return text

def main():
    llm = get_client()
    raw_text = WIKI_CORPUS_PATH.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    logger.info("Total chunks to process: %d", len(paragraphs))

    # Load partials
    all_triples = []
    processed_ids = set()
    if PARTIAL_PATH.exists():
        with PARTIAL_PATH.open(encoding="utf-8") as f:
            all_triples = json.load(f)
            processed_ids = {t.get("source_chunk_id") for t in all_triples if t.get("source_chunk_id") is not None}
            logger.info("Resuming from %d processed chunks.", len(processed_ids))

    to_process = [(i, p) for i, p in enumerate(paragraphs) if i not in processed_ids]
    
    if not to_process:
        logger.info("All chunks already processed.")
    else:
        logger.info("Processing %d remaining chunks with concurrency=%d", len(to_process), CONCURRENCY)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            future_to_chunk = {
                executor.submit(extract_triples_from_chunk, clean_wiki_text(p), i, llm): i 
                for i, p in to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    triples = future.result()
                    for t in triples:
                        t["source_chunk_id"] = chunk_id
                    all_triples.extend(triples)
                    # Periodically save partials
                    if len(all_triples) % 50 == 0:
                        with PARTIAL_PATH.open("w", encoding="utf-8") as f:
                            json.dump(all_triples, f, indent=2)
                except Exception as e:
                    logger.error("Chunk %d failed: %s", chunk_id, e)

    # Canonicalize
    logger.info("Deduplicating and canonicalizing %d raw triples...", len(all_triples))
    c_map = CanonicalMap()
    canonical_triples = []
    for t in all_triples:
        t["subject"] = c_map.resolve(t["subject"])
        t["object"] = c_map.resolve(t["object"])
        canonical_triples.append(t)
    
    # Final save
    with WIKI_TRIPLES_PATH.open("w", encoding="utf-8") as f:
        json.dump(canonical_triples, f, indent=2)
    
    logger.info("Saved %d triples to %s", len(canonical_triples), WIKI_TRIPLES_PATH)
    
    # Stats
    nodes = {t["subject"] for t in canonical_triples} | {t["object"] for t in canonical_triples}
    logger.info("Graph Stats: %d nodes, %d edges", len(nodes), len(canonical_triples))

if __name__ == "__main__":
    main()
