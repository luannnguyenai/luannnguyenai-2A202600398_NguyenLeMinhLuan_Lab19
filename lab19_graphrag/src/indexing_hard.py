"""
Indexing pipeline for sparse corpus (hard experiment).

Points to data/sparse_corpus.txt and outputs to outputs/triples_hard.json.

Run as:
    python -m src.indexing_hard
"""

from __future__ import annotations

from pathlib import Path
from src import indexing as base_indexing

# Override paths
ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "sparse_corpus.txt"
TRIPLES_PATH = ROOT / "outputs" / "triples_hard.json"

def run_indexing_hard():
    """Run indexing on sparse corpus, output to triples_hard.json."""
    import logging
    logger = logging.getLogger(__name__)

    # Temporarily patch the module-level paths
    original_corpus = base_indexing.CORPUS_PATH
    original_triples = base_indexing.TRIPLES_PATH

    try:
        base_indexing.CORPUS_PATH = CORPUS_PATH
        base_indexing.TRIPLES_PATH = TRIPLES_PATH
        logger.info("Running indexing on sparse corpus...")
        logger.info("Corpus: %s", CORPUS_PATH)
        logger.info("Output: %s", TRIPLES_PATH)
        return base_indexing.run_indexing()
    finally:
        # Restore original paths
        base_indexing.CORPUS_PATH = original_corpus
        base_indexing.TRIPLES_PATH = original_triples


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    triples = run_indexing_hard()
    print(f"\n=== Indexing complete ===")
    print(f"Extracted {len(triples)} triples from sparse corpus")
