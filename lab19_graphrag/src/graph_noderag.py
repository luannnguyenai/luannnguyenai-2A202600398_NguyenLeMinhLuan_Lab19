"""
NodeRAG integration: build the knowledge graph using the NodeRAG framework.

Run as:
    python -m src.graph_noderag

Falls back gracefully if NodeRAG is not installed or its API has changed.

NodeRAG (https://github.com/Terry-Xu-666/NodeRAG) is an all-in-one GraphRAG
framework. This module attempts to use its high-level API and prints a note
if the package is unavailable or the API doesn't match expectations.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRIPLES_PATH = ROOT / "outputs" / "triples.json"
OUTPUT_DIR = ROOT / "outputs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_noderag() -> bool:
    """
    Attempt to build a NodeRAG graph from the corpus.

    Returns True on success, False on graceful fallback.
    """
    try:
        import NodeRAG  # type: ignore[import]
    except ImportError:
        print(
            "\n[NodeRAG] NodeRAG package is not installed.\n"
            "To install: pip install NodeRAG\n"
            "NodeRAG requires additional setup (API keys, config files).\n"
            "See: https://github.com/Terry-Xu-666/NodeRAG\n"
            "Skipping NodeRAG demo — all other pipeline steps are unaffected."
        )
        return False

    corpus_path = ROOT / "data" / "tech_company_corpus.txt"

    try:
        # NodeRAG high-level API (may vary by version)
        # NodeRAG typically expects a config dict or YAML file
        config = {
            "working_dir": str(OUTPUT_DIR / "noderag_workspace"),
            "model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
        }

        # Attempt to initialize NodeRAG
        if hasattr(NodeRAG, "NodeConfig") and hasattr(NodeRAG, "NodeRAG"):
            node_config = NodeRAG.NodeConfig.from_dict(config)
            rag = NodeRAG.NodeRAG(config=node_config)
            rag.build(corpus_path.read_text(encoding="utf-8"))
            logger.info("NodeRAG graph built successfully.")
            print("\n[NodeRAG] Graph built at: %s", OUTPUT_DIR / "noderag_workspace")
            return True
        else:
            raise AttributeError("NodeRAG API does not match expected interface.")

    except AttributeError as exc:
        print(
            f"\n[NodeRAG] API mismatch: {exc}\n"
            "NodeRAG may have updated its interface since this code was written.\n"
            "Please check the NodeRAG documentation and update this module.\n"
            "Skipping NodeRAG demo — all other pipeline steps are unaffected."
        )
        return False
    except Exception as exc:
        print(
            f"\n[NodeRAG] Unexpected error: {exc}\n"
            "Skipping NodeRAG demo — all other pipeline steps are unaffected."
        )
        logger.exception("NodeRAG error details:")
        return False


def main() -> None:
    if not TRIPLES_PATH.exists():
        logger.warning(
            "triples.json not found at %s. Run indexing first (optional for NodeRAG).",
            TRIPLES_PATH,
        )

    success = run_noderag()
    if not success:
        print("\n[NodeRAG] Demo skipped. The rest of the GraphRAG pipeline is fully functional.")
        print("NetworkX graph (graph.graphml + graph_screenshot.png) covers the same data.")


if __name__ == "__main__":
    main()
    sys.exit(0)
