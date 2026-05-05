"""
NetworkX Graph builder for Wikipedia triples.
"""

import logging
from pathlib import Path
from src.graph_networkx import run_graph_networkx, TRIPLES_PATH as ORIG_TRIPLES_PATH, GRAPHML_PATH as ORIG_GRAPHML_PATH, PNG_PATH as ORIG_PNG_PATH

# Config
ROOT = Path(__file__).resolve().parent.parent
WIKI_TRIPLES_PATH = ROOT / "outputs" / "triples_wiki.json"
WIKI_GRAPHML_PATH = ROOT / "outputs" / "graph_wiki.graphml"
WIKI_PNG_PATH = ROOT / "outputs" / "graph_screenshot_wiki.png"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    import src.graph_networkx
    src.graph_networkx.TRIPLES_PATH = WIKI_TRIPLES_PATH
    src.graph_networkx.GRAPHML_PATH = WIKI_GRAPHML_PATH
    src.graph_networkx.PNG_PATH = WIKI_PNG_PATH
    
    run_graph_networkx()
    logger.info("Wikipedia graph rendered successfully.")

if __name__ == "__main__":
    main()
