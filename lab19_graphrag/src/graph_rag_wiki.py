"""
GraphRAG for Wikipedia: BFS_DEPTH=4 + MAX_FACTS=100.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.utils.llm_client import get_client, TRACKER
from src.utils.prompts import build_qa_messages
from src.graph_rag import (
    load_adjacency, 
    bfs_subgraph, 
    textualize_subgraph, 
    extract_question_entities, 
    fuzzy_match_entities
)

# Config
ROOT = Path(__file__).resolve().parent.parent
WIKI_TRIPLES_PATH = ROOT / "outputs" / "triples_wiki.json"
BFS_DEPTH = 4
MAX_FACTS = 120

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_wiki_triples_cache = None
_wiki_adj_cache = None
_wiki_nodes_cache = None

def _load_wiki_graph():
    global _wiki_triples_cache, _wiki_adj_cache, _wiki_nodes_cache
    if _wiki_triples_cache is None:
        if not WIKI_TRIPLES_PATH.exists():
            raise FileNotFoundError(f"Triples not found at {WIKI_TRIPLES_PATH}. Run indexing_wiki first.")
        with WIKI_TRIPLES_PATH.open(encoding="utf-8") as fh:
            _wiki_triples_cache = json.load(fh)
        _wiki_adj_cache = load_adjacency(_wiki_triples_cache)
        _wiki_nodes_cache = list(_wiki_adj_cache.keys())
    return _wiki_triples_cache, _wiki_adj_cache, _wiki_nodes_cache

def query_graph_rag_wiki(question: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    _, adj, nodes = _load_wiki_graph()
    llm = get_client()

    entities = extract_question_entities(question)
    # Use stage name for wiki
    # Note: extract_question_entities uses a hardcoded stage, 
    # but we can't change it without modifying src/graph_rag.py.
    # We will override the stage in our TRACKER record if needed, but let's stick to rules.
    
    matched_nodes = fuzzy_match_entities(entities, nodes)
    facts = bfs_subgraph(adj, matched_nodes, depth=BFS_DEPTH)
    
    if facts:
        # Override textualize to handle wiki max facts
        subgraph_text = "\n".join([f"{i+1}. {s} --[{r}]--> {o}" for i, (s,r,o) in enumerate(facts[:MAX_FACTS])])
        context = f"Knowledge Graph Facts:\n{subgraph_text}"
    else:
        context = "No relevant knowledge graph facts were found."
        subgraph_text = ""

    messages = build_qa_messages(context=context, question=question)
    answer = llm.chat(messages, stage="graph_rag_wiki_query", temperature=0.0, max_tokens=512)
    latency = (time.perf_counter() - t0) * 1000

    last_record = TRACKER.records[-1] if TRACKER.records else None
    tokens = (last_record.prompt_tokens + last_record.completion_tokens) if last_record else 0

    return {
        "answer": answer,
        "tokens": tokens,
        "latency_ms": latency,
        "subgraph_text": subgraph_text,
        "matched_nodes": matched_nodes
    }
