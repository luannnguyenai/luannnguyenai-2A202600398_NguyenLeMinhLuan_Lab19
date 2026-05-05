"""
GraphRAG for hard experiment: sparse corpus graph → BFS depth=4 → QA.

Hard variant of graph_rag.py:
  - Uses triples_hard.json from sparse corpus
  - BFS_DEPTH = 4 (vs 2) to handle 4-hop questions
  - MAX_FACTS = 80 (vs 60) for deeper subgraph exploration

Run as:
    python -m src.graph_rag_hard

Requires:
    outputs/triples_hard.json  (produced by src.indexing_hard)
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from src.utils.llm_client import get_client, TRACKER
from src.utils.prompts import build_qa_messages, build_entity_extraction_messages

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TRIPLES_PATH = ROOT / "outputs" / "triples_hard.json"
BFS_DEPTH = 4                 # Increased from 2 to handle 4-hop questions
FUZZY_THRESHOLD = 70          # Same as original
MAX_FACTS = 80                # Increased from 60 for deeper traversal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_adjacency(triples: list[dict[str, str]]) -> dict[str, list[tuple[str, str]]]:
    """
    Build a simple adjacency dict from triples for fast BFS.

    For each triple (S, R, O):
      adj[S] includes (R, O)
      adj[O] includes (INVERSE_R, S)   ← undirected BFS for better recall
    """
    adj: dict[str, list[tuple[str, str]]] = {}
    for t in triples:
        s, r, o = t["subject"], t["relation"], t["object"]
        adj.setdefault(s, []).append((r, o))
        adj.setdefault(o, []).append((f"INVERSE_{r}", s))
    return adj


# ---------------------------------------------------------------------------
# BFS subgraph retrieval
# ---------------------------------------------------------------------------

def bfs_subgraph(
    adj: dict[str, list[tuple[str, str]]],
    seeds: list[str],
    depth: int = BFS_DEPTH,
) -> list[tuple[str, str, str]]:
    """
    BFS from *seeds* up to *depth* hops.

    Returns a list of (subject, relation, object) fact tuples.
    """
    visited_nodes: set[str] = set()
    visited_edges: set[tuple[str, str, str]] = set()
    queue: deque[tuple[str, int]] = deque()

    for seed in seeds:
        if seed in adj:
            queue.append((seed, 0))
            visited_nodes.add(seed)

    while queue:
        node, d = queue.popleft()
        if d >= depth:
            continue
        for rel, neighbor in adj.get(node, []):
            edge = (node, rel, neighbor)
            if edge not in visited_edges:
                visited_edges.add(edge)
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, d + 1))

    return list(visited_edges)


# ---------------------------------------------------------------------------
# Textualization
# ---------------------------------------------------------------------------

def textualize_subgraph(facts: list[tuple[str, str, str]]) -> str:
    """
    Convert a list of (subject, relation, object) tuples into numbered sentences.
    """
    lines = [
        f"{i + 1}. {s} --[{r}]--> {o}"
        for i, (s, r, o) in enumerate(facts[:MAX_FACTS])
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------

def extract_question_entities(question: str) -> list[str]:
    """
    Use the LLM to extract named entities from a question.
    Falls back to an empty list on parse error.
    """
    llm = get_client()
    messages = build_entity_extraction_messages(question)
    raw = llm.chat(messages, stage="graph_rag_hard_entity_extraction", temperature=0.0, max_tokens=200)
    try:
        entities = json.loads(raw.strip())
        if isinstance(entities, list):
            return [str(e) for e in entities]
    except (json.JSONDecodeError, ValueError):
        logger.warning("Could not parse entity list from: %r", raw)
    return []


def fuzzy_match_entities(
    entities: list[str],
    graph_nodes: list[str],
    threshold: int = FUZZY_THRESHOLD,
) -> list[str]:
    """
    For each extracted entity, find the closest matching graph node.

    Returns a deduplicated list of matched node names.
    """
    matched: list[str] = []
    for entity in entities:
        if not graph_nodes:
            break
        result = process.extractOne(entity, graph_nodes, scorer=fuzz.partial_ratio)
        if result and result[1] >= threshold:
            matched.append(result[0])
            logger.debug("Matched '%s' → '%s' (score=%d)", entity, result[0], result[1])
        else:
            logger.debug("No match for entity '%s' (best score=%s).", entity, result[1] if result else "N/A")
    # Deduplicate preserving order
    seen: set[str] = set()
    deduped = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


# ---------------------------------------------------------------------------
# Query function
# ---------------------------------------------------------------------------

# Module-level cache so the graph is only loaded once per process
_triples_cache: list[dict[str, str]] | None = None
_adj_cache: dict[str, list[tuple[str, str]]] | None = None
_nodes_cache: list[str] | None = None


def _load_graph() -> tuple[list[dict[str, str]], dict, list[str]]:
    global _triples_cache, _adj_cache, _nodes_cache
    if _triples_cache is None:
        with TRIPLES_PATH.open(encoding="utf-8") as fh:
            _triples_cache = json.load(fh)
        _adj_cache = load_adjacency(_triples_cache)
        _nodes_cache = list(_adj_cache.keys())
        logger.info(
            "Loaded hard graph: %d triples, %d unique nodes.",
            len(_triples_cache),
            len(_nodes_cache),
        )
    return _triples_cache, _adj_cache, _nodes_cache  # type: ignore[return-value]


def query_graph_rag_hard(question: str) -> dict[str, Any]:
    """
    Full GraphRAG query pipeline for hard experiment.

    Steps:
        1. Extract named entities from question via LLM.
        2. Fuzzy-match entities to graph nodes.
        3. BFS up to depth=4 from matched nodes.
        4. Textualize subgraph facts.
        5. Feed into QA prompt → answer.

    Returns
    -------
    dict with keys:
        answer (str), tokens (int), latency_ms (float),
        subgraph_text (str), matched_nodes (list[str])
    """
    t0 = time.perf_counter()
    triples, adj, nodes = _load_graph()
    llm = get_client()

    # Step 1: entity extraction
    entities = extract_question_entities(question)
    logger.info("Extracted entities: %s", entities)

    # Step 2: fuzzy match to graph nodes
    matched_nodes = fuzzy_match_entities(entities, nodes)
    logger.info("Matched graph nodes: %s", matched_nodes)

    # Step 3: BFS subgraph (depth=4 for hard variant)
    facts = bfs_subgraph(adj, matched_nodes, depth=BFS_DEPTH)
    logger.info("BFS yielded %d facts from nodes %s.", len(facts), matched_nodes)

    # Step 4: textualize
    if facts:
        subgraph_text = textualize_subgraph(facts)
        context = f"Knowledge Graph Facts:\n{subgraph_text}"
    else:
        context = "No relevant knowledge graph facts were found for this question."
        subgraph_text = ""

    # Step 5: QA
    messages = build_qa_messages(context=context, question=question)
    answer = llm.chat(messages, stage="graph_rag_hard_query", temperature=0.0, max_tokens=512)

    total_latency_ms = (time.perf_counter() - t0) * 1000

    last_record = TRACKER.records[-1] if TRACKER.records else None
    tokens = (
        (last_record.prompt_tokens + last_record.completion_tokens)
        if last_record else 0
    )

    logger.info("GraphRAG (hard) answered in %.0f ms (%d tokens).", total_latency_ms, tokens)
    return {
        "answer": answer,
        "tokens": tokens,
        "latency_ms": total_latency_ms,
        "subgraph_text": subgraph_text,
        "matched_nodes": matched_nodes,
    }


if __name__ == "__main__":
    import sys
    if not TRIPLES_PATH.exists():
        print("ERROR: Run 'python -m src.indexing_hard' first to generate triples_hard.json")
        sys.exit(1)

    demo_q = "Who is the CEO of the company that acquired DeepMind?"
    print(f"\nDemo question: {demo_q}")
    result = query_graph_rag_hard(demo_q)
    print(f"\nMatched nodes: {result['matched_nodes']}")
    print(f"\nSubgraph:\n{result['subgraph_text'][:500]}...")
    print(f"\nAnswer: {result['answer']}")
    print(f"Tokens: {result['tokens']}, Latency: {result['latency_ms']:.0f} ms")
    sys.exit(0)
