"""
Flat RAG baseline: corpus → ChromaDB vector store → top-k retrieval → QA.

Run as:
    python -m src.flat_rag

Outputs:
    outputs/chroma_db/   (persistent ChromaDB storage)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.utils.llm_client import get_client, TRACKER
from src.utils.prompts import build_qa_messages

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "tech_company_corpus.txt"
OUTPUT_DIR = ROOT / "outputs"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
COLLECTION_NAME = "tech_corpus"
TOP_K = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def load_paragraphs(corpus_path: Path) -> list[str]:
    """Split corpus file into non-empty paragraphs."""
    raw = corpus_path.read_text(encoding="utf-8")
    return [p.strip() for p in raw.split("\n\n") if p.strip()]


def build_flat_index(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Build (or load) a persistent ChromaDB collection from the corpus.

    Parameters
    ----------
    force_rebuild : if True, drop existing collection and rebuild.

    Returns
    -------
    ChromaDB Collection ready for querying.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    llm = get_client()

    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing:
        if force_rebuild:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Dropped existing collection '%s' for rebuild.", COLLECTION_NAME)
        else:
            logger.info("Collection '%s' already exists — skipping rebuild.", COLLECTION_NAME)
            return client.get_collection(COLLECTION_NAME)

    paragraphs = load_paragraphs(CORPUS_PATH)
    logger.info("Embedding %d paragraphs with text-embedding-3-small...", len(paragraphs))

    embeddings = llm.embed(paragraphs, stage="flat_rag_indexing")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=paragraphs,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(paragraphs))],
        metadatas=[{"chunk_id": i} for i in range(len(paragraphs))],
    )
    logger.info("Added %d documents to ChromaDB collection '%s'.", len(paragraphs), COLLECTION_NAME)
    return collection


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_flat_rag(
    question: str,
    collection: chromadb.Collection | None = None,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    Retrieve top-k relevant paragraphs and generate an answer via QA prompt.

    Parameters
    ----------
    question : the question to answer.
    collection : pre-loaded ChromaDB collection (built if None).
    top_k : number of chunks to retrieve.

    Returns
    -------
    dict with keys:
        answer (str), tokens (int), latency_ms (float), context (str)
    """
    if collection is None:
        collection = build_flat_index()

    llm = get_client()

    # --- Retrieval ---
    t0 = time.perf_counter()
    q_embedding = llm.embed_single(question, stage="flat_rag_query")
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents"],
    )
    retrieved_docs: list[str] = results["documents"][0] if results["documents"] else []

    context = "\n\n".join(
        f"[Chunk {i + 1}]: {doc}" for i, doc in enumerate(retrieved_docs)
    )

    # --- Generation ---
    messages = build_qa_messages(context=context, question=question)
    # Track tokens via TRACKER inside llm.chat; also capture wall-clock latency
    t_gen_start = time.perf_counter()
    answer = llm.chat(messages, stage="flat_rag_query", temperature=0.0, max_tokens=512)
    total_latency_ms = (time.perf_counter() - t0) * 1000

    # Approximate token count from the last call record
    last_record = TRACKER.records[-1] if TRACKER.records else None
    tokens = (
        (last_record.prompt_tokens + last_record.completion_tokens)
        if last_record else 0
    )

    logger.info("Flat RAG answered question in %.0f ms (%d tokens).", total_latency_ms, tokens)
    return {
        "answer": answer,
        "tokens": tokens,
        "latency_ms": total_latency_ms,
        "context": context,
    }


# ---------------------------------------------------------------------------
# Main (demo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    col = build_flat_index()
    demo_q = "Who is the CEO of the company that acquired DeepMind?"
    print(f"\nDemo question: {demo_q}")
    result = query_flat_rag(demo_q, collection=col)
    print(f"Answer: {result['answer']}")
    print(f"Tokens: {result['tokens']}, Latency: {result['latency_ms']:.0f} ms")
    sys.exit(0)
