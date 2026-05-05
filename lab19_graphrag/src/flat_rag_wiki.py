"""
Flat RAG for Wikipedia: Sentence-level chunking + TOP_K=4.
"""

import logging
import time
import re
from pathlib import Path
from typing import Any, Optional

import chromadb
from src.utils.llm_client import get_client, TRACKER
from src.utils.prompts import build_qa_messages

# Config
ROOT = Path(__file__).resolve().parent.parent
WIKI_CORPUS_PATH = ROOT / "data" / "wiki_corpus.txt"
OUTPUT_DIR = ROOT / "outputs"
CHROMA_DIR = OUTPUT_DIR / "chroma_db_wiki"
COLLECTION_NAME = "wiki_corpus"
TOP_K = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def chunk_by_sentence(text: str) -> list[str]:
    """Simple sentence splitter using regex."""
    # Split on ". " followed by capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def build_flat_index_wiki(force_rebuild: bool = False) -> chromadb.Collection:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    llm = get_client()

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        if force_rebuild:
            client.delete_collection(COLLECTION_NAME)
        else:
            return client.get_collection(COLLECTION_NAME)

    raw_text = WIKI_CORPUS_PATH.read_text(encoding="utf-8")
    # Remove headers and timestamp
    lines = [l for l in raw_text.split("\n") if not l.startswith("===") and not l.startswith("Scrape Timestamp")]
    clean_text = " ".join(lines)
    
    sentences = chunk_by_sentence(clean_text)
    logger.info("Embedding %d sentences for Wikipedia corpus...", len(sentences))

    # Embed in batches to be safe
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        all_embeddings.extend(llm.embed(batch, stage="flat_rag_wiki_indexing"))

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(
        documents=sentences,
        embeddings=all_embeddings,
        ids=[f"sent_{i}" for i in range(len(sentences))],
        metadatas=[{"sent_id": i} for i in range(len(sentences))]
    )
    return collection

def query_flat_rag_wiki(question: str, collection: Optional[chromadb.Collection] = None) -> dict[str, Any]:
    if collection is None:
        collection = build_flat_index_wiki()

    llm = get_client()
    t0 = time.perf_counter()
    
    q_embedding = llm.embed_single(question, stage="flat_rag_wiki_query")
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=TOP_K,
        include=["documents"]
    )
    retrieved_docs = results["documents"][0] if results["documents"] else []
    context = "\n".join(f"- {doc}" for doc in retrieved_docs)

    messages = build_qa_messages(context=context, question=question)
    answer = llm.chat(messages, stage="flat_rag_wiki_query", temperature=0.0, max_tokens=512)
    latency = (time.perf_counter() - t0) * 1000

    last_record = TRACKER.records[-1] if TRACKER.records else None
    tokens = (last_record.prompt_tokens + last_record.completion_tokens) if last_record else 0

    return {
        "answer": answer,
        "tokens": tokens,
        "latency_ms": latency,
        "context": context
    }
