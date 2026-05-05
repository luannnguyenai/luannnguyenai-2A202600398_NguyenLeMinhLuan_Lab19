"""
Indexing pipeline: corpus → paragraphs → LLM extraction → normalized triples → JSON.

Run as:
    python -m src.indexing

Outputs:
    outputs/triples.json
"""

from __future__ import annotations

import json
import logging
import re
import string
import sys
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from src.utils.llm_client import get_client, TRACKER
from src.utils.prompts import (
    build_extraction_messages,
    build_extraction_retry_messages,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "tech_company_corpus.txt"
OUTPUT_DIR = ROOT / "outputs"
TRIPLES_PATH = OUTPUT_DIR / "triples.json"

FUZZY_THRESHOLD = 90          # rapidfuzz score 0-100
MAX_RETRIES = 2               # retry malformed JSON up to 2 times
ALLOWED_ENTITY_TYPES = {"Person", "Organization", "Product", "Location", "Year", "Event"}
ALLOWED_RELATIONS = {
    "FOUNDED_BY", "FOUNDED_IN", "CEO_OF", "ACQUIRED", "RELEASED",
    "HEADQUARTERED_IN", "SUBSIDIARY_OF", "INVESTED_IN",
    "COLLABORATES_WITH", "COMPETES_WITH",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Lowercase and strip punctuation for canonical matching."""
    name = name.lower().strip()
    name = name.translate(str.maketrans("", "", string.punctuation))
    return " ".join(name.split())


def _parse_json_triples(raw: str) -> list[dict[str, str]] | None:
    """
    Attempt to parse LLM output as JSON and return the list of triples.
    Returns None if parsing fails or schema is wrong.
    """
    try:
        # Strip any accidental markdown fences
        cleaned = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
        data = json.loads(cleaned)
        if "triples" not in data or not isinstance(data["triples"], list):
            return None
        # Validate each triple has required keys
        required = {"subject", "subject_type", "relation", "object", "object_type"}
        for t in data["triples"]:
            if not required.issubset(t.keys()):
                return None
        return data["triples"]
    except (json.JSONDecodeError, TypeError):
        return None


def _filter_triple(t: dict[str, str]) -> bool:
    """Return True if the triple passes type/relation validation."""
    return (
        t.get("subject_type") in ALLOWED_ENTITY_TYPES
        and t.get("object_type") in ALLOWED_ENTITY_TYPES
        and t.get("relation") in ALLOWED_RELATIONS
        and bool(t.get("subject", "").strip())
        and bool(t.get("object", "").strip())
    )


# ---------------------------------------------------------------------------
# Entity deduplication
# ---------------------------------------------------------------------------

class CanonicalMap:
    """
    Maps variant entity names to a single canonical form.

    A new name is added as canonical if no existing canonical name has
    a fuzzy similarity >= FUZZY_THRESHOLD. Otherwise the new name is mapped
    to the most-similar existing canonical.
    """

    def __init__(self, threshold: int = FUZZY_THRESHOLD) -> None:
        self.threshold = threshold
        self._canonicals: list[str] = []       # normalised canonical names
        self._raw_canonicals: list[str] = []   # original-case canonical names
        self._map: dict[str, str] = {}         # norm(variant) → raw canonical

    def resolve(self, name: str) -> str:
        """Return the canonical form for *name*, registering it if needed."""
        norm = _normalize(name)
        if norm in self._map:
            return self._map[norm]

        if not self._canonicals:
            # First entity ever
            self._canonicals.append(norm)
            self._raw_canonicals.append(name.strip())
            self._map[norm] = name.strip()
            return name.strip()

        result = process.extractOne(norm, self._canonicals, scorer=fuzz.ratio)
        if result and result[1] >= self.threshold:
            canonical_raw = self._raw_canonicals[self._canonicals.index(result[0])]
            self._map[norm] = canonical_raw
            return canonical_raw
        else:
            # New distinct entity
            self._canonicals.append(norm)
            self._raw_canonicals.append(name.strip())
            self._map[norm] = name.strip()
            return name.strip()


# ---------------------------------------------------------------------------
# LLM extraction with retry
# ---------------------------------------------------------------------------

def extract_triples_from_chunk(
    text: str,
    chunk_id: int,
    client: Any,
) -> list[dict[str, str]]:
    """
    Call the LLM to extract triples from *text* with up to MAX_RETRIES retries.

    Returns a (possibly empty) list of valid triple dicts.
    """
    messages = build_extraction_messages(text)
    raw = client.chat(messages, stage="indexing", temperature=0.0, max_tokens=1500)
    triples = _parse_json_triples(raw)

    retries = 0
    while triples is None and retries < MAX_RETRIES:
        retries += 1
        logger.warning("Chunk %d: malformed JSON (attempt %d), retrying...", chunk_id, retries)
        messages = build_extraction_retry_messages(text, raw)
        raw = client.chat(messages, stage="indexing", temperature=0.0, max_tokens=1500)
        triples = _parse_json_triples(raw)

    if triples is None:
        logger.error("Chunk %d: gave up after %d retries. Skipping chunk.", chunk_id, MAX_RETRIES)
        return []

    valid = [t for t in triples if _filter_triple(t)]
    logger.info("Chunk %d: extracted %d valid triples (raw=%d).", chunk_id, len(valid), len(triples))
    return valid


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_indexing() -> list[dict[str, str]]:
    """
    Full indexing pipeline.

    1. Read corpus and split into paragraphs.
    2. For each paragraph, call the LLM to extract triples.
    3. Normalize and deduplicate entity names.
    4. Save to outputs/triples.json.

    Returns the list of normalized triple dicts.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client()
    cmap = CanonicalMap()

    logger.info("Reading corpus from %s", CORPUS_PATH)
    raw_corpus = CORPUS_PATH.read_text(encoding="utf-8")

    # Split on blank lines; keep non-empty paragraphs
    paragraphs = [p.strip() for p in raw_corpus.split("\n\n") if p.strip()]
    logger.info("Corpus split into %d paragraphs.", len(paragraphs))

    all_triples: list[dict[str, str]] = []

    for chunk_id, paragraph in enumerate(paragraphs):
        raw_triples = extract_triples_from_chunk(paragraph, chunk_id, client)

        for t in raw_triples:
            # Normalise entity names through canonical map
            t["subject"] = cmap.resolve(t["subject"])
            t["object"] = cmap.resolve(t["object"])
            t["source_chunk_id"] = chunk_id
            all_triples.append(t)

    # Deduplicate identical triples
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for t in all_triples:
        key = (t["subject"], t["relation"], t["object"])
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    logger.info(
        "Total triples: %d (after dedup from %d raw).", len(deduped), len(all_triples)
    )
    TRIPLES_PATH.write_text(json.dumps(deduped, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved triples to %s", TRIPLES_PATH)

    # Print token summary and cache indexing tokens for warm runs
    by_stage = TRACKER.tokens_by_stage()
    if "indexing" in by_stage:
        s = by_stage["indexing"]
        logger.info(
            "Indexing token usage — prompt: %d, completion: %d, total: %d",
            s["prompt"], s["completion"], s["total"],
        )

        # Save indexing tokens to cache for evaluate.py (warm runs)
        indexing_cache = {
            "stage": "indexing",
            "prompt": s["prompt"],
            "completion": s["completion"],
            "total": s["total"],
            "latency_ms_total": sum(r.latency_ms for r in TRACKER.records if r.stage == "indexing"),
            "calls": sum(1 for r in TRACKER.records if r.stage == "indexing"),
        }
        cache_path = OUTPUT_DIR / "indexing_tokens.json"
        cache_path.write_text(json.dumps(indexing_cache, indent=2), encoding="utf-8")
        logger.info("Saved indexing tokens cache to %s", cache_path)

    return deduped


if __name__ == "__main__":
    triples = run_indexing()
    print(f"\n=== Sample triples (first 10 of {len(triples)}) ===")
    for t in triples[:10]:
        print(
            f"  [{t['subject_type']}] {t['subject']} --[{t['relation']}]--> "
            f"{t['object']} [{t['object_type']}]  (chunk {t['source_chunk_id']})"
        )
    sys.exit(0)
