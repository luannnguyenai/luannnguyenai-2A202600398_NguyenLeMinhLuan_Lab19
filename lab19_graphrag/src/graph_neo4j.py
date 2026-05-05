"""
Neo4j graph loader: merge nodes and relationships from triples.json into Neo4j.

Run as (requires running Neo4j instance):
    python -m src.graph_neo4j

Environment variables (from .env):
    NEO4J_URI      e.g. bolt://localhost:7687
    NEO4J_USER     e.g. neo4j
    NEO4J_PASSWORD e.g. yourpassword
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
TRIPLES_PATH = ROOT / "outputs" / "triples.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_triples(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_neo4j_graph(triples: list[dict[str, str]]) -> None:
    """
    Connect to Neo4j and MERGE all nodes and relationships from triples.

    Uses MERGE to be idempotent — safe to run multiple times.
    """
    try:
        from neo4j import GraphDatabase  # type: ignore[import]
    except ImportError:
        logger.error("neo4j package not installed. Run: pip install neo4j")
        return

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")

    if not password:
        logger.warning(
            "NEO4J_PASSWORD not set. Attempting connection without authentication."
        )

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", uri)
    except Exception as exc:
        logger.error(
            "Could not connect to Neo4j at %s: %s\n"
            "Skipping Neo4j loading. Start Neo4j and retry.",
            uri, exc,
        )
        return

    def _load(tx: Any, triple: dict[str, str]) -> None:
        cypher = (
            "MERGE (s:Entity {name: $subject}) "
            "SET s.type = $subject_type "
            "MERGE (o:Entity {name: $object}) "
            "SET o.type = $object_type "
            f"MERGE (s)-[r:{triple['relation']}]->(o)"
        )
        tx.run(
            cypher,
            subject=triple["subject"],
            subject_type=triple.get("subject_type", "Unknown"),
            object=triple["object"],
            object_type=triple.get("object_type", "Unknown"),
        )

    with driver.session() as session:
        for i, triple in enumerate(triples):
            try:
                session.execute_write(_load, triple)
            except Exception as exc:
                logger.warning("Failed to merge triple %d: %s", i, exc)

    driver.close()
    logger.info("Loaded %d triples into Neo4j.", len(triples))

    # Print Cypher snippet for the user to run in Neo4j Browser
    print("\n" + "=" * 60)
    print("To visualize the graph in Neo4j Browser, run:")
    print("=" * 60)
    print("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
    print("=" * 60)
    print("\nOr to see all Organization nodes:")
    print("MATCH (n:Entity {type: 'Organization'}) RETURN n LIMIT 50")
    print("=" * 60)


def main() -> None:
    if not TRIPLES_PATH.exists():
        logger.error("triples.json not found at %s. Run indexing first.", TRIPLES_PATH)
        sys.exit(1)

    triples = load_triples(TRIPLES_PATH)
    logger.info("Loaded %d triples for Neo4j loading.", len(triples))
    build_neo4j_graph(triples)


# Type hint fix for the closure
from typing import Any  # noqa: E402


if __name__ == "__main__":
    main()
    sys.exit(0)
