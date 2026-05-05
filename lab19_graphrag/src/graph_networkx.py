"""
Build a NetworkX MultiDiGraph from triples.json, save as GraphML,
and render a PNG visualization.

Run as:
    python -m src.graph_networkx

Outputs:
    outputs/graph.graphml
    outputs/graph_screenshot.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TRIPLES_PATH = ROOT / "outputs" / "triples.json"
OUTPUT_DIR = ROOT / "outputs"
GRAPHML_PATH = OUTPUT_DIR / "graph.graphml"
PNG_PATH = OUTPUT_DIR / "graph_screenshot.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Node colour mapping by entity type
TYPE_COLORS: dict[str, str] = {
    "Person": "#4A90D9",
    "Organization": "#E67E22",
    "Product": "#2ECC71",
    "Location": "#9B59B6",
    "Year": "#F39C12",
    "Event": "#E74C3C",
    "Unknown": "#95A5A6",
}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def load_triples(path: Path) -> list[dict[str, str]]:
    """Load triples from JSON file."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_graph(triples: list[dict[str, str]]) -> nx.MultiDiGraph:
    """
    Build a NetworkX MultiDiGraph from triple dicts.

    Each node stores its entity type as a 'type' attribute.
    Each edge stores the relation label.
    """
    G = nx.MultiDiGraph()

    for t in triples:
        subj = t["subject"]
        obj = t["object"]
        rel = t["relation"]

        # Add / update nodes with type attribute
        if subj not in G:
            G.add_node(subj, entity_type=t.get("subject_type", "Unknown"))
        if obj not in G:
            G.add_node(obj, entity_type=t.get("object_type", "Unknown"))

        # Add directed edge with relation label
        G.add_edge(subj, obj, label=rel)

    return G


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def render_graph(G: nx.MultiDiGraph, output_path: Path) -> None:
    """
    Render the graph with matplotlib using spring_layout.

    Nodes are coloured by entity type; edge labels show relation names.
    For readability with large graphs, only a subset of edge labels are drawn.
    """
    fig, ax = plt.subplots(figsize=(24, 18))
    ax.set_title("Tech Company Knowledge Graph", fontsize=18, fontweight="bold", pad=20)
    ax.axis("off")

    # Layout
    seed = 42
    pos = nx.spring_layout(G, seed=seed, k=2.5)

    # Node colours by type
    node_colors = [
        TYPE_COLORS.get(G.nodes[n].get("entity_type", "Unknown"), "#95A5A6")
        for n in G.nodes()
    ]

    # Draw nodes + labels
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=800,
        alpha=0.9,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7,
        font_weight="bold",
        font_color="white",
    )

    # Draw edges (use DiGraph view to avoid duplicate arcs for MultiDiGraph)
    simple_edges = list(G.edges())
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=simple_edges,
        edge_color="#555555",
        arrows=True,
        arrowsize=15,
        width=1.2,
        connectionstyle="arc3,rad=0.1",
        alpha=0.7,
        min_source_margin=20,
        min_target_margin=20,
    )

    # Edge labels (first relation per unique (u,v) pair)
    edge_label_dict: dict[tuple[str, str], str] = {}
    for u, v, data in G.edges(data=True):
        if (u, v) not in edge_label_dict:
            edge_label_dict[(u, v)] = data.get("label", "")

    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=edge_label_dict,
        font_size=5,
        font_color="#222222",
        label_pos=0.35,
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5),
    )

    # Legend
    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=color,
            markersize=10,
            label=etype,
        )
        for etype, color in TYPE_COLORS.items()
        if etype != "Unknown"
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Graph rendered and saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_graph_networkx() -> nx.MultiDiGraph:
    """Full pipeline: load triples → build graph → save graphml + png."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRIPLES_PATH.exists():
        logger.error("triples.json not found at %s. Run indexing first.", TRIPLES_PATH)
        sys.exit(1)

    triples = load_triples(TRIPLES_PATH)
    logger.info("Loaded %d triples.", len(triples))

    G = build_graph(triples)
    logger.info(
        "Graph: %d nodes, %d edges.",
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    # Save GraphML
    nx.write_graphml(G, str(GRAPHML_PATH))
    logger.info("Saved GraphML to %s", GRAPHML_PATH)

    # Render PNG
    render_graph(G, PNG_PATH)

    return G


if __name__ == "__main__":
    G = run_graph_networkx()
    print(f"\nGraph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"GraphML saved to: {GRAPHML_PATH}")
    print(f"Screenshot saved to: {PNG_PATH}")
    sys.exit(0)
