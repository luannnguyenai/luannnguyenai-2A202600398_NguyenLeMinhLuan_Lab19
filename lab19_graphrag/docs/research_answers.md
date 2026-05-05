# Lab 19: GraphRAG vs Flat RAG Research Findings (Wikipedia Experiment)

This document summarizes the findings from the "Hard" experiment using real-world data scraped from Wikipedia.

## 1. Corpus Statistics (Wikipedia)
- **Source**: 10 Related AI Wikipedia pages (OpenAI, DeepMind, Sam Altman, Demis Hassabis, etc.)
- **Chunks**: 283 paragraphs
- **Graph Nodes**: 704 entities
- **Graph Edges**: 2013 relationships

## 2. Evaluation Results (Iteration C - Final)

| Hops | Flat RAG Accuracy (TOP_K=2) | GraphRAG Accuracy (BFS=4) | Gap (Graph - Flat) |
|------|-----------------------------|---------------------------|--------------------|
| 1-hop| 100.0%                      | 100.0%                    | +0.0%              |
| 2-hop| 80.0%                       | 100.0%                    | **+20.0%**         |
| 3-hop| 100.0%                      | 100.0%                    | +0.0%              |
| 4-hop| 100.0%                      | 100.0%                    | +0.0%              |

**Note**: Flat RAG remains surprisingly resilient even with `TOP_K=2` due to the high reasoning capabilities of `gpt-4o-mini`. However, GraphRAG showed superior structural retrieval in the 2-hop category, capturing linked entities that keyword-based retrieval missed.

## 3. Key Observations

### Why GraphRAG Wins on Multi-Hop
In 2-hop scenarios, GraphRAG successfully traversed connections between entities (e.g., DeepMind -> Google -> Sundar Pichai) even when the source sentences were in completely different parts of the corpus. Flat RAG with small `TOP_K` failed when the embedding of the question didn't strongly align with the intermediate "bridge" sentences.

### Cost & Latency Analysis (Wiki)
- **Indexing Cost**: ~$0.15 (Parallel extraction of 2000+ triples)
- **Query Latency**:
    - Flat RAG: ~800ms
    - GraphRAG: ~1500ms (includes entity extraction + BFS traversal)

## 4. Conclusion
GraphRAG provides a significant advantage in **structural reliability**. While Flat RAG can "guess" correctly if it retrieves enough context, GraphRAG explicitly navigates the knowledge path, making it more robust for complex business logic or scientific research where exact chains of evidence are required.
