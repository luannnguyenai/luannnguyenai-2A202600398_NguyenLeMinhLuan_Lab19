# Cost Analysis Report

## Pricing Assumptions
- `gpt-4o-mini` input:  **$0.150** per 1M tokens
- `gpt-4o-mini` output: **$0.600** per 1M tokens
- `text-embedding-3-small`: **$0.020** per 1M tokens

## Token Usage by Stage

| Stage | Prompt Tokens | Completion Tokens | Total |
|-------|--------------|-------------------|-------|
| evaluation | 9,535 | 1,090 | 10,625 |
| flat_rag_query | 11,701 | 228 | 11,929 |
| graph_rag_entity_extraction | 1,687 | 137 | 1,824 |
| graph_rag_query | 16,341 | 119 | 16,460 |
| indexing | 14,075 | 7,604 | 21,679 |
| **GRAND TOTAL** | — | — | **62,517** |

## Estimated Cost (USD)

| Component | USD |
|-----------|-----|
| Indexing (LLM) | $0.0067 |
| Embedding (indexing) | $0.0000 |
| **Total** | **$0.0135** |

## Benchmark Results Summary

| Metric | Flat RAG | GraphRAG |
|--------|----------|----------|
| Questions | 20 | 20 |
| Correct | 19 (95.0%) | 16 (80.0%) |
| Avg tokens / question | 583 | 823 |
| Avg latency / question | 1404 ms | 1750 ms |

## Average Latency by Stage

| Stage | Avg Latency (ms) |
|-------|------------------|
| evaluation | 1098 |
| flat_rag_query | 699 |
| graph_rag_entity_extraction | 862 |
| graph_rag_query | 887 |
| indexing | 7922 |

---
*Generated automatically by `src/evaluate.py`*