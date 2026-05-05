# Cost Analysis Report

## Pricing Assumptions
- `gpt-4o-mini` input:  **$0.150** per 1M tokens
- `gpt-4o-mini` output: **$0.600** per 1M tokens
- `text-embedding-3-small`: **$0.020** per 1M tokens

## Token Usage by Stage

| Stage | Prompt Tokens | Completion Tokens | Total |
|-------|--------------|-------------------|-------|
| evaluation | 9,749 | 1,113 | 10,862 |
| flat_rag_indexing | 1,782 | 0 | 1,782 |
| flat_rag_query | 11,732 | 276 | 12,008 |
| graph_rag_entity_extraction | 1,701 | 146 | 1,847 |
| graph_rag_query | 18,752 | 226 | 18,978 |
| **GRAND TOTAL** | — | — | **45,477** |

## Estimated Cost (USD)

| Component | USD |
|-----------|-----|
| Indexing (LLM) | $0.0000 |
| Embedding (indexing) | $0.0000 |
| **Total** | **$0.0076** |

## Benchmark Results Summary

| Metric | Flat RAG | GraphRAG |
|--------|----------|----------|
| Questions | 20 | 20 |
| Correct | 18 (90.0%) | 13 (65.0%) |
| Avg tokens / question | 586 | 949 |
| Avg latency / question | 1466 ms | 2092 ms |

## Average Latency by Stage

| Stage | Avg Latency (ms) |
|-------|------------------|
| evaluation | 1270 |
| flat_rag_indexing | 3474 |
| flat_rag_query | 731 |
| graph_rag_entity_extraction | 941 |
| graph_rag_query | 1149 |

---
*Generated automatically by `src/evaluate.py`*