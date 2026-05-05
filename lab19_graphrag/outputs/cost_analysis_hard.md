# Hard Experiment Results

## Accuracy by Hop Count

| Hop Type | Flat RAG | GraphRAG | Gap |
|----------|----------|----------|-----|
| 2-hop    |   100% (5/5) |    60% (3/5) |   -40pp |
| 3-hop    |   100% (5/5) |    80% (4/5) |   -20pp |
| 4-hop    |   100% (5/5) |    80% (4/5) |   -20pp |
| single   |   100% (5/5) |    80% (4/5) |   -20pp |

## Overall Results

| System | Correct | Accuracy | Avg Tokens |
|--------|---------|----------|------------|
| Flat RAG  | 20/20 | 100.0% | 135 |
| GraphRAG  | 15/20 | 75.0% | 1078 |

## Token Usage by Stage

| Stage | Prompt Tokens | Completion Tokens | Total |
|-------|--------------|-------------------|-------|
| evaluation_hard | 9,218 | 1,103 | 10,321 |
| flat_rag_hard_indexing | 2,064 | 0 | 2,064 |
| flat_rag_hard_query | 2,829 | 111 | 2,940 |
| graph_rag_hard_entity_extraction | 1,654 | 96 | 1,750 |
| graph_rag_hard_query | 21,462 | 106 | 21,568 |

## Observations

**Flat RAG vs GraphRAG on Sparse Corpus:**

Flat RAG achieved **perfect 100% accuracy** (20/20) even on the sparse corpus with constrained TOP_K=3.
GraphRAG achieved **75% accuracy** (15/20), struggling on 5 questions despite deeper BFS traversal.

**Key Findings:**

1. **Embedding robustness**: Text-embedding-3-small successfully retrieves relevant paragraphs
   even when facts are scattered across atomic sentences. The embedding space captures
   semantic relationships well enough for sparse retrieval.

2. **Entity extraction brittleness**: GraphRAG's weakness lies in LLM-based entity extraction from
   questions. On several 4-hop questions, the entity extraction LLM failed to recognize
   company names (e.g., "Amazon" in Q20), preventing graph traversal entirely (0 facts retrieved).

3. **Distractor effectiveness**: The 90 distractor paragraphs did not confuse Flat RAG's
   semantic search enough to cause false positives or missed retrievals. The question
   embeddings remained distinct from noise.

4. **BFS depth vs extraction quality**: While GraphRAG's BFS_DEPTH=4 matched question complexity,
   the system's earlier bottleneck (entity extraction) prevented it from reaching the deeper
   nodes needed for complex answers.

**Implications:**

- On well-formed, atomic corpora (even sparse ones), vector retrieval is highly effective.
- GraphRAG's advantage emerges when: (a) entity extraction is reliable (domain-specific NER),
  or (b) the corpus enforces explicit relationships that text similarity cannot capture.
- To make GraphRAG win, we would need either:
  * Much sparser corpus where multi-hop facts never co-occur in retrieved chunks, OR
  * Weaker question-to-chunk semantic alignment (e.g., paraphrase questions to obscure terms), OR
  * Higher TOP_K=1 or 2 to severely limit Flat RAG, OR
  * Implement finetuned entity extraction for the specific domain.
