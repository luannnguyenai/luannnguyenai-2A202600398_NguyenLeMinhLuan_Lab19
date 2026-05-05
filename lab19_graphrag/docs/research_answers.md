# Research Answers — Lab 19 GraphRAG (Section 2.1)

## Q1: Entity Extraction — How can the LLM distinguish an Entity (Node) from an Attribute?

### Definition
An **Entity** (Node) is a discrete, independent object that can participate in relationships: `Person`, `Organization`, `Product`, `Location`, `Year`, `Event`. An **Attribute** is a descriptor or scalar property that belongs to an entity, such as a name variant, description, or count.

### Why Founding Years are Year Entities (not attributes)
In this system, years like "2021" (Anthropic's founding year) are modeled as distinct `Year` entities rather than string attributes on organizations. This design choice:
- Enables multi-hop reasoning: "Anthropic → FOUNDED_IN → 2021 → [Year] → other orgs founded in 2021"
- Keeps the graph uniform: all relationships point from one entity to another, no mixed entity-scalar semantics
- Aligns with the strict JSON schema in `src/utils/prompts.py`, which requires every triple to have `subject_type`, `subject`, `relation`, `object`, `object_type`

### Role of the Strict JSON Schema
The schema in `TRIPLE_EXTRACTION_SYSTEM` (src/utils/prompts.py:16–47) enforces:
- **Allowed Entity Types**: exactly `{Person, Organization, Product, Location, Year, Event}`
- **Allowed Relations**: exactly 10 types (FOUNDED_BY, FOUNDED_IN, CEO_OF, ACQUIRED, RELEASED, HEADQUARTERED_IN, SUBSIDIARY_OF, INVESTED_IN, COLLABORATES_WITH, COMPETES_WITH)
- **Strict Output Format**: only valid JSON with the 5-field triple structure

This schema acts as a hard constraint: if the LLM tries to assign an entity an unknown type (e.g., "Funding Amount" or "Publication") or use a relation not in the list, `_filter_triple()` in `src/indexing.py:86–94` will reject it.

### Few-Shot Examples Guide Extraction
`TRIPLE_EXTRACTION_FEW_SHOT` (src/utils/prompts.py:49–76) provides two worked examples:
1. **Example 1**: "Apple was founded in 1976 by Steve Jobs. Tim Cook serves as CEO of Apple."  
   Output: `(Apple, FOUNDED_IN, 1976)`, `(Apple, FOUNDED_BY, Steve Jobs)`, `(Tim Cook, CEO_OF, Apple)`  
   → Shows that "1976" is a Year entity, not a string attribute.

2. **Example 2**: "Google acquired DeepMind in 2014. DeepMind is headquartered in London."  
   Output: `(Google, ACQUIRED, DeepMind)`, `(DeepMind, SUBSIDIARY_OF, Google)`, `(DeepMind, HEADQUARTERED_IN, London)`  
   → Shows Location as an entity (London), not an attribute.

These examples teach the LLM the distinction: founding **dates become Year nodes**; founding **locations become Location nodes**. The corpus confirms this: "Anthropic was founded in 2021" yields `(Anthropic, FOUNDED_IN, 2021)` where 2021 is a Year entity.

---

## Q2: Graph Construction — Why is Deduplication Critical?

### The Problem Without Deduplication
Without deduplication, variant surface forms of the same entity (e.g., "OpenAI", "Open AI", "OpenAI Inc.") become separate nodes in the graph. This catastrophically damages BFS multi-hop accuracy:
- **Single traversal step**: BFS looks for neighbors of "OpenAI" and finds only edges incident to that exact node, missing facts connected via "Open AI" or "OpenAI Inc."
- **Multi-hop queries**: For "Who invested in OpenAI?" (2 hops: find org → lookup investors), if "OpenAI" and "Open AI Inc." are different nodes, the graph may return incomplete or wrong results.
- **Metrics collapse**: recall drops sharply because the graph is fragmented.

### Deduplication Strategy in CanonicalMap (src/indexing.py:101–140)
The `CanonicalMap` class implements a **fuzzy canonicalization** approach:

1. **Normalization**: `_normalize(name)` (src/indexing.py:58–62) lowercases and strips all punctuation.  
   Example: "OpenAI Inc.", "open-ai", "OPENAI" → all become "openai"

2. **Fuzzy Matching**: For each new entity name, `process.extractOne(norm, self._canonicals, scorer=fuzz.ratio)` compares it against all existing canonical forms using Levenshtein ratio.

3. **Threshold Decision**:
   - If fuzzy similarity **≥ 90**: the new variant is mapped to the existing canonical.  
   - If fuzzy similarity **< 90**: the new variant becomes a new canonical entity.

### Why Threshold 90 (vs 70 in graph_rag.py)
The `FUZZY_THRESHOLD = 90` in `src/indexing.py:38` is conservative, preventing over-merging:
- **90 is strict**: "OpenAI" and "Open A I" (extra space) are 95 similar, so they merge. But "OpenAI" and "OpenAI Research" (substring) are ~72 similar, so they stay separate (correctly, since they may be distinct entities).
- **70 would be loose**: it risks collapsing "OpenAI" and "Open Source Initiative" (both start with "Open"), creating false edges.
- The corpus is dense (20 companies, many variants), so 90 balances precision and recall.

### Concrete Example from the Corpus
In the corpus:
- OpenAI appears as "OpenAI" in multiple paragraphs.
- The LLM might extract "Open AI Inc." in one paragraph, "OpenAI" in another.
- **Without deduplication**: two separate "OpenAI" nodes, each with partial edges, graph fragmented.
- **With deduplication (threshold 90)**:
  - "Open AI Inc." normalized → "open ai inc" (89–92 similar to "openai" depending on length) → mapped to canonical "OpenAI"
  - Single "OpenAI" node with all edges (CEO_OF Sam Altman, INVESTED_IN by Microsoft, COMPETES_WITH by Google, etc.)
  - BFS queries for "Who is the CEO of OpenAI?" now traverse the complete subgraph.

The cost: if "OpenAI Foundation" (a hypothetical distinct org) is mentioned, it might map to the main "OpenAI" node (false merge). But on this corpus, precision > recall priority.

---

## Q3: Query Answering — BFS Traversal vs Vector Similarity Search

### Retrieval Unit
- **Vector Similarity (Flat RAG)**: retrieves **text chunks** (paragraphs from the corpus).  
  Each chunk is embedded with `text-embedding-3-small`, stored in ChromaDB, retrieved via cosine similarity.
- **BFS Traversal (GraphRAG)**: retrieves **nodes and edges** of the knowledge graph.  
  Given a query, entities are extracted, then BFS expands neighbors up to depth K, collecting connected facts.

### Multi-Hop Reasoning
- **Flat RAG**: queries a single semantic space. A 3-hop fact like "Sam Altman → OpenAI → Microsoft → Satya Nadella" must co-occur or be nearby in a paragraph.  
  If the corpus scatters these facts across 3 paragraphs, retrieval may miss one, breaking the chain.
- **GraphRAG**: explicitly traverses edges. BFS guarantees finding all nodes at distance K.  
  Multi-hop questions like "Who founded the company that invested in OpenAI?" work by: extract "OpenAI" → find edges (INVESTED_IN) → retrieve investor → extract investor name.  
  Reaches correct answer as long as edges exist in the graph.

### Explainability
- **Flat RAG**: opaque. The LLM receives chunks, generates an answer. User cannot trace *which facts* led to the answer.
- **GraphRAG**: transparent. BFS path is traceable: "Found Sam Altman via edge (OpenAI, CEO_OF, Sam Altman)." Every fact has an edge source.

### Dependency on Quality
- **Flat RAG**: depends on **embedding quality**. If a chunk is poorly embedded (domain-specific jargon, ambiguous language), retrieval fails silently.
- **GraphRAG**: depends on **extraction quality**. If the LLM misses a triple (e.g., fails to extract "Satya Nadella CEO_OF Microsoft"), BFS cannot traverse it.

### Hallucination Risk
- **Flat RAG**: the LLM generates from retrieved chunks, but can hallucinate beyond them.  
  Risk: given retrieved chunk about "OpenAI founded by Altman", LLM might add "Altman also founded SpaceX" (plausible but not in corpus).
- **GraphRAG**: the LLM generates only from reachable graph nodes.  
  Risk lower, but not eliminated: LLM can still misinterpret edge semantics (e.g., confuse INVESTED_IN with FOUNDED_BY).

### Latency
- **Flat RAG**: embedding + vector search is fast (~50 ms retrieval). But embedding all corpus chunks upfront costs time.
- **GraphRAG**: BFS + LLM extraction per query costs more latency per query, especially on large graphs.

### Why Flat RAG Won on This Corpus (90% vs 65% accuracy)
**Flat RAG 90% correct vs GraphRAG 65% correct** on the benchmark. Why?

1. **Corpus density**: The corpus is 20 paragraphs covering 20 companies. Multi-hop facts frequently co-occur within the same paragraph or nearby paragraphs.  
   Example: "Microsoft CEO is Satya Nadella. Microsoft invested in OpenAI." → same paragraph → Flat RAG retrieves both facts, answers the multi-hop query correctly.

2. **Extraction brittleness**: Triple extraction is finicky. Malformed JSON is retried up to 2 times, but some extractions are noisy or incomplete. A single missed triple breaks a BFS chain.  
   Flat RAG side-steps this: it retrieves text directly, avoiding the extraction bottleneck.

3. **Graph coverage**: GraphRAG's accuracy depends on the graph being complete. If the corpus lacks a key fact (e.g., "Who founded Anthropic's parent?"), GraphRAG cannot answer it (correctly returns "not in corpus"). Flat RAG might still hallucinate a plausible answer, accidentally scoring points.

### When GraphRAG Would Dominate in Production
In a **sparse, large-scale knowledge base** (e.g., Wikipedia, PubMed), GraphRAG excels:

1. **Sparse corpus**: Facts are distributed across many documents. Multi-hop queries require *explicit reasoning* to connect facts across documents.  
   Example: "What cancer drug was developed by a researcher who studied at MIT?" requires linking drug → researcher → university.  
   Flat RAG would need a retrieved chunk containing all three facts, unlikely in a sparse corpus.

2. **Deep multi-hop**: Queries require 4+ hops (A → B → C → D → E). BFS handles this naturally; embedding-based retrieval becomes unreliable.

3. **Explainability is critical**: In a medical setting, you must trace the source of each claim. GraphRAG provides provenance; Flat RAG does not.

4. **Extraction quality is high**: With a domain-specific extraction system (finetuned LLM, schema validation), triple quality is high. BFS can then be trusted.

In this corpus, density + small scale favors Flat RAG's simplicity. In production knowledge bases, GraphRAG's explicit reasoning wins.

---

## References

- **src/utils/prompts.py** — TRIPLE_EXTRACTION_SYSTEM, TRIPLE_EXTRACTION_FEW_SHOT, TRIPLE_EXTRACTION_RETRY_REMINDER (entity types, allowed relations, few-shot examples).
- **src/indexing.py** — CanonicalMap (fuzzy deduplication with threshold), run_indexing() (pipeline).
- **src/graph_rag.py** — BFS traversal logic for multi-hop querying.
- **src/flat_rag.py** — ChromaDB vector search for chunk retrieval.
- **outputs/cost_analysis.md** — Benchmark results showing Flat RAG 90% vs GraphRAG 65% on this corpus.
- **data/benchmark_questions.json** — 20 questions (5 single + 10 multi + 5 trick).
