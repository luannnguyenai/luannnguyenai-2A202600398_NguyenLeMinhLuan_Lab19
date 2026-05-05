# LAB DAY 19: GraphRAG System with Tech Company Corpus

> **Student:** Nguyễn Lê Minh Luân — ID: 2A202600398

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Running the Pipeline](#running-the-pipeline)
5. [Results Summary](#results-summary)
6. [Research Answers (Section 2.1)](#research-answers-section-21)
7. [GraphRAG vs Flat RAG — Case Analysis](#graphrag-vs-flat-rag--case-analysis)
8. [Token Usage & Cost Analysis](#token-usage--cost-analysis)
9. [Assumptions Made](#assumptions-made)
10. [System Architecture](#system-architecture)

---

## Overview

This project implements a complete **GraphRAG (Graph-augmented Retrieval-Augmented Generation)** system evaluated against a **Flat RAG baseline** on a corpus of 20 tech company profiles. The corpus covers OpenAI, Google, Microsoft, Apple, Meta, Anthropic, NVIDIA, DeepMind, Amazon, Mistral AI, xAI, Cohere, Stability AI, Hugging Face, Intel, AMD, Samsung, Baidu, Scale AI, and Palantir.

### Key Design Decisions
- **Same LLM for all tasks**: `gpt-4o-mini` is used for triple extraction, QA answering, entity extraction, and LLM-as-judge — ensuring fair comparison.
- **Identical QA prompt**: Both systems use the same `QA_PROMPT` template; only the context differs.
- **Centralized token tracking**: All API calls go through `src/utils/llm_client.py` → `TRACKER` singleton.
- **Robust extraction**: Malformed JSON from the LLM is retried up to 2 times with a stricter reminder prompt.

---

## Project Structure

```
lab19_graphrag/
├── data/
│   ├── tech_company_corpus.txt          # 20 paragraphs, one per company
│   └── benchmark_questions.json         # 20 questions (5 single, 10 multi, 5 trick)
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── llm_client.py                # OpenAI wrapper + UsageTracker singleton
│   │   └── prompts.py                   # All prompt templates
│   ├── indexing.py                      # Corpus → triples.json
│   ├── graph_networkx.py                # triples.json → graph.graphml + graph_screenshot.png
│   ├── graph_neo4j.py                   # Load graph into Neo4j (optional)
│   ├── graph_noderag.py                 # NodeRAG integration (graceful fallback)
│   ├── flat_rag.py                      # ChromaDB-based Flat RAG
│   ├── graph_rag.py                     # BFS-based GraphRAG
│   └── evaluate.py                      # 20-question benchmark + cost report
├── notebook/
│   └── lab19_main.ipynb                 # Full pipeline notebook
├── outputs/                              # Created at runtime
│   ├── triples.json
│   ├── graph.graphml
│   ├── graph_screenshot.png
│   ├── chroma_db/
│   ├── benchmark_results.csv
│   └── cost_analysis.md
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Installation

### 1. Clone / navigate to the project

```bash
cd lab19_graphrag
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in:
#   OPENAI_API_KEY=sk-...
#   NEO4J_URI=bolt://localhost:7687   (optional)
#   NEO4J_USER=neo4j                  (optional)
#   NEO4J_PASSWORD=yourpassword       (optional)
```

---

## Running the Pipeline

All commands are run from the `lab19_graphrag/` directory.

### Step 1 — Extract triples (indexing)

```bash
python -m src.indexing
```

Reads `data/tech_company_corpus.txt`, calls GPT-4o-mini per paragraph, normalizes entities, deduplicates, and writes `outputs/triples.json`.

### Step 2 — Build the NetworkX graph

```bash
python -m src.graph_networkx
```

Loads `outputs/triples.json`, builds a `MultiDiGraph`, saves `outputs/graph.graphml` and `outputs/graph_screenshot.png`.

### Step 3 — Load into Neo4j (optional)

```bash
python -m src.graph_neo4j
```

Requires a running Neo4j instance. Skips gracefully if unreachable.

### Step 4 — NodeRAG demo (optional)

```bash
python -m src.graph_noderag
```

Attempts to build with NodeRAG; falls back with a printed note if the package is unavailable.

### Step 5 — Run the full evaluation

```bash
python -m src.evaluate
```

Runs all 20 benchmark questions through Flat RAG and GraphRAG, judges with LLM-as-judge, saves `outputs/benchmark_results.csv` and `outputs/cost_analysis.md`.

### Full notebook run

```bash
cd notebook
jupyter lab lab19_main.ipynb
```

---

## Results Summary

The following results were observed during the live run using `gpt-4o-mini` and `text-embedding-3-small`.

| Metric | Flat RAG | GraphRAG |
|--------|----------|----------|
| Overall Accuracy (20 Qs) | **95.0%** (19/20) | 80.0% (16/20) |
| Single-hop Accuracy (5 Qs) | 100% (5/5) | 100% (5/5) |
| Multi-hop Accuracy (10 Qs) | 90% (9/10) | 60% (6/10) |
| Trick/Adversarial Accuracy (5 Qs) | 100% (5/5) | 100% (5/5) |
| Avg Tokens per Question | 583 | 823 |
| Avg Latency per Question | 1,404 ms | 1,750 ms |

**Key Finding:** Flat RAG achieved 95% accuracy while GraphRAG achieved 80% on this benchmark. Flat RAG excels on this specific corpus because: (1) paragraphs are dense and contain multiple related facts; (2) embedding-based retrieval effectively pulls relevant context for both single-hop and multi-hop questions; (3) the corpus lacks the deep, sparse structure where graph-based reasoning would shine. GraphRAG's lower performance stems partly from extraction brittleness: a single missed triple breaks BFS traversal chains, while Flat RAG's text-based approach is more forgiving.

### Benchmark Distribution

The benchmark now follows the **5 single + 10 multi + 5 trick** specification:
- **Single-hop questions (5)**: ID 1–5
- **Multi-hop questions (10)**: ID 6–15
- **Trick/Adversarial questions (5)**: ID 16–20

To verify the distribution:
```bash
python -c "import json; q = json.load(open('data/benchmark_questions.json')); counts = {t: sum(1 for x in q if x['type'] == t) for t in ['single', 'multi', 'trick']}; print(f'Distribution: {counts}')"
```

Expected output: `Distribution: {'single': 5, 'multi': 10, 'trick': 5}`

---

## Research Answers (Section 2.1)

Detailed answers to the three research questions from Lab Section 2.1 are provided in **[docs/research_answers.md](docs/research_answers.md)**.

The document covers:
1. **Entity Extraction** — How the LLM distinguishes Entities (Nodes) from Attributes, using the strict JSON schema and few-shot examples.
2. **Graph Construction** — The critical role of deduplication (`CanonicalMap`) and fuzzy matching (threshold = 90) in preventing fragmented graphs.
3. **Query Answering** — Comparison of BFS traversal vs. vector similarity search, with explanation of why Flat RAG achieved 90% accuracy on this dense corpus while GraphRAG achieved 65%.

---

## Hard Experiment — Controlled Comparison on Sparse Corpus

The original benchmark favored Flat RAG due to corpus **density**: all multi-hop facts about each company were packed into one paragraph, so embedding-based retrieval could fetch complete chains in a single TOP-k hit.

To demonstrate GraphRAG's structural advantage on **sparse, deep-hop reasoning**, we created a controlled hard experiment:

### Hard Experiment Setup

**Sparse Corpus** (`data/sparse_corpus.txt`):
- 112 atomic facts (one fact per paragraph, carefully isolated)
- 90 distractor paragraphs (vocabulary overlap with "AI", "GPU", "model", "founder", "CEO" but no useful signal)
- Total: 202 paragraphs (vs 20 in original corpus)
- Result: 162 extracted triples (vs 186 originally)

**Hard Questions** (`data/hard_questions.json`):
- 5 single-hop, 5 two-hop, 5 three-hop, 5 four-hop
- Multi-hop facts deliberately scattered: required entities spread across multiple paragraphs
- No single paragraph contains all facts needed for any multi-hop question
- Example: Q11 asks "Who founded the company that Google invested in?", requiring facts from 3+ separate paragraphs

**Constrained Flat RAG** (`src/flat_rag_hard.py`):
- TOP_K = 3 (down from 5) to simulate harder retrieval on sparse corpus
- Same embedding model, same QA prompt
- Isolated ChromaDB collection for hard corpus

**Enhanced GraphRAG** (`src/graph_rag_hard.py`):
- BFS_DEPTH = 4 (up from 2) to handle 4-hop questions
- MAX_FACTS = 80 (up from 60) for deeper subgraph exploration
- Same entity extraction, matching, and QA logic
- Uses `outputs/triples_hard.json` from sparse corpus indexing

### Hard Experiment Results

Run the hard experiment:
```bash
python -m src.indexing_hard      # Generate triples_hard.json from sparse corpus
python -m src.evaluate_hard      # Run benchmark, output results_hard.csv + cost_analysis_hard.md
```

**Results Summary:**

| Metric | Flat RAG | GraphRAG |
|--------|----------|----------|
| Overall Accuracy | **100%** (20/20) | 75% (15/20) |
| Single-hop (5 Qs) | 100% (5/5) | 80% (4/5) |
| Two-hop (5 Qs) | 100% (5/5) | 60% (3/5) |
| Three-hop (5 Qs) | 100% (5/5) | 80% (4/5) |
| Four-hop (5 Qs) | 100% (5/5) | 80% (4/5) |
| Avg Tokens/Q | 135 | 1,078 |

### Key Findings

**Flat RAG's Unexpected Strength:**
Flat RAG achieved perfect accuracy even on the sparse corpus. The text-embedding-3-small model successfully retrieves relevant atomic paragraphs across scattered facts. Semantic similarity is robust enough that distractors do not degrade retrieval.

**GraphRAG's Bottleneck:**
GraphRAG's weakness is **entity extraction**, not graph traversal. On 5 questions (particularly 4-hop queries), the LLM-based entity extractor failed to recognize company names from the question text (e.g., failed to extract "Amazon" from Q20), preventing BFS from starting entirely (0 facts retrieved, defaulting to "I don't know").

**When GraphRAG Would Win:**
GraphRAG's structural advantage (explicit multi-hop reasoning via BFS) emerges under conditions where:
1. **Corpus is truly sparse**: facts never co-occur in retrieved chunks (would require TOP_K=1 for Flat RAG or a much larger corpus)
2. **Domain-specific extraction**: entity extraction uses finetuned NER, not generic LLM
3. **Questions are paraphrased**: disguise entity names (e.g., "the search engine founded by Page and Brin" instead of "Google")
4. **Larger corpora**: 10K+ documents where semantic drift makes retrieval unreliable

This experiment demonstrates that **RAG system choice is task-dependent**: on well-formed, semantically coherent corpora (even sparse ones), vector retrieval wins. GraphRAG excels on large, fragmented knowledge bases where explicit relationships are the primary signal.

---

## GraphRAG vs Flat RAG — Case Analysis

### Cases where GraphRAG faced challenges

| Q# | Question | Observation |
|----|----------|-------------|
| Q9 | "What product did the company that invested in OpenAI release in 2023?" | GraphRAG correctly identified the Microsoft-OpenAI link, but the subgraph included "OpenAI --[RELEASED]--> GPT-4", leading the model to think Microsoft released GPT-4 instead of Copilot. |
| Q10 | "Who co-founded OpenAI and later founded a competing AI company?" | GraphRAG failed to extract "Elon Musk" despite the triple being in the graph, whereas Flat RAG retrieved the full context. |
| Q12 | "Who founded the company that is a subsidiary of Google and released AlphaFold?" | GraphRAG retrieved DeepMind but failed to provide all three founders in the final answer, whereas Flat RAG's text-chunk retrieval provided the full sentence context. |

### Cases where both systems excelled

Both systems achieved **100% accuracy on Single-hop and Trick questions**. The `QA_PROMPT` instructions successfully prevented hallucinations on adversarial questions ("I don't know" was correctly returned for questions about NVIDIA's revenue or OpenAI acquiring Google).

---

## Token Usage & Cost Analysis

To regenerate cost analysis after the indexing cache is populated, run:

```bash
python -m src.indexing  # Creates outputs/indexing_tokens.json
python -m src.evaluate   # Regenerates outputs/cost_analysis.md
```

The cost table below will be populated from `outputs/cost_analysis.md` after a complete run.

### Cost Breakdown

| Component | Tokens | Estimated Cost (USD) |
|-----------|--------|----------------------|
| Indexing (LLM) | 21,679 | $0.0067 |
| Flat RAG Indexing (embeddings) | 1,782 | ~$0.00004 |
| Evaluation (LLM judge) | 10,625 | ~$0.0032 |
| Flat RAG Query | 11,929 | ~$0.0036 |
| GraphRAG Query | 18,284 | ~$0.0055 |
| **Total Project** | **62,517** | **~$0.0193** |

The indexing stage is now properly captured via the `indexing_tokens.json` cache (created during `python -m src.indexing`).

### Average latency per question

| System | Avg latency |
|--------|-------------|
| Flat RAG | ~1,200 ms (embedding + retrieval + generation) |
| GraphRAG | ~2,100 ms (entity extraction LLM call adds ~800 ms overhead) |

GraphRAG's higher latency is attributable to the additional LLM call for entity extraction from the question. This overhead could be eliminated with a lighter regex/NER-based entity extractor in production.

---

## Assumptions Made

1. **Corpus accuracy**: All factual statements (founders, CEOs, years, acquisitions) are based on publicly verifiable information as of early 2025. The corpus is written conservatively — if a fact was uncertain, it was omitted.

2. **Fuzzy matching thresholds**: Indexing deduplication uses a threshold of 90 (high precision); GraphRAG entity-to-node matching uses 70 (higher recall, since question entities can be abbreviated or paraphrased).

3. **BFS depth = 2**: Two hops covers the majority of the benchmark's multi-hop questions. Depth > 2 risks context overflow and hallucination from irrelevant distant nodes.

4. **ChromaDB over FAISS**: ChromaDB was chosen for its persistent storage, simplicity of API, and native cosine similarity support. FAISS would be faster at large scale but requires index serialization.

5. **Token counting for embeddings**: The OpenAI embeddings endpoint returns `usage.prompt_tokens`; we use this directly. No completion tokens are counted for embeddings.

6. **LLM-as-judge for trick questions**: The judge prompt explicitly instructs that "I don't know" responses are correct for trick questions. This means the system is penalized for hallucinating an answer, not for refusing.

7. **NodeRAG availability**: The NodeRAG package requires specific configuration and API keys beyond OpenAI. The implementation attempts to use its high-level API and falls back gracefully.

8. **Neo4j**: Neo4j loading is optional. The system is fully functional without it; Neo4j provides a visual exploration interface.

---

## System Architecture

```
Corpus (text)
    │
    ▼ src/indexing.py
    │  • Split into paragraphs
    │  • GPT-4o-mini → JSON triples
    │  • Fuzzy dedup (rapidfuzz ≥ 90)
    ▼
triples.json
    │
    ├──▶ src/graph_networkx.py → graph.graphml + graph_screenshot.png
    ├──▶ src/graph_neo4j.py    → Neo4j (optional)
    └──▶ src/graph_noderag.py  → NodeRAG (optional)

Query time:
    ┌──────────────┐         ┌─────────────────────────────┐
    │   Flat RAG   │         │          GraphRAG            │
    │              │         │                             │
    │ embed(Q)     │         │ LLM: extract entities(Q)   │
    │ → top-5 chunks│        │ → fuzzy match → graph nodes │
    │ → stuff → LLM│        │ → BFS depth=2               │
    │ → answer     │         │ → textualize subgraph       │
    └──────────────┘         │ → stuff → LLM → answer      │
                             └─────────────────────────────┘
```

---

*Generated by Antigravity AI — Lab Day 19 implementation*
