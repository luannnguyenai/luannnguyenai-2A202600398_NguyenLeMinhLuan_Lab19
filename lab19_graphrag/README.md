# LAB DAY 19: GraphRAG System with Tech Company Corpus

> **Student:** Nguyễn Lê Minh Luân — ID: 2A202600398

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Running the Pipeline](#running-the-pipeline)
5. [Results Summary](#results-summary)
6. [GraphRAG vs Flat RAG — Case Analysis](#graphrag-vs-flat-rag--case-analysis)
7. [Token Usage & Cost Analysis](#token-usage--cost-analysis)
8. [Assumptions Made](#assumptions-made)
9. [System Architecture](#system-architecture)

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
| Overall Accuracy (20 Qs) | **90.0%** (18/20) | 65.0% (13/20) |
| Single-hop Accuracy (5 Qs) | 100% (5/5) | 100% (5/5) |
| Multi-hop Accuracy (11 Qs) | 82% (9/11) | 36% (4/11) |
| Trick/Adversarial Accuracy (4 Qs) | 100% (4/4) | 100% (4/4) |
| Avg Tokens per Question | 586 | 949 |
| Avg Latency per Question | 1,466 ms | 2,092 ms |

**Key Finding:** Surprisingly, Flat RAG performed better on the multi-hop questions in this specific test. This was largely due to the high density of the corpus paragraphs; since related facts were often in the same or adjacent paragraphs, the embedding-based retrieval was sufficient to pull the required context. GraphRAG, while retrieving the correct nodes, sometimes provided a "flat" list of facts that led the LLM to misattribute relationships (e.g., attributing a subsidiary's product to the parent company).

---

## GraphRAG vs Flat RAG — Case Analysis

### Cases where GraphRAG faced challenges

| Q# | Question | Observation |
|----|----------|-------------|
| Q9 | "What product did the company that invested in OpenAI release in 2023?" | GraphRAG correctly identified the Microsoft-OpenAI link, but the subgraph included "OpenAI --[RELEASED]--> GPT-4", leading the model to think Microsoft released GPT-4 instead of Copilot. |
| Q12 | "Who founded the company that is a subsidiary of Google and released AlphaFold?" | GraphRAG retrieved DeepMind but failed to provide all three founders in the final answer, whereas Flat RAG's text-chunk retrieval provided the full sentence context. |
| Q20 | "Which company that competed with NVIDIA released a GPU in 2023..." | This complex 3-hop query confused the GraphRAG traversal logic, leading to a "refusal" or "I don't know" when nodes were missed. |

### Cases where both systems excelled

Both systems achieved **100% accuracy on Single-hop and Trick questions**. The `QA_PROMPT` instructions successfully prevented hallucinations on adversarial questions ("I don't know" was correctly returned for questions about NVIDIA's revenue or OpenAI acquiring Google).

---

## Token Usage & Cost Analysis

Based on the `outputs/cost_analysis.md` generated during the run:

| Component | Tokens | Estimated Cost (USD) |
|-----------|--------|----------------------|
| Indexing (LLM Triples) | ~21,730 | ~$0.0067 |
| Indexing (Embeddings) | 1,782 | ~$0.00003 |
| Benchmarking (20 Qs) | ~45,477 | ~$0.0076 |
| **Total Project Run** | **~68,989** | **~$0.0143** |

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
