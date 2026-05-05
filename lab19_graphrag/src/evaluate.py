"""
Evaluation pipeline: run all 20 benchmark questions through Flat RAG and GraphRAG,
judge correctness with an LLM-as-judge, save CSV results + cost analysis.

Run as:
    python -m src.evaluate

Outputs:
    outputs/benchmark_results.csv
    outputs/cost_analysis.md
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.utils.llm_client import get_client, TRACKER
from src.utils.prompts import build_judge_messages
from src.flat_rag import build_flat_index, query_flat_rag
from src.graph_rag import query_graph_rag

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "benchmark_questions.json"
OUTPUT_DIR = ROOT / "outputs"
CSV_PATH = OUTPUT_DIR / "benchmark_results.csv"
COST_PATH = OUTPUT_DIR / "cost_analysis.md"

# OpenAI pricing (USD per 1M tokens) — gpt-4o-mini as of 2025
GPT4O_MINI_INPUT_PRICE_PER_1M = 0.150
GPT4O_MINI_OUTPUT_PRICE_PER_1M = 0.600
EMBED_SMALL_PRICE_PER_1M = 0.020

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

def judge_answer(
    question: str,
    q_type: str,
    ground_truth: str,
    system_answer: str,
) -> dict[str, Any]:
    """
    Use the LLM to judge whether *system_answer* is correct relative to *ground_truth*.

    Returns dict: {correct: bool, reason: str}
    """
    llm = get_client()
    messages = build_judge_messages(question, q_type, ground_truth, system_answer)
    raw = llm.chat(messages, stage="evaluation", temperature=0.0, max_tokens=300)
    try:
        import re
        cleaned = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        data = json.loads(cleaned)
        return {
            "correct": bool(data.get("correct", False)),
            "reason": data.get("reason", ""),
        }
    except (json.JSONDecodeError, TypeError):
        logger.warning("Judge returned non-JSON: %r", raw)
        return {"correct": False, "reason": f"Parse error: {raw[:100]}"}


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation() -> list[dict[str, Any]]:
    """
    Run all benchmark questions through both systems and save results.

    Returns list of result dicts.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load questions
    with QUESTIONS_PATH.open(encoding="utf-8") as fh:
        questions = json.load(fh)
    logger.info("Loaded %d benchmark questions.", len(questions))

    # Pre-build flat RAG index once
    logger.info("Building Flat RAG index...")
    collection = build_flat_index()

    results: list[dict[str, Any]] = []

    for q in questions:
        qid = q["id"]
        question = q["question"]
        q_type = q["type"]
        ground_truth = q["ground_truth"]

        logger.info("=" * 60)
        logger.info("Q%d [%s]: %s", qid, q_type, question)

        # --- Flat RAG ---
        try:
            flat_result = query_flat_rag(question, collection=collection)
            flat_answer = flat_result["answer"]
            flat_tokens = flat_result["tokens"]
            flat_latency = flat_result["latency_ms"]
        except Exception as exc:
            logger.error("Flat RAG failed on Q%d: %s", qid, exc)
            flat_answer = "ERROR"
            flat_tokens = 0
            flat_latency = 0.0

        # --- GraphRAG ---
        try:
            graph_result = query_graph_rag(question)
            graph_answer = graph_result["answer"]
            graph_tokens = graph_result["tokens"]
            graph_latency = graph_result["latency_ms"]
        except Exception as exc:
            logger.error("GraphRAG failed on Q%d: %s", qid, exc)
            graph_answer = "ERROR"
            graph_tokens = 0
            graph_latency = 0.0

        logger.info("Flat answer: %s", flat_answer[:120])
        logger.info("Graph answer: %s", graph_answer[:120])

        # --- Judge ---
        flat_judge = judge_answer(question, q_type, ground_truth, flat_answer)
        graph_judge = judge_answer(question, q_type, ground_truth, graph_answer)

        logger.info(
            "Flat correct=%s | Graph correct=%s",
            flat_judge["correct"],
            graph_judge["correct"],
        )

        results.append({
            "id": qid,
            "question": question,
            "type": q_type,
            "ground_truth": ground_truth,
            "flat_answer": flat_answer,
            "graph_answer": graph_answer,
            "flat_correct": flat_judge["correct"],
            "graph_correct": graph_judge["correct"],
            "flat_tokens": flat_tokens,
            "graph_tokens": graph_tokens,
            "flat_latency_ms": round(flat_latency, 1),
            "graph_latency_ms": round(graph_latency, 1),
            "flat_reason": flat_judge["reason"],
            "graph_reason": graph_judge["reason"],
        })

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def save_csv(results: list[dict[str, Any]]) -> None:
    """Save benchmark results to CSV."""
    fields = [
        "id", "question", "type", "ground_truth",
        "flat_answer", "graph_answer",
        "flat_correct", "graph_correct",
        "flat_tokens", "graph_tokens",
        "flat_latency_ms", "graph_latency_ms",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved results to %s", CSV_PATH)


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a summary table to stdout."""
    total = len(results)
    flat_correct = sum(1 for r in results if r["flat_correct"])
    graph_correct = sum(1 for r in results if r["graph_correct"])

    flat_avg_tokens = sum(r["flat_tokens"] for r in results) / total
    graph_avg_tokens = sum(r["graph_tokens"] for r in results) / total
    flat_avg_latency = sum(r["flat_latency_ms"] for r in results) / total
    graph_avg_latency = sum(r["graph_latency_ms"] for r in results) / total

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} {'Flat RAG':>12} {'GraphRAG':>12}")
    print("-" * 70)
    print(f"{'Correct answers':<35} {flat_correct:>12}/{total} {graph_correct:>12}/{total}")
    print(f"{'Accuracy':<35} {flat_correct/total*100:>11.1f}% {graph_correct/total*100:>11.1f}%")
    print(f"{'Avg tokens per question':<35} {flat_avg_tokens:>12.0f} {graph_avg_tokens:>12.0f}")
    print(f"{'Avg latency per question (ms)':<35} {flat_avg_latency:>12.0f} {graph_avg_latency:>12.0f}")
    print("=" * 70)

    # By type
    for q_type in ("single", "multi", "trick"):
        subset = [r for r in results if r["type"] == q_type]
        if not subset:
            continue
        fc = sum(1 for r in subset if r["flat_correct"])
        gc = sum(1 for r in subset if r["graph_correct"])
        n = len(subset)
        print(f"  [{q_type.upper()} x{n}] Flat={fc}/{n} ({fc/n*100:.0f}%)  Graph={gc}/{n} ({gc/n*100:.0f}%)")

    print("=" * 70)

    # Cases where graph was better
    print("\nCases where GraphRAG was correct but Flat RAG was NOT:")
    graph_wins = [r for r in results if r["graph_correct"] and not r["flat_correct"]]
    if graph_wins:
        for r in graph_wins:
            print(f"  Q{r['id']} [{r['type']}]: {r['question'][:80]}")
    else:
        print("  (none)")

    print("\nCases where Flat RAG was correct but GraphRAG was NOT:")
    flat_wins = [r for r in results if r["flat_correct"] and not r["graph_correct"]]
    if flat_wins:
        for r in flat_wins:
            print(f"  Q{r['id']} [{r['type']}]: {r['question'][:80]}")
    else:
        print("  (none)")


def generate_cost_analysis(results: list[dict[str, Any]]) -> None:
    """Generate cost_analysis.md with token totals and USD estimates."""
    by_stage = TRACKER.tokens_by_stage()
    avg_lat = TRACKER.avg_latency_by_stage()

    total = len(results)

    # Collect per-system totals from results (for querying stage)
    flat_tokens_total = sum(r["flat_tokens"] for r in results)
    graph_tokens_total = sum(r["graph_tokens"] for r in results)

    # Indexing tokens
    indexing_tokens = by_stage.get("indexing", {"prompt": 0, "completion": 0, "total": 0})
    embed_tokens = by_stage.get("flat_rag_indexing", {"prompt": 0, "completion": 0, "total": 0})

    def cost(prompt_t: int, comp_t: int, embed_t: int = 0) -> float:
        return (
            prompt_t / 1_000_000 * GPT4O_MINI_INPUT_PRICE_PER_1M
            + comp_t / 1_000_000 * GPT4O_MINI_OUTPUT_PRICE_PER_1M
            + embed_t / 1_000_000 * EMBED_SMALL_PRICE_PER_1M
        )

    total_cost = cost(
        sum(s.get("prompt", 0) for s in by_stage.values()),
        sum(s.get("completion", 0) for s in by_stage.values()),
    )

    flat_correct = sum(1 for r in results if r["flat_correct"])
    graph_correct = sum(1 for r in results if r["graph_correct"])
    flat_avg_latency = sum(r["flat_latency_ms"] for r in results) / total
    graph_avg_latency = sum(r["graph_latency_ms"] for r in results) / total

    lines = [
        "# Cost Analysis Report",
        "",
        "## Pricing Assumptions",
        f"- `gpt-4o-mini` input:  **${GPT4O_MINI_INPUT_PRICE_PER_1M:.3f}** per 1M tokens",
        f"- `gpt-4o-mini` output: **${GPT4O_MINI_OUTPUT_PRICE_PER_1M:.3f}** per 1M tokens",
        f"- `text-embedding-3-small`: **${EMBED_SMALL_PRICE_PER_1M:.3f}** per 1M tokens",
        "",
        "## Token Usage by Stage",
        "",
        "| Stage | Prompt Tokens | Completion Tokens | Total |",
        "|-------|--------------|-------------------|-------|",
    ]
    for stage, stats in sorted(by_stage.items()):
        lines.append(
            f"| {stage} | {stats['prompt']:,} | {stats['completion']:,} | {stats['total']:,} |"
        )

    grand_total = sum(s["total"] for s in by_stage.values())
    lines += [
        f"| **GRAND TOTAL** | — | — | **{grand_total:,}** |",
        "",
        "## Estimated Cost (USD)",
        "",
        "| Component | USD |",
        "|-----------|-----|",
        f"| Indexing (LLM) | ${cost(indexing_tokens['prompt'], indexing_tokens['completion']):.4f} |",
        f"| Embedding (indexing) | ${embed_tokens['total'] / 1_000_000 * EMBED_SMALL_PRICE_PER_1M:.4f} |",
        f"| **Total** | **${total_cost:.4f}** |",
        "",
        "## Benchmark Results Summary",
        "",
        "| Metric | Flat RAG | GraphRAG |",
        "|--------|----------|----------|",
        f"| Questions | {total} | {total} |",
        f"| Correct | {flat_correct} ({flat_correct/total*100:.1f}%) | {graph_correct} ({graph_correct/total*100:.1f}%) |",
        f"| Avg tokens / question | {flat_tokens_total/total:.0f} | {graph_tokens_total/total:.0f} |",
        f"| Avg latency / question | {flat_avg_latency:.0f} ms | {graph_avg_latency:.0f} ms |",
        "",
        "## Average Latency by Stage",
        "",
        "| Stage | Avg Latency (ms) |",
        "|-------|------------------|",
    ]
    for stage, lat in sorted(avg_lat.items()):
        lines.append(f"| {stage} | {lat:.0f} |")

    lines += [
        "",
        "---",
        "*Generated automatically by `src/evaluate.py`*",
    ]

    COST_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Cost analysis saved to %s", COST_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results = run_evaluation()
    save_csv(results)
    print_summary(results)
    generate_cost_analysis(results)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
    sys.exit(0)
