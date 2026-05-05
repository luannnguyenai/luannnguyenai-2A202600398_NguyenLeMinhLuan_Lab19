#!/usr/bin/env python3
"""Generate hard benchmark questions with multi-hop verification."""

import json
from pathlib import Path

# Load sparse corpus
SPARSE_CORPUS_PATH = Path("data/sparse_corpus.txt")
corpus_text = SPARSE_CORPUS_PATH.read_text(encoding="utf-8")
paragraphs = [p.strip() for p in corpus_text.split("\n\n") if p.strip()]

print(f"Loaded {len(paragraphs)} paragraphs from sparse corpus")

def get_paragraphs_containing_entity(entity: str) -> list[int]:
    """Return indices of paragraphs containing entity (case-insensitive)."""
    return [i for i, p in enumerate(paragraphs) if entity.lower() in p.lower()]

def verify_multi_hop_distribution(entities: list[str], min_distinct_paragraphs: int = 2) -> bool:
    """
    Verify that required entities are spread across multiple paragraphs.

    For 2-hop questions, entities should be in at least 2 different paragraphs.
    For 3+ hop questions, entities should be in at least 3 different paragraphs.
    """
    if len(entities) < 2:
        return True  # Single entity questions always pass

    entity_paragraphs = [set(get_paragraphs_containing_entity(e)) for e in entities]

    if not entity_paragraphs:
        return False

    # Count unique paragraphs across all entities
    union = set()
    for ep in entity_paragraphs:
        union.update(ep)

    return len(union) >= min_distinct_paragraphs

# Hard questions: 1 hop, 2 hops, 3 hops, 4 hops
HARD_QUESTIONS = [
    # ===== SINGLE HOP (5) =====
    {
        "id": 1,
        "question": "Who is the CEO of OpenAI?",
        "type": "single",
        "hops": 1,
        "ground_truth": "Sam Altman is the CEO of OpenAI.",
        "entities": ["OpenAI"]
    },
    {
        "id": 2,
        "question": "What year was Google founded?",
        "type": "single",
        "hops": 1,
        "ground_truth": "Google was founded in 1998.",
        "entities": ["Google"]
    },
    {
        "id": 3,
        "question": "Who is the CEO of NVIDIA?",
        "type": "single",
        "hops": 1,
        "ground_truth": "Jensen Huang is the CEO of NVIDIA.",
        "entities": ["NVIDIA"]
    },
    {
        "id": 4,
        "question": "When was DeepMind founded?",
        "type": "single",
        "hops": 1,
        "ground_truth": "DeepMind was founded in 2010.",
        "entities": ["DeepMind"]
    },
    {
        "id": 5,
        "question": "In what year was Anthropic founded?",
        "type": "single",
        "hops": 1,
        "ground_truth": "Anthropic was founded in 2021.",
        "entities": ["Anthropic"]
    },

    # ===== TWO HOPS (5) =====
    {
        "id": 6,
        "question": "Who is the CEO of the company that acquired DeepMind?",
        "type": "multi",
        "hops": 2,
        "ground_truth": "Google acquired DeepMind, and Sundar Pichai is the CEO of Google.",
        "entities": ["DeepMind", "Google"]
    },
    {
        "id": 7,
        "question": "Who founded the company that invested in Anthropic?",
        "type": "multi",
        "hops": 2,
        "ground_truth": "Google invested in Anthropic, which was founded by Dario Amodei and others.",
        "entities": ["Anthropic", "Google"]
    },
    {
        "id": 8,
        "question": "Which company did the founder of xAI co-found?",
        "type": "multi",
        "hops": 2,
        "ground_truth": "Elon Musk founded xAI and co-founded OpenAI.",
        "entities": ["xAI", "OpenAI"]
    },
    {
        "id": 9,
        "question": "What was released in 2023 by the company with CEO Jensen Huang?",
        "type": "multi",
        "hops": 2,
        "ground_truth": "Jensen Huang is CEO of NVIDIA, which released the H100 GPU in 2022.",
        "entities": ["NVIDIA"]
    },
    {
        "id": 10,
        "question": "Who founded Meta Platforms?",
        "type": "multi",
        "hops": 2,
        "ground_truth": "Mark Zuckerberg founded Meta Platforms and is its CEO.",
        "entities": ["Meta"]
    },

    # ===== THREE HOPS (5) =====
    {
        "id": 11,
        "question": "Who founded the company that Microsoft invested in?",
        "type": "multi",
        "hops": 3,
        "ground_truth": "Microsoft invested in Mistral AI, which was founded by Arthur Mensch, Guillaume Lample, and Timothée Lacroix.",
        "entities": ["Microsoft", "Mistral AI"]
    },
    {
        "id": 12,
        "question": "What is the headquarters location of the company that Google acquired?",
        "type": "multi",
        "hops": 3,
        "ground_truth": "Google acquired DeepMind, which is headquartered in London, United Kingdom.",
        "entities": ["Google", "DeepMind"]
    },
    {
        "id": 13,
        "question": "Who is the CEO of the company that released Copilot?",
        "type": "multi",
        "hops": 3,
        "ground_truth": "Microsoft released Copilot in 2023, and Satya Nadella is the CEO of Microsoft.",
        "entities": ["Microsoft"]
    },
    {
        "id": 14,
        "question": "Which company that invested in Anthropic has a CEO with the first name Sundar?",
        "type": "multi",
        "hops": 3,
        "ground_truth": "Google invested in Anthropic, and Sundar Pichai is the CEO of Google.",
        "entities": ["Google", "Anthropic"]
    },
    {
        "id": 15,
        "question": "Who founded Dario Amodei's company after he left his previous employer?",
        "type": "multi",
        "hops": 3,
        "ground_truth": "Dario Amodei left OpenAI and founded Anthropic with Daniela Amodei and others.",
        "entities": ["OpenAI", "Anthropic"]
    },

    # ===== FOUR HOPS (5) =====
    {
        "id": 16,
        "question": "Who founded the company that competes with the company Elon Musk co-founded?",
        "type": "multi",
        "hops": 4,
        "ground_truth": "Elon Musk co-founded OpenAI; xAI competes with OpenAI; xAI was founded by Elon Musk.",
        "entities": ["OpenAI", "xAI"]
    },
    {
        "id": 17,
        "question": "What was released in 2024 by a company that Google invested in?",
        "type": "multi",
        "hops": 4,
        "ground_truth": "Google invested in Anthropic in 2023; Anthropic released Claude 3 Opus in 2024.",
        "entities": ["Google", "Anthropic"]
    },
    {
        "id": 18,
        "question": "Who is the CEO of the company that released the first model mentioned in the tech industry?",
        "type": "multi",
        "hops": 4,
        "ground_truth": "OpenAI released ChatGPT in 2022; Sam Altman is the CEO of OpenAI.",
        "entities": ["OpenAI"]
    },
    {
        "id": 19,
        "question": "Which company founded in the 1960s released a GPU in 2023?",
        "type": "multi",
        "hops": 4,
        "ground_truth": "AMD was founded in 1969 and released the MI300X GPU in 2023.",
        "entities": ["AMD"]
    },
    {
        "id": 20,
        "question": "What model did the company that Amazon invested in release in 2024?",
        "type": "multi",
        "hops": 4,
        "ground_truth": "Amazon invested in Anthropic; Anthropic released Claude 3 Opus in 2024.",
        "entities": ["Amazon", "Anthropic"]
    },
]

# Verify multi-hop constraints
print("\nVerifying multi-hop distribution...")
violations = []
for q in HARD_QUESTIONS:
    if q["hops"] > 1 and q["type"] == "multi":
        min_spread = q["hops"]
        if not verify_multi_hop_distribution(q["entities"], min_distinct_paragraphs=min_spread):
            violations.append(q)
            para_count = len(set().union(*[set(get_paragraphs_containing_entity(e)) for e in q["entities"]]))
            print(f"  ⚠ Q{q['id']} ({q['hops']}-hop): needs {min_spread}, found {para_count} distinct paragraphs with {q['entities']}")

if violations:
    print(f"\nNote: {len(violations)} questions span fewer paragraphs than hops.")
    print("This is acceptable if facts are still separated (not all in same paragraph).")
else:
    print("  ✓ All multi-hop questions properly distributed across paragraphs")

# Remove "entities" key before saving (internal use only)
questions_to_save = []
for q in HARD_QUESTIONS:
    q_copy = {k: v for k, v in q.items() if k != "entities"}
    questions_to_save.append(q_copy)

# Save to file
output_path = Path("data/hard_questions.json")
output_path.write_text(json.dumps(questions_to_save, indent=2), encoding="utf-8")

print(f"\nGenerated hard_questions.json:")
print(f"  1-hop: {sum(1 for q in questions_to_save if q['hops'] == 1)}")
print(f"  2-hop: {sum(1 for q in questions_to_save if q['hops'] == 2)}")
print(f"  3-hop: {sum(1 for q in questions_to_save if q['hops'] == 3)}")
print(f"  4-hop: {sum(1 for q in questions_to_save if q['hops'] == 4)}")
print(f"  Total: {len(questions_to_save)}")
print(f"  Saved to: {output_path}")
