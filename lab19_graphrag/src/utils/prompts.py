"""
Prompt templates for triple extraction and question-answering.

Rules
-----
* TRIPLE_EXTRACTION_PROMPT uses strict JSON output with allowed entity/relation types.
* QA_PROMPT is identical for both Flat RAG and GraphRAG — only the context section differs.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Triple extraction
# ---------------------------------------------------------------------------

TRIPLE_EXTRACTION_SYSTEM = """\
You are an expert knowledge graph extraction assistant.
Extract factual (subject, relation, object) triples from the provided text.

ENTITY TYPES allowed (use exactly these strings):
  Person, Organization, Product, Location, Year, Event

RELATION TYPES allowed (use exactly these strings):
  FOUNDED_BY, FOUNDED_IN, CEO_OF, ACQUIRED, RELEASED,
  HEADQUARTERED_IN, SUBSIDIARY_OF, INVESTED_IN,
  COLLABORATES_WITH, COMPETES_WITH

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences, no explanation:
{
  "triples": [
    {
      "subject": "<entity name>",
      "subject_type": "<entity type>",
      "relation": "<relation>",
      "object": "<entity name>",
      "object_type": "<entity type>"
    }
  ]
}

RULES:
1. Every triple MUST use one of the allowed relation types.
2. Every entity MUST use one of the allowed entity types.
3. Do NOT infer facts not stated in the text.
4. For founding years, the subject is the organization and the object is the Year string (e.g. "2015").
5. Return ONLY the JSON object — no extra text before or after.
"""

TRIPLE_EXTRACTION_FEW_SHOT = """\
--- FEW-SHOT EXAMPLES ---

Example 1 Input:
  "Apple was founded in 1976 by Steve Jobs. Tim Cook serves as CEO of Apple."

Example 1 Output:
{
  "triples": [
    {"subject": "Apple", "subject_type": "Organization", "relation": "FOUNDED_IN", "object": "1976", "object_type": "Year"},
    {"subject": "Apple", "subject_type": "Organization", "relation": "FOUNDED_BY", "object": "Steve Jobs", "object_type": "Person"},
    {"subject": "Tim Cook", "subject_type": "Person", "relation": "CEO_OF", "object": "Apple", "object_type": "Organization"}
  ]
}

Example 2 Input:
  "Google acquired DeepMind in 2014. DeepMind is headquartered in London."

Example 2 Output:
{
  "triples": [
    {"subject": "Google", "subject_type": "Organization", "relation": "ACQUIRED", "object": "DeepMind", "object_type": "Organization"},
    {"subject": "DeepMind", "subject_type": "Organization", "relation": "SUBSIDIARY_OF", "object": "Google", "object_type": "Organization"},
    {"subject": "DeepMind", "subject_type": "Organization", "relation": "HEADQUARTERED_IN", "object": "London", "object_type": "Location"}
  ]
}
--- END FEW-SHOT EXAMPLES ---
"""

TRIPLE_EXTRACTION_USER_TEMPLATE = """\
{few_shot}

Now extract triples from the following text paragraph:

\"\"\"
{text}
\"\"\"

Remember: output ONLY the JSON object with a "triples" key.
"""

TRIPLE_EXTRACTION_RETRY_REMINDER = """\
Your previous response was not valid JSON or did not match the required schema.

Required schema:
{{"triples": [{{"subject": str, "subject_type": str, "relation": str, "object": str, "object_type": str}}]}}

Allowed entity types: Person, Organization, Product, Location, Year, Event
Allowed relation types: FOUNDED_BY, FOUNDED_IN, CEO_OF, ACQUIRED, RELEASED, HEADQUARTERED_IN, SUBSIDIARY_OF, INVESTED_IN, COLLABORATES_WITH, COMPETES_WITH

Respond with ONLY valid JSON. No markdown, no extra text.
"""


def build_extraction_messages(text: str) -> list[dict[str, str]]:
    """Build the messages list for triple extraction."""
    return [
        {"role": "system", "content": TRIPLE_EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": TRIPLE_EXTRACTION_USER_TEMPLATE.format(
                few_shot=TRIPLE_EXTRACTION_FEW_SHOT,
                text=text,
            ),
        },
    ]


def build_extraction_retry_messages(
    text: str, bad_response: str
) -> list[dict[str, str]]:
    """Build retry messages that include the failed response and a reminder."""
    return [
        {"role": "system", "content": TRIPLE_EXTRACTION_SYSTEM},
        {
            "role": "user",
            "content": TRIPLE_EXTRACTION_USER_TEMPLATE.format(
                few_shot=TRIPLE_EXTRACTION_FEW_SHOT,
                text=text,
            ),
        },
        {"role": "assistant", "content": bad_response},
        {"role": "user", "content": TRIPLE_EXTRACTION_RETRY_REMINDER},
    ]


# ---------------------------------------------------------------------------
# QA prompt (identical structure for both retrieval systems)
# ---------------------------------------------------------------------------

QA_SYSTEM = """\
You are a precise question-answering assistant.
Answer the question using ONLY the information provided in the context below.
If the answer cannot be found in the context, respond with exactly:
  "I don't know."
Do NOT use any external knowledge. Do NOT hallucinate facts.
"""

QA_USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer:"""


def build_qa_messages(context: str, question: str) -> list[dict[str, str]]:
    """Build the messages list for question answering (shared by both RAG systems)."""
    return [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": QA_USER_TEMPLATE.format(context=context, question=question)},
    ]


# ---------------------------------------------------------------------------
# LLM-as-judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an objective answer correctness judge.
Compare the system answer to the ground truth and decide if the system answer is correct.

For questions of type "trick": the correct answer is an explicit refusal or "I don't know" statement.
  - If the system says "I don't know" or clearly refuses to answer, mark correct=true.
  - If the system fabricates an answer, mark correct=false.

For all other question types: the system answer is correct if it captures the key factual
content of the ground truth, even if worded differently. Partial answers that miss key facts
should be marked correct=false.

Respond with ONLY valid JSON — no markdown, no explanation outside the JSON:
{"correct": true, "reason": "brief explanation"}
or
{"correct": false, "reason": "brief explanation"}
"""

JUDGE_USER_TEMPLATE = """\
Question: {question}
Question Type: {q_type}
Ground Truth: {ground_truth}
System Answer: {system_answer}

Is the system answer correct?
"""


def build_judge_messages(
    question: str,
    q_type: str,
    ground_truth: str,
    system_answer: str,
) -> list[dict[str, str]]:
    """Build the messages list for LLM-as-judge evaluation."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                question=question,
                q_type=q_type,
                ground_truth=ground_truth,
                system_answer=system_answer,
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Entity extraction from question (used by GraphRAG)
# ---------------------------------------------------------------------------

ENTITY_EXTRACTION_SYSTEM = """\
You are an entity extraction assistant.
Given a question, extract all named entities (people, organizations, products, locations).
Return ONLY a JSON array of strings, e.g.: ["OpenAI", "Sam Altman"]
No explanation, no markdown fences.
"""

ENTITY_EXTRACTION_USER_TEMPLATE = """\
Question: {question}

Named entities (JSON array only):"""


def build_entity_extraction_messages(question: str) -> list[dict[str, str]]:
    """Build messages to extract entities from a question for GraphRAG node matching."""
    return [
        {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM},
        {"role": "user", "content": ENTITY_EXTRACTION_USER_TEMPLATE.format(question=question)},
    ]
