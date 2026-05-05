"""
OpenAI wrapper that centrally tracks token usage and latency for every API call.

All LLM and embedding calls in this project MUST go through this module so that
per-stage cost accounting is accurate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """Stores metadata for a single LLM/embedding API call."""
    stage: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass
class UsageTracker:
    """Accumulates all API call records for the lifetime of the process."""
    records: list[CallRecord] = field(default_factory=list)

    def add(self, record: CallRecord) -> None:
        self.records.append(record)
        logger.info(
            "[%s] prompt=%d completion=%d total=%d latency=%.0fms",
            record.stage,
            record.prompt_tokens,
            record.completion_tokens,
            record.prompt_tokens + record.completion_tokens,
            record.latency_ms,
        )

    def total_tokens(self) -> int:
        return sum(r.prompt_tokens + r.completion_tokens for r in self.records)

    def tokens_by_stage(self) -> dict[str, dict[str, int]]:
        """Return prompt/completion/total tokens grouped by stage."""
        result: dict[str, dict[str, int]] = {}
        for r in self.records:
            bucket = result.setdefault(r.stage, {"prompt": 0, "completion": 0})
            bucket["prompt"] += r.prompt_tokens
            bucket["completion"] += r.completion_tokens
        for bucket in result.values():
            bucket["total"] = bucket["prompt"] + bucket["completion"]
        return result

    def avg_latency_by_stage(self) -> dict[str, float]:
        """Return average latency in ms grouped by stage."""
        sums: dict[str, list[float]] = {}
        for r in self.records:
            sums.setdefault(r.stage, []).append(r.latency_ms)
        return {stage: sum(lats) / len(lats) for stage, lats in sums.items()}

    def replay(
        self,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float = 0.0,
        calls: int = 1,
    ) -> None:
        """Replay cached token usage from a previous run into this tracker.

        Used to populate tracker with tokens from indexing stages when
        evaluating warm (when triples.json already exists).

        Parameters
        ----------
        stage : str
            Stage label (e.g. "indexing")
        prompt_tokens : int
            Prompt tokens for this stage
        completion_tokens : int
            Completion tokens for this stage
        latency_ms : float, optional
            Total latency in milliseconds (default 0.0)
        calls : int, optional
            Number of API calls aggregated (default 1)
        """
        # Add synthetic record(s) to match the aggregated totals
        # If multiple calls, spread latency evenly
        per_call_latency = latency_ms / calls if calls > 0 else 0.0
        per_call_prompt = prompt_tokens // calls if calls > 0 else 0
        per_call_completion = completion_tokens // calls if calls > 0 else 0
        remainder_prompt = prompt_tokens % calls if calls > 0 else 0
        remainder_completion = completion_tokens % calls if calls > 0 else 0

        for i in range(calls):
            # Distribute remainder to first call(s)
            p_toks = per_call_prompt + (1 if i < remainder_prompt else 0)
            c_toks = per_call_completion + (1 if i < remainder_completion else 0)
            record = CallRecord(
                stage=stage,
                prompt_tokens=p_toks,
                completion_tokens=c_toks,
                latency_ms=per_call_latency,
            )
            self.add(record)


# Global singleton tracker shared across all modules
TRACKER = UsageTracker()


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around OpenAI client that records token usage and latency.

    Parameters
    ----------
    model : str
        Chat completion model name, e.g. "gpt-4o-mini".
    embedding_model : str
        Embedding model name, e.g. "text-embedding-3-small".
    """

    CHAT_MODEL: str = "gpt-4o-mini"
    EMBED_MODEL: str = "text-embedding-3-small"

    def __init__(
        self,
        model: str = CHAT_MODEL,
        embedding_model: str = EMBED_MODEL,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self._client = OpenAI()
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        stage: str = "unknown",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """
        Call the chat completions endpoint and record usage.

        Parameters
        ----------
        messages : list of message dicts (role/content)
        stage : label used in the usage tracker (e.g. "indexing", "graph_rag")
        temperature : sampling temperature
        max_tokens : max completion tokens

        Returns
        -------
        str : the assistant message content
        """
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        TRACKER.add(CallRecord(
            stage=stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        ))

        content = response.choices[0].message.content or ""
        return content

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, texts: list[str], stage: str = "embedding") -> list[list[float]]:
        """
        Embed a list of texts and record usage (estimated via tiktoken).

        Parameters
        ----------
        texts : list of strings to embed
        stage : label used in the usage tracker

        Returns
        -------
        list of float vectors
        """
        t0 = time.perf_counter()
        response = self._client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Embedding endpoint does not return completion tokens
        prompt_tokens = response.usage.prompt_tokens if response.usage else sum(
            len(self._enc.encode(t)) for t in texts
        )

        TRACKER.add(CallRecord(
            stage=stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            latency_ms=latency_ms,
        ))

        return [item.embedding for item in response.data]

    def embed_single(self, text: str, stage: str = "embedding") -> list[float]:
        """Convenience method to embed a single string."""
        return self.embed([text], stage=stage)[0]


# Module-level default client instance
_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Return (or create) the default module-level LLMClient."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
