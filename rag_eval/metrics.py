"""Scores for retrieval and for answers.

Retrieval (did we find the right chunk?)
- hit rate@k: share of questions where a relevant chunk is in the top k.
- MRR (mean reciprocal rank): 1/rank of the first relevant chunk, averaged.
  Rank 1 scores 1.0, rank 2 scores 0.5, not found scores 0.

Answers (is what the model said right, and is it backed by the sources?)
- correctness: share of the expected facts that appear in the answer.
- refusal: on questions the documents can't answer, did it say so?
- number grounding: share of numbers in the answer that also appear in the
  sources. A cheap, deterministic check for made-up figures.
- faithfulness: an LLM judge splits the answer into claims and marks each
  one supported or not by the sources (see `judge_faithfulness`).
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .corpus import Chunk


def norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 1,797 -> 1797
    text = re.sub(r"[`*_]", "", text)
    return re.sub(r"\s+", " ", text)


def is_relevant(chunk: Chunk, item: dict) -> bool:
    if chunk.doc != item["doc"]:
        return False
    body = norm(chunk.text)
    return any(norm(e) in body for e in item["evidence"])


def first_relevant_rank(ranked: list[Chunk], item: dict) -> int | None:
    for rank, c in enumerate(ranked, 1):
        if is_relevant(c, item):
            return rank
    return None


def hit_at_k(ranked: list[Chunk], item: dict, k: int) -> float:
    r = first_relevant_rank(ranked[:k], item)
    return 1.0 if r else 0.0


def reciprocal_rank(ranked: list[Chunk], item: dict) -> float:
    r = first_relevant_rank(ranked, item)
    return 1.0 / r if r else 0.0


def precision_at_k(ranked: list[Chunk], item: dict, k: int) -> float:
    """Share of the top k chunks that are relevant.

    TODO(Beki): write this one yourself (see tests/test_metrics.py).
    Example: top 3 = [relevant, not, relevant] -> 2/3.
    """
    raise NotImplementedError


def correctness(answer: str, facts: list[list[str]]) -> float:
    a = norm(answer)
    if not facts:
        return 1.0
    return sum(any(norm(alt) in a for alt in fact) for fact in facts) / len(facts)


REFUSAL = re.compile(r"(don'?t|do not|does not|doesn'?t|cannot|can'?t) (know|contain|say|mention|provide|find)"
                     r"|not (mentioned|stated|specified|available|in the sources)|no information", re.I)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL.search(answer.replace("’", "'")))


def numbers(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", " ", norm(text))  # drop citation markers
    return re.findall(r"\d+(?:\.\d+)?", text)


def number_grounding(answer: str, chunks: list[Chunk]) -> float | None:
    nums = numbers(answer)
    if not nums:
        return None
    ctx = set(numbers(" ".join(c.text for c in chunks)))
    return sum(n in ctx for n in nums) / len(nums)


JUDGE_PROMPT = """You check whether an answer is supported by source text.
Split the ANSWER into its factual claims. For each claim, decide if the SOURCES state it
(paraphrase is fine; anything not in the sources is unsupported).
Return JSON only: {{"claims": [{{"claim": "...", "supported": true}}]}}

SOURCES:
{sources}

ANSWER:
{answer}"""


def judge_faithfulness(answer: str, chunks: list[Chunk],
                       chat: Callable[[list[dict]], str]) -> float | None:
    """Fraction of the answer's claims the judge marks as supported.

    Returns None when the judge's reply can't be parsed or lists no claims.
    """
    from .generate import format_context
    prompt = JUDGE_PROMPT.format(sources=format_context(chunks), answer=answer)
    raw = chat([{"role": "user", "content": prompt}])
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        claims = json.loads(m.group(0))["claims"] if m else []
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    flags = [bool(c.get("supported")) for c in claims if isinstance(c, dict)]
    return sum(flags) / len(flags) if flags else None
