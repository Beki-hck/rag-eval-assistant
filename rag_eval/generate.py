"""Answer a question from retrieved chunks.

Two answerers:
- `LLMAnswerer`: a local model writes the answer from the numbered sources.
- `ExtractiveAnswerer`: no model at all. It copies the source sentences
  that best match the question. It can't make things up, so it is the
  floor every LLM should beat, and it runs anywhere (the live demo).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .corpus import Chunk
from .retrieve import tokenize

IDK = "I don't know based on these documents."

SYSTEM = (
    "You answer questions about a developer's portfolio projects using ONLY the "
    "numbered sources. Cite sources like [1]. Keep the answer under 80 words. "
    f"If the sources don't contain the answer, reply exactly: {IDK}"
)


@dataclass
class Answer:
    text: str
    sources: list[Chunk] = field(default_factory=list)


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.doc} / {c.section})\n{c.text}"
                       for i, c in enumerate(chunks, 1))


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Sources:\n{format_context(chunks)}\n\nQuestion: {question}"}]


class LLMAnswerer:
    def __init__(self, chat: Callable[[list[dict]], str]):
        self.chat = chat

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        return Answer(self.chat(build_messages(question, chunks)).strip(), chunks)


def sentences(text: str) -> list[str]:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*|") for p in parts if len(tokenize(p)) >= 3]


class ExtractiveAnswerer:
    """Pick the 2 source sentences that share the most question words.

    Says "I don't know" when no sentence shares at least `min_overlap`
    of the question's content words.
    """

    def __init__(self, n_sentences: int = 2, min_overlap: float = 0.5):
        self.n, self.min_overlap = n_sentences, min_overlap

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        q = set(tokenize(question))
        cands = []
        for i, c in enumerate(chunks, 1):
            ctx = set(tokenize(c.section + " " + c.doc))
            for s in sentences(c.text):
                words = set(tokenize(s)) | ctx
                cands.append((len(q & words) / max(len(q), 1), -i, s, i))
        cands.sort(reverse=True)
        if not cands or cands[0][0] < self.min_overlap:
            return Answer(IDK, chunks)
        picked = cands[: self.n]
        return Answer(" ".join(f"{s} [{i}]" for _, _, s, i in picked), chunks)
