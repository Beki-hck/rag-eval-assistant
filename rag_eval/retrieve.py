"""Retrievers: find the chunks most related to a question.

- BM25: keyword search, the classic search-engine formula.
- Embedding: meaning search. Each chunk becomes a vector of numbers from an
  embedding model; chunks whose vectors point the same way as the
  question's vector (cosine similarity) are returned.
- Hybrid: merges both ranked lists with reciprocal rank fusion.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Protocol

from .corpus import Chunk

STOPWORDS = set(
    "a an the and or of to in on for with by is are was were be it its this that "
    "these those as at from what which who how why does do did i my me you your "
    "can there their they them than then so if not no but into about".split()
)


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return [t for t in toks if t not in STOPWORDS]


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]: ...


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # Heading words count too: "Results" sections should match "results".
        self.docs = [Counter(tokenize(c.section + " " + c.doc + " " + c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[float]:
        q = tokenize(query)
        scores = []
        for d, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * ln / self.avg))
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], scores[i]) for i in order]


Embedder = Callable[[list[str]], list[list[float]]]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class CachedEmbedder:
    """Wraps an embedder so each text is embedded once, saved to a JSON file."""

    def __init__(self, embed: Embedder, name: str, cache_dir: Path | str = ".cache"):
        self.embed, self.path = embed, Path(cache_dir) / f"emb-{re.sub(r'[^a-zA-Z0-9]+', '_', name)}.json"
        self.cache: dict[str, list[float]] = (
            json.loads(self.path.read_text()) if self.path.exists() else {})

    def __call__(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [t for t, key in zip(texts, keys) if key not in self.cache]
        for i in range(0, len(missing), 32):
            batch = missing[i:i + 32]
            for t, v in zip(batch, self.embed(batch)):
                self.cache[hashlib.sha1(t.encode()).hexdigest()] = v
        if missing:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache))
        return [self.cache[key] for key in keys]


class EmbeddingRetriever:
    def __init__(self, chunks: list[Chunk], embed: Embedder,
                 doc_prefix: str = "search_document: ", query_prefix: str = "search_query: "):
        # nomic-embed-text expects these task prefixes; other models ignore them.
        self.chunks, self.embed, self.qp = chunks, embed, query_prefix
        self.vecs = embed([f"{doc_prefix}{c.doc} / {c.section}\n{c.text}" for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        qv = self.embed([self.qp + query])[0]
        scored = sorted(((cosine(qv, v), i) for i, v in enumerate(self.vecs)), reverse=True)[:k]
        return [(self.chunks[i], s) for s, i in scored]


class Hybrid:
    """Reciprocal rank fusion: score = sum of 1 / (60 + rank) over both lists."""

    def __init__(self, retrievers: list[Retriever], depth: int = 20, c: int = 60):
        self.retrievers, self.depth, self.c = retrievers, depth, c

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for rank, (chunk, _) in enumerate(r.search(query, self.depth), 1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1 / (self.c + rank)
                by_id[chunk.id] = chunk
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(by_id[cid], s) for cid, s in best]
