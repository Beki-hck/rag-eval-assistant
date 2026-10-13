"""Run the eval set against a retriever and an answerer."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Callable

from . import metrics as M
from .corpus import Chunk
from .generate import IDK
from .retrieve import BM25, EmbeddingRetriever, Hybrid, Retriever

DEFAULT_EVAL = Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"


def load_eval(path: Path | str = DEFAULT_EVAL) -> list[dict]:
    items = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for it in items:
        it.setdefault("answerable", True)
    return items


def build_retriever(name: str, chunks: list[Chunk], embed=None) -> Retriever:
    if name == "bm25":
        return BM25(chunks)
    if embed is None:
        raise ValueError(f"retriever {name!r} needs an embedding model")
    if name == "embed":
        return EmbeddingRetriever(chunks, embed)
    if name == "hybrid":
        return Hybrid([BM25(chunks), EmbeddingRetriever(chunks, embed)])
    raise ValueError(f"unknown retriever {name!r}")


def eval_retrieval(retriever: Retriever, items: list[dict], ks=(1, 3, 5)) -> dict:
    rows = []
    for it in (i for i in items if i["answerable"]):
        ranked = [c for c, _ in retriever.search(it["question"], max(ks))]
        rows.append({"id": it["id"], "rank": M.first_relevant_rank(ranked, it),
                     **{f"hit@{k}": M.hit_at_k(ranked, it, k) for k in ks},
                     "rr": M.reciprocal_rank(ranked, it)})
    summary = {f"hit@{k}": mean(r[f"hit@{k}"] for r in rows) for k in ks}
    summary["mrr"] = mean(r["rr"] for r in rows)
    summary["n"] = len(rows)
    return {"summary": summary, "rows": rows}


def eval_answers(retriever: Retriever, answerer, items: list[dict], k: int = 3,
                 judge: Callable[[list[dict]], str] | None = None,
                 cache_path: Path | None = None) -> dict:
    """Answer every question; score correctness, refusals, grounding and faithfulness.

    With `cache_path`, finished rows are saved as JSONL and skipped on rerun.
    """
    done: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    rows = []
    for it in items:
        if it["id"] in done:
            rows.append(done[it["id"]])
            continue
        chunks = [c for c, _ in retriever.search(it["question"], k)]
        ans = answerer.answer(it["question"], chunks).text
        refused = M.is_refusal(ans) or ans.strip() == IDK
        row = {"id": it["id"], "answerable": it["answerable"], "answer": ans,
               "sources": [c.id for c in chunks], "refused": refused,
               "retrieved_relevant": bool(it["answerable"] and M.first_relevant_rank(chunks, it))}
        if it["answerable"]:
            row["correctness"] = M.correctness(ans, it["facts"])
        if not refused:
            row["grounding"] = M.number_grounding(ans, chunks)
            row["faithfulness"] = M.judge_faithfulness(ans, chunks, judge) if judge else None
        rows.append(row)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"summary": summarize(rows), "rows": rows}


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    found = [r for r in ans if r["retrieved_relevant"]]
    return {
        "correctness": _avg(r["correctness"] for r in ans),
        "correctness_when_retrieved": _avg(r["correctness"] for r in found),
        "false_refusals": sum(r["refused"] for r in ans),
        "n_answerable": len(ans),
        "correct_refusals": sum(r["refused"] for r in unans),
        "n_unanswerable": len(unans),
        "grounding": _avg(r.get("grounding") for r in rows),
        "faithfulness": _avg(r.get("faithfulness") for r in rows),
    }


def rescore(rows: list[dict], items: list[dict], chunks: list[Chunk]) -> list[dict]:
    """Recompute model-free scores on saved answers (no model calls).

    Faithfulness is kept as saved; a row that was a refusal before has none.
    """
    by_id, by_chunk = {i["id"]: i for i in items}, {c.id: c for c in chunks}
    out = []
    for r in rows:
        it, srcs = by_id[r["id"]], [by_chunk[c] for c in r["sources"]]
        r = dict(r, refused=M.is_refusal(r["answer"]))
        if it["answerable"]:
            r["correctness"] = M.correctness(r["answer"], it["facts"])
        if r["refused"]:
            r.pop("grounding", None); r.pop("faithfulness", None)
        else:
            r["grounding"] = M.number_grounding(r["answer"], srcs)
            r.setdefault("faithfulness", None)
        out.append(r)
    return out
