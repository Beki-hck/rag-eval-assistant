"""rag-eval command line: ask, retrieval, answers, serve."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import DEFAULT_CORPUS, load_chunks
from .evaluate import build_retriever, eval_answers, eval_retrieval, load_eval
from .generate import ExtractiveAnswerer, LLMAnswerer
from .llm import ollama_chat, ollama_embedder
from .retrieve import CachedEmbedder


def make_embedder(name: str):
    return CachedEmbedder(ollama_embedder(name), name)


def make_chat(spec: str):
    if not spec.startswith("ollama:"):
        raise SystemExit(f"model must look like ollama:<name>, got {spec!r}")
    model = spec.split(":", 1)[1]
    return lambda messages: ollama_chat(model, messages)


def make_answerer(spec: str):
    return ExtractiveAnswerer() if spec == "extractive" else LLMAnswerer(make_chat(spec))


def fmt(v, pct=True):
    if v is None:
        return "n/a"
    return f"{100 * v:.1f}" if pct else f"{v:.3f}"


def cmd_ask(a):
    chunks = load_chunks(a.corpus, a.chunk_size)
    embed = make_embedder(a.embed_model) if a.retriever != "bm25" else None
    ret = build_retriever(a.retriever, chunks, embed)
    hits = [c for c, _ in ret.search(a.question, a.k)]
    print(make_answerer(a.model).answer(a.question, hits).text, "\n")
    for i, c in enumerate(hits, 1):
        print(f"[{i}] {c.doc} / {c.section}")


def cmd_retrieval(a):
    items = load_eval(a.eval)
    embed = make_embedder(a.embed_model) if set(a.retrievers) - {"bm25"} else None
    out, lines = [], ["| Retriever | Chunk size | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR |",
                      "|---|---|---|---|---|---|---|"]
    for size in a.chunk_sizes:
        chunks = load_chunks(a.corpus, size)
        for name in a.retrievers:
            res = eval_retrieval(build_retriever(name, chunks, embed), items)
            s = res["summary"]
            out.append({"retriever": name, "chunk_size": size, "n_chunks": len(chunks), **res})
            lines.append(f"| {name} | {size} | {len(chunks)} | {fmt(s['hit@1'])} | {fmt(s['hit@3'])} "
                         f"| {fmt(s['hit@5'])} | {fmt(s['mrr'], False)} |")
            print(lines[-1], flush=True)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    (od / "retrieval.json").write_text(json.dumps(out, indent=1))
    (od / "retrieval.md").write_text("\n".join(lines) + "\n")


def cmd_answers(a):
    items = load_eval(a.eval)
    chunks = load_chunks(a.corpus, a.chunk_size)
    embed = make_embedder(a.embed_model) if a.retriever != "bm25" else None
    ret = build_retriever(a.retriever, chunks, embed)
    judge = make_chat(a.judge) if a.judge else None
    od = Path(a.out)
    rows_md = od / "answers.md"
    header = ("| Answerer | Retriever | Correct facts | Correct when right chunk found | False refusals "
              "| Refused unanswerable | Number grounding | Faithfulness (judge) |\n|---|---|---|---|---|---|---|---|\n")
    for spec in a.model:
        slug = spec.replace(":", "_").replace("/", "_")
        res = eval_answers(ret, make_answerer(spec), items, a.k, judge,
                           cache_path=od / f"answers__{slug}__{a.retriever}{a.chunk_size}.jsonl")
        s = res["summary"]
        line = (f"| {spec} | {a.retriever}, {a.chunk_size} chars, k={a.k} | {fmt(s['correctness'])} "
                f"| {fmt(s['correctness_when_retrieved'])} | {s['false_refusals']}/{s['n_answerable']} "
                f"| {s['correct_refusals']}/{s['n_unanswerable']} | {fmt(s['grounding'])} "
                f"| {fmt(s['faithfulness'])} |")
        print(line, flush=True)
        od.mkdir(parents=True, exist_ok=True)
        if not rows_md.exists():
            rows_md.write_text(header)
        with rows_md.open("a") as f:
            f.write(line + "\n")


def cmd_serve(a):
    from .server import serve
    serve(a.host, a.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag-eval")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--eval", default=None)
    p.add_argument("--embed-model", default="nomic-embed-text")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ask", help="answer one question")
    s.add_argument("question")
    s.add_argument("-m", "--model", default="extractive", help="extractive or ollama:<model>")
    s.add_argument("-r", "--retriever", default="bm25", choices=["bm25", "embed", "hybrid"])
    s.add_argument("--chunk-size", type=int, default=600)
    s.add_argument("-k", type=int, default=3)
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("retrieval", help="score retrievers and chunk sizes")
    s.add_argument("-r", "--retrievers", nargs="+", default=["bm25"], choices=["bm25", "embed", "hybrid"])
    s.add_argument("--chunk-sizes", nargs="+", type=int, default=[300, 600, 1200])
    s.add_argument("-o", "--out", default="results")
    s.set_defaults(fn=cmd_retrieval)

    s = sub.add_parser("answers", help="score answers from one or more answerers")
    s.add_argument("-m", "--model", action="append", required=True, help="extractive or ollama:<model>")
    s.add_argument("-r", "--retriever", default="bm25", choices=["bm25", "embed", "hybrid"])
    s.add_argument("--chunk-size", type=int, default=600)
    s.add_argument("-k", type=int, default=3)
    s.add_argument("--judge", default=None, help="ollama:<model> for faithfulness")
    s.add_argument("-o", "--out", default="results")
    s.set_defaults(fn=cmd_answers)

    s = sub.add_parser("serve", help="web demo")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    if a.eval is None:
        from .evaluate import DEFAULT_EVAL
        a.eval = DEFAULT_EVAL
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
