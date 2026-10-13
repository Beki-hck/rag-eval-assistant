import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from rag_eval import metrics as M
from rag_eval.corpus import Chunk, chunk_text, clean_markdown, load_chunks, split_sections
from rag_eval.evaluate import build_retriever, eval_answers, eval_retrieval, load_eval
from rag_eval.generate import IDK, ExtractiveAnswerer, LLMAnswerer
from rag_eval.retrieve import BM25, CachedEmbedder, EmbeddingRetriever, Hybrid, tokenize


def fake_embed(texts):
    """Bag-of-words hashed into 64 dims: similar words -> similar vectors."""
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in tokenize(t):
            v[hash(w) % 64] += 1
        out.append(v)
    return out


C = [Chunk("a#x#0", "a", "Results", "The model scored 91% on math."),
     Chunk("b#y#0", "b", "Install", "Run pip install to set it up."),
     Chunk("c#z#0", "c", "License", "Released under the MIT license.")]


def test_clean_and_sections():
    md = "![badge](x)\n# Title\nintro\n<!-- c -->\n## Results\n```\n# not a heading\n```\nbody"
    secs = split_sections(clean_markdown(md))
    assert [h for h, _ in secs] == ["Title", "Results"]
    assert "# not a heading" in secs[1][1]


def test_chunk_text_respects_size():
    text = "\n\n".join(f"paragraph {i} " + "word " * 30 for i in range(10))
    chunks = chunk_text(text, 300)
    assert len(chunks) > 3 and all(len(c) <= 300 for c in chunks)
    assert "paragraph 0" in chunks[0]


def test_chunk_overlap_repeats_tail():
    text = "\n\n".join(f"para{i} " + "x " * 60 for i in range(4))
    a = chunk_text(text, 200, overlap=50)
    assert len(a) > 1 and a[1].split()[0] == "x"


def test_bm25_ranks_keyword_match_first():
    top = BM25(C).search("what license is it under", 3)
    assert top[0][0].doc == "c"


def test_embedding_and_hybrid(tmp_path):
    emb = CachedEmbedder(fake_embed, "fake", tmp_path)
    r = EmbeddingRetriever(C, emb)
    assert r.search("pip install setup", 1)[0][0].doc == "b"
    assert (tmp_path / "emb-fake.json").exists()
    h = Hybrid([BM25(C), r]).search("MIT license", 2)
    assert h[0][0].doc == "c"


def test_cached_embedder_calls_once(tmp_path):
    calls = []
    def emb(t):
        calls.append(len(t))
        return fake_embed(t)
    e = CachedEmbedder(emb, "x", tmp_path)
    e(["hello world", "foo bar"]); e(["hello world"])
    assert calls == [2]


def test_retrieval_metrics():
    item = {"doc": "a", "evidence": ["91%"]}
    ranked = [C[1], C[0], C[2]]
    assert M.first_relevant_rank(ranked, item) == 2
    assert M.reciprocal_rank(ranked, item) == 0.5
    assert M.hit_at_k(ranked, item, 1) == 0 and M.hit_at_k(ranked, item, 2) == 1
    assert M.reciprocal_rank([C[1]], item) == 0


def test_answer_metrics():
    assert M.correctness("It got 1,797 digits and 91%", [["1797"], ["91%"], ["zzz"]]) == pytest.approx(2 / 3)
    assert M.is_refusal(IDK) and M.is_refusal("The sources do not mention this.")
    assert not M.is_refusal("It scored 91% [1].")
    assert M.number_grounding("It scored 91% [1].", C) == 1.0
    assert M.number_grounding("It scored 95% [1].", C) == 0.0
    assert M.number_grounding("No numbers here.", C) is None


def test_judge_faithfulness_parses_and_fails_soft():
    reply = 'Sure: {"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}'
    assert M.judge_faithfulness("x", C, lambda m: reply) == 0.5
    assert M.judge_faithfulness("x", C, lambda m: "no json") is None


def test_extractive_answers_and_refuses():
    ex = ExtractiveAnswerer()
    assert "91%" in ex.answer("What did the model score on math?", C).text
    assert ex.answer("Who won the football world cup?", C).text == IDK


def test_llm_answerer_sends_numbered_sources():
    seen = {}
    def chat(m):
        seen["m"] = m
        return " It scored 91% [1]. "
    a = LLMAnswerer(chat).answer("score?", C[:1])
    assert a.text == "It scored 91% [1]."
    assert "[1] (a / Results)" in seen["m"][1]["content"]


def test_eval_set_evidence_exists_for_every_chunk_size():
    items = load_eval()
    assert len(items) >= 50 and sum(not i["answerable"] for i in items) >= 8
    for size in (300, 600, 1200):
        chunks = load_chunks(size=size)
        for it in items:
            if it["answerable"]:
                assert any(M.is_relevant(c, it) for c in chunks), (size, it["id"])


def test_eval_pipeline_end_to_end(tmp_path):
    items = load_eval()
    chunks = load_chunks(size=600)
    res = eval_retrieval(build_retriever("bm25", chunks), items)
    assert 0.5 < res["summary"]["hit@5"] <= 1
    cache = tmp_path / "a.jsonl"
    out = eval_answers(BM25(chunks), ExtractiveAnswerer(), items, judge=lambda m: '{"claims": []}', cache_path=cache)
    s = out["summary"]
    assert s["n_answerable"] == 43 and s["n_unanswerable"] == 8
    assert len(cache.read_text().splitlines()) == len(items)
    again = eval_answers(BM25(chunks), None, items, cache_path=cache)  # all cached: answerer unused
    assert again["summary"] == s


def test_embed_retriever_needs_embedder():
    with pytest.raises(ValueError):
        build_retriever("embed", C)


def test_server_answers(monkeypatch):
    monkeypatch.setenv("RAG_ANSWERER", "extractive")
    from rag_eval.server import App, make_handler
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"Ask about my AI projects" in urllib.request.urlopen(base + "/").read()
        d = json.loads(urllib.request.urlopen(base + "/api/ask?q=What+license+are+the+Amharic+items+under").read())
        assert d["answerer"].startswith("extractive") and len(d["sources"]) == 3
    finally:
        srv.shutdown()


def test_refusal_only_for_short_declines():
    assert M.is_refusal("[1] I don't know based on these documents.")
    long = "Confident learning flagged 0 of 150 iris labels [1]. " * 3 + "The sources do not mention CIFAR."
    assert not M.is_refusal(long)


def test_grounding_ignores_citation_lists_and_numbering():
    assert M.number_grounding("It scored 91% [1, 3].", C) == 1.0
    assert M.number_grounding("Stats:\n1. scored 91%\n2. done", C) == 1.0
