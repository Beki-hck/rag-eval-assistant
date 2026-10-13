# rag-eval-assistant

![ci](https://github.com/Beki-hck/rag-eval-assistant/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![deps](https://img.shields.io/badge/dependencies-none-brightgreen)

A question-answering assistant over my portfolio's project READMEs, built with retrieval-augmented generation (RAG), plus the evaluation that tells you whether it actually works: did it find the right passage, is the answer correct, and is every claim backed by a source?

```
python -m rag_eval ask "Which detector had the best F1 on digits?" -m ollama:llama3.2:3b
```

## How RAG works here

1. **Chunking.** Each README is split by heading, then packed into chunks of about N characters (300, 600 or 1,200 are compared).
2. **Retrieval.** For a question, find the top k chunks three ways:
   - `bm25`: keyword search (the classic search-engine scoring formula).
   - `embed`: meaning search. An embedding model (`nomic-embed-text` through Ollama) turns text into a vector; chunks whose vectors point the same way as the question's (cosine similarity) win.
   - `hybrid`: both lists merged with reciprocal rank fusion.
3. **Generation.** A local model answers from the numbered chunks only, cites them like `[1]`, and must say "I don't know" when the chunks don't contain the answer. An `extractive` answerer that just copies the best-matching source sentences is the no-LLM baseline.

## How it's evaluated

`data/eval.jsonl` has 51 questions I wrote against the five READMEs: 43 answerable, each with the document and a short evidence phrase that marks a correct chunk (so the labels work at any chunk size) and the facts a correct answer must contain, plus 8 questions the documents can't answer.

| Metric | What it measures |
|---|---|
| Hit rate@k | share of questions where a correct chunk is in the top k |
| MRR | mean of 1/rank of the first correct chunk (1.0 = always first) |
| Correct facts | share of required facts present in the answer |
| False refusals / refused unanswerable | "I don't know" on answerable vs unanswerable questions |
| Number grounding | share of numbers in the answer that appear in the retrieved chunks (deterministic hallucination check) |
| Faithfulness | an LLM judge splits the answer into claims and marks each supported or not by the chunks |

## Results

<!-- results:start -->
Measured 2026-10-13 on my PC with Ollama (temperature 0). Embeddings: `nomic-embed-text`. Judge: `qwen2.5:3b`. Every answer is in [`benchmarks/2026-10-13/`](benchmarks/2026-10-13/).

**Retrieval (43 answerable questions)**

| Retriever | Chunk size | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| bm25 | 300 | 69.8 | 83.7 | 88.4 | 0.775 |
| embed | 300 | 74.4 | 86.0 | 93.0 | 0.814 |
| hybrid | 300 | 76.7 | 86.0 | 90.7 | 0.819 |
| bm25 | 600 | 65.1 | 83.7 | 90.7 | 0.749 |
| embed | 600 | 65.1 | 81.4 | 95.3 | 0.759 |
| **hybrid** | **600** | **76.7** | **93.0** | 95.3 | **0.830** |
| bm25 | 1200 | 55.8 | 88.4 | 95.3 | 0.705 |
| embed | 1200 | 51.2 | 79.1 | 93.0 | 0.662 |
| hybrid | 1200 | 58.1 | 86.0 | 95.3 | 0.721 |

**Answers (hybrid, 600-character chunks, top 3)**

| Answerer | Correct facts | When the right chunk was retrieved | False refusals | Refused unanswerable | Number grounding | Faithfulness (judge) |
|---|---|---|---|---|---|---|
| extractive (no LLM) | 62.8 | 67.5 | 10/43 | 6/8 | 100.0 | 77.4 |
| llama3.2:3b | 73.6 | 79.2 | 4/43 | 8/8 | 95.2 | 83.3 |
| qwen2.5:3b | 73.3 | 76.2 | 4/43 | 8/8 | 99.0 | 68.4 |
| **gemma3:4b** | **86.0** | **91.2** | **2/43** | **8/8** | **100.0** | **91.3** |

What I found:

- **Hybrid search wins.** Merging keyword and meaning search put the right chunk in the top 3 for 93% of questions, against 84% for keyword search alone. Embeddings found the paraphrased questions that BM25 missed, and BM25 kept the exact names and numbers that embeddings blurred.
- **Chunk size matters as much as the retriever.** At 1,200 characters every retriever got worse at ranking the right chunk first (hit@1 51 to 58), because each chunk mixes several topics.
- **Gemma 3 4B is the best answerer by a wide margin:** 86% of required facts, and it refused all 8 unanswerable questions while wrongly refusing only 2 answerable ones. All three models refused every unanswerable question; the copy-paste baseline answered 2 of them with unrelated sentences.
- **No model made up a number.** The only ungrounded numbers were "7 scorers" (the source says "Seven") and "45 tasks" (Llama and Qwen added 20 + 15 + 10 themselves).
- **The small judge isn't trustworthy.** It gave the copy-paste baseline, which can only repeat the sources, just 77% faithfulness, and it scored its own model (Qwen) lowest. Faithfulness from a 3B judge is a rough signal; number grounding and the fact checks are more reliable here.
- **Checking the metrics changed a number.** My first refusal and grounding checks counted citation lists like `[1, 3]` and list numbering as numbers, and they treated long answers with a hedge as refusals. I fixed both and rescored the saved answers with `rag-eval rescore` (no model calls). That moved Llama's grounding from 92.4 to 95.2 and Gemma's from 95.4 to 100.
<!-- results:end -->

## Run it

```
pip install -e ".[dev]"
python -m rag_eval ask "What license are the Amharic benchmark items released under?"
python -m rag_eval retrieval -o results                       # BM25, no model needed
ollama pull nomic-embed-text
python -m rag_eval retrieval -r bm25 embed hybrid -o results
python -m rag_eval answers -m extractive -m ollama:llama3.2:3b -r hybrid --judge ollama:qwen2.5:3b -o results
python -m rag_eval rescore results                            # re-score saved answers after a metric change
python -m rag_eval serve                                      # web demo on http://localhost:8000
pytest -q
```

`answers` is resumable: finished questions are saved to `results/answers__*.jsonl` and skipped on rerun. Embeddings are cached in `.cache/`.

## Docker and deployment

```
docker build -t rag-eval-assistant .
docker run -p 8000:8000 rag-eval-assistant
```

The web demo uses a local model when it can reach Ollama (`OLLAMA_HOST`, `RAG_MODEL`, e.g. `-e OLLAMA_HOST=http://host.docker.internal:11434`). On a free cloud host with no GPU it falls back to the extractive answerer and says so on the page. The image has no dependencies beyond Python, and it deploys as-is to any host that builds a Dockerfile and sets `PORT` (Render, Fly.io, Hugging Face Spaces).

## Layout

| File | Purpose |
|---|---|
| `rag_eval/corpus.py` | markdown cleaning, heading split, chunking |
| `rag_eval/retrieve.py` | BM25, embedding retriever with a disk cache, hybrid fusion |
| `rag_eval/generate.py` | prompt with numbered sources, LLM and extractive answerers |
| `rag_eval/metrics.py` | hit rate, MRR, correctness, refusals, number grounding, LLM-judge faithfulness |
| `rag_eval/evaluate.py` | runs the eval set, caches answers, summarizes |
| `rag_eval/server.py` | standard-library web demo and JSON API |
| `rag_eval/llm.py` | Ollama chat and embedding client over plain HTTP |
| `data/corpus/` | the five READMEs (snapshot of 2026-10-09) |
| `data/eval.jsonl` | the 51 eval questions |

## Limits

- 51 questions on 5 short documents: differences of a few points are noise.
- I wrote the questions after reading the documents, so their wording is closer to the text than a stranger's would be.
- Correctness is substring matching on required facts; a right answer phrased differently can be missed. A judge model checking its own family's answers may be lenient.
- One run per setting at temperature 0.

## License

MIT
