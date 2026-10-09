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
Measured 2026-10-09. Raw rows in [`benchmarks/2026-10-09/`](benchmarks/2026-10-09/).

**Retrieval (BM25, 43 answerable questions)**

| Chunk size | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| 300 | 138 | 69.8 | 83.7 | 88.4 | 0.775 |
| 600 | 77 | 65.1 | 83.7 | 90.7 | 0.749 |
| 1200 | 50 | 55.8 | 88.4 | 95.3 | 0.705 |

**Answers (BM25, 600-character chunks, top 3)**

| Answerer | Correct facts | When the right chunk was retrieved | False refusals | Refused unanswerable | Number grounding |
|---|---|---|---|---|---|
| extractive (no LLM) | 60.5 | 69.4 | 10/43 | 7/8 | 100.0 |

_Pending: embedding and hybrid retrieval, and answers from llama3.2:3b, qwen2.5:3b and gemma3:4b with a faithfulness judge (run on my PC with Ollama)._

What the numbers show so far:

- **Chunk size is a trade-off.** Small chunks put the right passage first more often (Hit@1 69.8 at 300 characters vs 55.8 at 1,200), big chunks are more likely to contain it somewhere in the top 5 (95.3 vs 88.4), but each one hands the model more irrelevant text.
- **Keyword search misses paraphrases.** The 4 questions BM25 never finds in the top 5 at 600 characters use words the README doesn't, like "third-party Python packages" for "standard library only" and "password login" for "sign-in is by name only". That is the gap meaning search should close.
- **Copying sentences can't read tables.** Most of the extractive baseline's misses are numbers that live in markdown tables (Gemma's 53.6 average, cleanlab's 0.93 F1): the right chunk was retrieved but no single sentence matched the question. It never invents a number (grounding 100%), but it answered one unanswerable question with an unrelated sentence.
<!-- results:end -->

## Run it

```
pip install -e ".[dev]"
python -m rag_eval ask "What license are the Amharic benchmark items released under?"
python -m rag_eval retrieval -o results                       # BM25, no model needed
ollama pull nomic-embed-text
python -m rag_eval retrieval -r bm25 embed hybrid -o results
python -m rag_eval answers -m extractive -m ollama:llama3.2:3b -r hybrid --judge ollama:qwen2.5:3b -o results
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
