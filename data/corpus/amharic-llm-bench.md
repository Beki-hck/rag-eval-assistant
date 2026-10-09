# amharic-llm-bench

![tests](https://github.com/Beki-hck/amharic-llm-bench/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

A small Amharic benchmark for large language models. It covers reading comprehension, multiple choice, translation in both directions and summarization, with scoring built for how Amharic is actually written. It runs free local models through [Ollama](https://ollama.com) and is built on [evalforge](https://github.com/Beki-hck/evalforge), my LLM evaluation harness.

```bash
amharic-bench run -m ollama:gemma3:4b -m ollama:qwen2.5:3b
```

## Why

Amharic has more than 50 million speakers, but most open models are tuned and evaluated almost entirely on English. Existing Amharic datasets such as FLORES-200, IrokoBench and AmQA are useful, but they are public and may have leaked into training data. Their scoring also ignores problems specific to Amharic:

- **Spelling variants.** ሀ/ሐ/ኀ, ሰ/ሠ, አ/ዐ and ጸ/ፀ sound the same and are used interchangeably ("ሐኪም" and "ሀኪም" are the same word). A string comparison marks a correct answer wrong.
- **Affixes.** "በአዲስ አበባ" means "in Addis Ababa". Word-level metrics treat it as a different word from "አዲስ አበባ".
- **Wrong-script answers.** A model can answer in English, in Latin transliteration, or in a mix of scripts. A loose metric still gives it partial credit for shared names and numbers.

## What's in it

All 113 items were written for this benchmark, so no model has seen them in training, and I reviewed every one as an Amharic speaker. The passages are fictional.

| Suite | Items | Task | Score |
|---|---|---|---|
| `am_qa` | 26 | Read a short Amharic passage, answer a question in a few words (lookup, numbers, inference) | % correct |
| `am_mcq` | 27 | Multiple choice in Amharic: culture, history, geography, vocabulary, grammar, arithmetic | % correct |
| `en_am` | 25 | Translate English to Amharic | chrF |
| `am_en` | 25 | Translate Amharic to English (the same 25 sentence pairs reversed) | chrF |
| `am_sum` | 10 | Summarize an Amharic news-style passage in one Amharic sentence | chrF |

Both translation directions use the same sentence pairs, so the gap between them reflects direction, not content.

## How scoring works

- **Normalization** (`normalize.py`): folds the homophone letter families, converts Ethiopic punctuation (። ፣ ፧) and Ethiopic numerals (፲፱፻፹፯ → 1987), then lowercases and collapses whitespace. Every comparison goes through it.
- **QA** (`am_qa` scorer): passes if the normalized gold answer appears inside a short reply, which handles prefixes like በ- and ለ-, or if word-overlap F1 is at least 0.5. A long reply can't pass by pasting the passage back.
- **chrF**: character n-gram F-score, which handles Amharic affixes much better than word-based BLEU. I re-implemented sacreBLEU's chrF, and the tests check it against sacreBLEU on fixed and 200 random strings (exact match).
- **Script check**: for Amharic-output tasks, a reply whose letters are less than 50% Fidel scores 0. Lead-in lines like "Here is the translation:" are stripped first.
- **Script drift**: counts Amharic-output answers that slip into a third script (Chinese, Korean, Cyrillic and so on).
- **Confidence intervals**: every score has a 95% bootstrap interval over items. With 25 items, differences of a few points are noise.
- **Tested data**: the tests check that every gold answer gets full marks from its own scorer, that the multiple-choice options are distinct, and that the correct letters are spread across A–D.

## Results

Run on 2026-10-08 on my PC with Ollama, at temperature 0 with answers capped at 512 tokens. QA and MCQ are % correct; the other suites are mean sentence chrF (0–100). Brackets are 95% bootstrap intervals.

<!-- results:start -->
| Model | Average | QA | MCQ | EN→AM | AM→EN | Summary | Answered in Fidel | Drifted to other scripts |
|---|---|---|---|---|---|---|---|---|
| gemma3:4b | **53.6** | 92.3% [81–100] | 63.0% [44–81] | 7.4 [4–11] | 68.7 [61–76] | 36.6 [26–47] | 60% | 34% |
| qwen2.5:3b | **15.8** | 7.7% [0–19] | 18.5% [4–33] | 4.5 [3–7] | 16.0 [12–21] | 32.4 [25–39] | 83% | 23% |
| llama3.2:3b | **14.8** | 0.0% [0–0] | 22.2% [7–37] | 4.8 [4–6] | 17.0 [14–20] | 29.9 [19–40] | 100% | 0% |
<!-- results:end -->

Every answer is in [`benchmarks/2026-10-08/`](benchmarks/2026-10-08/), and `details.html` there shows each prompt and reply.

### What I found

1. **Gemma reads Amharic but can't write it.** Gemma 3 4B answered 92% of the reading questions and translated Amharic to English at chrF 68.7. Translating English to Amharic, it scored 7.4.
2. **It writes by drifting between scripts.** 34% of Gemma's Amharic outputs slipped into other scripts and languages mid-sentence. For "The market opens early on Saturday morning." it wrote `በ saturday ቀን ከ temprano መ abrió።`, which is Amharic, English and Spanish together. For "Please drink plenty of water when you have a fever." it answered entirely in Chinese: `请发烧时大量饮水。`
3. **Fidel is not the same as Amharic.** Llama 3.2 3B wrote 100% Fidel script, but its translations are made-up words (`ሳላት ሳከም ልካም ...`, chrF 4.8). A check that only looks at the script would rank it best, which is why chrF against human references matters.
4. **The 3B models are at chance on multiple choice.** Llama scored 22% and Qwen 19%, against 25% for random guessing. Llama only ever picked A or C (11 times each). Qwen gave no option letter at all on 13 of 27 questions.
5. **Llama 3.2 3B echoes instead of answering.** It scored 0/26 on QA, and 12 of its replies began by repeating the "ጽሑፍ፦" or "ጥያቄ፦" label from the prompt.

Gemma's lead is far outside the confidence intervals. Llama and Qwen can't be separated on this data.

### Limitations

- 113 items is small, so read the intervals, not just the averages.
- chrF misses meaning errors in single words. Gemma translated "Saturday" as "Friday" and still scored 61.9 on that sentence.
- Each summary has one reference, so chrF on `am_sum` is a rough signal.
- Answer key changes are recorded below. After the first full run I read every QA failure for all three models and accepted two answers that were correct but missing from the key.
- The items were drafted with AI assistance. On 2026-10-08 I reviewed all 113, as an Amharic speaker, and found nothing to correct. `python scripts/export_review.py` writes the review sheet I used.

## Quick start

```bash
git clone https://github.com/Beki-hck/amharic-llm-bench
cd amharic-llm-bench
pip install -e .            # also installs evalforge from GitHub

ollama pull gemma3:4b
amharic-bench run -m ollama:gemma3:4b      # writes results/leaderboard.html
amharic-bench list                         # the suites
amharic-bench normalize "ሐሙስ ፲፱፻፹፯ ዓ.ም።"  # -> ሀሙስ 1987 አ.ም.
```

On Windows, `python -m amharic_bench.cli run ...` works if the `amharic-bench` command isn't on your PATH. Any evalforge model works, including `openai:<model>` and `anthropic:<model>` if you set an API key.

After changing a scorer or the answer key, re-score saved answers without calling the model again:

```bash
amharic-bench rescore results/*__*.json -o results
```

## Project layout

```
src/amharic_bench/
  normalize.py     homophone folding, Ethiopic numerals, script detection
  metrics.py       chrF (sacreBLEU-compatible) and token F1
  scorers.py       am_qa, am_choice, chrf: registered into evalforge
  models.py        evalforge clients with an output-length cap
  leaderboard.py   per-model table, bootstrap CIs, script and drift rates
  cli.py           run / rescore / leaderboard / list / normalize
  suites/*.jsonl   generated by scripts/build_suites.py
scripts/
  build_suites.py  the benchmark data (edit here, then regenerate)
  export_review.py review sheet for a native speaker
```

## Answer key changes

- 2026-10-08: `qa-12` also accepts "በውሃ ተጥለቀለቁ" ("they flooded"), and `qa-15` also accepts "ውድድሩ" ("the race"). Both are correct answers whose last letter changes form with the suffix, so substring matching missed them.

## License

Code is MIT. The benchmark items are released under CC BY 4.0.
