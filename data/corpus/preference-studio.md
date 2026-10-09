# Preference Studio

![ci](https://github.com/Beki-hck/preference-studio/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![stack](https://img.shields.io/badge/stack-FastAPI%20%C2%B7%20React%20%C2%B7%20PostgreSQL-informational)

A web app for collecting **human preference data**, the kind used to train and evaluate language models with RLHF and DPO. Raters see a prompt and two responses, judge them against a rubric, and pick the better one on a 5-point scale. The app tracks how much raters agree, flags unreliable raters, and exports clean `{prompt, chosen, rejected}` pairs ready for training.

Local models can join in as raters too (LLM-as-judge, through [evalforge](https://github.com/Beki-hck/evalforge)), so you can measure how well a model's judgments match people's.

![demo](docs/demo.gif)

## Why I built it

Preference data is only useful if it's trustworthy. A single rater clicking quickly, a rater who always picks whichever answer is on the left, or a rubric nobody reads can quietly poison a dataset. Labeling teams handle this with overlap (several raters per pair), gold questions with known answers, and agreement statistics. This project implements those practices end to end.

## Features

**Rating**
- Side-by-side comparison with a configurable rubric (helpfulness, accuracy, safety and clarity by default), a 5-point overall scale and an optional rationale.
- **Randomized left/right order** for every assignment, stored with the label, so position bias can be measured and doesn't leak into the data.
- **Overlap scheduling:** each pair goes to N raters (configurable), least-labeled pairs first, so coverage grows evenly.
- **Gold pairs** with a known better answer are shown to every rater as attention checks.
- Keyboard shortcuts (1–5 to rate, Ctrl+Enter to submit) and time-per-label tracking.

**Quality metrics** (all implemented from scratch in [`metrics.py`](backend/prefstudio/metrics.py) and tested against published worked examples)
- Krippendorff's α (nominal on A/tie/B, and interval on the 5-point scale). It handles missing labels, which is the normal case when raters label different subsets.
- Fleiss' κ, pairwise Cohen's κ and percent agreement.
- Per rater: agreement with the other raters' majority, gold accuracy, **left-side pick rate** (position bias), tie rate and median time.
- Per pair: label split, majority, agreement, mean score, with a filter for disagreements.

**Export**
- DPO JSONL (`prompt`, `chosen`, `rejected`, `meta`) in the format TRL's `DPOTrainer` reads, with filters for minimum labels, agreement and margin (to drop near-ties) and for excluding raters.
- Raw labels JSONL with criteria, rationale, order shown and timing.

**AI raters**
- `prefstudio ai-rate` labels a project with any `ollama:*`, `openai:*` or `anthropic:*` model.
- Each pair is judged **in both orders**. If the verdict flips when the order flips, it's recorded as a tie and counted, which gives an order-consistency score for the model.

## Quick start

```bash
git clone https://github.com/Beki-hck/preference-studio
cd preference-studio

# Backend (SQLite by default, no database setup needed)
pip install -e "backend[dev]"

# Frontend
cd frontend && npm install && npm run build && cd ..

prefstudio serve            # open http://localhost:8000
```

Click **Load sample project** to get 40 hand-written pairs (10 of them gold), sign in with any name and start rating. Open the project in a second browser with a different name to see agreement numbers appear.

**With PostgreSQL (Docker):**

```bash
docker compose up --build   # app on http://localhost:8000, data in Postgres
```

Or point it at any Postgres database: `DATABASE_URL=postgresql://user:pass@host/db prefstudio serve`.

**Frontend development:** run `prefstudio serve` and `npm run dev` in `frontend/` (Vite proxies `/api` to port 8000).

## Command line

```bash
prefstudio seed                                   # create the sample project
prefstudio ai-rate 1 -m ollama:llama3.2:3b -m ollama:qwen2.5:3b -o results/ai.json
prefstudio stats 1                                # agreement table in the terminal
prefstudio export 1 dpo -o dpo.jsonl --min-labels 3 --min-agreement 0.66
prefstudio export 1 labels -o labels.jsonl
```

Pairs are imported as JSONL, one object per line:

```json
{"prompt": "...", "response_a": "...", "response_b": "...", "model_a": "optional", "model_b": "optional", "gold": "A"}
```

## Results

### Two local models as raters

I had two small local models label the 40 sample pairs with `prefstudio ai-rate` (Ollama on a Windows PC, temperature 0). Each pair was judged twice, once in each order:

| Rater | Order consistency | Picks first-shown answer | Gold accuracy (10 pairs) | Tie rate |
|---|---|---|---|---|
| llama3.2:3b | 55% | 72% | 70% | 45% |
| qwen2.5:3b | 92% | 54% | 90% | 55% |

- **llama3.2:3b has a strong position bias.** Its verdict flipped in 18 of 40 pairs when the two answers swapped places, and it preferred whichever answer came first 72% of the time. qwen2.5:3b flipped in only 3 pairs and was close to the unbiased 50%.
- **The two models barely agree with each other:** Cohen's κ = 0.21 ("fair") on the 30 non-gold pairs, Krippendorff's α = 0.19.
- Order-flipped verdicts are recorded as ties, which is why tie rates are high and why a biased judge loses gold accuracy.

The takeaway: a small model shouldn't be trusted as a preference judge without checking position consistency first, and even a consistent one needs human labels to compare against.

Reproduce it with:

```bash
prefstudio seed
prefstudio ai-rate 1 -m ollama:llama3.2:3b -m ollama:qwen2.5:3b -o results/ai.json
prefstudio stats 1
```

### Metric validation with simulated raters

To check that the agreement metrics behave correctly, [`scripts/simulate_raters.py`](scripts/simulate_raters.py) gives 400 pairs a hidden true score and simulates three raters who report it with a given probability ("skill") and guess randomly otherwise. Agreement should fall to about zero as skill falls to zero, and it does:

| Rater skill | Percent agreement | Krippendorff α (A/tie/B) | α (5-point) | Fleiss κ |
|---|---|---|---|---|
| 1.00 | 100% | 1.00 | 1.00 | 1.00 |
| 0.90 | 88% | 0.81 | 0.84 | 0.81 |
| 0.75 | 74% | 0.59 | 0.60 | 0.59 |
| 0.50 | 54% | 0.26 | 0.27 | 0.26 |
| 0.25 | 40% | 0.06 | 0.08 | 0.06 |
| 0.00 | 36% | 0.00 | -0.02 | 0.00 |

A simulated rater who guesses "left" whenever unsure is caught by the position-bias metric: they pick the left side 76% of the time, versus 51% for careful raters. CI runs this script on every push.

The dashboard below shows a project labeled by four simulated raters (`python scripts/simulate_raters.py --demo`). The weakest rater, `sim-dawit`, is flagged for low gold accuracy and a 90% left-side pick rate.

![dashboard](docs/dashboard.png)

## How it works

```
frontend/  React + TypeScript (Vite). Pages: sign in, projects, rate, dashboard.
backend/prefstudio/
  api.py        FastAPI routes under /api; also serves the built frontend
  models.py     SQLAlchemy tables: projects, items, raters, labels
  service.py    assignment (overlap + gold), labeling, stats and export queries
  analysis.py   per-rater, per-pair and project statistics; DPO pair selection
  metrics.py    Krippendorff's alpha, Fleiss' and Cohen's kappa, percent agreement
  ai_rater.py   LLM-as-judge rater using evalforge model clients
  cli.py        prefstudio serve | seed | ai-rate | stats | export
```

Labels are stored in a canonical form: score −2 (A much better) to +2 (B much better), whatever side each response was shown on, plus a flag for whether the order was swapped. That keeps exports simple and makes position bias measurable.

## Tests

```bash
cd backend && pytest                    # 40 tests
TEST_DATABASE_URL=postgresql://... pytest   # same suite on PostgreSQL
cd frontend && npm test
```

The metric tests check results against published examples: Krippendorff's 2011 worked example (α = 0.743 nominal, 0.815 ordinal, 0.849 interval), the Fleiss' κ example from Wikipedia (κ = 0.210), and a Cohen's κ example (κ = 0.40). CI runs the backend suite on both SQLite and PostgreSQL 16.

## Limitations

- Sign-in is by name only, with no passwords. It's meant for a trusted team or a demo, not a public labeling site.
- The sample pairs are hand-written for the demo, not drawn from a real model.
- There are no per-rater assignments or rater-specific projects yet.

## License

MIT
