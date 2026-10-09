# notes-to-fhir

Turns clinical notes into FHIR R4 records with a free local model (Ollama), and measures field-level accuracy against gold labels.

All notes are synthetic. No real patient data is used.

## What it does

1. **Notes with gold labels.** 50 template notes (seeded generator, 4 writing styles) and 20 hand-written notes kept as a held-out set. The rule baseline was frozen in git before the hand-written notes existed, so they test it fairly.
2. **Traps.** Each note hides cases that trip up extraction: negated conditions ("denies chest pain"), family history, resolved conditions, stopped drugs, dose changes, drugs started today, brand names, earlier lab values, and units to convert (lb, °F, mmol/L).
3. **Extraction.** A model reads the note and returns JSON (patient, conditions, medications, allergies, observations). Ollama's structured output keeps it to the schema; replies are parsed defensively (code fences, prose, truncation).
4. **Coding and FHIR.** Names map to SNOMED CT, ICD-10-CM, RxNorm and LOINC, and each note becomes a FHIR R4 Bundle validated with `fhir.resources`.
5. **Scoring.** Field-level precision, recall and F1 against the gold label, with bootstrap 95% confidence intervals, per-section scores, per-trap counts and attribute accuracy.

## Results (2026-10-09, laptop CPU, Ollama)

| Model | Field F1, all notes | Template notes | Hand-written notes | Seconds/note |
|---|---|---|---|---|
| Rule baseline | **97.4** [95.8–98.7] | 100.0 | 88.7 | 0.0 |
| Llama 3.2 3B | **94.0** [92.6–95.3] | 95.5 | **89.4** | 4.2 |
| Qwen 2.5 3B | **90.1** [88.0–91.9] | 91.7 | 85.3 | 4.2 |
| Gemma 3 4B (no format, see below) | **78.8** [77.3–80.3] | 79.2 | 77.6 | 6.8 |

Full tables (per section, per trap, per attribute) are in [benchmarks/2026-10-09/leaderboard.md](benchmarks/2026-10-09/leaderboard.md), with every model reply in the `predictions.jsonl` files. `notes-to-fhir report -o benchmarks/2026-10-09` rescores them without a model.

What the numbers say:

- **The rules win on the notes they were written for, not on new ones.** The baseline scores 100 on template notes but 88.7 on the hand-written set, where Llama 3.2 3B edges it (89.4).
- **Small models read negation well and status changes badly.** All three skip negated conditions (40/40) and family history, but Llama marked only 6 of 19 stopped drugs as stopped, and Qwen used the new dose in only 11 of 22 dose changes (its dose accuracy was 68%).
- **Gemma 3 4B can't use Ollama's JSON modes here.** With the schema, and with plain JSON mode, it answered `{}` for 67 of 70 notes. With no format (`--no-format`) it wrote real JSON, but returned observations as a name-to-number dictionary without units in 69 of 70 notes, so its observation score is 0 and its schema-valid rate is 1%. Its other sections are strong (medications 97.0, allergies 98.8).

Medical codes were checked against public terminology services (NLM RxNav and Clinical Tables for RxNorm and ICD-10-CM, tx.fhir.org for SNOMED CT and LOINC) by `scripts/verify_codes.py`: all codes matched their official names except one lookup that timed out (SNOMED 47625008, intravenous route). See [benchmarks/2026-10-09/code-check.txt](benchmarks/2026-10-09/code-check.txt).

## Run it

```
pip install -e ".[dev]"
notes-to-fhir list                       # note sets and trap counts
notes-to-fhir show hw-01 --gold          # one note and its gold label
notes-to-fhir run -m baseline -m ollama:llama3.2:3b -o results
notes-to-fhir run -m ollama:gemma3:4b --no-format -o results
notes-to-fhir convert note.txt -m ollama:llama3.2:3b -o bundle.json
pytest -q
```

`run` is resumable: it skips notes already in `predictions.jsonl`. `-m anthropic:<model>` works too if `ANTHROPIC_API_KEY` is set.

## Limits

- 70 notes is small; the confidence intervals show how much that matters.
- The terminology tables cover the conditions, drugs, allergies and labs in these notes, not all of medicine.
- One run per model at temperature 0, on one machine.
