# label-auditor

Finds examples in a dataset whose label is probably wrong, so a human can re-check the few that matter instead of all of them.

It trains several models with cross-validation (each example is scored only by models that never saw its label), then compares four detectors:

| Detector | Rule |
|---|---|
| `argmax` | flag when the averaged model's top class differs from the label (baseline) |
| `ensemble_vote` | flag when most of the models (logistic regression, k-NN, random forest) disagree with the label |
| `confident_learning` | per-class thresholds and the confident joint (Northcutt et al., 2021), written from scratch in `detect.py` |
| `cleanlab` | the `cleanlab` library's `find_label_issues`, used as a reference to check the from-scratch version |

## Results

To measure a detector you need to know which labels are wrong, so the benchmark flips a known share of labels (5%, 10%, 20%), either uniformly to any other class or systematically (class k becomes k+1, which mimics a confused annotator), and checks what each detector catches. 3 seeds per setting, 5-fold cross-validation. Full table: [benchmarks/2026-10-08/results.md](benchmarks/2026-10-08/results.md).

Averaged over all noise rates and types:

| Dataset | Detector | Precision | Recall | F1 |
|---|---|---|---|---|
| iris (3 flower species, 150) | argmax | 0.68 | 0.99 | 0.80 |
| | ensemble_vote | 0.67 | 0.98 | 0.79 |
| | confident_learning (ours) | **0.93** | 0.84 | 0.88 |
| | cleanlab | 0.85 | 0.91 | 0.88 |
| digits (8x8 images, 1,797) | argmax | 0.78 | 1.00 | 0.87 |
| | ensemble_vote | 0.78 | 1.00 | 0.88 |
| | confident_learning (ours) | **0.99** | 0.78 | 0.87 |
| | cleanlab | 0.94 | 0.92 | **0.93** |

What this shows:

- Plain disagreement catches almost every flipped label but about a quarter of what it flags is a false alarm. On iris, most false alarms are real versicolor/virginica flowers that sit in the overlap between the two species.
- Confident learning trades recall for precision: when it flags something it is right 93 to 99% of the time, because each class gets its own threshold.
- The from-scratch version matches cleanlab's F1 on iris and trails it on digits (0.87 vs 0.93). cleanlab additionally calibrates the confident joint and prunes by estimated noise rate per class pair, which recovers more of the flips. That is the next thing to add.
- On the original, unmodified labels confident learning flags 0 of 150 iris and 0 of 1,797 digits. The ranked list still surfaces the most borderline examples (for example `digits-5`, labeled 5, which the models read as a 9) for a person to look at.

Limits: the noise is synthetic, so these numbers describe how well each detector recovers flips we planted, not the real error rate of either dataset.

## Use

```
pip install -e ".[dev]"
python -m label_auditor bench iris digits -o benchmarks/today
python -m label_auditor audit digits --top 25 -o audit.csv
python -m label_auditor audit csv:my_data.csv          # last column is the label
```

Image folders (one subfolder per class, e.g. bird species) are embedded with a frozen pretrained ResNet-18 first:

```
pip install -e ".[images]"
python -m label_auditor audit folder:path/to/species_photos --top 50 -o audit.csv
```

`audit.csv` lists each suspect with its given label, the suggested label and both probabilities.

## Layout

- `data.py` loads datasets and injects reproducible label noise
- `embed.py` turns an image folder into feature vectors (optional, needs torch)
- `probs.py` out-of-fold predicted probabilities for each model
- `detect.py` the four detectors
- `metrics.py` precision, recall, F1, precision at k
- `cli.py` the `audit` and `bench` commands

Runs on CPU with free open-source libraries only; the iris + digits benchmark takes under two minutes.
