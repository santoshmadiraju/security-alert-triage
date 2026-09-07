# Security Alert Triage — Fine-Tuning DistilBERT on BGL Logs

Phase 1, Project 3 of an AI engineer learning roadmap. Supervised sequel to
[log-anomaly-detection](https://github.com/santoshmadiraju/log-anomaly-detection)
(Project 2): that project trained a char-level transformer from scratch to
score *unlabeled* HDFS logs by next-token surprise. This project fine-tunes a
**pretrained** Hugging Face model (DistilBERT) on a **labeled** classification
task instead — predicting whether a log line is a real alert or normal noise,
framed as the kind of triage a SOC analyst does by hand.

## Results

Held out, chronologically split, never touched until this single evaluation pass.

| metric | value |
|---|---|
| precision | 0.9978 |
| recall | 0.8702 |
| F1 | 0.9297 |
| ROC-AUC | 0.9390 |
| PR-AUC | 0.8904 |
| TP / FP / FN / TN | 40,273 / 89 / 6,005 / 665,828 |

Test set: 712,195 lines, 6.5% real alerts. PR-AUC of 0.89 against a ~0.065
random baseline is roughly a **13.7x lift**.

### vs. a naive baseline

"Predict alert iff log level is FATAL or FAILURE" — the cheapest possible
rule, since every real alert does carry one of those two levels:

| | baseline (level-only) | fine-tuned model |
|---|---|---|
| precision | 0.520 | 0.998 |
| recall | 1.000 | 0.870 |
| F1 | 0.684 | 0.930 |
| false positives | 42,690 | 89 |

The baseline never misses an alert, but at the cost of 42,690 false alarms
out of 665,917 normal lines — exactly the alert-fatigue problem this project
is framed around; no analyst can triage a queue that noisy. The fine-tuned
model cuts false positives by 99.8% while still catching 87% of real alerts.

## The key finding

**84.4% of test-set message templates never appeared anywhere in training**
(train has 31,380 unique masked templates, test has 9,899, and 8,351 of
those are entirely novel). This is the direct explanation for the gap
between validation performance (F1 0.986) and test performance (F1 0.930):
precision barely moved (0.9999 → 0.9978), but recall dropped from 97% to
87%. The model is very good at recognizing alert phrasing it has seen
before and conservative about phrasing it hasn't — a much more honest
failure mode than a random one, and the reason this project splits
**chronologically** rather than randomly. A random line-level split (common
in published log-anomaly-detection work) would have let the model see
nearly every test template during training, making the task deceptively
easy and hiding this generalization gap entirely.

## Method

- **Data:** [BGL](https://github.com/logpai/loghub) (Blue Gene/L
  supercomputer logs, loghub family — same corpus family as Project 2's
  HDFS data). 4,747,963 lines, 348,460 labeled alerts (7.34%). Split
  chronologically 70/15/15 by line order (train on the earliest data, test
  on the latest).
- **Input:** only the free-text log message, masked (`<NUM>`, `<HEX>`,
  `<NODE>` in place of numbers, hex addresses, and BGL node identifiers) and
  registered as real vocabulary tokens so each placeholder is one clean
  token instead of being fragmented by WordPiece. `type`/`component`/`level`
  are deliberately excluded from the input — every alert does have
  `level ∈ {FATAL, FAILURE}`, but that alone is a weak signal (FATAL-level
  lines are 59% normal), so the model has to learn from message content,
  not lean on a metadata shortcut. `level` is instead used for the baseline
  comparison above.
- **Model:** `distilbert-base-uncased` fine-tuned with HF `Trainer` for
  binary classification, 300,000-row random subsample of the 3.3M-row train
  split, `max_length=64`, batch size 32, lr 2e-5, fp16.
- **Training:** 2 of 3 planned epochs completed (see Hardware note below).
  Epoch 1 → epoch 2 validation F1 was 0.986 → 0.986 — already converged, so
  the missing 3rd epoch is not believed to be a meaningful gap.

## Hardware note

Training on this laptop (RTX 3050 Ti, 4.3GB VRAM) repeatedly hit thermal
throttling (GPU SM clock pinned at 255MHz of a possible 2100MHz at 89°C)
and, on sustained load, the machine hard-rebooted outright — confirmed via
Windows Reliability Monitor as unexpected shutdowns with no blue screen,
consistent with a hardware-level thermal/power protection cutoff rather
than a software crash. This happened enough times that the 3rd epoch was
abandoned in favor of evaluating the already-converged 2-epoch checkpoint.
Lesson carried forward: don't run multi-hour unattended GPU jobs on this
machine without addressing cooling first.

## Files

| File | Purpose |
|---|---|
| `download_data.py` | fetch + unpack BGL from loghub/Zenodo |
| `prepare_data.py` | parse raw BGL lines, mask, chronological split |
| `report.py` | Level/label crosstab, template-overlap check |
| `dataset.py` | HF Dataset / tokenizer wrapper, mask-token registration |
| `train.py` | fine-tuning loop (HF Trainer) |
| `evaluate.py` | held-out test evaluation |
| `baseline_check.py` | FATAL/FAILURE-only baseline for comparison |
| `analyze_errors.py` | false-negative template-novelty analysis (not yet run — see below) |

## Known gaps / possible follow-ups

- Error analysis on the 6,005 false negatives (does the miss rate skew
  toward unseen templates, as the val/test gap suggests?) — deferred due to
  the hardware issues above, not yet run.
- Only 2 of 3 epochs completed, and training used a 300k-row subsample of
  the 3.3M available train rows — both were practical concessions to this
  laptop's thermal limits, not principled stopping points. Worth revisiting
  if cooling gets fixed.
- No deployed demo yet (Project 2 had a live Streamlit app for this).
