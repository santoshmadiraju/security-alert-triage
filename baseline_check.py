"""
Cheap baseline for comparison: predict alert iff level in {FATAL, FAILURE}.
Uses data/processed/test.jsonl directly (not the tokenized dataset), since
that's where the `level` field still lives.
"""
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

HERE = Path(__file__).parent
TEST_PATH = HERE / "data" / "processed" / "test.jsonl"

y_true, y_pred = [], []
with open(TEST_PATH, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        y_true.append(rec["label"])
        y_pred.append(1 if rec["level"] in ("FATAL", "FAILURE") else 0)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("=== FATAL/FAILURE-only baseline, test set ===")
print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
