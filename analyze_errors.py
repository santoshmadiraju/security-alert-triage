"""
Phase 1 Project 3 - error analysis on the held-out test set.

Hypothesis: false negatives skew toward message templates never seen in
training, rather than being spread randomly - this checks that directly.
"""
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)

from prepare_data import mask_content

HERE = Path(__file__).parent
TOKENIZED_DIR = HERE / "data" / "tokenized"
PROCESSED_DIR = HERE / "data" / "processed"
CHECKPOINT = HERE / "runs" / "distilbert-v1" / "checkpoint-18750"

def get_predictions():
    tokenized = load_from_disk(str(TOKENIZED_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))
    model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(HERE / "runs" / "eval_tmp"),
                                per_device_eval_batch_size=64, report_to=[]),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    preds = trainer.predict(tokenized["test"])
    return np.argmax(preds.predictions, axis=-1)

def load_train_templates():
    templates = set()
    with open(PROCESSED_DIR / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            templates.add(mask_content(rec["text_raw"]))
    return templates

def load_test_records():
    with open(PROCESSED_DIR / "test.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    print("Loading train templates (reads all 3.3M train rows once)...")
    train_templates = load_train_templates()
    print(f"  {len(train_templates):,} unique train templates")

    print("Loading test records...")
    test_records = load_test_records()

    print("Running predictions on test...")
    pred_labels = get_predictions()
    assert len(pred_labels) == len(test_records), "row count mismatch - check split order assumption"

    fn_unseen = fn_seen = 0
    tp_unseen = tp_seen = 0
    fn_examples = []

    for rec, pred in zip(test_records, pred_labels):
        template = mask_content(rec["text_raw"])
        unseen = template not in train_templates

        if rec["label"] == 1 and pred == 0:          # false negative
            if unseen:
                fn_unseen += 1
                note = "unseen template"
            else:
                fn_seen += 1
                note = "SEEN in train - unexpected miss"
            if len(fn_examples) < 8:
                fn_examples.append((rec["alert_type"], note, rec["text_raw"]))
        elif rec["label"] == 1 and pred == 1:         # true positive
            tp_unseen += unseen
            tp_seen += not unseen

    fn_total, tp_total = fn_unseen + fn_seen, tp_unseen + tp_seen
    print(f"\n=== False negatives (missed alerts): {fn_total:,} ===")
    print(f"  unseen-template: {fn_unseen:,} ({fn_unseen/fn_total:.1%})")
    print(f"  seen-in-train:   {fn_seen:,} ({fn_seen/fn_total:.1%})")

    print(f"\n=== True positives (caught alerts): {tp_total:,} ===")
    print(f"  unseen-template: {tp_unseen:,} ({tp_unseen/tp_total:.1%})")
    print(f"  seen-in-train:   {tp_seen:,} ({tp_seen/tp_total:.1%})")

    print("\n=== Sample false negatives ===")
    for alert_type, note, text in fn_examples:
        print(f"  [{alert_type}] ({note}) {text[:120]}")


if __name__ == "__main__":
    main()
