from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import (
    average_precision_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding

HERE = Path(__file__).parent
TOKENIZED_DIR = HERE / "data" / "tokenized"
CHECKPOINT = HERE / "runs" / "distilbert-v1" / "checkpoint-18750"


def main():
    tokenized = load_from_disk(str(TOKENIZED_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))
    model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(HERE / "runs" / "eval_tmp"), per_device_eval_batch_size=64, report_to=[]),
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    preds = trainer.predict(tokenized["test"])
    logits, labels = preds.predictions, preds.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    pred_labels = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels, average="binary")
    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()

    print("=== HELD-OUT TEST RESULTS (checkpoint-18750, 2 epochs) ===")
    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print(f"roc_auc={roc_auc_score(labels, probs):.4f} pr_auc={average_precision_score(labels, probs):.4f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")


if __name__ == "__main__":
    main()
