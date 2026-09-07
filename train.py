"""
Phase 1 Project 3 - fine-tune DistilBERT on BGL security alert triage, using HF Trainer.
"""
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).parent
TOKENIZED_DIR = HERE / "data" / "tokenized"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = HERE / "runs" / "distilbert-v1"

def load_data():
    tokenized = load_from_disk(str(TOKENIZED_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZED_DIR / "tokenizer"))
    return tokenized, tokenizer

def load_model(tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    return model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
    }

def make_training_args():
    return TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),   # halves memory use on your 4.3GB GPU
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],   # no wandb/etc, just local logs
    )

def main():
    tokenized, tokenizer = load_data()
    model = load_model(tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=make_training_args(),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nFinal validation metrics:")
    print(trainer.evaluate())

    final_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nSaved final model to {final_dir}")


if __name__ == "__main__":
    main()
