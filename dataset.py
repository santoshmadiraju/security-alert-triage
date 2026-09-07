from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from prepare_data import mask_content  # reuse the same masking function

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "processed"
TOKENIZED_DIR = HERE / "data" / "tokenized"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64          # covers ~p99 of message lengths, see project log
TRAIN_SUBSAMPLE = 300_000  # None = use the full 3.3M rows
SEED = 42

def load_raw():
    return load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "val": str(DATA_DIR / "val.jsonl"),
            "test": str(DATA_DIR / "test.jsonl"),
        },
    )


def add_masked_text(batch):
    return {"text": [mask_content(t) for t in batch["text_raw"]]}


def make_tokenize_fn(tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
    return tokenize

def build_dataset():
    raw = load_raw()

    if TRAIN_SUBSAMPLE is not None:
        raw["train"] = raw["train"].shuffle(seed=SEED).select(range(TRAIN_SUBSAMPLE))
        print(f"Subsampled train to {TRAIN_SUBSAMPLE:,} rows")

    raw = raw.map(add_masked_text, batched=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(["<NUM>", "<HEX>", "<NODE>"])

    tokenized = raw.map(make_tokenize_fn(tokenizer), batched=True)

    # Trainer expects a column literally named "labels"
    tokenized = tokenized.rename_column("label", "labels")

    # keep only what training needs, drop the rest to save memory/disk
    keep_cols = {"input_ids", "attention_mask", "labels"}
    drop_cols = [c for c in tokenized["train"].column_names if c not in keep_cols]
    tokenized = tokenized.remove_columns(drop_cols)

    tokenized.set_format("torch")
    return tokenized, tokenizer

def main():
    tokenized, tokenizer = build_dataset()
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(TOKENIZED_DIR))
    tokenizer.save_pretrained(str(TOKENIZED_DIR / "tokenizer"))

    print("\nSaved tokenized dataset to", TOKENIZED_DIR)
    for split in tokenized:
        print(f"  {split}: {len(tokenized[split]):,} rows")
    print("\nSample row:")
    print(tokenized["train"][0])


if __name__ == "__main__":
    main()
