"""
Reads the processed train/val/test.jsonl and prints:
  1. split sizes by label
  2. Level vs label crosstab (is Level a giveaway for the label?)
  3. template-overlap check: what fraction of test-set message templates
     were never seen anywhere in train (recomputes the masked template from
     text_raw using the same mask_content() prepare_data.py uses).

Run this once prepare_data.py has finished writing all three files.
"""
import json
from collections import Counter
from pathlib import Path

from prepare_data import mask_content

OUT_DIR = Path(__file__).parent / "data" / "processed"


def load_stats(path: Path, collect_templates: bool):
    level_counts = Counter()
    label_counts = Counter()
    templates = set() if collect_templates else None
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            level_counts[(rec["level"], rec["label"])] += 1
            label_counts[rec["label"]] += 1
            if collect_templates:
                templates.add(mask_content(rec["text_raw"]))
    return n, level_counts, label_counts, templates


def main():
    all_level_counts = Counter()
    for split in ("train", "val", "test"):
        path = OUT_DIR / f"{split}.jsonl"
        collect = split in ("train", "test")
        n, level_counts, label_counts, templates = load_stats(path, collect)
        print(f"[{split}] n={n:,} label=0:{label_counts.get(0,0):,} label=1:{label_counts.get(1,0):,}")
        all_level_counts.update(level_counts)
        if split == "train":
            train_templates = templates
        elif split == "test":
            test_templates = templates

    print("\n[Level vs label crosstab] - is Level a giveaway for the label?")
    for lvl in sorted({lvl for (lvl, _lab) in all_level_counts}):
        n0 = all_level_counts.get((lvl, 0), 0)
        n1 = all_level_counts.get((lvl, 1), 0)
        total = n0 + n1
        frac_alert = n1 / total if total else 0.0
        print(f"  level={lvl:10s} normal={n0:>9,} alert={n1:>9,} alert_frac={frac_alert:6.2%}")

    unseen = test_templates - train_templates
    frac_unseen = len(unseen) / len(test_templates) if test_templates else 0.0
    print(f"\n[template overlap check] unique templates: train={len(train_templates):,} "
          f"test={len(test_templates):,}")
    print(f"  test templates NEVER seen in train: {len(unseen):,} ({frac_unseen:.1%})")
    print("  (this is the number that would look artificially better under a random split)")


if __name__ == "__main__":
    main()
