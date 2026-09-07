"""
Phase 1 Project 3 - parse and CHRONOLOGICALLY split the BGL log data.

Why chronological, not random: BGL has a small number of repeating message
templates. A random split lets the model see the same template in both
train and test, which inflates reported metrics and doesn't reflect a real
deployment (a production triage model constantly meets new phrasing). We
split by line order, which is already time-ordered in the raw file. Run
report.py afterwards to see how many test-set templates were NEVER seen in
train - the number that would look artificially better under a random split.

Note: text_masked is NOT stored here (the connected-folder mount is
bandwidth-limited, ~3.5MB/s, so halving each record's size by not
duplicating the message text roughly halves total write time). Masking is
cheap to recompute from text_raw - see mask_content() below, reused by
report.py and meant to be reused again in dataset.py.

This script deliberately does NOT decide whether Type/Component/Level
should be fed to the model alongside the message content - report.py prints
a crosstab of Level vs label so that decision gets made with evidence in
front of us (same spirit as Project 2: measure, don't assume).

Chunked by design (the mount's write bandwidth means the full 4.7M lines
takes a few minutes, longer than one shell call's timeout) - use
--start-at/--limit/--append to process in slices that append to the same
three output files. Example, 3 chunks of ~1.58M lines each:
    python3 prepare_data.py --start-at 0       --limit 1582655
    python3 prepare_data.py --start-at 1582655 --limit 1582655 --append
    python3 prepare_data.py --start-at 3165310                --append

Output: data/processed/{train,val,test}.jsonl, one JSON object per line:
    {
      "line_no": int, "epoch_ts": int, "node": str,
      "type": str, "component": str, "level": str,
      "label": 0|1, "alert_type": str|null,
      "text_raw": str, "split": "train"|"val"|"test"
    }
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RAW_LOG_DEFAULT = HERE / "data" / "raw" / "BGL.log"
OUT_DIR_DEFAULT = HERE / "data" / "processed"
FLUSH_EVERY = 50000

HEX_RE = re.compile(r"0x[0-9a-fA-F]+")
NODEID_RE = re.compile(r"\b[A-Z]\d{2}-[A-Z]\d-[A-Z]\d(?:-[A-Z])?(?::[A-Z]\d{2})?(?:-[A-Z]\d{2})?\b")
NUM_RE = re.compile(r"\b\d+\b")


def mask_content(text: str) -> str:
    """Shared with report.py (and meant to be reused in dataset.py) so
    masking is defined in exactly one place."""
    text = HEX_RE.sub("<HEX>", text)
    text = NODEID_RE.sub("<NODE>", text)
    text = NUM_RE.sub("<NUM>", text)
    return text


def parse_line(raw: str, line_no: int):
    raw = raw.rstrip("\n")
    parts = raw.split(None, 9)
    if len(parts) < 9:
        return None
    if len(parts) == 9:
        parts.append("")  # legitimate: line has no message content after the level field
    label, epoch_ts, _date, node, _full_ts, _node_repeat, log_type, component, level, content = parts
    try:
        epoch_ts = int(epoch_ts)
    except ValueError:
        return None
    return {
        "line_no": line_no,
        "epoch_ts": epoch_ts,
        "node": node,
        "type": log_type,
        "component": component,
        "level": level,
        "label": 0 if label == "-" else 1,
        "alert_type": None if label == "-" else label,
        "text_raw": content,
    }


class BufferedWriter:
    def __init__(self, path: Path, mode: str):
        self.fh = open(path, mode, encoding="utf-8")
        self.buf = []

    def add(self, s: str) -> None:
        self.buf.append(s)
        if len(self.buf) >= FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if self.buf:
            self.fh.write("\n".join(self.buf) + "\n")
            self.buf = []

    def close(self) -> None:
        self.flush()
        self.fh.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-log", type=Path, default=RAW_LOG_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--n-lines", type=int, default=4747963)
    ap.add_argument("--start-at", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    if not args.raw_log.exists():
        print(f"[error] {args.raw_log} not found - run download_data.py first", file=sys.stderr)
        return 1

    n_lines = args.n_lines
    train_end = int(n_lines * args.train_frac)
    val_end = int(n_lines * (args.train_frac + args.val_frac))
    print(f"[split boundaries] train[0:{train_end:,}) val[{train_end:,}:{val_end:,}) "
          f"test[{val_end:,}:{n_lines:,})")
    print(f"[this run] start_at={args.start_at:,} limit={args.limit} append={args.append}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    out_files = {s: BufferedWriter(args.out_dir / f"{s}.jsonl", mode)
                 for s in ("train", "val", "test")}

    skipped = 0
    processed = 0
    t0 = time.time()

    with open(args.raw_log, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(args.start_at):
            next(f, None)
        for i, raw in enumerate(f, start=args.start_at):
            if args.limit is not None and processed >= args.limit:
                break
            rec = parse_line(raw, i)
            if rec is None:
                skipped += 1
                processed += 1
                continue
            split = "train" if i < train_end else ("val" if i < val_end else "test")
            rec["split"] = split
            out_files[split].add(json.dumps(rec))
            processed += 1

    for fh in out_files.values():
        fh.close()

    elapsed = time.time() - t0
    rate = processed / elapsed if elapsed > 0 else 0
    print(f"[done] processed={processed:,} (skipped {skipped} malformed) in {elapsed:.1f}s "
          f"({rate:,.0f} lines/sec) - last raw line index {args.start_at + processed - 1:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
