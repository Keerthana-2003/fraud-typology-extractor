"""Split data, prevent eval contamination, and run the hand-verification workflow for the test set.

Step 1 - split and export the test candidates for review:
    python -m typology.prepare_data split --raw data/raw.jsonl
    -> data/train.jsonl, data/val.jsonl (SFT chat format), data/test_to_verify.csv

Step 2 - open data/test_to_verify.csv, check every row, fix labels if needed, put "y" in the `verified`
column (or "n" to drop the row), then import:
    python -m typology.prepare_data import --csv data/test_to_verify.csv
    -> data/test.jsonl (only verified rows)

Contamination control: the test set is carved out first, and any train/val narrative whose word 5-gram
Jaccard similarity with a test narrative exceeds --max-sim is dropped.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .prompts import messages
from .schema import Extraction


def _shingles(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9$.,']+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def drop_near_duplicates(pool: list[dict], test: list[dict], max_sim: float) -> tuple[list[dict], int]:
    """Remove pool items too similar to any test item. Uses an inverted index on shingles to stay fast."""
    test_sh = [_shingles(t["narrative"]) for t in test]
    index: dict[str, list[int]] = defaultdict(list)
    for i, sh in enumerate(test_sh):
        for s in sh:
            index[s].append(i)
    kept, dropped = [], 0
    for rec in pool:
        sh = _shingles(rec["narrative"])
        candidates = {i for s in sh for i in index.get(s, [])}
        if any(jaccard(sh, test_sh[i]) > max_sim for i in candidates):
            dropped += 1
        else:
            kept.append(rec)
    return kept, dropped


def _write_sft(path: Path, recs: list[dict]) -> None:
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps({"id": r["id"], "messages": messages(r["narrative"], target=r["label"]),
                                "label": r["label"]}) + "\n")


def split(raw: Path, out_dir: Path, test_size: int, val_frac: float, max_sim: float, seed: int) -> None:
    recs = [json.loads(line) for line in open(raw)]
    seen, uniq = set(), []
    for r in recs:  # exact-duplicate removal
        key = r["narrative"].strip().lower()
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    rng = random.Random(seed)
    by_typ: dict[str, list[dict]] = defaultdict(list)
    for r in uniq:
        by_typ[r["label"]["typology"]].append(r)
    test, rest = [], []
    per = max(1, test_size // len(by_typ))
    for typ, items in by_typ.items():  # stratified test carve-out
        rng.shuffle(items)
        test += items[:per]
        rest += items[per:]
    rest, dropped = drop_near_duplicates(rest, test, max_sim)
    rng.shuffle(rest)
    n_val = int(len(rest) * val_frac)
    val, train = rest[:n_val], rest[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_sft(out_dir / "train.jsonl", train)
    _write_sft(out_dir / "val.jsonl", val)
    with open(out_dir / "test_to_verify.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "narrative", "typology", "red_flags", "amounts", "channel", "urgency", "verified", "notes"])
        for r in test:
            lab = r["label"]
            w.writerow([r["id"], r["narrative"], lab["typology"], "; ".join(lab["red_flags"]),
                        "; ".join(str(a) for a in lab["amounts"]), lab["channel"], lab["urgency"], "", ""])
    print(f"unique={len(uniq)} train={len(train)} val={len(val)} test_candidates={len(test)} "
          f"dropped_near_duplicates={dropped}")
    print(f"Now review {out_dir / 'test_to_verify.csv'} and run: python -m typology.prepare_data import")


def import_verified(csv_path: Path, out: Path) -> None:
    kept, skipped, errors = [], 0, []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["verified"].strip().lower() not in {"y", "yes", "1", "true"}:
                skipped += 1
                continue
            try:
                label = Extraction(
                    typology=row["typology"].strip(),
                    red_flags=[x.strip() for x in row["red_flags"].split(";") if x.strip()],
                    amounts=[float(x) for x in row["amounts"].split(";") if x.strip()],
                    channel=row["channel"].strip(), urgency=row["urgency"].strip()).model_dump()
            except Exception as e:
                errors.append(f"{row['id']}: {e}")
                continue
            kept.append({"id": row["id"], "narrative": row["narrative"], "label": label, "verified": True})
    if errors:
        print("Rows with invalid labels (fix and re-import):\n  " + "\n  ".join(errors[:20]))
    with open(out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(kept)} verified test examples to {out} (skipped {skipped} unverified)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--raw", type=Path, default=Path("data/raw.jsonl"))
    s.add_argument("--out-dir", type=Path, default=Path("data"))
    s.add_argument("--test-size", type=int, default=330, help="candidates; ~300 usually survive review")
    s.add_argument("--val-frac", type=float, default=0.1)
    s.add_argument("--max-sim", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=21)
    i = sub.add_parser("import")
    i.add_argument("--csv", type=Path, default=Path("data/test_to_verify.csv"))
    i.add_argument("--out", type=Path, default=Path("data/test.jsonl"))
    a = ap.parse_args()
    if a.cmd == "split":
        split(a.raw, a.out_dir, a.test_size, a.val_frac, a.max_sim, a.seed)
    else:
        import_verified(a.csv, a.out)
