"""Regenerate the committed sample dataset from the full raw CSV.

The sample is a faithful miniature: it keeps the original (malformed) header
line byte-for-byte and the full 40-field row layout, so it exercises the same
column contract as the real dataset.

Usage:  python scripts/make_sample.py
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recommender import config  # noqa: E402

N_FAMOUS = 300  # highest review counts -> recognisable games for the demo
N_SPREAD = 250  # stratified across the popularity range
N_JUNK = 50  # zero-review / playtest rows, so the filter has work to do

POSITIVE_IDX = 23
NEGATIVE_IDX = 24
TAGS_IDX = 37
NAME_IDX = 1


def main() -> None:
    csv.field_size_limit(10**9)
    random.seed(0)

    with open(config.RAW_CSV, encoding="utf-8", newline="") as fh:
        header_line = fh.readline()  # keep the malformed header verbatim
        rows = [r for r in csv.reader(fh) if len(r) == 40]

    def reviews(row: list[str]) -> int:
        try:
            return int(row[POSITIVE_IDX] or 0) + int(row[NEGATIVE_IDX] or 0)
        except ValueError:
            return 0

    scored = sorted(rows, key=reviews, reverse=True)
    famous = scored[:N_FAMOUS]

    mid = [r for r in scored[N_FAMOUS:] if reviews(r) >= 10 and r[TAGS_IDX].strip()]
    spread = random.sample(mid, min(N_SPREAD, len(mid)))

    junk = [r for r in rows if reviews(r) == 0 or "playtest" in r[NAME_IDX].lower()]
    junk = random.sample(junk, min(N_JUNK, len(junk)))

    selected = famous + spread + junk
    random.shuffle(selected)

    config.SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SAMPLE_CSV, "w", encoding="utf-8", newline="") as fh:
        fh.write(header_line)
        csv.writer(fh, lineterminator="\n").writerows(selected)

    size_mb = config.SAMPLE_CSV.stat().st_size / 1024**2
    print(f"wrote {len(selected)} games -> {config.SAMPLE_CSV} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
