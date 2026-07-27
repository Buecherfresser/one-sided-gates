"""Result writing that does not destroy previous runs.

Every analysis script here evaluates whatever adapters happen to be staged when it runs, and the
original pattern was a plain `write_text(json.dumps(results))`. That silently deletes every config
from earlier runs. It bit us once for real: the data-efficiency sweep staged only the r3_standing and
c0_matched variants, so it wiped the r1_literal / r2_class rows written by the previous run. The
numbers survived only because they were also in the logs.

Two helpers, both idempotent and both preserving provenance:

  save_merged   dict-of-config results -> merge into a canonical JSON, newer wins per key, plus an
                unmerged per-run snapshot so you can always tell which run produced which number.
  append_jsonl  record streams -> append with a run tag rather than truncate.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _run_tag() -> str:
    """Stable-ish identifier for this process, for provenance in snapshots."""
    return f"{int(time.time())}-{os.getpid()}"


def save_merged(path: str | Path, results: dict, snapshot: bool = True) -> dict:
    """Merge `results` into the JSON at `path` (newer wins per top-level key).

    Returns the merged dict. Writes a sibling `<stem>_run_<tag>.json` snapshot of just this run's
    results unless `snapshot=False`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    merged: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, dict):
                merged = existing
        except json.JSONDecodeError:
            # A corrupt canonical file must not take the new results down with it; keep it aside.
            path.rename(path.with_suffix(f".corrupt-{_run_tag()}.json"))

    before = len(merged)
    merged.update(results)
    path.write_text(json.dumps(merged, indent=2, default=str))

    if snapshot:
        snap = path.with_name(f"{path.stem}_run_{_run_tag()}.json")
        snap.write_text(json.dumps(results, indent=2, default=str))

    print(f"  {path.name}: merged {len(results)} keys "
          f"({before} -> {len(merged)} total)")
    return merged


def append_jsonl(path: str | Path, records, run_tag: str | None = None) -> int:
    """Append records to a JSONL file, stamping each with a run tag. Never truncates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = run_tag or _run_tag()
    n = 0
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(dict(r, _run=tag), default=str) + "\n")
            n += 1
    print(f"  {path.name}: appended {n} records (run {tag})")
    return n
