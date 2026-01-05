#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 三、使用範例（直接照用）
# ✅ 範例 1：抽舊資料 20% 混進新資料（最推薦）
# python /Users/shushu/Development/python_llm/tools/merge_replay_jsonl.py \
#   --new_jsonl /Users/shushu/Development/python_llm/datasets/Traditional-Chinese-Medicine-Knowledge_instruction.jsonl \
#   --old_jsonl /Users/shushu/Development/python_llm/datasets/Sampled_Dataset.jsonl \
#   --ratio 0.2 \
#   --out_jsonl /Users/shushu/Development/python_llm/datasets/train_mix_v2.jsonl
# 👉 結果：
# 新資料：100%
# 舊資料：20%（隨機）
# 輸出：train_mix_v2.jsonl


import argparse
import json
import random
from pathlib import Path
from typing import List


def load_jsonl(path: Path) -> List[str]:
    """Load jsonl as raw lines (keep original content)."""
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            # basic sanity check
            try:
                json.loads(ln)
            except json.JSONDecodeError:
                print(f"[WARN] skip invalid json line in {path}")
                continue
            lines.append(ln)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_jsonl", type=str, required=True, help="New data jsonl (kept all)")
    ap.add_argument("--old_jsonl", type=str, required=True, help="Old data jsonl (sample from)")
    ap.add_argument("--out_jsonl", type=str, required=True, help="Output merged jsonl")

    # sampling controls
    ap.add_argument("--ratio", type=float, default=None, help="Sample ratio from old (e.g. 0.2)")
    ap.add_argument("--count", type=int, default=None, help="Sample fixed count from old")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.ratio is None and args.count is None:
        raise SystemExit("ERROR: Provide either --ratio or --count")

    if args.ratio is not None and args.count is not None:
        raise SystemExit("ERROR: Use only one of --ratio or --count")

    rng = random.Random(args.seed)

    new_path = Path(args.new_jsonl)
    old_path = Path(args.old_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_lines = load_jsonl(new_path)
    old_lines = load_jsonl(old_path)

    if not new_lines:
        raise SystemExit("ERROR: new_jsonl is empty or invalid")
    if not old_lines:
        raise SystemExit("ERROR: old_jsonl is empty or invalid")

    # determine sample size
    if args.ratio is not None:
        if not (0 < args.ratio < 1):
            raise SystemExit("ERROR: --ratio must be between 0 and 1")
        sample_n = max(1, int(len(old_lines) * args.ratio))
    else:
        sample_n = min(args.count, len(old_lines))

    rng.shuffle(old_lines)
    sampled_old = old_lines[:sample_n]

    # merge: new first, then sampled old
    merged = list(new_lines) + sampled_old

    # optional shuffle merged dataset (usually good for training)
    rng.shuffle(merged)

    with out_path.open("w", encoding="utf-8") as wf:
        for ln in merged:
            wf.write(ln + "\n")

    print("Merge done.")
    print(f"New data      : {len(new_lines)}")
    print(f"Old data      : {len(old_lines)}")
    print(f"Sampled old   : {len(sampled_old)}")
    print(f"Total output  : {len(merged)}")
    print(f"Output -> {out_path}")


if __name__ == "__main__":
    main()
