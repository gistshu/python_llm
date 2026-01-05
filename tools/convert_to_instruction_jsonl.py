#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 下面給你一個可直接用的 convert_to_instruction_jsonl.py，把你舊的「conversation 格式」資料：
# [
#   {"conversation":[{"system":"...","input":"...","output":"..."}]},
#   ...
# ]
# 轉成你要拿去訓練的新格式（JSONL，每行一筆）：
# {"instruction":"...","input":"...","output":"..."}

# python /Users/shushu/Development/python_llm/tools/convert_to_instruction_jsonl.py \
#   --in_path /Users/shushu/Development/python_llm/datasets/Traditional-Chinese-Medicine-Knowledge.json \
#   --out_path /Users/shushu/Development/python_llm/datasets/Traditional-Chinese-Medicine-Knowledge_instruction.jsonl
#   --add_disclaimer



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_SYSTEM = (
    "您是一位非常专业的的中医药学教授。"
    "您始终根据提问者的问题提供准确、全面和详细的答案。"
)

DISCLAIMER = (
    "以上内容仅供健康科普参考，不能替代医生面诊。"
    "若症状持续或加重，请及时就医。"
)


def strip_or_empty(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def load_any_json(path: Path) -> List[Any]:
    """
    Robust loader:
    - JSON array
    - JSONL
    - skips invalid lines
    """
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return []

    # Try full JSON first
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    # Fallback JSONL
    items = []
    for i, ln in enumerate(raw.splitlines(), start=1):
        ln = ln.strip()
        if not ln:
            continue
        try:
            items.append(json.loads(ln))
        except json.JSONDecodeError:
            print(f"[WARN] skip invalid json at line {i}")
            continue
    return items


def convert_item_to_messages(
    item: Dict[str, Any],
    system_text: str,
    add_disclaimer: bool,
) -> Optional[Dict[str, Any]]:
    """
    Supports:
    A) {"conversation":[{"system","input","output"}]}
    B) {"instruction","input","output"}
    """

    # A) conversation
    if "conversation" in item:
        conv = item.get("conversation")
        if isinstance(conv, list) and conv:
            turn = conv[0]
            instr = strip_or_empty(turn.get("system")) or system_text
            inp = strip_or_empty(turn.get("input"))
            out = strip_or_empty(turn.get("output"))
        else:
            return None

    # B) instruction/input/output
    elif all(k in item for k in ("instruction", "input", "output")):
        instr = system_text
        inp = strip_or_empty(item.get("instruction")) + "\n" + strip_or_empty(item.get("input"))
        out = strip_or_empty(item.get("output"))

    else:
        return None

    if not inp or not out:
        return None

    if add_disclaimer and DISCLAIMER not in out:
        out = out.rstrip() + "\n\n" + DISCLAIMER

    return {
        "messages": [
            {"role": "system", "content": instr},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": out},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    ap.add_argument("--add_disclaimer", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_any_json(in_path)

    n_total = 0
    n_written = 0
    n_skipped = 0

    with out_path.open("w", encoding="utf-8") as wf:
        for obj in items:
            n_total += 1
            if not isinstance(obj, dict):
                n_skipped += 1
                continue

            converted = convert_item_to_messages(
                obj,
                system_text=args.system,
                add_disclaimer=args.add_disclaimer,
            )
            if not converted:
                n_skipped += 1
                continue

            wf.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_written += 1

    print("Done.")
    print(f"Total read : {n_total}")
    print(f"Written    : {n_written}")
    print(f"Skipped    : {n_skipped}")
    print(f"Output -> {out_path}")


if __name__ == "__main__":
    main()
