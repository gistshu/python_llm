
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py

Normalize various dataset formats into JSONL with {"messages":[...]} suitable for chat SFT.

Supported input sample formats (mixed allowed):
1) {"messages":[{"role":"system","content":...},{"role":"user","content":...},{"role":"assistant","content":...}]}
2) {"conversation":[{"system":..., "input":..., "output":...}]}   (your earlier format)
3) {"instruction":..., "input":..., "output":...}

Key feature:
- Enforce a single, consistent SYSTEM prompt (role persona) for all samples.
- Move any question-title mistakenly placed in system into the user turn (optional heuristic).
- Optionally prepend "instruction" (or detected title) to user content.
- Adds a safety footer (optional) to assistant outputs for medical content.
- Filters empty / too-short samples.
- Truncates very long user/assistant content by character limit (simple, CPU-friendly).
- Outputs JSONL: one sample per line.

建議你直接這樣跑（符合你目前資料狀況）
你先把角色固定成中醫教授（你之前的 system）：
python prepare_data.py \
  --in /Users/shushu/Development/python_llm/datasets/Sampled_Dataset.json \
  --out /Users/shushu/Development/python_llm/datasets/Sampled_Dataset.jsonl \
  --system_prompt "您是一位非常专业的的中医药学教授。您始终根据提问者的问题提供准确、全面和详细的答案。" \
  --prepend_instruction_to_user \
  --move_bad_system_to_user \
  --max_user_chars 4000 \
  --max_assistant_chars 2000
如果你要加「醫療安全尾巴」（建議商用一定要）
例如：
python prepare_data.py \
  --in /Users/shushu/Development/python_llm/datasets/Sampled_Dataset.json \
  --out /Users/shushu/Development/python_llm/datasets/Sampled_Dataset.jsonl \
  --system_prompt "您是一位非常专业的的中医药学教授。您始终根据提问者的问题提供准确、全面和详细的答案。" \
  --prepend_instruction_to_user \
  --move_bad_system_to_user \
  --medical_footer "以上内容仅供健康科普参考，不能替代医生面诊。若症状持续或加重，请及时就医。" \
  --max_user_chars 4000 \
  --max_assistant_chars 2000


Notes:
- This script does not require ML libraries. Pure Python.
- For best SFT results, keep system_prompt stable across training & inference.
"""

import argparse
import json
import re
from typing import Any, Dict, List, Optional


def strip_or_empty(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def safe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    # Keep head and tail for context
    head = text[: max_chars - 200]
    tail = text[-200:]
    return head + "\n...(truncated)...\n" + tail


def looks_like_question_title(s: str) -> bool:
    """
    Heuristic: detect if 'system' is actually a short question title rather than a persona.
    """
    s = s.strip()
    if not s:
        return False
    # too short and contains question-like markers
    if len(s) <= 30 and (("?" in s) or ("？" in s) or ("怎么办" in s) or ("如何" in s) or ("怎么" in s)):
        return True
    # if it's short and does not contain typical persona words
    persona_markers = ["您是", "你是", "角色", "教授", "医生", "助理", "請", "始终", "根据", "回答", "专业"]
    if len(s) <= 40 and not any(m in s for m in persona_markers):
        return True
    return False


def add_medical_safety_footer(ans: str, footer: str) -> str:
    ans = ans.strip()
    if not ans:
        return ans
    if not footer:
        return ans
    # Avoid duplicating if footer already present (rough check)
    if footer.strip() in ans:
        return ans
    return ans + "\n\n" + footer.strip()


def normalize_to_messages(
    item: Dict[str, Any],
    system_prompt: str,
    prepend_instruction_to_user: bool,
    move_bad_system_to_user: bool,
    medical_footer: str,
) -> Optional[List[Dict[str, str]]]:
    """
    Return messages list or None if cannot normalize.
    """

    # Case A: already messages
    if "messages" in item and isinstance(item["messages"], list):
        msgs = item["messages"]
        # Find user and assistant
        sys_content = ""
        user_content = ""
        asst_content = ""
        for m in msgs:
            role = m.get("role")
            content = strip_or_empty(m.get("content"))
            if role == "system":
                sys_content = content
            elif role == "user":
                user_content = content
            elif role == "assistant":
                asst_content = content

        # If system is bad (looks like a question title), move it into user
        if move_bad_system_to_user and looks_like_question_title(sys_content):
            if user_content:
                user_content = sys_content + "\n" + user_content
            else:
                user_content = sys_content
            sys_content = ""

        # Enforce unified system prompt
        sys_final = system_prompt.strip()

        # Safety footer
        asst_final = add_medical_safety_footer(asst_content, medical_footer)

        if not user_content or not asst_final:
            return None

        return [
            {"role": "system", "content": sys_final},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": asst_final.strip()},
        ]

    # Case B: conversation list with system/input/output keys
    if "conversation" in item and isinstance(item["conversation"], list) and item["conversation"]:
        turn = item["conversation"][0]
        sys_ = strip_or_empty(turn.get("system"))
        inp_ = strip_or_empty(turn.get("input"))
        out_ = strip_or_empty(turn.get("output"))

        # If system is bad (title-like), move to user (prepend)
        if move_bad_system_to_user and looks_like_question_title(sys_):
            if inp_:
                inp_ = sys_ + "\n" + inp_
            else:
                inp_ = sys_
            sys_ = ""

        sys_final = system_prompt.strip()

        asst_final = add_medical_safety_footer(out_, medical_footer)
        if not inp_ or not asst_final:
            return None

        return [
            {"role": "system", "content": sys_final},
            {"role": "user", "content": inp_.strip()},
            {"role": "assistant", "content": asst_final.strip()},
        ]

    # Case C: instruction/input/output
    if all(k in item for k in ("instruction", "input", "output")):
        instr = strip_or_empty(item.get("instruction"))
        inp_ = strip_or_empty(item.get("input"))
        out_ = strip_or_empty(item.get("output"))

        user_content = inp_
        if prepend_instruction_to_user and instr:
            user_content = instr + "\n" + inp_ if inp_ else instr

        sys_final = system_prompt.strip()
        asst_final = add_medical_safety_footer(out_, medical_footer)
        if not user_content or not asst_final:
            return None

        return [
            {"role": "system", "content": sys_final},
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": asst_final.strip()},
        ]

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON file (list or object).")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL path.")
    ap.add_argument(
        "--system_prompt",
        required=True,
        help="Unified system prompt to enforce for all samples.",
    )
    ap.add_argument(
        "--prepend_instruction_to_user",
        action="store_true",
        help="If input has instruction field, prepend it to user content.",
    )
    ap.add_argument(
        "--move_bad_system_to_user",
        action="store_true",
        help="Heuristic: if system looks like a short question title, move it into user content.",
    )
    ap.add_argument(
        "--medical_footer",
        default="",
        help="Append this footer to assistant answers (e.g., medical safety disclaimer). Leave empty to disable.",
    )
    ap.add_argument("--max_user_chars", type=int, default=0, help="Truncate user content to this many chars (0 disables).")
    ap.add_argument("--max_assistant_chars", type=int, default=0, help="Truncate assistant content to this many chars (0 disables).")
    ap.add_argument("--min_user_chars", type=int, default=1, help="Drop samples with user content shorter than this.")
    ap.add_argument("--min_assistant_chars", type=int, default=1, help="Drop samples with assistant content shorter than this.")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N samples (0 = all).")

    args = ap.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Allow root object or list
    if isinstance(data, dict):
        # If a dict contains a list field, try common containers; otherwise wrap it
        if "data" in data and isinstance(data["data"], list):
            items = data["data"]
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        raise SystemExit("Input JSON root must be a list or object.")

    if args.limit and args.limit > 0:
        items = items[: args.limit]

    out_count = 0
    skipped = 0

    with open(args.out_path, "w", encoding="utf-8") as wf:
        for item in items:
            if not isinstance(item, dict):
                skipped += 1
                continue

            messages = normalize_to_messages(
                item=item,
                system_prompt=args.system_prompt,
                prepend_instruction_to_user=args.prepend_instruction_to_user,
                move_bad_system_to_user=args.move_bad_system_to_user,
                medical_footer=args.medical_footer,
            )

            if not messages:
                skipped += 1
                continue

            # Truncate if requested
            user = messages[1]["content"]
            asst = messages[2]["content"]

            user = safe_truncate(user, args.max_user_chars)
            asst = safe_truncate(asst, args.max_assistant_chars)

            if len(user) < args.min_user_chars or len(asst) < args.min_assistant_chars:
                skipped += 1
                continue

            messages[1]["content"] = user
            messages[2]["content"] = asst

            wf.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            out_count += 1

    print(f"Done. Wrote {out_count} samples to {args.out_path}. Skipped {skipped} samples.")


if __name__ == "__main__":
    main()
