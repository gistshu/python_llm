#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chat_cli.py (final)

Interactive chat on terminal:
- Natural language input
- Wrap into messages + chat_template prompt
- Base model + LoRA adapter (PEFT)
- Stable decode (token-based slicing)
- Medical disclaimer: remove ALL duplicates from model output, then append EXACTLY ONCE
- Optional Traditional Chinese output (system instruction + OpenCC post-convert)

Install:
  pip install -U torch transformers peft sentencepiece protobuf safetensors

Optional (force Traditional output):
  pip install opencc-python-reimplemented

Run:
python chat_cli.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --lora_dir /Users/shushu/Development/python_llm/outputs_tcm_lora_v2 \
  --max_new_tokens 256 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.85 \
  --repetition_penalty 1.15 \
  --force_traditional

Sampling (optional):
  --do_sample --temperature 0.8 --top_p 0.9
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ---------- Defaults ----------
DEFAULT_SYSTEM = (
    "您是一位非常專業的中醫藥學教授，擅長中醫辨證與日常調養建議。"
    "請一律使用繁體中文回答，用詞自然、專業、條理清楚。"
    "回答以健康科普為目的，必要時提醒就醫。"
)

# Official disclaimer (exactly-once policy, enforced by code)
DISCLAIMER = (
    "以上內容僅供健康科普參考，不能替代醫師面診。"
    "若症狀持續或加重，請及時就醫。"
)

# Some common variants that the model may generate; we will remove them too.
DISCLAIMER_VARIANTS = [
    "以上内容仅供健康科普参考，不能替代医生面诊。若症状持续或加重，请及时就医。",
    "以上內容僅供健康科普參考，不能替代醫師面診。若症狀持續或加重，請及時就醫。",
    "请注意：中药不能替代药物治疗，仅供参考。若症状持续或加重，请及时就医。",
    "請注意：中藥不能替代藥物治療，僅供參考。若症狀持續或加重，請及時就醫。",
]


def try_init_opencc(enable: bool):
    if not enable:
        return None
    try:
        from opencc import OpenCC
        return OpenCC("s2t")  # 简 -> 繁
    except Exception:
        print("[WARN] OpenCC not available. Install with: pip install opencc-python-reimplemented")
        return None


def normalize_medical_output(text: str, append_disclaimer: bool = True) -> str:
    """
    1) Remove any disclaimer-like lines that the model may have generated (even repeated).
    2) Clean extra blank lines.
    3) Append the official disclaimer exactly once (optional).
    """
    if not text:
        return text

    # Remove official disclaimer if model already generated it
    for phrase in [DISCLAIMER] + DISCLAIMER_VARIANTS:
        while phrase in text:
            text = text.replace(phrase, "")

    # Also remove obvious repeated blocks starting with key fragments
    key_frags = [
        "以上内容仅供健康科普参考",
        "以上內容僅供健康科普參考",
        "不能替代医生面诊",
        "不能替代醫師面診",
        "若症状持续或加重",
        "若症狀持續或加重",
        "请及时就医",
        "請及時就醫",
        "中药不能替代药物治疗",
        "中藥不能替代藥物治療",
    ]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Drop disclaimer-ish lines
        if any(k in stripped for k in key_frags):
            continue
        lines.append(stripped)

    cleaned = "\n".join(lines).strip()

    if append_disclaimer:
        if cleaned:
            return f"{cleaned}\n\n{DISCLAIMER}"
        return DISCLAIMER

    return cleaned


def build_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Prefer model chat_template when present, else fallback tags.
    Always uses add_generation_prompt=True to prompt an assistant continuation.
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # fallback
    sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
    lines = []
    if sys_msgs:
        lines.append(f"<system>\n{sys_msgs[0]}\n</system>")
    for m in messages:
        if m["role"] == "user":
            lines.append(f"<user>\n{m['content']}\n</user>")
        elif m["role"] == "assistant":
            lines.append(f"<assistant>\n{m['content']}\n</assistant>")
    lines.append("<assistant>\n")
    return "\n".join(lines)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    """
    Stable generation:
    - Token-based slicing (no brittle string matching)
    - Greedy by default
    - Only passes sampling params when do_sample=True
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
    )

    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
        )
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    gen_ids = out[0][prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lora_dir", type=str, required=True)
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM)

    ap.add_argument("--max_new_tokens", type=int, default=256)

    # default OFF for stability
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)

    ap.add_argument("--max_turns", type=int, default=10)

    # Output controls
    ap.add_argument("--force_traditional", action="store_true", help="Force Traditional output via OpenCC (if installed).")
    ap.add_argument("--no_disclaimer", action="store_true", help="Do not append official disclaimer (not recommended).")

    args = ap.parse_args()

    # Validate LoRA dir
    lora_path = Path(args.lora_dir)
    cfg = lora_path / "adapter_config.json"
    if not cfg.exists():
        raise SystemExit(
            f"[ERROR] LoRA adapter not found: {cfg}\n"
            f"Please point --lora_dir to a folder containing adapter_config.json and adapter_model.*"
        )

    # Optional OpenCC
    cc = try_init_opencc(args.force_traditional)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,  # torch_dtype deprecated in some envs
        device_map=None,
    )
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()

    messages: List[Dict[str, str]] = [{"role": "system", "content": args.system}]

    print("Enter your question. Type /exit to quit, /reset to clear history.\n")

    while True:
        try:
            q = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in ("/exit", "exit", "quit"):
            print("Bye.")
            break
        if q.lower() == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("History cleared.\n")
            continue

        messages.append({"role": "user", "content": q})

        # keep system + last 2*max_turns messages
        if args.max_turns > 0:
            keep = 1 + 2 * args.max_turns
            if len(messages) > keep:
                messages = [messages[0]] + messages[-(keep - 1):]

        prompt = build_prompt(tok, messages)

        try:
            ans = generate(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        except RuntimeError as e:
            print("\n[ERROR] Generation failed.")
            print("Cause:", str(e))
            print("Mitigations:")
            print("  1) Run WITHOUT --do_sample (greedy).")
            print("  2) Reduce --max_new_tokens (e.g. 128).")
            print("  3) If this persists, re-check your LoRA training stability.\n")
            continue

        # Normalize: remove any model-generated disclaimers, add exactly once (unless disabled)
        ans = normalize_medical_output(ans, append_disclaimer=(not args.no_disclaimer))

        # Force Traditional output if requested and OpenCC available
        if cc is not None:
            ans = cc.convert(ans)

        print(f"Assistant: {ans}\n")
        messages.append({"role": "assistant", "content": ans})


if __name__ == "__main__":
    main()
