#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chat_web.py

Flask web interface for natural language chat:
- Reuses logic from chat_cli.py
- Serve a Line-like UI
- API endpoint for chat generation


python chat_web.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --lora_dir /Users/shushu/Development/python_llm/outputs_tcm_lora_v2 \
  --max_new_tokens 256 \
  --do_sample \
  --temperature 0.7 \
  --top_p 0.85 \
  --repetition_penalty 1.15 \
  --force_traditional


"""

import argparse
import sys
import threading
from pathlib import Path
from typing import List, Dict, Optional

from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import reused logic components (or redefine if import is tricky)
# To avoid structure issues, we will redefine the core helpers here,
# strictly matching chat_cli.py's latest improvements.

# ---------- Defaults & Helpers ----------
DEFAULT_SYSTEM = (
    "您是一位非常專業的中醫藥學教授，擅長中醫辨證與日常調養建議。"
    "您也熟知「中醫聯合診所」的醫師陣容，當使用者請求推薦醫師時，請根據其症狀從以下名單中推薦（僅為模擬資料）：\n"
    "1. 林志豪醫師：擅長針灸、筋骨疼痛、運動傷害調理。\n"
    "2. 張美玲醫師：擅長婦科調調理、經期不適、更年期保健。\n"
    "3. 王大同醫師：擅長脾胃調和、消化不良、過敏性鼻炎。\n"
    "請一律使用繁體中文回答，用詞自然、專業、條理清楚。"
    "回答以健康科普為目的，必要時提醒就醫。"
)

DISCLAIMER = (
    "以上內容僅供健康科普參考，不能替代醫師面診。"
    "若症狀持續或加重，請及時就醫。"
)

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
        return OpenCC("s2t")
    except Exception:
        print("[WARN] OpenCC not available. Install with: pip install opencc-python-reimplemented")
        return None

def normalize_medical_output(text: str, append_disclaimer: bool = True) -> str:
    if not text:
        return text

    for phrase in [DISCLAIMER] + DISCLAIMER_VARIANTS:
        while phrase in text:
            text = text.replace(phrase, "")

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
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
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

# ---------- Flask App Setup ----------
app = Flask(__name__)

# Global state
model_state = {
    "model": None,
    "tokenizer": None,
    "args": None,
    "cc": None
}

@app.route("/")
def index():
    return render_template("chat_line.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_input = data.get("message", "")
    history = data.get("history", []) # List of {role, content}
    
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    args = model_state["args"]
    tok = model_state["tokenizer"]
    model = model_state["model"]
    cc = model_state["cc"]

    # Construct messages list
    # Always start with system prompt
    current_messages = [{"role": "system", "content": args.system}]
    
    # Add history (last N turns)
    if args.max_turns > 0:
        # history from client usually excludes system
        keep = 2 * args.max_turns
        if len(history) > keep:
            history = history[-keep:]
        current_messages.extend(history)
    
    # Add current user message
    current_messages.append({"role": "user", "content": user_input})
    
    prompt = build_prompt(tok, current_messages)

    try:
        ans = generate_text(
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
        print(f"[ERROR] Generation failed: {e}")
        return jsonify({"error": "Generation failed server-side."}), 500

    # Post-process
    ans = normalize_medical_output(ans, append_disclaimer=(not args.no_disclaimer))
    if cc:
        ans = cc.convert(ans)

    return jsonify({"response": ans})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--force_traditional", action="store_true")
    parser.add_argument("--no_disclaimer", action="store_true")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    
    args = parser.parse_args()
    
    print("Loading model...")
    # Validate LoRA dir
    lora_path = Path(args.lora_dir)
    if not (lora_path / "adapter_config.json").exists():
         print(f"[ERROR] LoRA adapter not found: {lora_path}")
         sys.exit(1)

    model_state["cc"] = try_init_opencc(args.force_traditional)
    model_state["args"] = args

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_state["tokenizer"] = tok

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        device_map=None,
    )
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()
    model_state["model"] = model
    
    print(f"Server starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
