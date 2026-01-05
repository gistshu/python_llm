#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# python infer.py \
#   --base_model Qwen/Qwen2.5-0.5B-Instruct \
#   --lora_dir outputs_tcm_lora \
#   --question "豆蔻有何用？"

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lora_dir", type=str, required=True, help="Path to trained adapter dir (output_dir)")
    ap.add_argument("--system", type=str, default="您是一位非常专业的的中医药学教授。您始终根据提问者的问题提供准确、全面和详细的答案。")
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        device_map=None,
    )

    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.question},
    ]

    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"<system>\n{args.system}\n</system>\n<user>\n{args.question}\n</user>\n<assistant>\n"

    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
