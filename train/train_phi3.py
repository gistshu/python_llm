#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 可增量訓練（v1→v2→…）的最新版 train.py，特點：
# 支援 從既有 LoRA adapter 繼續訓練：--resume_from_lora outputs_tcm_lora_v1
# 支援 replay（混入舊資料避免遺忘）：--replay_path old.jsonl --replay_ratio 0.2
# 支援 JSON / JSONL，且可混用三種格式：messages / conversation / instruction+input+output
# 只在 assistant token 計算 loss（prompt masking）
# CPU 友善（torch.float32，num_workers=0）
# 自動在 output_dir 寫入：
# train_args.json（完整參數）
# data_manifest.txt（資料路徑與比例）
# 預設用 LoRA 訓練；不依賴 TRL（避免版本坑）

# 你要怎麼用（v1 → v2 持續訓練）
# 1) 先做 v1（首次訓練）
# python train.py \
#   --data_path /path/to/v1_data.jsonl \
#   --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
#   --output_dir /Users/shushu/Development/python_llm/outputs_tcm_lora_v1 \
#   --max_seq_len 256 \
#   --num_train_epochs 1 \
#   --learning_rate 5e-5 \
#   --save_steps 50 \
#   --logging_steps 5 \
#   --lora_target_modules qkv_proj,o_proj,gate_up_proj,down_proj

# 2) v2：在 v1 adapter 上繼續訓練（可混入部分 v1 資料防遺忘）
# python train.py \
#   --data_path /Users/shushu/Development/python_llm/datasets/train_merge.jsonl \
#   --replay_path /Users/shushu/Development/python_llm/datasets/Sampled_Dataset.jsonl \
#   --replay_ratio 0.2 \
#   --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
#   --resume_from_lora /Users/shushu/Development/python_llm/outputs_tcm_lora_v1 \
#   --output_dir /Users/shushu/Development/python_llm/outputs_tcm_lora_v2 \
#   --max_seq_len 256 \
#   --num_train_epochs 1 \
#   --learning_rate 2e-5 \
#   --save_steps 50 \
#   --logging_steps 5 \
#   --lora_target_modules qkv_proj,o_proj,gate_up_proj,down_proj

# 經驗值：增量訓練通常 learning_rate 用 2e-5 或 1e-5 會比較穩；replay_ratio 建議 0.1～0.3。

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, PeftModel


# -----------------------------
# Helpers: IO / normalization
# -----------------------------

def strip_or_empty(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def normalize_item_to_messages(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize a single item to:
      {"messages":[{"role":"system","content":...},{"role":"user","content":...},{"role":"assistant","content":...}]}
    Supports:
      A) {"messages":[...]}
      B) {"conversation":[{"system":..., "input":..., "output":...}]}
      C) {"instruction":..., "input":..., "output":...}
    """
    # A) already messages
    if "messages" in item and isinstance(item["messages"], list):
        msgs = item["messages"]
        norm = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = strip_or_empty(m.get("content"))
            if role in ("system", "user", "assistant") and content != "":
                norm.append({"role": role, "content": content})
        roles = {m["role"] for m in norm}
        if "user" in roles and "assistant" in roles:
            sys = next((m["content"] for m in norm if m["role"] == "system"), "")
            usr = next((m["content"] for m in norm if m["role"] == "user"), "")
            ast = next((m["content"] for m in norm if m["role"] == "assistant"), "")
            if usr and ast:
                return {"messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": usr},
                    {"role": "assistant", "content": ast},
                ]}
        return None

    # B) conversation format
    if "conversation" in item and isinstance(item["conversation"], list) and item["conversation"]:
        turn = item["conversation"][0]
        if not isinstance(turn, dict):
            return None
        sys_ = strip_or_empty(turn.get("system"))
        inp_ = strip_or_empty(turn.get("input"))
        out_ = strip_or_empty(turn.get("output"))
        if inp_ and out_:
            return {"messages": [
                {"role": "system", "content": sys_},
                {"role": "user", "content": inp_},
                {"role": "assistant", "content": out_},
            ]}
        return None

    # C) instruction/input/output format
    if all(k in item for k in ("instruction", "input", "output")):
        instr = strip_or_empty(item.get("instruction"))
        inp_ = strip_or_empty(item.get("input"))
        out_ = strip_or_empty(item.get("output"))
        if inp_ and out_:
            return {"messages": [
                {"role": "system", "content": instr},
                {"role": "user", "content": inp_},
                {"role": "assistant", "content": out_},
            ]}
        return None

    return None


def load_dataset_any(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSON or JSONL file into a list of normalized {"messages":[...]} samples.
    """
    path = str(path)
    lower = path.lower()
    samples: List[Dict[str, Any]] = []

    def handle_obj(obj: Any):
        if isinstance(obj, dict):
            norm = normalize_item_to_messages(obj)
            if norm:
                samples.append(norm)

    if lower.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                obj = json.loads(ln)
                handle_obj(obj)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                handle_obj(obj)
        elif isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                for obj in data["data"]:
                    handle_obj(obj)
            else:
                handle_obj(data)

    return samples


# -----------------------------
# Chat template -> prompt text
# -----------------------------

def apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

    # Fallback
    sys = messages[0]["content"]
    user = messages[1]["content"]
    assistant = messages[2]["content"] if len(messages) > 2 else ""
    if add_generation_prompt:
        return f"<system>\n{sys}\n</system>\n<user>\n{user}\n</user>\n<assistant>\n"
    return f"<system>\n{sys}\n</system>\n<user>\n{user}\n</user>\n<assistant>\n{assistant}\n</assistant>\n"


def build_full_and_prompt_text(tokenizer, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    full_text includes assistant content.
    prompt_text ends right at assistant start (no assistant content) for masking.
    """
    full_text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    prompt_messages = messages[:2] + [{"role": "assistant", "content": ""}]
    prompt_text = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    return full_text, prompt_text


# -----------------------------
# Tokenize + label masking
# -----------------------------

def tokenize_and_mask(example: Dict[str, Any], tokenizer, max_len: int) -> Dict[str, Any]:
    messages = example["messages"]
    full_text, prompt_text = build_full_and_prompt_text(tokenizer, messages)

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors=None,
    )
    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_len,
        padding=False,
        return_tensors=None,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_len = len(prompt["input_ids"])
    labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[min(prompt_len, len(input_ids)):]
    labels = labels[: len(input_ids)]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class DataCollatorForCausalLMWithLabels:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            lab = f["labels"]
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            labels.append(lab)

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


# -----------------------------
# Replay mixing (anti-forgetting)
# -----------------------------

def build_train_dataset(
    new_path: str,
    replay_path: Optional[str],
    replay_ratio: float,
    seed: int,
) -> Tuple[Dataset, str]:
    """
    Returns (dataset, manifest_text)
    replay_ratio means: fraction of replay samples relative to total.
    Example replay_ratio=0.2 => 20% replay, 80% new (if enough replay data).
    """
    new_samples = load_dataset_any(new_path)
    if not new_samples:
        raise SystemExit(f"No valid samples loaded from new data: {new_path}")
    ds_new = Dataset.from_list(new_samples)

    manifest_lines = [
        f"new_data: {new_path}",
        f"new_count: {len(ds_new)}",
    ]

    if replay_path and replay_ratio > 0:
        replay_samples = load_dataset_any(replay_path)
        if not replay_samples:
            raise SystemExit(f"Replay path provided but no valid samples: {replay_path}")
        ds_replay_all = Dataset.from_list(replay_samples)

        # Target replay count
        # replay_ratio = replay / (new + replay)  => replay = (ratio/(1-ratio)) * new
        target_replay = int((replay_ratio / max(1e-9, (1.0 - replay_ratio))) * len(ds_new))
        target_replay = max(1, target_replay)

        # Sample replay deterministically
        rng = random.Random(seed)
        idxs = list(range(len(ds_replay_all)))
        rng.shuffle(idxs)
        idxs = idxs[: min(target_replay, len(ds_replay_all))]
        ds_replay = ds_replay_all.select(idxs)

        ds_mix = concatenate_datasets([ds_new, ds_replay]).shuffle(seed=seed)

        manifest_lines += [
            f"replay_data: {replay_path}",
            f"replay_count_total: {len(ds_replay_all)}",
            f"replay_ratio_target: {replay_ratio}",
            f"replay_count_used: {len(ds_replay)}",
            f"mixed_count: {len(ds_mix)}",
        ]
        return ds_mix, "\n".join(manifest_lines)

    manifest_lines += [
        "replay_data: (none)",
        "mixed_count: (same as new_count)",
    ]
    return ds_new, "\n".join(manifest_lines)



# -----------------------------
# LoRA target module resolver
# -----------------------------

def resolve_lora_target_modules(model, spec: str) -> List[str]:
    """
    Resolve LoRA target modules for PEFT.

    - spec="auto": infer a good set based on what modules actually exist in the model.
      * Phi-3 mini family commonly uses: qkv_proj, o_proj, gate_up_proj, down_proj
      * LLaMA/Qwen-like models commonly use: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj
    - spec="a,b,c": explicit comma-separated list.

    This function returns only modules that are present in the base model, ordered as in the candidate list.
    """
    spec = (spec or "").strip()
    if spec and spec.lower() != "auto":
        modules = [m.strip() for m in spec.split(",") if m.strip()]
        if not modules:
            raise ValueError("--lora_target_modules provided but empty after parsing.")
        return modules

    # Collect leaf module names (last path component) for fast membership test
    module_names = set()
    for name, _ in model.named_modules():
        module_names.add(name.split(".")[-1])

    candidates = [
        # Phi-3 (Mini/Small/Medium) attention uses fused QKV
        ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        # LLaMA/Qwen family
        ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        # Phi-2 / GPT-style MLP naming (best-effort fallback)
        ["query_key_value", "dense", "fc1", "fc2"],
    ]

    best = None
    best_score = -1
    for cand in candidates:
        score = sum(1 for m in cand if m in module_names)
        if score > best_score:
            best_score = score
            best = cand

    resolved = [m for m in (best or []) if m in module_names]
    if not resolved:
        # As a last resort, surface some hints for debugging
        sample = sorted(list(module_names))[:50]
        raise ValueError(
            "Could not infer LoRA target modules (auto). "
            "Pass --lora_target_modules explicitly. "
            f"Sample module names: {sample}"
        )
    return resolved

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--data_path", type=str, required=True, help="New data (JSON/JSONL).")
    ap.add_argument("--replay_path", type=str, default=None, help="Old data to mix in (JSON/JSONL).")
    ap.add_argument("--replay_ratio", type=float, default=0.0, help="Fraction of replay in total (e.g. 0.2).")

    # Model / LoRA
    ap.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--resume_from_lora", type=str, default=None, help="Path to existing LoRA adapter (v1) to continue training.")
    ap.add_argument("--output_dir", type=str, required=True, help="New output dir (e.g. outputs_tcm_lora_v2).")

    # Training
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-5)  # safer for continued training
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA hyperparams
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="auto", help="LoRA target modules. Use 'auto' to infer by model architecture; or provide comma-separated list, e.g. qkv_proj,o_proj,gate_up_proj,down_proj")

    args = ap.parse_args()
    set_seed(args.seed)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Persist training metadata
    with (outdir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset (with optional replay)
    ds_raw, manifest = build_train_dataset(
        new_path=args.data_path,
        replay_path=args.replay_path,
        replay_ratio=float(args.replay_ratio),
        seed=args.seed,
    )
    (outdir / "data_manifest.txt").write_text(manifest + "\n", encoding="utf-8")

    # Tokenize
    tokenized = ds_raw.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_seq_len),
        remove_columns=ds_raw.column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer)

    # Load base model (CPU)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
        device_map=None,
    )

    # LoRA config (must match between v1 and continued training)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=resolve_lora_target_modules(base, args.lora_target_modules),
    )

    if args.resume_from_lora:
        # Create a LoRA-wrapped model with the same config, then load existing adapter weights as trainable
        model = get_peft_model(base, lora_cfg)
        model = PeftModel.from_pretrained(model, args.resume_from_lora, is_trainable=True)
    else:
        model = get_peft_model(base, lora_cfg)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(outdir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=False,
        bf16=False,
        report_to=[],
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=args.max_grad_norm,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    trainer.model.save_pretrained(str(outdir))
    tokenizer.save_pretrained(str(outdir))
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()