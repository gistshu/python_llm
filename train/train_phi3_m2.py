#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Phi-3 Mini Instruct LoRA trainer (Apple Silicon friendly).

This is a tuned variant of train_phi3.py for Apple M2 Pro (16GB) and similar.

Key differences vs the CPU-f32 baseline:
  - Supports --device auto|mps|cpu
  - Supports --torch_dtype float16|bfloat16|float32 (default: float16 on MPS, float32 on CPU)
  - Enables gradient checkpointing (optional) to reduce activation memory
  - Uses low_cpu_mem_usage=True for more memory-efficient loading
  - Sets model.config.use_cache=False when training


建議你直接用這組參數（M2 Pro / 16GB 的保守穩定組）
首次訓練（v1）
python train_phi3_m2.py \
  --data_path /Users/shushu/Development/python_llm/datasets/train_merge.jsonl \
  --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
  --output_dir /Users/shushu/Development/python_llm/outputs_phi3_lora_v1 \
  --device mps \
  --torch_dtype float16 \
  --gradient_checkpointing \
  --max_seq_len 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --save_steps 50 \
  --logging_steps 5
  --lora_target_modules qkv_proj,o_proj,gate_up_proj,down_proj

增量訓練（v2：接 v1 adapter）
python train_phi3_m2.py \
  --data_path /Users/shushu/Development/python_llm/datasets/train_merge.jsonl \
  --replay_path /Users/shushu/Development/python_llm/datasets/train_001.jsonl \
  --replay_ratio 0.2 \
  --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
  --resume_from_lora /Users/shushu/Development/python_llm/outputs_phi3_lora_v1 \
  --output_dir /Users/shushu/Development/python_llm/outputs_phi3_lora_v2 \
  --device mps \
  --torch_dtype float16 \
  --gradient_checkpointing \
  --max_seq_len 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --save_steps 50 \
  --logging_steps 5
  --lora_target_modules qkv_proj,o_proj,gate_up_proj,down_proj
  
如果還是爆記憶體，照這個順序降載（不需要猜）
--max_seq_len 192（最有效）
--gradient_accumulation_steps 8（降低步數累積，速度會快，但等效 batch 變小）
--lora_r 4 --lora_alpha 8（LoRA 參數更小，效果可能略降但更穩）
--replay_ratio 降到 0.1




"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def strip_or_empty(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def normalize_item_to_messages(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a single sample into OpenAI-style messages."""
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
                return {
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": usr},
                        {"role": "assistant", "content": ast},
                    ]
                }
        return None

    if "conversation" in item and isinstance(item["conversation"], list) and item["conversation"]:
        turn = item["conversation"][0]
        if not isinstance(turn, dict):
            return None
        sys_ = strip_or_empty(turn.get("system"))
        inp_ = strip_or_empty(turn.get("input"))
        out_ = strip_or_empty(turn.get("output"))
        if inp_ and out_:
            return {
                "messages": [
                    {"role": "system", "content": sys_},
                    {"role": "user", "content": inp_},
                    {"role": "assistant", "content": out_},
                ]
            }
        return None

    if all(k in item for k in ("instruction", "input", "output")):
        instr = strip_or_empty(item.get("instruction"))
        inp_ = strip_or_empty(item.get("input"))
        out_ = strip_or_empty(item.get("output"))
        if inp_ and out_:
            return {
                "messages": [
                    {"role": "system", "content": instr},
                    {"role": "user", "content": inp_},
                    {"role": "assistant", "content": out_},
                ]
            }
        return None

    return None


def load_dataset_any(path: Union[str, Path]) -> List[Dict[str, Any]]:
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
                handle_obj(json.loads(ln))
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


def apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    sys = messages[0]["content"]
    user = messages[1]["content"]
    assistant = messages[2]["content"] if len(messages) > 2 else ""
    if add_generation_prompt:
        return f"<system>\n{sys}\n</system>\n<user>\n{user}\n</user>\n<assistant>\n"
    return f"<system>\n{sys}\n</system>\n<user>\n{user}\n</user>\n<assistant>\n{assistant}\n</assistant>\n"


def build_full_and_prompt_text(tokenizer, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    full_text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    prompt_messages = messages[:2] + [{"role": "assistant", "content": ""}]
    prompt_text = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    return full_text, prompt_text


def tokenize_and_mask(example: Dict[str, Any], tokenizer, max_len: int) -> Dict[str, Any]:
    messages = example["messages"]
    full_text, prompt_text = build_full_and_prompt_text(tokenizer, messages)

    full = tokenizer(full_text, truncation=True, max_length=max_len, padding=False, return_tensors=None)
    prompt = tokenizer(prompt_text, truncation=True, max_length=max_len, padding=False, return_tensors=None)

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    prompt_len = len(prompt["input_ids"])
    labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[min(prompt_len, len(input_ids)) :]
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


def build_train_dataset(new_path: str, replay_path: Optional[str], replay_ratio: float, seed: int) -> Tuple[Dataset, str]:
    new_samples = load_dataset_any(new_path)
    if not new_samples:
        raise SystemExit(f"No valid samples loaded from new data: {new_path}")
    ds_new = Dataset.from_list(new_samples)

    manifest_lines = [f"new_data: {new_path}", f"new_count: {len(ds_new)}"]

    if replay_path and replay_ratio > 0:
        replay_samples = load_dataset_any(replay_path)
        if not replay_samples:
            raise SystemExit(f"Replay path provided but no valid samples: {replay_path}")
        ds_replay_all = Dataset.from_list(replay_samples)

        target_replay = int((replay_ratio / max(1e-9, (1.0 - replay_ratio))) * len(ds_new))
        target_replay = max(1, target_replay)

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

    manifest_lines += ["replay_data: (none)", "mixed_count: (same as new_count)"]
    return ds_new, "\n".join(manifest_lines)


def resolve_lora_target_modules(model, spec: str) -> List[str]:
    spec = (spec or "").strip()
    if spec and spec.lower() != "auto":
        modules = [m.strip() for m in spec.split(",") if m.strip()]
        if not modules:
            raise ValueError("--lora_target_modules provided but empty after parsing.")
        return modules

    module_names = set()
    for name, _ in model.named_modules():
        module_names.add(name.split(".")[-1])

    candidates = [
        ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
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
        sample = sorted(list(module_names))[:50]
        raise ValueError(
            "Could not infer LoRA target modules (auto). "
            "Pass --lora_target_modules explicitly. "
            f"Sample module names: {sample}"
        )
    return resolved


def pick_device(device_arg: str) -> torch.device:
    device_arg = (device_arg or "auto").lower().strip()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("--device mps requested, but MPS is not available in this PyTorch build.")
        return torch.device("mps")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    dtype_arg = (dtype_arg or "auto").lower().strip()
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    # auto
    if device.type == "mps":
        return torch.float16
    return torch.float32


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--replay_path", type=str, default=None)
    ap.add_argument("--replay_ratio", type=float, default=0.0)

    ap.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--resume_from_lora", type=str, default=None)
    ap.add_argument("--output_dir", type=str, required=True)

    # Apple Silicon / device
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"], help="Training device")
    ap.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype. On MPS default is float16; on CPU default is float32.",
    )
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="auto")

    args = ap.parse_args()
    set_seed(args.seed)

    device = pick_device(args.device)
    dtype = pick_dtype(args.torch_dtype, device)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump({**vars(args), "resolved_device": str(device), "resolved_dtype": str(dtype)}, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_raw, manifest = build_train_dataset(args.data_path, args.replay_path, float(args.replay_ratio), args.seed)
    (outdir / "data_manifest.txt").write_text(manifest + "\n", encoding="utf-8")

    tokenized = ds_raw.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_seq_len),
        remove_columns=ds_raw.column_names,
        desc="Tokenizing",
    )
    data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer)

    # Load base model (memory-efficient) then move to device
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )

    # Training safety / memory
    base.config.use_cache = False
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable()

    if device.type != "cpu":
        base.to(device)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=resolve_lora_target_modules(base, args.lora_target_modules),
    )

    if args.resume_from_lora:
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
        fp16=(dtype == torch.float16 and device.type != "cpu"),
        bf16=(dtype == torch.bfloat16 and device.type != "cpu"),
        report_to=[],
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(str(outdir))
    tokenizer.save_pretrained(str(outdir))
    print(f"Saved to: {outdir}")


if __name__ == "__main__":
    main()
