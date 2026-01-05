##下面給你一套「在 Mac Pro（CPU only） 也能完整跑起來」的最小可行訓練工程，包含：
# requirements.txt
# accelerate CPU 設定
# train.py（SFT + LoRA；支援你這種 conversation[{system,input,output}] 資料）
# infer.py（載入 base model + LoRA adapter 推理）
# 可選參數：8bit/4bit/DeepSpeed（但我會明確標註：Mac CPU 實務上通常不可用或不建議）
# 同時我會把模型選擇調整成 CPU 可承受：建議先用 Qwen2.5-0.5B-Instruct 或 Qwen2.5-1.5B-Instruct 做驗證（7B 以上 CPU 訓練基本不可行）。

# project/
#   requirements.txt
#   accelerate_cpu.yaml
#   train.py
#   infer.py
#   data/
#     tcm_dataset.json 

# 2) accelerate CPU 設定（accelerate_cpu.yaml）
# compute_environment: LOCAL_MACHINE
# debug: false
# distributed_type: NO
# downcast_bf16: "no"
# enable_cpu_affinity: true
# machine_rank: 0
# main_training_function: main
# mixed_precision: "no"
# num_machines: 1
# num_processes: 1
# rdzv_backend: static
# same_network: true
# tpu_use_cluster: false
# tpu_use_sudo: false
# use_cpu: true


"""
train.py (CPU-friendly, Mac-ready)
Mac CPU 友善、不依賴 TRL 的 SFTTrainer、支援 JSON / JSONL、支援 messages / conversation / instruction 三種資料格式、並且 只在 assistant tokens 計算 loss）。

Features:
- Works on CPU only (Mac Pro)
- No TRL dependency; uses Transformers Trainer for stability across versions
- Supports input data formats (mixed allowed):
  1) JSONL/JSON: {"messages":[{"role":"system","content":...},{"role":"user","content":...},{"role":"assistant","content":...}]}
  2) JSON: {"conversation":[{"system":..., "input":..., "output":...}]} (single-turn; first element used)
  3) JSON: {"instruction":..., "input":..., "output":...}
- Converts to chat text using model chat_template when available
- Creates labels with prompt masked as -100 => loss only on assistant completion
- Handles truncation/padding reliably => avoids nesting/pad tensor errors
- Trains LoRA adapters (PEFT) to keep CPU training feasible

Recommended model for CPU smoke tests:
  Qwen/Qwen2.5-0.5B-Instruct  (or 1.5B if you can tolerate more slowness)

Install:
  pip install -U torch transformers datasets peft accelerate sentencepiece protobuf safetensors

Run:
python train_v1.py \
  --data_path /Users/shushu/Development/python_llm/datasets/Sampled_Dataset.jsonl \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir /Users/shushu/Development/python_llm/outputs_tcm_lora \
  --max_seq_len 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --save_steps 50 \
  --logging_steps 5

"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model


# -----------------------------
# Data loading / normalization
# -----------------------------

def strip_or_empty(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def normalize_item_to_messages(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Returns {"messages":[...]} or None.
    """
    # A) already messages
    if "messages" in item and isinstance(item["messages"], list):
        msgs = item["messages"]
        # minimal validation
        roles = [m.get("role") for m in msgs if isinstance(m, dict)]
        if "user" in roles and "assistant" in roles:
            # ensure standard keys
            norm = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                content = strip_or_empty(m.get("content"))
                if role in ("system", "user", "assistant") and content != "":
                    norm.append({"role": role, "content": content})
            # ensure we have user/assistant at least
            roles2 = {m["role"] for m in norm}
            if "user" in roles2 and "assistant" in roles2:
                # add empty system if missing (ok)
                if "system" not in roles2:
                    norm = [{"role": "system", "content": ""}] + norm
                # Keep only first system/user/assistant for single-turn training
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
        # map instruction -> system, input -> user, output -> assistant
        if inp_ and out_:
            return {"messages": [
                {"role": "system", "content": instr},
                {"role": "user", "content": inp_},
                {"role": "assistant", "content": out_},
            ]}
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
                obj = json.loads(ln)
                handle_obj(obj)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                handle_obj(obj)
        elif isinstance(data, dict):
            # allow container {"data":[...]}
            if "data" in data and isinstance(data["data"], list):
                for obj in data["data"]:
                    handle_obj(obj)
            else:
                handle_obj(data)

    return samples


# -----------------------------
# Chat template -> text
# -----------------------------

def apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """
    Uses model chat_template if present; otherwise falls back to a simple template.
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

    # Fallback: simple tags
    sys = messages[0]["content"]
    user = messages[1]["content"]
    assistant = messages[2]["content"] if len(messages) > 2 else ""
    if add_generation_prompt:
        return f"<system>\n{sys}\n</system>\n<user>\n{user}\n</user>\n<assistant>\n"
    return f"<system>\n{sys}\n</system>\n<user>\n{user}\n</user>\n<assistant>\n{assistant}\n</assistant>\n"


def build_full_and_prompt_text(tokenizer, messages: List[Dict[str, str]]) -> (str, str):
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
    # mask prompt tokens
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
# Main training
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--output_dir", type=str, default="outputs_tcm_lora")
    ap.add_argument("--seed", type=int, default=42)

    # CPU friendly
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CPU load
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
        device_map=None,
    )

    # LoRA targets (common for Qwen2.5)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    raw_samples = load_dataset_any(args.data_path)
    if not raw_samples:
        raise SystemExit("No valid samples loaded. Check your data_path and format.")

    ds = Dataset.from_list(raw_samples)

    tokenized = ds.map(
        lambda ex: tokenize_and_mask(ex, tokenizer, args.max_seq_len),
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to=[],
        optim="adamw_torch",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA adapter and tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
