"""
models/finetune.py
------------------
Fine-tunes LLaVA-1.5-7B (or a smaller mock model for Codespaces CPU)
on your custom image-text dataset using LoRA / QLoRA.

Usage:
  # Full fine-tune (needs GPU):
  python models/finetune.py --data_dir data/processed --mode product

  # Codespaces CPU-safe dry run (tiny model, 2 steps):
  python models/finetune.py --data_dir data/processed --dry_run

After training, weights are saved to models/checkpoints/
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    # Model
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    dry_run_model: str = "Salesforce/blip2-opt-2.7b"   # small CPU-friendly fallback

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj")

    # Training
    num_epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    fp16: bool = False          # set True if GPU
    bf16: bool = False          # set True if Ampere GPU

    # I/O
    output_dir: str = "models/checkpoints"
    logging_steps: int = 10
    save_steps: int = 100
    max_seq_len: int = 512


CFG = FinetuneConfig()


# ── dataset ───────────────────────────────────────────────────────────────────

class VLMDataset(Dataset):
    def __init__(self, json_path: Path, image_dir: Path, processor, max_len: int = 512):
        with open(json_path) as f:
            self.records = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.image_dir / rec["image"]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (336, 336), color=(128, 128, 128))

        # Build text from conversation
        convs = rec.get("conversations", [])
        human_text = next((c["value"] for c in convs if c["from"] == "human"), "Describe this image.")
        gpt_text   = next((c["value"] for c in convs if c["from"] == "gpt"),   "")

        # Strip <image> token from prompt — processor adds it
        human_text = human_text.replace("<image>\n", "").replace("<image>", "").strip()
        full_text  = f"USER: {human_text}\nASSISTANT: {gpt_text}"

        encoding = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        # Flatten batch dim added by processor
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = item["input_ids"].clone()
        return item


# ── LoRA setup ────────────────────────────────────────────────────────────────

def apply_lora(model, cfg: FinetuneConfig):
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── trainer ───────────────────────────────────────────────────────────────────

def build_trainer(model, processor, train_ds, val_ds, cfg: FinetuneConfig, dry_run: bool):
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=1 if dry_run else cfg.num_epochs,
        max_steps=2 if dry_run else -1,
        per_device_train_batch_size=1 if dry_run else cfg.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if dry_run else cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_dir=f"{cfg.output_dir}/logs",
        logging_steps=1 if dry_run else cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="epoch" if not dry_run else "no",
        save_total_limit=2,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if not dry_run else None,
        tokenizer=processor.tokenizer,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="data/processed")
    parser.add_argument("--mode",      choices=["medical", "product"], default="product")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Use a small model for 2 steps — safe on Codespaces CPU")
    parser.add_argument("--use_qlora", action="store_true",
                        help="Use 4-bit quantisation (needs GPU + bitsandbytes)")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    image_dir = data_dir / "images"
    ckpt_dir  = Path(CFG.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_id = CFG.dry_run_model if args.dry_run else CFG.model_name
    print(f"\nModel  : {model_id}")
    print(f"Dry run: {args.dry_run}")
    print(f"QLoRA  : {args.use_qlora}\n")

    # ── load processor ──
    print("Loading processor …")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # ── load model ──
    print("Loading model …")
    quant_cfg = None
    if args.use_qlora and not args.dry_run:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32 if args.dry_run else torch.float16,
        trust_remote_code=True,
    )

    # ── apply LoRA ──
    if not args.dry_run:
        model = apply_lora(model, CFG)
    else:
        print("[dry-run] Skipping LoRA — using base model weights for smoke test")

    # ── datasets ──
    print("Building datasets …")
    train_ds = VLMDataset(data_dir / "train.json", image_dir, processor, CFG.max_seq_len)
    val_ds   = VLMDataset(data_dir / "val.json",   image_dir, processor, CFG.max_seq_len)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    # ── train ──
    print("\nStarting training …")
    trainer = build_trainer(model, processor, train_ds, val_ds, CFG, args.dry_run)
    trainer.train()

    # ── save ──
    save_path = ckpt_dir / "final"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"\nModel saved → {save_path}")
    print("Next step: python eval/run_eval.py --checkpoint models/checkpoints/final")


if __name__ == "__main__":
    main()
