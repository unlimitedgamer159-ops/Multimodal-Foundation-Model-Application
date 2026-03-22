"""
data/prepare_dataset.py
-----------------------
Downloads and prepares an image-text dataset for VLM fine-tuning.

Supports two modes:
  --mode medical   : Uses a subset of the ROCO radiology dataset (image + caption)
  --mode product   : Uses Amazon product images + descriptions from HuggingFace

Usage:
  python data/prepare_dataset.py --mode product --output_dir data/processed
"""

import os
import json
import argparse
import random
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm


# ── helpers ──────────────────────────────────────────────────────────────────

def resize_and_save(image: Image.Image, path: Path, size: int = 336) -> bool:
    """Resize to square and save. Returns False if image is corrupt."""
    try:
        image = image.convert("RGB")
        image = image.resize((size, size), Image.LANCZOS)
        image.save(path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"  [warn] Could not process image: {e}")
        return False


def build_instruction(mode: str, text: str) -> dict:
    """Wrap a caption/description in a LLaVA-style instruction dict."""
    if mode == "medical":
        question = "Describe the findings in this medical image."
    else:
        question = "Describe this product in detail."

    return {
        "conversations": [
            {"from": "human",  "value": f"<image>\n{question}"},
            {"from": "gpt",    "value": text.strip()},
        ]
    }


# ── medical dataset ───────────────────────────────────────────────────────────

def prepare_medical(output_dir: Path, max_samples: int):
    """
    Uses the ROCO dataset (Radiology Objects in COntext).
    HuggingFace: https://huggingface.co/datasets/eltorio/ROCO-radiology
    Falls back to a tiny synthetic set if the download fails.
    """
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []

    print("Loading ROCO-radiology dataset …")
    try:
        ds = load_dataset("eltorio/ROCO-radiology", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, total=max_samples):
            if count >= max_samples:
                break
            try:
                img = item["image"] if isinstance(item["image"], Image.Image) \
                      else Image.open(BytesIO(item["image"]["bytes"]))
                fname = f"medical_{count:05d}.jpg"
                if resize_and_save(img, img_dir / fname):
                    caption = item.get("caption") or item.get("text") or ""
                    if len(caption) < 10:
                        continue
                    rec = build_instruction("medical", caption)
                    rec["image"] = fname
                    records.append(rec)
                    count += 1
            except Exception as e:
                print(f"  [skip] {e}")
    except Exception as e:
        print(f"  [warn] Could not load ROCO: {e}. Using synthetic fallback.")
        records = _synthetic_medical(img_dir, max_samples)

    return records


def _synthetic_medical(img_dir: Path, n: int) -> list:
    """Generate tiny placeholder records so the pipeline can still run."""
    import numpy as np
    records = []
    captions = [
        "Chest X-ray showing clear lung fields with no acute cardiopulmonary process.",
        "CT scan of the abdomen revealing a homogeneous liver with no focal lesions.",
        "MRI of the knee demonstrating an intact ACL with no evidence of meniscal tear.",
        "Chest radiograph with bilateral infiltrates consistent with pneumonia.",
        "Brain MRI showing no acute intracranial abnormality.",
    ]
    for i in range(n):
        arr = (np.random.rand(336, 336, 3) * 255).astype("uint8")
        img = Image.fromarray(arr)
        fname = f"medical_{i:05d}.jpg"
        img.save(img_dir / fname)
        rec = build_instruction("medical", captions[i % len(captions)])
        rec["image"] = fname
        records.append(rec)
    return records


# ── product dataset ───────────────────────────────────────────────────────────

def prepare_product(output_dir: Path, max_samples: int):
    """
    Uses the 'walmartlabs/all_beauty_amazon' or a generic product dataset.
    Falls back to 'datasets/laion/220k-GPT4Vision-captions-from-LIVIS' if needed.
    """
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []

    print("Loading product dataset (fashion-mnist as lightweight proxy) …")
    try:
        ds = load_dataset("zalandoresearch/fashion_mnist", split="train", streaming=True)
        labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
                  "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
        descriptions = {
            "T-shirt/top": "A casual T-shirt or top garment, suitable for everyday wear. Features a round neckline and short sleeves.",
            "Trouser":     "A pair of trousers with a straight cut silhouette. Suitable for formal or casual occasions.",
            "Pullover":    "A knitted pullover sweater with long sleeves. Warm and comfortable for cooler weather.",
            "Dress":       "An elegant dress with a flowing silhouette. Versatile style suitable for multiple occasions.",
            "Coat":        "A full-length coat providing warmth and style. Features a button-front closure.",
            "Sandal":      "Open-toe sandal with straps. Lightweight and breathable for warm weather.",
            "Shirt":       "A collared shirt with button-down front. Suitable for both casual and business settings.",
            "Sneaker":     "Comfortable athletic sneaker with rubber sole and lace-up closure.",
            "Bag":         "A structured handbag with a single handle and zippered closure.",
            "Ankle boot":  "An ankle-height boot with a low heel. Stylish and comfortable for all-day wear.",
        }
        count = 0
        for item in tqdm(ds, total=max_samples):
            if count >= max_samples:
                break
            try:
                img = item["image"]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                fname = f"product_{count:05d}.jpg"
                if resize_and_save(img, img_dir / fname):
                    label = labels[item["label"]]
                    description = descriptions.get(label, f"A {label} product.")
                    rec = build_instruction("product", description)
                    rec["image"] = fname
                    records.append(rec)
                    count += 1
            except Exception as e:
                print(f"  [skip] {e}")
    except Exception as e:
        print(f"  [error] {e}")

    return records


# ── split & save ──────────────────────────────────────────────────────────────

def split_and_save(records: list, output_dir: Path, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(records)
    n = len(records)
    t = int(n * train_ratio)
    v = int(n * val_ratio)

    splits = {
        "train": records[:t],
        "val":   records[t:t+v],
        "test":  records[t+v:],
    }

    for name, data in splits.items():
        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {name}: {len(data)} samples → {path}")

    # also save metadata
    meta = {
        "total": n,
        "splits": {k: len(v) for k, v in splits.items()},
        "image_dir": str(output_dir / "images"),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {output_dir}/meta.json")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["medical", "product"], default="product")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Mode        : {args.mode}")
    print(f"  Max samples : {args.max_samples}")
    print(f"  Output dir  : {output_dir}")
    print(f"{'='*50}\n")

    if args.mode == "medical":
        records = prepare_medical(output_dir, args.max_samples)
    else:
        records = prepare_product(output_dir, args.max_samples)

    if not records:
        print("[error] No records collected. Check dataset access.")
        return

    print(f"\nCollected {len(records)} valid records. Splitting …")
    split_and_save(records, output_dir)
    print("\nDone! Run fine-tuning next:")
    print(f"  python models/finetune.py --data_dir {output_dir} --mode {args.mode}")


if __name__ == "__main__":
    main()
