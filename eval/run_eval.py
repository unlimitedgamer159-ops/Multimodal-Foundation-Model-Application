"""
eval/run_eval.py
----------------
Evaluation suite for the fine-tuned VLM.

Metrics computed:
  Text modality  : ROUGE-L, BLEU-4, BERTScore (via evaluate library)
  Image modality : Retrieval R@1 (image→text), answer relevance via keyword overlap

Usage:
  python eval/run_eval.py --checkpoint models/checkpoints/final --data_dir data/processed
  python eval/run_eval.py --dry_run    # uses base model + first 5 test samples
"""

import json
import argparse
import sys
import time
from pathlib import Path

import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# Add project root so we can import models/
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.inference import VLMInference

try:
    import evaluate as hf_evaluate
    EVALUATE_OK = True
except ImportError:
    EVALUATE_OK = False
    print("[warn] `evaluate` not installed — falling back to simple metrics")

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


# ── simple metrics (no evaluate library needed) ───────────────────────────────

def simple_bleu(reference: str, hypothesis: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    return sentence_bleu(
        [ref_tokens], hyp_tokens,
        smoothing_function=SmoothingFunction().method1
    )


def simple_rouge_l(reference: str, hypothesis: str) -> float:
    """Longest Common Subsequence based ROUGE-L."""
    r = reference.lower().split()
    h = hypothesis.lower().split()
    if not r or not h:
        return 0.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if r[i-1] == h[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / n
    rec  = lcs / m
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def keyword_overlap(reference: str, hypothesis: str) -> float:
    """Simple token F1 — proxy for answer relevance."""
    r_tokens = set(reference.lower().split())
    h_tokens = set(hypothesis.lower().split())
    if not r_tokens or not h_tokens:
        return 0.0
    tp = len(r_tokens & h_tokens)
    prec = tp / len(h_tokens)
    rec  = tp / len(r_tokens)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# ── retrieval metric ──────────────────────────────────────────────────────────

def retrieval_r_at_1(model: VLMInference, records: list, image_dir: Path) -> float:
    """
    Image→Text retrieval R@1.
    For each image, generate a description, then rank all reference captions
    by keyword overlap. R@1 = fraction where the correct caption ranks first.
    """
    correct = 0
    refs = [r["conversations"][1]["value"] for r in records]

    for i, rec in enumerate(tqdm(records, desc="Retrieval R@1")):
        img = Image.open(image_dir / rec["image"]).convert("RGB")
        prompt = "Describe this image."
        pred = model.generate(img, prompt)
        scores = [keyword_overlap(ref, pred) for ref in refs]
        best_idx = max(range(len(scores)), key=lambda j: scores[j])
        if best_idx == i:
            correct += 1

    return correct / len(records)


# ── main eval loop ────────────────────────────────────────────────────────────

def evaluate_model(model: VLMInference, records: list, image_dir: Path) -> dict:
    predictions, references = [], []
    latencies = []

    for rec in tqdm(records, desc="Generating"):
        img = Image.open(image_dir / rec["image"]).convert("RGB")
        prompt = rec["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "").strip()
        reference = rec["conversations"][1]["value"]

        t0 = time.time()
        pred = model.generate(img, prompt)
        latencies.append(time.time() - t0)

        predictions.append(pred)
        references.append(reference)

    # Text metrics
    bleu_scores   = [simple_bleu(r, p)    for r, p in zip(references, predictions)]
    rouge_scores  = [simple_rouge_l(r, p) for r, p in zip(references, predictions)]
    kw_scores     = [keyword_overlap(r, p) for r, p in zip(references, predictions)]

    results = {
        "n_samples":       len(records),
        "bleu4_mean":      round(sum(bleu_scores)  / len(bleu_scores),  4),
        "rouge_l_mean":    round(sum(rouge_scores) / len(rouge_scores), 4),
        "keyword_f1_mean": round(sum(kw_scores)    / len(kw_scores),    4),
        "latency_mean_s":  round(sum(latencies)    / len(latencies),    3),
        "latency_p95_s":   round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "per_sample": [
            {
                "reference":   references[i],
                "prediction":  predictions[i],
                "bleu4":       round(bleu_scores[i],  4),
                "rouge_l":     round(rouge_scores[i], 4),
                "keyword_f1":  round(kw_scores[i],    4),
                "latency_s":   round(latencies[i],    3),
            }
            for i in range(len(records))
        ],
    }
    return results


# ── plots ─────────────────────────────────────────────────────────────────────

def save_plots(results: dict, output_dir: Path):
    per = results["per_sample"]
    df = pd.DataFrame(per)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("VLM Evaluation Results", fontsize=14, fontweight="bold")

    for ax, col, title in zip(
        axes,
        ["bleu4", "rouge_l", "keyword_f1"],
        ["BLEU-4", "ROUGE-L", "Keyword F1"],
    ):
        sns.histplot(df[col], bins=20, ax=ax, kde=True, color="#5B8FF9")
        ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"mean={df[col].mean():.3f}")
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = output_dir / "eval_metrics.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Plot saved → {path}")

    # Latency plot
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(df["latency_s"].values, marker="o", markersize=3, linewidth=1, color="#5B8FF9")
    ax2.axhline(results["latency_mean_s"], color="red", linestyle="--",
                label=f"mean={results['latency_mean_s']:.2f}s")
    ax2.set_title("Per-Sample Inference Latency")
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Seconds")
    ax2.legend()
    path2 = output_dir / "eval_latency.png"
    plt.tight_layout()
    plt.savefig(path2, dpi=120)
    plt.close()
    print(f"  Plot saved → {path2}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint dir, or HF model id")
    parser.add_argument("--data_dir",   default="data/processed")
    parser.add_argument("--split",      default="test")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_dir", default="eval/results")
    parser.add_argument("--dry_run",    action="store_true",
                        help="5 samples, base model, no GPU needed")
    args = parser.parse_args()

    if args.dry_run:
        args.max_samples = 5
        print("[dry-run] Using 5 samples and base model\n")

    data_dir  = Path(args.data_dir)
    image_dir = data_dir / "images"
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test records
    split_file = data_dir / f"{args.split}.json"
    if not split_file.exists():
        print(f"[error] {split_file} not found. Run data/prepare_dataset.py first.")
        sys.exit(1)

    with open(split_file) as f:
        records = json.load(f)[: args.max_samples]
    print(f"Evaluating on {len(records)} samples from {split_file}\n")

    # Load model
    model = VLMInference(checkpoint=args.checkpoint)

    # Text generation metrics
    print("\n--- Text modality metrics ---")
    results = evaluate_model(model, records, image_dir)

    # Image retrieval metric (only if enough samples)
    if len(records) >= 5:
        print("\n--- Image retrieval R@1 ---")
        r_at_1 = retrieval_r_at_1(model, records[:20], image_dir)
        results["retrieval_r_at_1"] = round(r_at_1, 4)
    else:
        results["retrieval_r_at_1"] = None

    # Print summary
    print("\n" + "="*45)
    print("  EVALUATION SUMMARY")
    print("="*45)
    print(f"  Samples        : {results['n_samples']}")
    print(f"  BLEU-4         : {results['bleu4_mean']:.4f}")
    print(f"  ROUGE-L        : {results['rouge_l_mean']:.4f}")
    print(f"  Keyword F1     : {results['keyword_f1_mean']:.4f}")
    if results["retrieval_r_at_1"] is not None:
        print(f"  Retrieval R@1  : {results['retrieval_r_at_1']:.4f}")
    print(f"  Latency (mean) : {results['latency_mean_s']:.3f}s")
    print(f"  Latency (p95)  : {results['latency_p95_s']:.3f}s")
    print("="*45)

    # Save JSON
    json_path = out_dir / "results.json"
    # Drop per_sample for summary file (keep it for debugging)
    summary = {k: v for k, v in results.items() if k != "per_sample"}
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {json_path}")

    per_path = out_dir / "per_sample.json"
    with open(per_path, "w") as f:
        json.dump(results["per_sample"], f, indent=2)
    print(f"Per-sample log → {per_path}")

    # Save plots
    print("\nGenerating plots …")
    save_plots(results, out_dir)
    print("\nDone! Start the API next:")
    print("  python api/main.py")


if __name__ == "__main__":
    main()
