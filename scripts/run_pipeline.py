"""
scripts/run_pipeline.py
-----------------------
Runs the full pipeline end-to-end:
  1. Prepare dataset
  2. Fine-tune (dry run by default)
  3. Evaluate
  4. Start API

Designed to work on GitHub Codespaces (CPU, 8 GB RAM) in dry-run mode.
Pass --full for a proper GPU run (takes ~2–6 hrs depending on hardware).

Usage:
  python scripts/run_pipeline.py             # dry run — safe on Codespaces
  python scripts/run_pipeline.py --full      # full training
  python scripts/run_pipeline.py --skip_train  # data + eval + API only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], step: str):
    print(f"\n{'='*55}")
    print(f"  STEP: {step}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'='*55}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  → {status}  ({elapsed:.1f}s)")
    if result.returncode != 0:
        print("  Continuing to next step …")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        choices=["medical", "product"], default="product")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--full",        action="store_true", help="Full training (needs GPU)")
    parser.add_argument("--skip_train",  action="store_true")
    parser.add_argument("--skip_api",    action="store_true")
    args = parser.parse_args()

    py = sys.executable
    dry = not args.full

    print("\n" + "★"*55)
    print("  Multimodal VLM — Full Pipeline")
    print(f"  Mode      : {args.mode}")
    print(f"  Dry run   : {dry}")
    print(f"  Samples   : {args.max_samples}")
    print("★"*55)

    # 1. Dataset
    run(
        [py, "data/prepare_dataset.py",
         "--mode", args.mode,
         "--max_samples", str(args.max_samples),
         "--output_dir", "data/processed"],
        "Prepare dataset"
    )

    # 2. Fine-tune
    if not args.skip_train:
        ft_cmd = [py, "models/finetune.py",
                  "--data_dir", "data/processed",
                  "--mode", args.mode]
        if dry:
            ft_cmd.append("--dry_run")
        run(ft_cmd, "Fine-tune VLM")

    # 3. Evaluate
    eval_cmd = [py, "eval/run_eval.py",
                "--data_dir", "data/processed",
                "--output_dir", "eval/results"]
    if dry:
        eval_cmd.append("--dry_run")
    run(eval_cmd, "Evaluate model")

    # 4. API
    if not args.skip_api:
        print("\n" + "="*55)
        print("  STEP: Start API")
        print("  The API will start on port 8000.")
        print("  In Codespaces, click the 'Open in Browser' popup,")
        print("  or open the PORTS tab and click the local address.")
        print("  Run tests with: python api/test_api.py")
        print("="*55)
        subprocess.run([py, "api/main.py"])


if __name__ == "__main__":
    main()
