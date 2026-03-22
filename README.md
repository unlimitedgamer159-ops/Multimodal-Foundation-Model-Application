# Multimodal Foundation Model — VLM Fine-tuning + API

Fine-tune a vision-language model (LLaVA / BLIP-2) on a custom image-text dataset,
evaluate it across both modalities, and serve it via a REST API.

## Project layout

```
multimodal-vlm/
├── .devcontainer/devcontainer.json   # GitHub Codespaces config
├── requirements.txt
│
├── data/
│   └── prepare_dataset.py            # Download + preprocess image-text pairs
│
├── models/
│   ├── finetune.py                   # LoRA / QLoRA fine-tuning
│   └── inference.py                  # Shared inference wrapper
│
├── eval/
│   └── run_eval.py                   # BLEU-4 · ROUGE-L · retrieval R@1 · latency
│
├── api/
│   ├── main.py                       # FastAPI service
│   └── test_api.py                   # Endpoint smoke tests
│
└── scripts/
    └── run_pipeline.py               # One command: data → train → eval → API
```

---

## Quick start (GitHub Codespaces — CPU, dry run)

```bash
# 1. Install dependencies (done automatically by devcontainer, or run manually)
pip install -r requirements.txt

# 2. Run the full pipeline in dry-run mode (CPU-safe, ~5 min)
python scripts/run_pipeline.py

# 3. In a second terminal, test the API
python api/test_api.py
```

---

## Step-by-step

### Step 1 — Prepare dataset

```bash
# Product images (fashion-mnist proxy, downloads ~50 MB)
python data/prepare_dataset.py --mode product --max_samples 500

# Medical images (ROCO radiology, needs HF account for some splits)
python data/prepare_dataset.py --mode medical --max_samples 500
```

Outputs to `data/processed/`:
- `train.json`, `val.json`, `test.json`  — LLaVA-format conversation records
- `images/`  — resized 336×336 JPEGs
- `meta.json`

---

### Step 2 — Fine-tune

**Dry run (CPU, 2 steps, verifies the pipeline):**
```bash
python models/finetune.py --dry_run
```

**Full fine-tune (needs GPU — e.g. Codespaces with GPU or Colab):**
```bash
python models/finetune.py --data_dir data/processed --mode product
```

**QLoRA (4-bit, GPU only):**
```bash
python models/finetune.py --data_dir data/processed --use_qlora
```

Checkpoint saved to `models/checkpoints/final/`.

---

### Step 3 — Evaluate

```bash
# Dry run (5 samples, base model)
python eval/run_eval.py --dry_run

# Full eval against fine-tuned checkpoint
python eval/run_eval.py \
  --checkpoint models/checkpoints/final \
  --data_dir data/processed \
  --max_samples 200
```

Outputs:
- `eval/results/results.json`  — summary metrics
- `eval/results/per_sample.json`
- `eval/results/eval_metrics.png`
- `eval/results/eval_latency.png`

**Metrics explained:**

| Metric | What it measures |
|---|---|
| BLEU-4 | n-gram precision vs reference text |
| ROUGE-L | Longest common subsequence F1 |
| Keyword F1 | Token overlap — proxy for answer relevance |
| Retrieval R@1 | Does the model's output best match the correct caption? |
| Latency p95 | 95th-percentile inference time per sample |

---

### Step 4 — API

```bash
# Start server
python api/main.py
# → http://localhost:8000
```

In Codespaces, port 8000 is forwarded automatically. Click the notification or
open the **PORTS** tab.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check + model info |
| POST | `/predict` | Base64 image + prompt → text |
| POST | `/predict/file` | Upload image file directly |
| POST | `/predict/batch` | Up to 16 items at once |
| GET | `/eval` | Latest eval metrics |

**Interactive docs:** `http://localhost:8000/docs`

**Example curl:**
```bash
# Encode an image
B64=$(base64 -w 0 my_image.jpg)

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$B64\", \"prompt\": \"Describe this image.\", \"max_new_tokens\": 100}"
```

---

## Model choices

| Model | VRAM | Notes |
|---|---|---|
| `Salesforce/blip2-opt-2.7b` | ~6 GB (fp16) | Default, CPU-runnable in fp32 |
| `llava-hf/llava-1.5-7b-hf` | ~14 GB | Best quality, needs GPU |
| `llava-hf/llava-1.5-13b-hf` | ~26 GB | Highest quality |

Change the model in `models/finetune.py` → `FinetuneConfig.model_name`.

---

## LoRA config (models/finetune.py)

```python
lora_r         = 16     # rank — higher = more params
lora_alpha     = 32     # scaling factor
lora_dropout   = 0.05
target_modules = ("q_proj", "v_proj", "k_proj", "o_proj")
```

---

## Environment variables (optional)

Create a `.env` file:
```
HF_TOKEN=hf_...        # for gated HuggingFace datasets / models
CUDA_VISIBLE_DEVICES=0
```
