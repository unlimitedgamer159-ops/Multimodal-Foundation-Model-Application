"""
api/main.py
-----------
FastAPI service exposing the fine-tuned VLM.

Endpoints:
  GET  /              — health check + model info
  POST /predict       — image + optional prompt → generated text
  POST /predict/batch — list of images + prompts → list of texts
  GET  /eval          — latest evaluation metrics (reads eval/results/results.json)
  POST /predict/file  — upload an image file directly

Run:
  python api/main.py
  # or
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import json
import sys
import base64
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.inference import VLMInference


# ── app setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multimodal VLM API",
    description="Vision-language model fine-tuned on custom image-text pairs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── model (lazy-loaded on first request) ─────────────────────────────────────

_model: Optional[VLMInference] = None
CHECKPOINT_DIR = Path("models/checkpoints/final")
EVAL_RESULTS   = Path("eval/results/results.json")


def get_model() -> VLMInference:
    global _model
    if _model is None:
        checkpoint = str(CHECKPOINT_DIR) if CHECKPOINT_DIR.exists() else None
        _model = VLMInference(checkpoint=checkpoint)
    return _model


# ── schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image (JPEG/PNG)")
    prompt: str    = Field(
        default="Describe this image.",
        description="Text prompt to condition generation"
    )
    max_new_tokens: int = Field(default=128, ge=1, le=512)


class PredictResponse(BaseModel):
    prediction:   str
    latency_s:    float
    model_used:   str
    prompt_used:  str


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_latency_s: float


# ── helpers ───────────────────────────────────────────────────────────────────

def b64_to_pil(b64_str: str) -> Image.Image:
    try:
        # Strip data URI prefix if present
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    checkpoint = str(CHECKPOINT_DIR) if CHECKPOINT_DIR.exists() else "base (no fine-tuned checkpoint found)"
    return {
        "status": "ok",
        "model": checkpoint,
        "endpoints": ["/predict", "/predict/batch", "/predict/file", "/eval"],
    }


@app.post("/predict", response_model=PredictResponse, summary="Single image prediction")
def predict(req: PredictRequest):
    model = get_model()
    image = b64_to_pil(req.image_b64)

    model.max_new_tokens = req.max_new_tokens
    t0   = time.time()
    pred = model.generate(image, req.prompt)
    lat  = round(time.time() - t0, 3)

    return PredictResponse(
        prediction=pred,
        latency_s=lat,
        model_used=model.checkpoint,
        prompt_used=req.prompt,
    )


@app.post("/predict/file", summary="Upload image file directly")
async def predict_file(
    file: UploadFile = File(...),
    prompt: str = "Describe this image.",
    max_new_tokens: int = 128,
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    model = get_model()
    raw   = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")

    model.max_new_tokens = max_new_tokens
    t0   = time.time()
    pred = model.generate(image, prompt)
    lat  = round(time.time() - t0, 3)

    return {
        "filename":    file.filename,
        "prediction":  pred,
        "latency_s":   lat,
        "model_used":  model.checkpoint,
        "prompt_used": prompt,
    }


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="Batch predictions")
def predict_batch(req: BatchPredictRequest):
    if len(req.items) > 16:
        raise HTTPException(status_code=400, detail="Max 16 items per batch")

    model = get_model()
    results = []
    t_total = time.time()

    for item in req.items:
        image = b64_to_pil(item.image_b64)
        model.max_new_tokens = item.max_new_tokens
        t0   = time.time()
        pred = model.generate(image, item.prompt)
        lat  = round(time.time() - t0, 3)
        results.append(PredictResponse(
            prediction=pred,
            latency_s=lat,
            model_used=model.checkpoint,
            prompt_used=item.prompt,
        ))

    return BatchPredictResponse(
        results=results,
        total_latency_s=round(time.time() - t_total, 3),
    )


@app.get("/eval", summary="Latest evaluation metrics")
def get_eval():
    if not EVAL_RESULTS.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation results found. Run: python eval/run_eval.py"
        )
    with open(EVAL_RESULTS) as f:
        return json.load(f)


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
