"""
models/inference.py
-------------------
Shared inference class used by both the evaluation suite and the API.

Loads a fine-tuned (or base) VLM and runs image+prompt → text generation.
Handles both LLaVA-style and BLIP-2-style models transparently.
"""

import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


class VLMInference:
    """
    Thin wrapper around a HuggingFace vision-language model.

    Args:
        checkpoint: path to fine-tuned checkpoint, OR a HF model id string.
        device: 'cuda', 'cpu', or None (auto-detect).
        max_new_tokens: generation budget.
    """

    DEFAULT_MODEL = "Salesforce/blip2-opt-2.7b"   # CPU-safe default

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 128,
    ):
        self.checkpoint = checkpoint or self.DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self._load()

    def _load(self):
        print(f"[VLMInference] Loading '{self.checkpoint}' on {self.device} …")
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        print("[VLMInference] Ready.")

    @torch.inference_mode()
    def generate(self, image: Image.Image, prompt: str) -> str:
        """
        Run a single image + text prompt through the model.

        Returns the generated text string.
        """
        image = image.convert("RGB")
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        # Strip the prompt tokens from the output
        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def batch_generate(self, items: list[dict]) -> list[str]:
        """
        Generate for a list of {'image': PIL.Image, 'prompt': str} dicts.
        Falls back to sequential to keep memory low on CPU.
        """
        return [self.generate(it["image"], it["prompt"]) for it in items]
