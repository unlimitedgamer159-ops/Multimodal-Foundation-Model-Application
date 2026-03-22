"""
api/test_api.py
---------------
Quick smoke-test for all API endpoints.
Generates a synthetic test image so no dataset is needed.

Usage:
  # with the API running in another terminal:
  python api/test_api.py

  # or point to a different host:
  python api/test_api.py --base_url http://localhost:8000
"""

import argparse
import base64
import io
import json
import sys
import time

import httpx
from PIL import Image, ImageDraw, ImageFont
import numpy as np


BASE_URL = "http://localhost:8000"


def make_test_image(text: str = "TEST") -> str:
    """Create a simple synthetic image and return as base64."""
    img = Image.new("RGB", (336, 336), color=(200, 220, 240))
    draw = ImageDraw.Draw(img)
    # Draw a rectangle and label
    draw.rectangle([60, 60, 276, 276], outline=(80, 80, 200), width=4)
    draw.text((130, 150), text, fill=(30, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def print_result(name: str, resp: httpx.Response):
    ok = "✓" if resp.status_code < 300 else "✗"
    print(f"\n{ok}  {name}  [{resp.status_code}]")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2)[:800])
    except Exception:
        print(resp.text[:400])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default=BASE_URL)
    args = parser.parse_args()
    url = args.base_url.rstrip("/")

    client = httpx.Client(timeout=120)
    img_b64 = make_test_image("VLM")

    # 1. Health check
    r = client.get(f"{url}/")
    print_result("GET /", r)

    # 2. Single predict
    payload = {
        "image_b64":      img_b64,
        "prompt":         "What do you see in this image?",
        "max_new_tokens": 60,
    }
    r = client.post(f"{url}/predict", json=payload)
    print_result("POST /predict", r)

    # 3. Batch predict
    batch_payload = {
        "items": [
            {"image_b64": img_b64, "prompt": "Describe this image.", "max_new_tokens": 50},
            {"image_b64": img_b64, "prompt": "What colors are present?", "max_new_tokens": 40},
        ]
    }
    r = client.post(f"{url}/predict/batch", json=batch_payload)
    print_result("POST /predict/batch", r)

    # 4. File upload
    buf = io.BytesIO(base64.b64decode(img_b64))
    buf.name = "test.jpg"
    r = client.post(
        f"{url}/predict/file",
        files={"file": ("test.jpg", buf, "image/jpeg")},
        params={"prompt": "Describe this image.", "max_new_tokens": 50},
    )
    print_result("POST /predict/file", r)

    # 5. Eval metrics
    r = client.get(f"{url}/eval")
    print_result("GET /eval", r)

    client.close()
    print("\nAll tests done.")


if __name__ == "__main__":
    main()
