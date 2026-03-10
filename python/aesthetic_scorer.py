"""
Aesthetic Scorer — async CLIP-based frame quality estimator.

Scores each generated frame against positive/negative aesthetic anchors:
    score = sim(frame, POSITIVE_TEXTS) - sim(frame, NEGATIVE_TEXTS)  → [0, 1]

Runs in a background thread so it never blocks the generation loop.
The score is fed back into Susa's PerformanceMemory so prompt elements
that produce beautiful frames get reinforced over time (the learning loop).

Usage:
    scorer = AestheticScorer()
    scorer.start()

    def on_score(score, context):
        susa.record_aesthetic(context["tokens"], score)

    scorer.submit(pil_image, on_score, context={"tokens": last_tokens})
"""

import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

# ── Aesthetic anchor texts ────────────────────────────────────────────────────
# Positive: what we want the AI visuals to look like
_POSITIVE = [
    "stunning vibrant colorful abstract digital art",
    "beautiful high quality photorealistic art",
    "vivid neon glowing luminous artwork",
    "incredible detailed cinematic masterpiece",
    "rich saturated vibrant dynamic composition",
]

# Negative: what we want to avoid
_NEGATIVE = [
    "ugly blurry low quality image",
    "distorted pixelated artifact noise",
    "dark muddy dull desaturated boring",
    "poorly rendered low resolution garbage",
]

CLIP_MODEL = "openai/clip-vit-base-patch32"

# Score normalization params — tuned empirically for CLIP ViT-B/32 in v18 environment
_SCORE_CENTER = 0.01   # centered on observed raw delta (avg ~0.1-0.2 becomes 0.5)
_SCORE_RANGE  = 0.15   # +/- range around center for increased sensitivity


class AestheticScorer:
    """
    Async CLIP-based aesthetic scorer.

    Attributes
    ----------
    ready : bool
        True once CLIP is loaded and scoring is available.
    recent_scores : deque[float]
        Ring buffer of recent scores for monitoring.
    """

    def __init__(self):
        self._model     = None
        self._processor = None
        self._pos_emb   = None   # (1, D) averaged positive text embedding
        self._neg_emb   = None   # (1, D) averaged negative text embedding

        self._queue   = deque(maxlen=8)
        self._lock    = threading.Lock()
        self._running = False
        self.ready    = False
        self.recent_scores: deque[float] = deque(maxlen=120)

    def start(self):
        """Non-blocking start — loads CLIP and begins scoring in background."""
        self._running = True
        t = threading.Thread(target=self._init_and_run, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(
        self,
        image: Image.Image,
        callback: Callable,
        context: Optional[dict] = None,
    ):
        """
        Submit a frame for async aesthetic scoring.

        Parameters
        ----------
        image    : PIL Image to score
        callback : called as callback(score: float, context: dict) when done
        context  : arbitrary dict passed back to callback (e.g. {"tokens": [...]})
        """
        if not self.ready:
            return
        with self._lock:
            self._queue.append((image.copy(), callback, context or {}))

    def score_sync(self, image: Image.Image) -> float:
        """Synchronous score — blocks until complete. Only use when not in loop."""
        if not self.ready:
            return 0.5
        return self._score_image(image)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_and_run(self):
        try:
            from transformers import CLIPModel, CLIPProcessor

            print("[SCORER] Loading CLIP ViT-B/32 for aesthetic learning...")
            # Use CPU to avoid VRAM contention with SD pipeline
            self._model = CLIPModel.from_pretrained(
                CLIP_MODEL, torch_dtype=torch.float32
            ).to("cpu")
            self._model.eval()
            self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

            self._pos_emb = self._encode_texts(_POSITIVE)
            self._neg_emb = self._encode_texts(_NEGATIVE)

            self.ready = True
            print("[SCORER] Aesthetic scorer ready — learning loop active.")

        except Exception as e:
            print(f"[SCORER] CLIP load failed: {e}. Aesthetic scoring disabled.")
            self.ready = False
            return

        while self._running:
            with self._lock:
                item = self._queue.popleft() if self._queue else None

            if item is None:
                time.sleep(0.04)
                continue

            image, callback, context = item
            try:
                score = self._score_image(image)
                self.recent_scores.append(score)
                callback(score, context)
            except Exception as e:
                print(f"[SCORER] Scoring error: {e}")

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of texts and return mean unit-norm embedding (1, D)."""
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to("cpu")
        with torch.inference_mode():
            emb = self._model.get_text_features(**inputs).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.mean(dim=0, keepdim=True)   # (1, D)

    def _score_image(self, image: Image.Image) -> float:
        """
        Score a PIL image. Returns float in [0, 1].
        0 = ugly/bad, 0.5 = neutral, 1 = beautiful/vibrant.
        """
        img_small = image.resize((224, 224), Image.BILINEAR)
        inputs = self._processor(images=img_small, return_tensors="pt").to("cpu")

        with torch.inference_mode():
            img_emb = self._model.get_image_features(**inputs).float()
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        pos = float((img_emb @ self._pos_emb.T).squeeze())
        neg = float((img_emb @ self._neg_emb.T).squeeze())
        raw = pos - neg

        # Map to [0, 1] — center at neutral, clip extremes
        score = (raw - _SCORE_CENTER) / _SCORE_RANGE + 0.5
        return float(np.clip(score, 0.0, 1.0))

    def stats(self) -> dict:
        """Return scoring stats for monitoring."""
        if not self.recent_scores:
            return {"avg": 0.5, "min": 0.0, "max": 1.0, "n": 0}
        arr = list(self.recent_scores)
        return {
            "avg": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n":   len(arr),
        }
