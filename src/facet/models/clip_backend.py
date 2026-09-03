"""CLIP image-embedding adapter.

Included as a contrasting representation for experiment E2: CLIP encodes styling, photo
quality and aesthetic convention, where ArcFace encodes identity-discriminative geometry.
Which of those better predicts attractiveness ratings is an open question and a known
confound - see docs/RESEARCH.md sections 2.3 and 13.1.

LICENSING: OpenAI CLIP is MIT - commercially usable.
"""
from __future__ import annotations

import numpy as np


class ClipEmbedder:
    commercial_use = True
    license = "MIT"

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", use_gpu: bool = True):
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.name = f"clip_{model_id.split('/')[-1]}"
        self.version = f"hf:{model_id}"
        self.torch = torch
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(self.device).eval()
        self.dim = int(self.model.config.projection_dim)

    def encode(self, crops: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """`crops`: (N, H, W, 3) BGR uint8. Returns (N, dim) L2-normalised float32."""
        outs = []
        for i in range(0, len(crops), batch_size):
            # .copy() is required: the ::-1 BGR->RGB view has a negative stride, which
            # torch.from_numpy rejects.
            batch = [np.ascontiguousarray(c[..., ::-1]) for c in crops[i : i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                e = self.model(**inputs).image_embeds
            e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            outs.append(e.cpu().numpy())
        return np.concatenate(outs).astype(np.float32)
