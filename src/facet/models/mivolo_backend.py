"""MiVOLO age/gender adapter.

MiVOLO is the production age/gender recommendation from docs/RESEARCH.md 2.5: Apache-2.0
code *and* weights, unlike everything else in the stack.

**Licensing care taken here.** MiVOLO's repository depends on Ultralytics YOLOv8 for its
detector, which is AGPL-3.0 and network-viral. We already have SCRFD, so this adapter uses
only MiVOLO's *model definition* (vendored under `facet.third_party.mivolo`, Apache-2.0) with
our own detector. `yolo_detector.py` is deliberately not vendored and `ultralytics` is never
imported - there is a test asserting this.

Residual licensing note recorded in docs/LICENSING.md: MiVOLO's weights are Apache-2.0 but
were trained partly on IMDB-clean, which derives from IMDb data under non-commercial terms.
Apache-2.0 on weights does not launder training-data provenance. A research/personal system
is unaffected; a commercial one needs advice.
"""
from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving aspect ratio, pad to square - MiVOLO's own preprocessing."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    out = np.full((size, size, 3), 114, dtype=img.dtype)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top : top + nh, left : left + nw] = resized
    return out


class MiVOLOPredictor:
    """Face-only MiVOLO inference.

    MiVOLO v2 takes 6 channels: a face crop and a person crop concatenated. We detect faces,
    not people, so the person half is filled the way MiVOLO itself fills a missing body -
    a zero image passed through the same normalisation. This is MiVOLO's documented
    face-only mode, not a hack, but it does forgo the body cues that are part of why the
    published numbers are as good as they are.
    """

    name = "mivolo_v2_d1_384"
    commercial_use = True
    license = "Apache-2.0 (weights and code); see docs/LICENSING.md on training-data provenance"

    def __init__(self, weights: str | None = None, device: str = "cuda", input_size: int = 384):
        import timm
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from ..third_party import mivolo  # noqa: F401  (registers mivolo_d1_384 with timm)

        self.torch = torch
        self.device = device
        self.input_size = input_size
        # From the published config.json.
        self.min_age, self.max_age, self.avg_age = 0.0, 122.0, 61.0
        self.version = "hf:iitolstykh/mivolo_v2"

        path = weights or hf_hub_download("iitolstykh/mivolo_v2", "model.safetensors")
        self.model = timm.create_model("mivolo_d1_384", num_classes=3, in_chans=6,
                                       pretrained=False)
        # Checkpoint keys are prefixed "mivolo.model."; strip it to match timm's names.
        prefix = "mivolo.model."
        raw = load_file(path)
        state = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        if not state:
            state = raw
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        self.load_report = {"missing": len(missing), "unexpected": len(unexpected)}
        # A silent partial load produces plausible-looking constant predictions rather than
        # an error, so refuse to continue on one.
        if missing or unexpected:
            raise RuntimeError(
                f"MiVOLO weights did not load cleanly: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys. Refusing to run on partly random weights."
            )
        self.model.eval().to(device)

    def _prep(self, crops: np.ndarray) -> "np.ndarray":
        """BGR uint8 crops -> (N, 6, S, S) float32, face in 0:3 and a zero body in 3:6."""
        s = self.input_size
        faces = np.empty((len(crops), 3, s, s), dtype=np.float32)
        for i, c in enumerate(crops):
            x = letterbox(c, s)[:, :, ::-1].astype(np.float32) / 255.0
            faces[i] = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        # MiVOLO represents a missing body as a zero image put through the same
        # normalisation, i.e. (0 - mean) / std - not as literal zeros.
        blank = ((0.0 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
        bodies = np.broadcast_to(blank[:, None, None], (3, s, s))
        bodies = np.repeat(bodies[None], len(crops), axis=0)
        return np.concatenate([faces, bodies], axis=1)

    def predict(self, crops: np.ndarray, batch_size: int = 32):
        """Returns (p_female, age_years)."""
        probs, ages = [], []
        for i in range(0, len(crops), batch_size):
            x = self.torch.from_numpy(self._prep(crops[i : i + batch_size])).to(self.device)
            with self.torch.no_grad():
                out = self.model(x).float().cpu().numpy()
            g = out[:, :2]
            e = np.exp(g - g.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            probs.append(p[:, 1])                      # config: 0=male, 1=female
            ages.append(out[:, 2] * (self.max_age - self.min_age) + self.avg_age)
        return np.concatenate(probs), np.concatenate(ages)
