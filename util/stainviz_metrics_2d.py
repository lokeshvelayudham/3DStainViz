"""2D paired-anchor metrics for StainViz-3D evaluation."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from skimage import color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _as_float01(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.size == 0:
        return values
    if np.nanmin(values) < 0.0:
        values = (values + 1.0) * 0.5
    if np.nanmax(values) > 1.0:
        max_value = 65535.0 if np.nanmax(values) > 255.0 else 255.0
        values = values / max_value
    return np.clip(values, 0.0, 1.0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    left = a.reshape(-1).astype(np.float64)
    right = b.reshape(-1).astype(np.float64)
    if np.allclose(left, right) and np.std(left) == 0.0:
        return 1.0
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def image_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Return SSIM, PSNR, CIEDE2000, and intensity correlation."""
    pred = _as_float01(prediction)
    tgt = _as_float01(target)
    if pred.shape != tgt.shape:
        raise ValueError("prediction and target must have the same shape")
    if np.array_equal(pred, tgt):
        ssim = 1.0
        psnr = math.inf
    else:
        channel_axis = -1 if pred.ndim == 3 and pred.shape[-1] in {3, 4} else None
        win_size = min(7, pred.shape[0], pred.shape[1]) if pred.ndim >= 2 else 1
        if win_size % 2 == 0:
            win_size -= 1
        ssim = float(structural_similarity(pred, tgt, data_range=1.0, channel_axis=channel_axis, win_size=max(win_size, 3)))
        psnr = float(peak_signal_noise_ratio(tgt, pred, data_range=1.0))
    ciede2000 = float("nan")
    if pred.ndim == 3 and pred.shape[-1] >= 3:
        delta = color.deltaE_ciede2000(color.rgb2lab(pred[..., :3]), color.rgb2lab(tgt[..., :3]))
        ciede2000 = float(np.nanmean(delta))
    return {
        "ssim": ssim,
        "psnr": psnr,
        "ciede2000": ciede2000,
        "intensity_pearson": _pearson(pred, tgt),
    }
