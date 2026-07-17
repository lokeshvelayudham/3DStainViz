"""Whole-volume tiled inference and output packaging for StainViz-3D."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


def tile_starts(size: int, tile_size: int, overlap: int) -> List[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap_xy must satisfy 0 <= overlap < tile_size")
    if tile_size >= size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, max(size - tile_size + 1, 1), step))
    last = size - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _blend_window(height: int, width: int, device: torch.device) -> torch.Tensor:
    wy = torch.hann_window(max(height, 2), periodic=False, device=device)[:height].clamp_min(1e-3)
    wx = torch.hann_window(max(width, 2), periodic=False, device=device)[:width].clamp_min(1e-3)
    return (wy[:, None] * wx[None, :]).float()


def _context_positions(center: int, depth: int, context_slices: int, missing_mask: torch.Tensor) -> Tuple[List[int], List[bool], List[int]]:
    radius = context_slices // 2
    requested = list(range(center - radius, center + radius + 1))
    positions: List[int] = []
    valid: List[bool] = []
    offsets: List[int] = []
    for value in requested:
        in_bounds = 0 <= value < depth
        clamped = min(max(value, 0), depth - 1)
        is_valid = in_bounds and not bool(missing_mask[clamped])
        positions.append(clamped)
        valid.append(is_valid)
        offsets.append(value - center)
    return positions, valid, offsets


@torch.no_grad()
def infer_volume_tiled(
    model: torch.nn.Module,
    volume: torch.Tensor,
    context_slices: int = 3,
    tile_size: int = 512,
    overlap_xy: int = 64,
    device: Optional[torch.device] = None,
    missing_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Run deterministic XY-tiled, z-slabbed inference over ``[Z,C,H,W]``."""
    if context_slices < 1 or context_slices % 2 == 0:
        raise ValueError("context_slices must be a positive odd integer")
    if volume.ndim != 4:
        raise ValueError("volume must have shape [Z,C,H,W]")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()
    model.to(device)

    source = volume.detach().float()
    depth, channels, height, width = source.shape
    missing = torch.zeros(depth, dtype=torch.bool) if missing_mask is None else missing_mask.cpu().bool()
    sums = torch.zeros_like(source, device=device)
    weights = torch.zeros(depth, 1, height, width, device=device)
    z_counts = [0 for _ in range(depth)]
    y_starts = tile_starts(height, tile_size, overlap_xy)
    x_starts = tile_starts(width, tile_size, overlap_xy)

    for center in range(depth):
        if bool(missing[center]):
            continue
        positions, valid, offsets = _context_positions(center, depth, context_slices, missing)
        for position, is_valid in zip(positions, valid):
            if is_valid:
                z_counts[position] += 1
        valid_tensor = torch.tensor(valid, dtype=torch.bool, device=device).unsqueeze(0)
        z_tensor = torch.tensor(offsets, dtype=torch.float32, device=device).unsqueeze(0)
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + tile_size, height)
                x1 = min(x0 + tile_size, width)
                slabs = []
                for position, is_valid in zip(positions, valid):
                    if is_valid:
                        slabs.append(source[position, :, y0:y1, x0:x1])
                    else:
                        slabs.append(torch.zeros(channels, y1 - y0, x1 - x0))
                slab = torch.stack(slabs).unsqueeze(0).to(device)
                result = model(slab, z_offsets=z_tensor, neighbor_valid=valid_tensor)
                prediction = result["prediction"] if isinstance(result, dict) else result
                if prediction.ndim == 4:
                    prediction = prediction.unsqueeze(1)
                window = _blend_window(y1 - y0, x1 - x0, device).view(1, y1 - y0, x1 - x0)
                for slab_index, (position, is_valid) in enumerate(zip(positions, valid)):
                    if not is_valid:
                        continue
                    sums[position, :, y0:y1, x0:x1] += prediction[0, slab_index] * window
                    weights[position, :, y0:y1, x0:x1] += window
    output = (sums / weights.clamp_min(1e-6)).cpu()
    output[missing] = 0.0
    if was_training:
        model.train()
    metadata = {
        "context_slices": context_slices,
        "tile_size": tile_size,
        "overlap_xy": overlap_xy,
        "z_prediction_counts": z_counts,
        "missing_slices": [int(index) for index, value in enumerate(missing.tolist()) if value],
        "tile_grid": {"y": y_starts, "x": x_starts},
    }
    return output, metadata


def _slice_to_uint8(slice_tensor: torch.Tensor) -> np.ndarray:
    values = slice_tensor.detach().cpu().float().numpy()
    if values.min() < 0.0:
        values = (values + 1.0) * 0.5
    values = np.clip(values, 0.0, 1.0)
    if values.shape[0] == 1:
        return (values[0] * 255.0 + 0.5).astype(np.uint8)
    return (np.moveaxis(values[:3], 0, -1) * 255.0 + 0.5).astype(np.uint8)


def save_volume_package(
    output_dir: Union[str, Path],
    volume: torch.Tensor,
    metadata: Dict[str, object],
    provenance: Dict[str, object],
    save_tiff: bool = False,
) -> None:
    """Write the stable StainViz-3D inference output contract."""
    output = Path(output_dir)
    slices_dir = output / "slices"
    output.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)
    np.save(output / "volume.npy", volume.detach().cpu().numpy().astype(np.float32))
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    counts = np.asarray(metadata.get("z_prediction_counts", []), dtype=np.float32)
    if counts.size:
        np.save(output / "confidence.npy", counts)
    for index, plane in enumerate(volume):
        image = Image.fromarray(_slice_to_uint8(plane))
        image.save(slices_dir / f"{index:06d}.png")
    if save_tiff:
        try:
            import tifffile

            array = np.stack([_slice_to_uint8(plane) for plane in volume])
            tifffile.imwrite(output / "volume.tiff", array)
        except ImportError:
            metadata = dict(metadata)
            metadata["tiff_fallback"] = "tifffile unavailable; PNG slices and volume.npy were written"
            (output / "metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
