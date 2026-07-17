"""Evaluate K=1 and volumetric StainViz-3D outputs against sparse anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from util.stainviz_metrics_2d import image_metrics
from util.stainviz_metrics_3d import volume_discontinuity, z_smoothing_control


def _load_volume(path: str) -> torch.Tensor:
    array = np.load(path)
    if array.ndim == 3:
        array = array[:, None, :, :]
    elif array.ndim == 4 and array.shape[-1] in {1, 3, 4}:
        array = np.moveaxis(array[..., :3], -1, 1)
    if array.ndim != 4:
        raise ValueError(f"expected [Z,C,H,W] or [Z,H,W,C], got {array.shape}")
    return torch.from_numpy(array).float()


def _to_hwc(plane: torch.Tensor) -> np.ndarray:
    array = plane.detach().cpu().numpy()
    return np.moveaxis(array[:3], 0, -1) if array.shape[0] > 1 else array[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare StainViz-3D K=1 and volumetric outputs")
    parser.add_argument("--baseline-volume", required=True, help="K=1 baseline volume.npy")
    parser.add_argument("--volumetric-volume", required=True, help="2.5D/volumetric volume.npy")
    parser.add_argument("--target-volume", default="", help="optional sparse-anchor target volume.npy")
    parser.add_argument("--valid-indices", default="", help="comma-separated paired anchor indices")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = _load_volume(args.baseline_volume)
    volumetric = _load_volume(args.volumetric_volume)
    if baseline.shape != volumetric.shape:
        raise ValueError("baseline and volumetric outputs must have the same shape")
    report = {
        "baseline_discontinuity": float(volume_discontinuity(baseline).item()),
        "volumetric_discontinuity": float(volume_discontinuity(volumetric).item()),
        "volumetric_edge_difference_l1": float((baseline[:, :, 1:] - volumetric[:, :, 1:]).abs().mean().item()),
    }
    smoothed = z_smoothing_control(volumetric)
    report["z_smoothing_control_discontinuity"] = float(volume_discontinuity(smoothed).item())
    report["z_smoothing_control_l1_to_volumetric"] = float((smoothed - volumetric).abs().mean().item())
    if args.target_volume:
        target = _load_volume(args.target_volume)
        indices = [int(value) for value in args.valid_indices.split(",") if value.strip()]
        if not indices:
            indices = list(range(min(target.shape[0], volumetric.shape[0])))
        paired = []
        for index in indices:
            paired.append(
                {
                    "slice_index": index,
                    "baseline": image_metrics(_to_hwc(baseline[index]), _to_hwc(target[index])),
                    "volumetric": image_metrics(_to_hwc(volumetric[index]), _to_hwc(target[index])),
                }
            )
        report["paired_anchor_metrics"] = paired
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
