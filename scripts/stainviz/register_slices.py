"""Registration helpers for StainViz-3D blockface sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch

from data.stainviz_manifest import ManifestRecord, load_manifest, resolve_manifest_path, validate_manifest, write_manifest
from models.stainviz_warp import identity_grid


def pixel_affine_to_grid(matrix: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Convert a pixel-space output-to-input affine into a normalized grid.

    ``matrix`` maps output pixel coordinates ``[x, y, 1]`` to input pixel
    coordinates. The returned grid is directly usable with
    ``torch.nn.functional.grid_sample(..., align_corners=False)``.
    """
    transform = torch.as_tensor(matrix, dtype=torch.float32)
    if transform.shape == (2, 3):
        bottom = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        transform = torch.cat([transform, bottom], dim=0)
    if transform.shape != (3, 3):
        raise ValueError("matrix must have shape [2,3] or [3,3]")
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij")
    ones = torch.ones_like(x)
    points = torch.stack((x, y, ones), dim=-1).reshape(-1, 3).T
    source = (transform @ points).T.reshape(height, width, 3)
    x_norm = (source[..., 0] + 0.5) * (2.0 / width) - 1.0
    y_norm = (source[..., 1] + 0.5) * (2.0 / height) - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


def _as_gray01(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        values = values[..., :3].mean(axis=-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def registration_confidence(fixed: np.ndarray, warped: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """Estimate local confidence from post-warp similarity in [0, 1]."""
    fixed_gray = _as_gray01(fixed)
    warped_gray = _as_gray01(warped)
    if fixed_gray.shape != warped_gray.shape:
        raise ValueError("fixed and warped images must have the same shape")
    diff = np.abs(fixed_gray - warped_gray)
    confidence = np.exp(-float(sigma) * diff)
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _load_simpleitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except ImportError as exc:
        raise RuntimeError("SimpleITK is required for estimated registration; install SimpleITK or use identity-grids") from exc
    return sitk


def _read_scalar_sitk(sitk, path: Path):
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if image.GetNumberOfComponentsPerPixel() <= 1:
        return sitk.Cast(image, sitk.sitkFloat32)
    channels = [
        sitk.VectorIndexSelectionCast(image, index, sitk.sitkFloat32)
        for index in range(image.GetNumberOfComponentsPerPixel())
    ]
    result = channels[0]
    for channel in channels[1:]:
        result = result + channel
    return result / float(len(channels))


def _initial_transform(sitk, fixed, moving, kind: str):
    if kind == "rigid":
        transform = sitk.Euler2DTransform()
    elif kind == "affine":
        transform = sitk.AffineTransform(2)
    else:
        raise ValueError("transform kind must be rigid or affine")
    return sitk.CenteredTransformInitializer(fixed, moving, transform, sitk.CenteredTransformInitializerFilter.GEOMETRY)


def _sitk_transform_to_grid(sitk, fixed, moving, transform) -> torch.Tensor:
    width, height = fixed.GetSize()
    moving_width, moving_height = moving.GetSize()
    grid = np.zeros((height, width, 2), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            fixed_point = fixed.TransformIndexToPhysicalPoint((x, y))
            moving_point = transform.TransformPoint(fixed_point)
            moving_index = moving.TransformPhysicalPointToContinuousIndex(moving_point)
            grid[y, x, 0] = (moving_index[0] + 0.5) * (2.0 / moving_width) - 1.0
            grid[y, x, 1] = (moving_index[1] + 0.5) * (2.0 / moving_height) - 1.0
    return torch.from_numpy(grid)


def estimate_registration_grid(
    fixed_path: Path,
    moving_path: Path,
    metric: str = "mattes",
    transform_kind: str = "affine",
    iterations: int = 100,
) -> Tuple[torch.Tensor, np.ndarray, dict]:
    """Estimate a rigid/affine SimpleITK transform and convert it to a StainViz grid."""
    sitk = _load_simpleitk()
    fixed = _read_scalar_sitk(sitk, fixed_path)
    moving = _read_scalar_sitk(sitk, moving_path)
    registration = sitk.ImageRegistrationMethod()
    if metric == "mattes":
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == "correlation":
        registration.SetMetricAsCorrelation()
    else:
        raise ValueError("metric must be mattes or correlation")
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(_initial_transform(sitk, fixed, moving, transform_kind), inPlace=False)
    transform = registration.Execute(fixed, moving)
    warped = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    confidence = registration_confidence(sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(warped))
    grid = _sitk_transform_to_grid(sitk, fixed, moving, transform)
    status = {
        "metric": metric,
        "transform": transform_kind,
        "iterations": iterations,
        "optimizer_stop_condition": registration.GetOptimizerStopConditionDescription(),
        "final_metric": float(registration.GetMetricValue()),
    }
    return grid, confidence, status


def _write_grid(path: Path, grid: torch.Tensor, confidence: np.ndarray) -> Tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_path = path.with_suffix(".npz")
    confidence_path = path.with_name(path.stem + "_confidence.npy")
    np.savez_compressed(grid_path, grid=grid.cpu().numpy().astype(np.float32))
    np.save(confidence_path, confidence.astype(np.float32))
    return grid_path.as_posix(), confidence_path.as_posix()


def _manifest_path(path: str, root: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate.as_posix()


def write_identity_grids(records: Iterable[ManifestRecord], root: Path, output_dir: Path) -> Tuple[list, dict]:
    """Write explicit identity registration grids for already aligned volumes."""
    records = list(records)
    validate_manifest(records, root)
    grouped = {}
    for row in records:
        grouped.setdefault((row.volume_id, row.domain), []).append(row)
    updated = []
    status = {"written": 0, "skipped_last_planes": 0}
    for rows in grouped.values():
        rows.sort(key=lambda row: row.slice_index)
        for row, next_row in zip(rows, rows[1:]):
            if next_row.slice_index != row.slice_index + 1:
                updated.append(row)
                continue
            grid = identity_grid(1, 1).numpy()
            try:
                from PIL import Image

                with Image.open(resolve_manifest_path(row.image_path, root)) as image:
                    width, height = image.size
                grid = identity_grid(height, width)
            except Exception:
                grid = identity_grid(1, 1)
            base = output_dir / f"{row.volume_id}_{row.domain}_{row.slice_index:06d}_to_next"
            grid_name, confidence_name = _write_grid(base, grid, np.ones(grid.shape[:2], dtype=np.float32))
            updated.append(
                ManifestRecord(
                    **{
                        **row.__dict__,
                        "registration_grid_to_next": _manifest_path(grid_name, root),
                        "registration_confidence_to_next": _manifest_path(confidence_name, root),
                    }
                )
            )
            status["written"] += 1
        if rows:
            status["skipped_last_planes"] += 1
    updated_keys = {(row.volume_id, row.domain, row.slice_index) for row in updated}
    updated.extend(row for row in records if (row.volume_id, row.domain, row.slice_index) not in updated_keys)
    updated.sort(key=lambda row: (row.volume_id, row.domain, row.slice_index))
    return updated, status


def main() -> int:
    parser = argparse.ArgumentParser(description="StainViz-3D registration preprocessing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity = subparsers.add_parser("identity-grids", help="write explicit identity grids for already registered slices")
    identity.add_argument("--manifest", required=True)
    identity.add_argument("--manifest-root", required=True)
    identity.add_argument("--output-manifest", required=True)
    identity.add_argument("--output-dir", required=True)
    identity.add_argument("--assume-registered", action="store_true", required=True)

    estimate = subparsers.add_parser("estimate", help="estimate a rigid/affine SimpleITK registration")
    estimate.add_argument("--fixed", required=True)
    estimate.add_argument("--moving", required=True)
    estimate.add_argument("--output-grid", required=True)
    estimate.add_argument("--output-confidence", required=True)
    estimate.add_argument("--metric", choices=("mattes", "correlation"), default="mattes")
    estimate.add_argument("--transform", choices=("rigid", "affine"), default="affine")
    estimate.add_argument("--iterations", type=int, default=100)

    args = parser.parse_args()
    if args.command == "identity-grids":
        root = Path(args.manifest_root)
        records = load_manifest(args.manifest)
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        updated, status = write_identity_grids(records, root, output_dir)
        write_manifest(updated, args.output_manifest)
        print(json.dumps(status, sort_keys=True))
        return 0
    if args.command == "estimate":
        grid, confidence, status = estimate_registration_grid(
            Path(args.fixed),
            Path(args.moving),
            metric=args.metric,
            transform_kind=args.transform,
            iterations=args.iterations,
        )
        Path(args.output_grid).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_confidence).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_grid, grid=grid.numpy().astype(np.float32))
        np.save(args.output_confidence, confidence.astype(np.float32))
        print(json.dumps(status, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
