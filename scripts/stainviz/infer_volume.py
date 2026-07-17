"""Run deterministic whole-volume StainViz-3D inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from data.stainviz_image_io import load_image_tensor
from data.stainviz_manifest import ManifestRecord, load_manifest, resolve_manifest_path, validate_manifest, write_manifest
from data.stainviz_normalization import PercentileNormalizer
from models.stainviz_3d_networks import StainViz25DGenerator
from util.stainviz_volume_io import infer_volume_tiled, save_volume_package


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def _select_volume(records: List[ManifestRecord], domain: str, volume_id: Optional[str], split: Optional[str]) -> List[ManifestRecord]:
    selected = [row for row in records if row.domain == domain and (split is None or row.split == split)]
    volumes = sorted({row.volume_id for row in selected})
    if volume_id:
        selected = [row for row in selected if row.volume_id == volume_id]
    elif len(volumes) == 1:
        volume_id = volumes[0]
        selected = [row for row in selected if row.volume_id == volume_id]
    else:
        raise ValueError(f"--volume-id is required when manifest has volumes: {volumes}")
    if not selected:
        raise ValueError(f"no records found for domain={domain!r}, volume_id={volume_id!r}")
    selected.sort(key=lambda row: row.slice_index)
    return selected


def _load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint must be a state_dict or contain state_dict")
    return {key.removeprefix("module."): value for key, value in state.items()}


def _load_plane(path: Path, channels: int, normalizer: Optional[PercentileNormalizer]) -> torch.Tensor:
    if normalizer is None:
        tensor, _ = load_image_tensor(path, channels=channels, normalize=True)
        return tensor
    tensor, _ = load_image_tensor(path, channels=channels, normalize=False)
    normalized = normalizer(tensor.numpy())
    return torch.from_numpy(normalized).float().mul(2.0).sub(1.0)


def _load_volume(rows: List[ManifestRecord], root: str, channels: int, normalizer: Optional[PercentileNormalizer]) -> Tuple[torch.Tensor, torch.Tensor]:
    reference = next((row for row in rows if not row.missing), None)
    if reference is None:
        raise ValueError("selected volume has no readable planes")
    reference_tensor = _load_plane(resolve_manifest_path(reference.image_path, root), channels, normalizer)
    planes = []
    missing = []
    for row in rows:
        if row.missing:
            planes.append(torch.zeros_like(reference_tensor))
            missing.append(True)
        else:
            planes.append(_load_plane(resolve_manifest_path(row.image_path, root), channels, normalizer))
            missing.append(False)
    return torch.stack(planes), torch.tensor(missing, dtype=torch.bool)


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer a full StainViz-3D volume from a manifest and generator checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--domain", default="blockface")
    parser.add_argument("--split", default="")
    parser.add_argument("--volume-id", default="")
    parser.add_argument("--checkpoint", required=True, help="path to latest_net_G.pth or latest_net_G_A.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-stain", default="HE")
    parser.add_argument("--input-nc", type=int, default=3)
    parser.add_argument("--output-nc", type=int, default=3)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--context-slices", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap-xy", type=int, default=64)
    parser.add_argument("--normalization-json", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-tiff", action="store_true")
    parser.add_argument("--assume-registered", action="store_true")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    validate_manifest(records, args.manifest_root)
    selected = _select_volume(records, args.domain, args.volume_id or None, args.split or None)
    normalizer = PercentileNormalizer.load(args.normalization_json) if args.normalization_json else None
    volume, missing = _load_volume(selected, args.manifest_root, args.input_nc, normalizer)

    model = StainViz25DGenerator(args.input_nc, args.output_nc, args.ngf, args.fusion_heads)
    load_report = model.load_state_dict(_load_state(Path(args.checkpoint)), strict=False)
    if load_report.missing_keys or load_report.unexpected_keys:
        raise RuntimeError(
            "checkpoint is incompatible: "
            f"missing={load_report.missing_keys}, unexpected={load_report.unexpected_keys}"
        )
    output, metadata = infer_volume_tiled(
        model,
        volume,
        context_slices=args.context_slices,
        tile_size=args.tile_size,
        overlap_xy=args.overlap_xy,
        device=torch.device(args.device),
        missing_mask=missing,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(selected, output_dir / "manifest_used.csv")
    provenance = {
        "research_use_only": True,
        "target_stain": args.target_stain,
        "checkpoint_path_name": Path(args.checkpoint).name,
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "code_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "normalization": normalizer.to_dict() if normalizer else "image_dtype_to_minus1_plus1",
        "spatial_calibration": {
            "z_um": [row.z_um for row in selected],
            "microns_per_pixel": selected[0].microns_per_pixel,
        },
        "slice_range": [selected[0].slice_index, selected[-1].slice_index],
        "excluded_slices": [row.slice_index for row in selected if row.missing],
        "output_scaling": "volume.npy float32 in model range; PNG/TIFF slices uint8 display range",
        "registration_status": "assume_registered" if args.assume_registered else "not_applied_during_inference",
        "interpolation": "overlapping XY tiles with Hann blending; z predictions confidence-count accumulated",
    }
    save_volume_package(output_dir, output, metadata, provenance, save_tiff=args.save_tiff)
    print(json.dumps({"output_dir": str(output_dir), "slices": len(selected), **metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
