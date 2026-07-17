"""Shared ordered-slab loading for StainViz paired and unpaired datasets."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .stainviz_image_io import load_image_tensor, load_mask_tensor
from .stainviz_manifest import ManifestRecord, load_manifest, resolve_manifest_path, validate_manifest
from models.stainviz_warp import identity_grid


def parse_context_offsets(context_slices: int, context_stride: int, text: str = "") -> List[int]:
    if text:
        offsets = [int(value.strip()) for value in text.split(",") if value.strip()]
    else:
        if context_slices < 1 or context_slices % 2 == 0:
            raise ValueError("context_slices must be a positive odd integer")
        radius = context_slices // 2
        offsets = [value * context_stride for value in range(-radius, radius + 1)]
    if 0 not in offsets:
        raise ValueError("context offsets must include 0")
    return offsets


class OrderedSlabLoader:
    def __init__(self, opt, domain: str):
        self.root = Path(getattr(opt, "manifest_root", "") or opt.dataroot)
        all_records = load_manifest(opt.manifest_path)
        validate_manifest(all_records, self.root)
        split = "val" if getattr(opt, "phase", "train") == "val" else getattr(opt, "phase", "train")
        self.records = [row for row in all_records if row.domain == domain and row.split == split]
        self.groups: Dict[str, List[ManifestRecord]] = defaultdict(list)
        for row in self.records:
            self.groups[row.volume_id].append(row)
        for rows in self.groups.values():
            rows.sort(key=lambda row: row.slice_index)
        self.centers = [(volume, position) for volume, rows in sorted(self.groups.items()) for position in range(len(rows))]
        self.offsets = parse_context_offsets(opt.context_slices, opt.context_stride, getattr(opt, "context_offsets", ""))
        self.boundary_padding = opt.boundary_padding
        self.assume_registered = opt.assume_registered

    def __len__(self):
        return len(self.centers)

    def _position(self, requested: int, length: int) -> Tuple[int, bool]:
        if 0 <= requested < length:
            return requested, True
        if self.boundary_padding == "invalid" or self.boundary_padding == "replicate":
            return min(max(requested, 0), length - 1), False
        if self.boundary_padding == "reflect":
            if length == 1:
                return 0, False
            period = 2 * length - 2
            folded = requested % period
            return (folded if folded < length else period - folded), False
        raise ValueError(f"unsupported boundary padding: {self.boundary_padding}")

    def _load_grid(self, row: ManifestRecord, height: int, width: int):
        if row.registration_grid_to_next:
            loaded = np.load(resolve_manifest_path(row.registration_grid_to_next, self.root))
            array = loaded["grid"] if hasattr(loaded, "files") and "grid" in loaded.files else loaded
            grid = torch.from_numpy(np.asarray(array)).float()
            confidence = torch.ones(1, height, width)
            if row.registration_confidence_to_next:
                confidence = load_mask_tensor(resolve_manifest_path(row.registration_confidence_to_next, self.root))
            return grid, confidence
        if self.assume_registered:
            return identity_grid(height, width), torch.ones(1, height, width)
        return identity_grid(height, width), torch.zeros(1, height, width)

    def load(self, dataset_index: int, channels: int, load_targets: bool = False) -> Dict[str, object]:
        volume, center_position = self.centers[dataset_index]
        rows = self.groups[volume]
        selected, validity = zip(*(self._position(center_position + offset, len(rows)) for offset in self.offsets))
        slab_rows = [rows[position] for position in selected]
        sources, masks, targets, pair_masks, pair_valid = [], [], [], [], []
        for row in slab_rows:
            source, _ = load_image_tensor(resolve_manifest_path(row.image_path, self.root), channels=channels)
            sources.append(source)
            masks.append(
                load_mask_tensor(resolve_manifest_path(row.tissue_mask_path, self.root))
                if row.tissue_mask_path
                else torch.ones(1, *source.shape[-2:])
            )
            if load_targets and row.pair_valid and row.paired_target_path:
                target, _ = load_image_tensor(resolve_manifest_path(row.paired_target_path, self.root), channels=channels)
                targets.append(target)
                pair_masks.append(
                    load_mask_tensor(resolve_manifest_path(row.pair_confidence_path, self.root))
                    if row.pair_confidence_path
                    else torch.ones(1, *source.shape[-2:])
                )
                pair_valid.append(True)
            elif load_targets:
                targets.append(torch.zeros(channels, *source.shape[-2:]))
                pair_masks.append(torch.zeros(1, *source.shape[-2:]))
                pair_valid.append(False)
        height, width = sources[0].shape[-2:]
        grids, confidences = [], []
        for left, right in zip(slab_rows, slab_rows[1:]):
            if right.slice_index == left.slice_index + 1:
                grid, confidence = self._load_grid(left, height, width)
            else:
                grid, confidence = identity_grid(height, width), torch.zeros(1, height, width)
            grids.append(grid)
            confidences.append(confidence)
        return {
            "rows": slab_rows,
            "A": torch.stack(sources),
            "B": torch.stack(targets) if load_targets else None,
            "neighbor_valid_A": torch.tensor(validity, dtype=torch.bool),
            "pair_valid": torch.tensor(pair_valid, dtype=torch.bool) if load_targets else None,
            "tissue_mask_A": torch.stack(masks),
            "pair_confidence": torch.stack(pair_masks) if load_targets else None,
            "warp_A_to_next": torch.stack(grids) if grids else torch.empty(0, height, width, 2),
            "warp_conf_A_to_next": torch.stack(confidences) if confidences else torch.empty(0, 1, height, width),
            "center_position": center_position,
        }

