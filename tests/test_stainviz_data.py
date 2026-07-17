import csv
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from data.blockface_paired_dataset import BlockfacePairedDataset
from data.blockface_unaligned_dataset import BlockfaceUnalignedDataset
from data.stainviz_image_io import load_image_tensor
from data.stainviz_transforms import JointVolumeTransform
from models.stainviz_warp import identity_grid


def test_16bit_image_scaling_is_deterministic(tmp_path):
    array = np.array([[0, 32768], [65535, 16384]], dtype=np.uint16)
    path = tmp_path / "slice.tif"
    Image.fromarray(array).save(path)

    first, metadata = load_image_tensor(path, channels=1)
    second, _ = load_image_tensor(path, channels=1)

    assert torch.equal(first, second)
    assert first.shape == (1, 2, 2)
    assert torch.isclose(first[0, 0, 0], torch.tensor(-1.0))
    assert torch.isclose(first[0, 1, 0], torch.tensor(1.0))
    assert metadata["bit_depth"] == 16


def test_joint_transform_keeps_slices_masks_and_identity_grid_aligned():
    image = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    images = torch.cat([image, image + 100], dim=0)
    mask = (image > 20).float().repeat(2, 1, 1, 1)
    grid = identity_grid(8, 8).unsqueeze(0)
    transform = JointVolumeTransform(output_size=(4, 4))

    result = transform(
        {"A": images, "tissue_mask_A": mask, "warp_A_to_next": grid},
        crop=(2, 1, 4, 4),
        horizontal_flip=True,
    )

    assert torch.equal(result["A"][1] - result["A"][0], torch.full((1, 4, 4), 100.0))
    assert torch.equal(result["tissue_mask_A"][0], (result["A"][0] > 20).float())
    assert torch.equal(result["tissue_mask_A"][0], result["tissue_mask_A"][1])
    assert torch.allclose(result["warp_A_to_next"][0], identity_grid(4, 4), atol=1e-5)


def _save_rgb(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (value, value, value)).save(path)


def _write_manifest(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _base_opt(tmp_path, manifest):
    return SimpleNamespace(
        dataroot=str(tmp_path),
        manifest_path=str(manifest),
        manifest_root=str(tmp_path),
        source_domain="blockface",
        target_domain="HE",
        phase="train",
        context_slices=3,
        context_stride=1,
        context_offsets="",
        boundary_padding="replicate",
        assume_registered=True,
        input_nc=3,
        output_nc=3,
        preprocess="none",
        load_size=16,
        crop_size=16,
        no_flip=True,
        serial_batches=True,
        max_dataset_size=float("inf"),
    )


def test_paired_dataset_returns_slab_contract_with_sparse_anchor(tmp_path):
    rows = []
    for index in range(3):
        source = f"source/{index:03d}.png"
        _save_rgb(tmp_path / source, 20 + index)
        row = {
            "specimen_id": "spec_1",
            "volume_id": "vol_1",
            "domain": "blockface",
            "split": "train",
            "slice_index": index,
            "z_um": index * 20,
            "image_path": source,
            "microns_per_pixel": 10,
        }
        if index == 1:
            target = "target/001.png"
            _save_rgb(tmp_path / target, 180)
            row.update({"paired_target_path": target, "pair_valid": 1, "target_stain": "HE"})
        rows.append(row)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    opt = _base_opt(tmp_path, manifest)
    opt.crop_size = 8
    sample = BlockfacePairedDataset(opt)[1]

    assert sample["A"].shape == (3, 3, 8, 8)
    assert sample["B"].shape == (3, 3, 8, 8)
    assert sample["pair_valid"].tolist() == [False, True, False]
    assert sample["neighbor_valid_A"].tolist() == [True, True, True]
    assert sample["warp_A_to_next"].shape == (2, 8, 8, 2)
    assert sample["center_index"].item() == 1


def test_unaligned_dataset_uses_single_valid_target_for_unordered_collection(tmp_path):
    rows = []
    for index in range(3):
        source = f"source/{index:03d}.png"
        _save_rgb(tmp_path / source, 20 + index)
        rows.append({
            "specimen_id": "spec_a",
            "volume_id": "vol_a",
            "domain": "blockface",
            "split": "train",
            "slice_index": index,
            "z_um": index * 20,
            "image_path": source,
            "microns_per_pixel": 10,
        })
    for index in range(2):
        target = f"target/{index:03d}.png"
        _save_rgb(tmp_path / target, 180 + index)
        rows.append({
            "specimen_id": f"spec_b{index}",
            "volume_id": f"unordered_{index}",
            "domain": "HE",
            "split": "train",
            "slice_index": 0,
            "z_um": 0,
            "image_path": target,
            "microns_per_pixel": 10,
        })
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    sample = BlockfaceUnalignedDataset(_base_opt(tmp_path, manifest))[1]

    assert sample["A"].shape[0] == 3
    assert sample["B"].shape[0] == 1
    assert sample["neighbor_valid_B"].tolist() == [True]
    assert sample["warp_B_to_next"].shape[0] == 0
