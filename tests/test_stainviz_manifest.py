import csv
from pathlib import Path

import pytest
from PIL import Image

from data.stainviz_manifest import (
    ManifestValidationError,
    load_manifest,
    prepare_manifest_from_folders,
    validate_manifest,
)


FIELDS = [
    "specimen_id",
    "volume_id",
    "domain",
    "split",
    "slice_index",
    "z_um",
    "image_path",
    "microns_per_pixel",
]


def _write_image(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (value, value, value)).save(path)


def _write_manifest(path: Path, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(image_path: str, specimen: str = "spec_01", split: str = "train", index: int = 0):
    return {
        "specimen_id": specimen,
        "volume_id": f"{specimen}_volume",
        "domain": "blockface",
        "split": split,
        "slice_index": index,
        "z_um": index * 20.0,
        "image_path": image_path,
        "microns_per_pixel": 10.0,
    }


def test_valid_manifest_loads_relative_paths(tmp_path):
    _write_image(tmp_path / "images" / "000.png")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [_row("images/000.png")])

    records = load_manifest(manifest)
    report = validate_manifest(records, root=tmp_path)

    assert records[0].image_path == "images/000.png"
    assert report["specimens"] == 1
    assert report["volumes"] == 1


def test_specimen_leakage_is_rejected(tmp_path):
    _write_image(tmp_path / "0.png")
    _write_image(tmp_path / "1.png")
    records_path = tmp_path / "manifest.csv"
    _write_manifest(
        records_path,
        [_row("0.png", split="train", index=0), _row("1.png", split="test", index=1)],
    )

    with pytest.raises(ManifestValidationError, match="appears in multiple splits"):
        validate_manifest(load_manifest(records_path), root=tmp_path)


def test_duplicate_plane_is_rejected(tmp_path):
    _write_image(tmp_path / "0.png")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [_row("0.png"), _row("0.png")])

    with pytest.raises(ManifestValidationError, match="duplicate plane"):
        validate_manifest(load_manifest(manifest), root=tmp_path)


def test_raw_folder_preparation_is_specimen_safe_and_deterministic(tmp_path):
    raw = tmp_path / "raw"
    for specimen in ("mouse_a", "mouse_b", "mouse_c"):
        _write_image(raw / specimen / "slice_000.png")
        _write_image(raw / specimen / "slice_001.png")

    first = prepare_manifest_from_folders(
        raw,
        domain="blockface",
        filename_regex=r"slice_(?P<index>\d+)",
        z_spacing_um=20.0,
        microns_per_pixel=10.0,
        split_ratios=(0.67, 0.0, 0.33),
        seed=13,
    )
    second = prepare_manifest_from_folders(
        raw,
        domain="blockface",
        filename_regex=r"slice_(?P<index>\d+)",
        z_spacing_um=20.0,
        microns_per_pixel=10.0,
        split_ratios=(0.67, 0.0, 0.33),
        seed=13,
    )

    assert [(row.specimen_id, row.split) for row in first] == [
        (row.specimen_id, row.split) for row in second
    ]
    assert all(len({r.split for r in first if r.specimen_id == sid}) == 1 for sid in {r.specimen_id for r in first})
    specimen_ids = sorted({r.specimen_id for r in first})
    assert all(sid.startswith("spec_") for sid in specimen_ids)
    assert all("mouse" not in sid for sid in specimen_ids)
    assert [r.slice_index for r in first if r.specimen_id == specimen_ids[0]] == [0, 1]
