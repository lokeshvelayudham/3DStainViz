"""Manifest contracts and specimen-safe raw-folder preparation for StainViz-3D."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class ManifestValidationError(ValueError):
    """Raised when a manifest cannot safely be used for training or inference."""


def _optional_text(value: object) -> Optional[str]:
    text = "" if value is None else str(value).strip()
    return text or None


def _optional_float(value: object) -> Optional[float]:
    text = _optional_text(value)
    return None if text is None else float(text)


def _optional_bool(value: object, default: bool = False) -> bool:
    text = _optional_text(value)
    if text is None:
        return default
    if text.lower() in {"1", "true", "yes", "y"}:
        return True
    if text.lower() in {"0", "false", "no", "n"}:
        return False
    raise ManifestValidationError(f"invalid boolean value: {value!r}")


@dataclass(frozen=True)
class ManifestRecord:
    """One acquired plane or one target-domain image."""

    specimen_id: str
    volume_id: str
    domain: str
    split: str
    slice_index: int
    z_um: float
    image_path: str
    microns_per_pixel: float
    channels: Optional[str] = None
    target_stain: Optional[str] = None
    paired_target_path: Optional[str] = None
    pair_valid: bool = False
    pair_confidence_path: Optional[str] = None
    tissue_mask_path: Optional[str] = None
    qc_score: float = 1.0
    missing: bool = False
    registration_grid_to_next: Optional[str] = None
    registration_confidence_to_next: Optional[str] = None
    scanner_id: Optional[str] = None
    batch_id: Optional[str] = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ManifestRecord":
        required = (
            "specimen_id",
            "volume_id",
            "domain",
            "split",
            "slice_index",
            "z_um",
            "image_path",
            "microns_per_pixel",
        )
        missing = [name for name in required if _optional_text(row.get(name)) is None]
        if missing:
            raise ManifestValidationError(f"missing required fields: {', '.join(missing)}")
        return cls(
            specimen_id=str(row["specimen_id"]).strip(),
            volume_id=str(row["volume_id"]).strip(),
            domain=str(row["domain"]).strip(),
            split=str(row["split"]).strip(),
            slice_index=int(str(row["slice_index"]).strip()),
            z_um=float(str(row["z_um"]).strip()),
            image_path=str(row["image_path"]).strip(),
            microns_per_pixel=float(str(row["microns_per_pixel"]).strip()),
            channels=_optional_text(row.get("channels")),
            target_stain=_optional_text(row.get("target_stain")),
            paired_target_path=_optional_text(row.get("paired_target_path")),
            pair_valid=_optional_bool(row.get("pair_valid")),
            pair_confidence_path=_optional_text(row.get("pair_confidence_path")),
            tissue_mask_path=_optional_text(row.get("tissue_mask_path")),
            qc_score=_optional_float(row.get("qc_score")) or 1.0,
            missing=_optional_bool(row.get("missing")),
            registration_grid_to_next=_optional_text(row.get("registration_grid_to_next")),
            registration_confidence_to_next=_optional_text(row.get("registration_confidence_to_next")),
            scanner_id=_optional_text(row.get("scanner_id")),
            batch_id=_optional_text(row.get("batch_id")),
        )


def load_manifest(path: Union[str, Path]) -> List[ManifestRecord]:
    """Load a CSV or JSONL manifest."""
    manifest_path = Path(path)
    if manifest_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    else:
        with manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        raise ManifestValidationError("manifest contains no records")
    return [ManifestRecord.from_mapping(row) for row in rows]


def resolve_manifest_path(path: str, root: Union[str, Path]) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def validate_manifest(
    records: Sequence[ManifestRecord], root: Union[str, Path], allow_nonmonotonic_z: bool = False
) -> Dict[str, object]:
    """Validate biological isolation, ordering, files, and volume-level consistency."""
    if not records:
        raise ManifestValidationError("manifest contains no records")
    root_path = Path(root)
    errors: List[str] = []
    split_by_specimen: Dict[str, set] = defaultdict(set)
    keys = Counter()
    groups: Dict[Tuple[str, str], List[ManifestRecord]] = defaultdict(list)
    for record in records:
        split_by_specimen[record.specimen_id].add(record.split)
        keys[(record.volume_id, record.domain, record.slice_index)] += 1
        groups[(record.volume_id, record.domain)].append(record)
        if record.split not in {"train", "val", "test"}:
            errors.append(f"unsupported split {record.split!r} for {record.specimen_id}")
        for label, value in (("image", record.image_path), ("paired target", record.paired_target_path)):
            if label == "image" and record.missing:
                continue
            if value and not resolve_manifest_path(value, root_path).exists():
                errors.append(f"missing {label} file for {record.volume_id}:{record.slice_index}")
        if record.pair_valid and not record.paired_target_path:
            errors.append(f"pair_valid is true but paired target is unavailable for {record.volume_id}:{record.slice_index}")
        if record.missing and record.pair_valid:
            errors.append(f"missing source plane cannot also be a valid pair for {record.volume_id}:{record.slice_index}")
        if record.microns_per_pixel <= 0:
            errors.append(f"microns_per_pixel must be positive for {record.volume_id}:{record.slice_index}")
    for specimen, splits in split_by_specimen.items():
        if len(splits) > 1:
            errors.append(f"specimen {specimen!r} appears in multiple splits: {sorted(splits)}")
    for key, count in keys.items():
        if count > 1:
            errors.append(f"duplicate plane {key!r}")
    for (volume, domain), rows in groups.items():
        rows = sorted(rows, key=lambda row: row.slice_index)
        z_values = [row.z_um for row in rows]
        if not allow_nonmonotonic_z and any(right <= left for left, right in zip(z_values, z_values[1:])):
            errors.append(f"non-monotonic z order for volume={volume!r}, domain={domain!r}")
        mpp = {row.microns_per_pixel for row in rows}
        channels = {row.channels for row in rows if row.channels is not None}
        if len(mpp) > 1:
            errors.append(f"inconsistent microns_per_pixel for volume={volume!r}, domain={domain!r}")
        if len(channels) > 1:
            errors.append(f"inconsistent channels for volume={volume!r}, domain={domain!r}")
    if errors:
        raise ManifestValidationError("; ".join(errors))
    return {
        "records": len(records),
        "specimens": len(split_by_specimen),
        "volumes": len({row.volume_id for row in records}),
        "domains": dict(Counter(row.domain for row in records)),
        "splits": dict(Counter(row.split for row in records)),
    }


def _pseudonym(folder_name: str) -> str:
    return "spec_" + hashlib.sha256(folder_name.encode("utf-8")).hexdigest()[:12]


def _assign_splits(specimens: Sequence[str], ratios: Tuple[float, float, float], seed: int) -> Dict[str, str]:
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6 or any(value < 0 for value in ratios):
        raise ManifestValidationError("split_ratios must contain three non-negative values summing to 1")
    shuffled = sorted(specimens)
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * ratios[0])
    val_end = train_end + round(len(shuffled) * ratios[1])
    labels = (["train"] * train_end) + (["val"] * (val_end - train_end))
    labels += ["test"] * (len(shuffled) - len(labels))
    return dict(zip(shuffled, labels))


def prepare_manifest_from_folders(
    input_root: Union[str, Path],
    domain: str,
    filename_regex: str,
    z_spacing_um: float,
    microns_per_pixel: float,
    split_map: Optional[Mapping[str, str]] = None,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 13,
) -> List[ManifestRecord]:
    """Inventory one specimen/volume directory per child into pseudonymous records."""
    root = Path(input_root)
    pattern = re.compile(filename_regex)
    folders = sorted(path for path in root.iterdir() if path.is_dir())
    if not folders:
        raise ManifestValidationError("input root has no specimen directories")
    raw_names = [folder.name for folder in folders]
    assignments = dict(split_map) if split_map is not None else _assign_splits(raw_names, split_ratios, seed)
    records: List[ManifestRecord] = []
    for folder in folders:
        if folder.name not in assignments:
            raise ManifestValidationError(f"split map has no entry for specimen directory {folder.name!r}")
        specimen_id = _pseudonym(folder.name)
        volume_id = f"{specimen_id}_volume"
        matched = []
        for image in sorted(path for path in folder.rglob("*") if path.is_file()):
            match = pattern.search(image.stem)
            if not match:
                continue
            try:
                index_text = match.group("index")
            except IndexError as exc:
                raise ManifestValidationError("filename_regex must define a named 'index' group") from exc
            matched.append((int(index_text), image))
        if not matched:
            raise ManifestValidationError(f"no filenames matched the configured regex in {folder.name!r}")
        for slice_index, image in sorted(matched):
            records.append(
                ManifestRecord(
                    specimen_id=specimen_id,
                    volume_id=volume_id,
                    domain=domain,
                    split=assignments[folder.name],
                    slice_index=slice_index,
                    z_um=slice_index * z_spacing_um,
                    image_path=image.relative_to(root).as_posix(),
                    microns_per_pixel=microns_per_pixel,
                )
            )
    return records


def write_manifest(records: Iterable[ManifestRecord], path: Union[str, Path]) -> None:
    """Write records as CSV or JSONL using the complete stable schema."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    if output.suffix.lower() == ".jsonl":
        output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
        return
    names = [field.name for field in fields(ManifestRecord)]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
