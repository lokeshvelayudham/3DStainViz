"""Prepare specimen-safe StainViz-3D manifests from raw folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from data.stainviz_manifest import prepare_manifest_from_folders, validate_manifest, write_manifest


def _split_ratios(text: str) -> Tuple[float, float, float]:
    values = tuple(float(value.strip()) for value in text.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("split ratios must be train,val,test")
    return values


def _load_split_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    split_path = Path(path)
    if split_path.suffix.lower() == ".json":
        payload = json.loads(split_path.read_text())
        if isinstance(payload, dict):
            return {str(key): str(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return {str(row["folder"]): str(row["split"]) for row in payload}
        raise ValueError("JSON split map must be an object or list of {folder, split}")
    with split_path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        result = {}
        for row in rows:
            folder = row.get("folder") or row.get("directory") or row.get("specimen")
            split = row.get("split")
            if not folder or not split:
                raise ValueError("CSV split map requires folder/specimen and split columns")
            result[folder] = split
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a StainViz-3D raw-folder manifest")
    parser.add_argument("--raw-root", required=True, help="directory with one specimen/volume per child folder")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--qc-summary", default="")
    parser.add_argument("--domain", required=True, help="source or target domain label, e.g. blockface or HE")
    parser.add_argument("--filename-regex", required=True, help="regex with named capture group (?P<index>...)")
    parser.add_argument("--z-spacing-um", type=float, required=True)
    parser.add_argument("--microns-per-pixel", type=float, required=True)
    parser.add_argument("--split-map", default="")
    parser.add_argument("--split-ratios", type=_split_ratios, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    records = prepare_manifest_from_folders(
        input_root=args.raw_root,
        domain=args.domain,
        filename_regex=args.filename_regex,
        z_spacing_um=args.z_spacing_um,
        microns_per_pixel=args.microns_per_pixel,
        split_map=_load_split_map(args.split_map),
        split_ratios=args.split_ratios,
        seed=args.seed,
    )
    report = validate_manifest(records, root=args.raw_root)
    write_manifest(records, args.output_manifest)
    if args.qc_summary:
        summary = {
            **report,
            "domain": args.domain,
            "raw_root_name": Path(args.raw_root).name,
            "z_spacing_um": args.z_spacing_um,
            "microns_per_pixel": args.microns_per_pixel,
        }
        Path(args.qc_summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.qc_summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
