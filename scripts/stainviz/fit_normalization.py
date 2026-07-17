"""Fit training-set normalization statistics for StainViz-3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.stainviz_image_io import load_image_tensor
from data.stainviz_manifest import load_manifest, resolve_manifest_path, validate_manifest
from data.stainviz_normalization import PercentileNormalizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit percentile normalization from manifest training images")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--low", type=float, default=1.0)
    parser.add_argument("--high", type=float, default=99.0)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    validate_manifest(records, args.manifest_root)
    selected = [
        row
        for row in records
        if row.domain == args.domain and row.split == args.split and not row.missing
    ]
    if args.max_records:
        selected = selected[: args.max_records]
    arrays = []
    for row in selected:
        tensor, _ = load_image_tensor(resolve_manifest_path(row.image_path, args.manifest_root), channels=args.channels, normalize=False)
        arrays.append(tensor.numpy())
    normalizer = PercentileNormalizer.fit(arrays, low=args.low, high=args.high)
    normalizer.save(args.output)
    print(json.dumps({"records": len(selected), "output": Path(args.output).name, **normalizer.to_dict()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
