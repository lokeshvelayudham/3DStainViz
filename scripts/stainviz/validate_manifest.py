"""Validate a StainViz-3D manifest before training or inference."""

from __future__ import annotations

import argparse
import json

from data.stainviz_manifest import load_manifest, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate StainViz-3D manifests")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--allow-nonmonotonic-z", action="store_true")
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    report = validate_manifest(records, args.manifest_root, allow_nonmonotonic_z=args.allow_nonmonotonic_z)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
