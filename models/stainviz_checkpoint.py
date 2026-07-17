"""Explicit generator initialization helpers for paired-to-unpaired curricula."""

from pathlib import Path
from typing import Dict

import torch


def load_generator_checkpoint(net, path: str, allow_partial: bool = False) -> Dict[str, object]:
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"generator checkpoint does not exist: {checkpoint}")
    target = net.module if hasattr(net, "module") else net
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    incompatible = target.load_state_dict(state, strict=not allow_partial)
    report = {
        "path": str(checkpoint),
        "loaded": len(state) - len(incompatible.unexpected_keys),
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }
    print(
        "StainViz generator initialization: "
        f"loaded={report['loaded']} missing={len(report['missing'])} "
        f"unexpected={len(report['unexpected'])}"
    )
    return report
