"""Volumetric consistency metrics for StainViz-3D outputs."""

from __future__ import annotations

from typing import Optional

import torch

from models.stainviz_warp import warp_image


def volume_discontinuity(
    volume: torch.Tensor,
    grids: Optional[torch.Tensor] = None,
    confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean adjacent-slice absolute discontinuity, optionally warp-compensated."""
    values = volume if torch.is_tensor(volume) else torch.as_tensor(volume)
    values = values.float()
    if values.ndim != 4:
        raise ValueError("volume must have shape [Z,C,H,W]")
    if values.shape[0] < 2:
        return values.new_tensor(0.0)
    losses = []
    for index in range(values.shape[0] - 1):
        left = values[index:index + 1]
        right = values[index + 1:index + 2]
        if grids is not None:
            left = warp_image(left, grids[index:index + 1].to(values.device).float())
        diff = (left - right).abs()
        if confidence is not None:
            weight = confidence[index:index + 1].to(values.device).float()
            losses.append((diff * weight).sum() / weight.sum().clamp_min(1.0))
        else:
            losses.append(diff.mean())
    return torch.stack(losses).mean()


def z_smoothing_control(volume: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Simple z-axis smoothing baseline for exposing artificially uniform output."""
    values = volume.float()
    if kernel <= 1 or values.shape[0] < 2:
        return values.clone()
    radius = kernel // 2
    padded = torch.cat([values[:1].expand(radius, -1, -1, -1), values, values[-1:].expand(radius, -1, -1, -1)], dim=0)
    smoothed = []
    for index in range(values.shape[0]):
        smoothed.append(padded[index:index + kernel].mean(dim=0))
    return torch.stack(smoothed)
