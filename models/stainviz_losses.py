"""Masked paired and deformation-compensated cross-slice objectives."""

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .stainviz_warp import warp_image


def masked_robust_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "charbonnier",
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Compute L1 or Charbonnier loss only over valid pixels and channels."""
    residual = prediction - target
    if kind == "l1":
        error = residual.abs()
    elif kind == "charbonnier":
        error = torch.sqrt(residual.square() + epsilon ** 2)
    else:
        raise ValueError(f"unsupported reconstruction loss: {kind}")
    weights = torch.broadcast_to(mask.to(error.dtype), error.shape)
    denominator = weights.sum()
    if denominator.item() == 0:
        return prediction.sum() * 0.0
    return (error * weights).sum() / denominator.clamp_min(epsilon)


def masked_ssim_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Differentiable local SSIM averaged over valid regions."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(prediction, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    var_x = F.avg_pool2d(prediction * prediction, 3, 1, 1) - mu_x.square()
    var_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y.square()
    covariance = F.avg_pool2d(prediction * target, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    ).clamp_min(1e-8)
    weights = torch.broadcast_to(mask.to(ssim.dtype), ssim.shape)
    if weights.sum().item() == 0:
        return prediction.sum() * 0.0
    return ((1.0 - ssim) * 0.5 * weights).sum() / weights.sum().clamp_min(1.0)


def sobel_features(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    kernel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    kernel_y = kernel_x.t()
    kernels = torch.stack((kernel_x, kernel_y)).unsqueeze(1)
    kernels = kernels.repeat(channels, 1, 1, 1)
    gradients = F.conv2d(image, kernels, padding=1, groups=channels)
    return gradients


class CrossSliceConsistencyLoss(nn.Module):
    """Compare adjacent generated planes after precomputed deformation."""

    def __init__(self, space: str = "sobel", source_change_beta: float = 5.0, epsilon: float = 1e-6):
        super().__init__()
        if space not in {"rgb", "sobel", "generator"}:
            raise ValueError(f"unsupported cross-slice space: {space}")
        self.space = space
        self.source_change_beta = source_change_beta
        self.epsilon = epsilon

    def _features(self, image: torch.Tensor) -> torch.Tensor:
        return sobel_features(image) if self.space == "sobel" else image

    def forward(
        self,
        prediction: torch.Tensor,
        grids: torch.Tensor,
        confidence: torch.Tensor,
        neighbor_valid: torch.Tensor,
        tissue_mask: Optional[torch.Tensor] = None,
        source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prediction.ndim != 5:
            raise ValueError("prediction must have shape [B,K,C,H,W]")
        pair_losses = []
        for index in range(prediction.shape[1] - 1):
            left = self._features(prediction[:, index])
            right = self._features(prediction[:, index + 1])
            warped = warp_image(left, grids[:, index])
            weights = confidence[:, index].to(prediction.dtype)
            valid = (neighbor_valid[:, index] & neighbor_valid[:, index + 1]).to(prediction.dtype)
            weights = weights * valid.view(-1, 1, 1, 1)
            if tissue_mask is not None:
                weights = weights * tissue_mask[:, index] * tissue_mask[:, index + 1]
            if source is not None and self.source_change_beta > 0:
                source_left = warp_image(source[:, index], grids[:, index])
                source_change = (source[:, index + 1] - source_left).abs().mean(dim=1, keepdim=True)
                weights = weights * torch.exp(-self.source_change_beta * source_change)
            expanded = torch.broadcast_to(weights, right.shape)
            denominator = expanded.sum()
            if denominator.item() > 0:
                robust = torch.sqrt((right - warped).square() + self.epsilon ** 2) - self.epsilon
                pair_losses.append((robust * expanded).sum() / denominator.clamp_min(self.epsilon))
        if not pair_losses:
            return prediction.sum() * 0.0
        return torch.stack(pair_losses).mean()

