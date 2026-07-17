"""Registration-grid helpers using the StainViz align_corners=False convention."""

import torch
import torch.nn.functional as F


def identity_grid(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return a normalized ``[H,W,2]`` identity sampling grid."""
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def warp_image(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Warp ``[B,C,H,W]`` images with ``[B,H,W,2]`` normalized grids."""
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)
