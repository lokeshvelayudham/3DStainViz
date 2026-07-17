"""Joint spatial transforms for image slabs, masks, confidence, and warp grids."""

import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from models.stainviz_warp import identity_grid


class JointVolumeTransform:
    """Apply one resize/crop/flip realization to every spatial tensor in a sample."""

    def __init__(self, output_size: Tuple[int, int], random_flip: bool = False):
        self.output_size = output_size
        self.random_flip = random_flip

    @staticmethod
    def _crop(tensor: torch.Tensor, crop: Tuple[int, int, int, int]) -> torch.Tensor:
        top, left, height, width = crop
        return tensor[..., top : top + height, left : left + width]

    @staticmethod
    def _transform_grid(
        grid: torch.Tensor, original_hw: Tuple[int, int], crop: Tuple[int, int, int, int], flip: bool
    ) -> torch.Tensor:
        top, left, height, width = crop
        original_h, original_w = original_hw
        field = grid.permute(0, 3, 1, 2)
        field = JointVolumeTransform._crop(field, crop).permute(0, 2, 3, 1).contiguous()
        x_pixels = ((field[..., 0] + 1.0) * original_w - 1.0) / 2.0 - left
        y_pixels = ((field[..., 1] + 1.0) * original_h - 1.0) / 2.0 - top
        field[..., 0] = (2.0 * x_pixels + 1.0) / width - 1.0
        field[..., 1] = (2.0 * y_pixels + 1.0) / height - 1.0
        if flip:
            field = torch.flip(field, dims=(2,))
            field[..., 0] = -field[..., 0]
        return field

    def __call__(
        self,
        sample: Dict[str, torch.Tensor],
        crop: Optional[Tuple[int, int, int, int]] = None,
        horizontal_flip: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        spatial = next(value for value in sample.values() if torch.is_tensor(value) and value.ndim >= 3)
        original_h, original_w = spatial.shape[-2:]
        out_h, out_w = self.output_size
        if crop is None:
            crop_h, crop_w = min(out_h, original_h), min(out_w, original_w)
            top = random.randint(0, max(0, original_h - crop_h))
            left = random.randint(0, max(0, original_w - crop_w))
            crop = (top, left, crop_h, crop_w)
        flip = self.random_flip and random.random() > 0.5 if horizontal_flip is None else horizontal_flip
        result = dict(sample)
        for key, value in sample.items():
            if not torch.is_tensor(value):
                continue
            if key.startswith("warp_") and value.ndim == 4 and value.shape[-1] == 2:
                transformed = self._transform_grid(value.clone(), (original_h, original_w), crop, flip)
                if transformed.shape[1:3] != (out_h, out_w):
                    transformed = F.interpolate(
                        transformed.permute(0, 3, 1, 2), size=(out_h, out_w), mode="bilinear", align_corners=False
                    ).permute(0, 2, 3, 1)
                # Reconstruct exact identity to avoid interpolation drift.
                if value.numel() and torch.allclose(value[0], identity_grid(original_h, original_w), atol=1e-6):
                    transformed = identity_grid(out_h, out_w, value.device, value.dtype).expand(value.shape[0], -1, -1, -1).clone()
                result[key] = transformed
            elif value.ndim >= 3 and value.shape[-2:] == (original_h, original_w):
                transformed = self._crop(value, crop)
                if flip:
                    transformed = torch.flip(transformed, dims=(-1,))
                if transformed.shape[-2:] != (out_h, out_w):
                    leading = transformed.shape[:-3]
                    channels = transformed.shape[-3]
                    flat = transformed.reshape(-1, channels, *transformed.shape[-2:])
                    mode = "nearest" if "mask" in key or "valid" in key else "bilinear"
                    kwargs = {} if mode == "nearest" else {"align_corners": False}
                    flat = F.interpolate(flat, size=(out_h, out_w), mode=mode, **kwargs)
                    transformed = flat.reshape(*leading, channels, out_h, out_w)
                result[key] = transformed
        return result
