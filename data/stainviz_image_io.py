"""Image I/O with explicit bit-depth scaling for StainViz-3D."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


def _read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile

            return np.asarray(tifffile.imread(path))
        except ImportError:
            pass
    with Image.open(path) as image:
        return np.asarray(image)


def load_image_tensor(
    path: Union[str, Path], channels: Optional[int] = None, normalize: bool = True
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Load an image as ``[C,H,W]`` and optionally scale it to ``[-1,1]``."""
    image_path = Path(path)
    array = _read_array(image_path)
    original_dtype = array.dtype
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"unsupported image shape {array.shape} for {image_path.name}")
    if channels == 1 and array.shape[-1] != 1:
        array = np.mean(array[..., :3], axis=-1, keepdims=True)
    elif channels == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif channels is not None and array.shape[-1] != channels:
        raise ValueError(f"expected {channels} channels but found {array.shape[-1]} in {image_path.name}")
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()
    bit_depth = int(np.iinfo(original_dtype).bits) if np.issubdtype(original_dtype, np.integer) else 32
    if normalize:
        if np.issubdtype(original_dtype, np.integer):
            maximum = float(np.iinfo(original_dtype).max)
        else:
            maximum = float(np.nanmax(array)) if float(np.nanmax(array)) > 1.0 else 1.0
        tensor = tensor.div(maximum).clamp(0.0, 1.0).mul(2.0).sub(1.0)
    return tensor, {
        "dtype": str(original_dtype),
        "bit_depth": bit_depth,
        "shape": list(array.shape),
        "scale": "[-1,1]" if normalize else "raw",
    }


def load_mask_tensor(path: Union[str, Path]) -> torch.Tensor:
    """Load a mask as a binary ``[1,H,W]`` tensor."""
    tensor, _ = load_image_tensor(path, channels=1, normalize=False)
    return (tensor > 0).float()

