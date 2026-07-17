"""Memory-bounded 2.5D generator with shared 2D encoding and z-aware fusion."""

import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.ReLU(True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels, affine=True),
            nn.ReLU(True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SinusoidalZEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.arange(half, device=offsets.device, dtype=offsets.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = offsets.unsqueeze(-1) * frequencies.unsqueeze(0).unsqueeze(0)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[-1]))
        return embedding


class SliceAttentionFusion(nn.Module):
    """Fuse the z sequence independently at each bottleneck location."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        if channels % heads:
            raise ValueError("bottleneck channels must be divisible by fusion_heads")
        self.z_embedding = SinusoidalZEmbedding(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    @staticmethod
    def _slice_confidence(confidence: torch.Tensor, slices: int) -> torch.Tensor:
        if confidence.ndim > 2:
            confidence = confidence.flatten(2).mean(-1)
        if confidence.shape[1] == slices:
            return confidence
        if confidence.shape[1] != max(0, slices - 1):
            raise ValueError("registration_confidence must describe K slices or K-1 adjacent pairs")
        if slices == 1:
            return torch.ones(confidence.shape[0], 1, device=confidence.device, dtype=confidence.dtype)
        result = torch.ones(confidence.shape[0], slices, device=confidence.device, dtype=confidence.dtype)
        result[:, 0] = confidence[:, 0]
        result[:, -1] = confidence[:, -1]
        if slices > 2:
            result[:, 1:-1] = (confidence[:, :-1] + confidence[:, 1:]) * 0.5
        return result.clamp(0.0, 1.0)

    def forward(
        self,
        encoded: torch.Tensor,
        z_offsets: torch.Tensor,
        neighbor_valid: torch.Tensor,
        registration_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, slices, channels, height, width = encoded.shape
        valid = neighbor_valid.bool().clone()
        all_invalid = ~valid.any(dim=1)
        if all_invalid.any():
            valid[all_invalid, slices // 2] = True
        features = encoded + self.z_embedding(z_offsets).view(batch, slices, channels, 1, 1)
        if registration_confidence is not None:
            weights = self._slice_confidence(registration_confidence, slices)
            features = features * weights.view(batch, slices, 1, 1, 1)
        sequence = features.permute(0, 3, 4, 1, 2).reshape(batch * height * width, slices, channels)
        padding = (~valid).view(batch, 1, 1, slices).expand(batch, height, width, slices)
        padding = padding.reshape(batch * height * width, slices)
        attended, _ = self.attention(sequence, sequence, sequence, key_padding_mask=padding, need_weights=False)
        sequence = self.norm(sequence + attended)
        return sequence.reshape(batch, height, width, slices, channels).permute(0, 3, 4, 1, 2).contiguous()


class StainViz25DGenerator(nn.Module):
    """Shared-encoder generator accepting legacy 4D or volumetric 5D input."""

    def __init__(self, input_nc: int, output_nc: int, ngf: int = 64, fusion_heads: int = 4):
        super().__init__()
        bottleneck = ngf * 4
        self.encoder_1 = ConvBlock(input_nc, ngf)
        self.encoder_2 = ConvBlock(ngf, ngf * 2, stride=2)
        self.encoder_3 = ConvBlock(ngf * 2, bottleneck, stride=2)
        self.fusion = SliceAttentionFusion(bottleneck, fusion_heads)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck, ngf * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(True),
            nn.Conv2d(ngf, output_nc, 3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        z_offsets: Optional[torch.Tensor] = None,
        neighbor_valid: Optional[torch.Tensor] = None,
        registration_confidence: Optional[torch.Tensor] = None,
        target_stain_id: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del target_stain_id
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError("StainViz25DGenerator expects [B,C,H,W] or [B,K,C,H,W]")
        batch, slices, channels, height, width = x.shape
        flat = x.reshape(batch * slices, channels, height, width)
        encoded = self.encoder_3(self.encoder_2(self.encoder_1(flat)))
        encoded = encoded.reshape(batch, slices, *encoded.shape[1:])
        if z_offsets is None:
            z_offsets = torch.arange(slices, device=x.device, dtype=x.dtype) - slices // 2
            z_offsets = z_offsets.unsqueeze(0).expand(batch, -1)
        else:
            z_offsets = z_offsets.to(device=x.device, dtype=x.dtype)
            z_offsets = z_offsets - z_offsets[:, slices // 2 : slices // 2 + 1]
        if neighbor_valid is None:
            neighbor_valid = torch.ones(batch, slices, device=x.device, dtype=torch.bool)
        else:
            neighbor_valid = neighbor_valid.to(device=x.device, dtype=torch.bool)
        fused = self.fusion(encoded, z_offsets, neighbor_valid, registration_confidence)
        decoded = self.decoder(fused.reshape(batch * slices, *fused.shape[2:]))
        prediction = decoded.reshape(batch, slices, *decoded.shape[1:])
        output = {
            "prediction": prediction,
            "center_prediction": prediction[:, slices // 2],
            "log_scale": None,
            "features": None,
        }
        if return_features:
            output["features"] = {"encoded": encoded, "fused": fused}
        return output


def define_stainviz_generator(input_nc: int, output_nc: int, ngf: int, fusion_heads: int, gpu_ids):
    """Create and initialize a StainViz generator using upstream initialization."""
    from . import networks

    generator = StainViz25DGenerator(input_nc, output_nc, ngf, fusion_heads)
    return networks.init_net(generator, "normal", 0.02, gpu_ids)

