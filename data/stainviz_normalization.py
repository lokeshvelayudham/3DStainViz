"""Training-set normalization utilities for StainViz-3D."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Union

import numpy as np


@dataclass(frozen=True)
class PercentileNormalizer:
    """Clip and scale images using percentiles fitted on training data."""

    minimum: float
    maximum: float
    low_percentile: float = 1.0
    high_percentile: float = 99.0
    eps: float = 1e-6

    @classmethod
    def fit(
        cls,
        arrays: Iterable[np.ndarray],
        low: float = 1.0,
        high: float = 99.0,
        max_samples: int = 2_000_000,
    ) -> "PercentileNormalizer":
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("low and high percentiles must satisfy 0 <= low < high <= 100")
        chunks = []
        remaining = max_samples
        for array in arrays:
            values = np.asarray(array, dtype=np.float32).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            if values.size > remaining:
                indices = np.linspace(0, values.size - 1, remaining, dtype=np.int64)
                values = values[indices]
            chunks.append(values)
            remaining -= values.size
            if remaining <= 0:
                break
        if not chunks:
            raise ValueError("cannot fit normalization from empty training arrays")
        pooled = np.concatenate(chunks)
        minimum, maximum = np.percentile(pooled, [low, high])
        if maximum <= minimum:
            maximum = minimum + 1.0
        return cls(float(minimum), float(maximum), float(low), float(high))

    def __call__(self, array: np.ndarray) -> np.ndarray:
        values = np.asarray(array, dtype=np.float32)
        scaled = (values - self.minimum) / max(self.maximum - self.minimum, self.eps)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["kind"] = "percentile"
        return payload

    def save(self, path: Union[str, Path]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PercentileNormalizer":
        payload = json.loads(Path(path).read_text())
        if payload.get("kind") not in {None, "percentile"}:
            raise ValueError(f"unsupported normalizer kind: {payload.get('kind')!r}")
        return cls(
            minimum=float(payload["minimum"]),
            maximum=float(payload["maximum"]),
            low_percentile=float(payload.get("low_percentile", 1.0)),
            high_percentile=float(payload.get("high_percentile", 99.0)),
            eps=float(payload.get("eps", 1e-6)),
        )
