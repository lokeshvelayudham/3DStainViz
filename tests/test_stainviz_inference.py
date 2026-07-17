import json

import numpy as np
import torch

from data.stainviz_normalization import PercentileNormalizer
from scripts.stainviz.register_slices import pixel_affine_to_grid, registration_confidence
from util.stainviz_metrics_2d import image_metrics
from util.stainviz_metrics_3d import volume_discontinuity
from util.stainviz_volume_io import infer_volume_tiled, save_volume_package
from models.stainviz_warp import identity_grid


class IdentitySlabModel(torch.nn.Module):
    def forward(self, x, **kwargs):
        return {"prediction": x, "center_prediction": x[:, x.shape[1] // 2]}


def test_pixel_affine_identity_produces_grid_sample_identity():
    grid = pixel_affine_to_grid(np.eye(3), height=7, width=9)
    assert torch.allclose(grid, identity_grid(7, 9), atol=1e-6)


def test_registration_confidence_is_high_for_equal_images_and_low_for_mismatch():
    image = np.zeros((16, 16), dtype=np.float32)
    image[4:12, 4:12] = 1.0

    equal = registration_confidence(image, image)
    mismatch = registration_confidence(image, 1.0 - image)

    assert equal.mean() > 0.9
    assert mismatch.mean() < equal.mean() * 0.5


def test_percentile_normalizer_round_trips_serialized_training_statistics(tmp_path):
    training = [np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)]
    normalizer = PercentileNormalizer.fit(training, low=0.0, high=100.0)
    path = tmp_path / "normalization.json"
    normalizer.save(path)

    restored = PercentileNormalizer.load(path)
    transformed = restored(np.array([[0.0, 15.0, 30.0]], dtype=np.float32))

    assert np.allclose(transformed, [[0.0, 0.5, 1.0]])


def test_tiled_volume_inference_reconstructs_identity_without_seams():
    volume = torch.linspace(-1, 1, 4 * 3 * 13 * 15).reshape(4, 3, 13, 15)

    output, metadata = infer_volume_tiled(
        IdentitySlabModel(), volume, context_slices=3, tile_size=8, overlap_xy=4, device=torch.device("cpu")
    )

    assert torch.allclose(output, volume, atol=1e-5)
    assert metadata["z_prediction_counts"] == [2, 3, 3, 2]


def test_missing_plane_remains_flagged_and_package_has_provenance(tmp_path):
    volume = torch.zeros(3, 3, 8, 8)
    output, metadata = infer_volume_tiled(
        IdentitySlabModel(),
        volume,
        context_slices=3,
        tile_size=8,
        overlap_xy=2,
        missing_mask=torch.tensor([False, True, False]),
        device=torch.device("cpu"),
    )
    provenance = {"research_use_only": True, "target_stain": "HE", "excluded_slices": [1]}
    save_volume_package(tmp_path, output, metadata, provenance)

    assert torch.count_nonzero(output[1]) == 0
    assert (tmp_path / "volume.npy").exists()
    assert (tmp_path / "slices" / "000000.png").exists()
    assert json.loads((tmp_path / "provenance.json").read_text())["research_use_only"] is True


def test_metrics_report_perfect_2d_match_and_zero_volume_discontinuity():
    image = np.zeros((8, 8, 3), dtype=np.float32)
    metrics = image_metrics(image, image)
    volume = torch.zeros(3, 3, 8, 8)

    assert metrics["ssim"] == 1.0
    assert metrics["psnr"] == float("inf")
    assert volume_discontinuity(volume).item() == 0.0
