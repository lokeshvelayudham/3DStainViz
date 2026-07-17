import torch

from models.stainviz_3d_networks import StainViz25DGenerator


def _generator():
    return StainViz25DGenerator(input_nc=3, output_nc=3, ngf=8, fusion_heads=2)


def test_generator_accepts_4d_and_5d_inputs():
    generator = _generator()

    single = generator(torch.randn(2, 3, 32, 32))
    slab = generator(
        torch.randn(2, 3, 3, 32, 32),
        z_offsets=torch.tensor([[-20.0, 0.0, 20.0], [-20.0, 0.0, 20.0]]),
        neighbor_valid=torch.tensor([[True, True, True], [False, True, False]]),
    )

    assert single["prediction"].shape == (2, 1, 3, 32, 32)
    assert single["center_prediction"].shape == (2, 3, 32, 32)
    assert slab["prediction"].shape == (2, 3, 3, 32, 32)
    assert slab["center_prediction"].shape == (2, 3, 32, 32)


def test_all_invalid_neighbors_fall_back_without_nan_and_backward_is_finite():
    generator = _generator()
    inputs = torch.randn(1, 3, 3, 32, 32, requires_grad=True)

    output = generator(inputs, neighbor_valid=torch.zeros(1, 3, dtype=torch.bool))
    output["center_prediction"].mean().backward()

    assert torch.isfinite(output["prediction"]).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_return_features_and_confidence_inputs_are_supported():
    generator = _generator()
    output = generator(
        torch.randn(1, 3, 3, 32, 32),
        registration_confidence=torch.ones(1, 2, 1, 32, 32),
        return_features=True,
    )

    assert set(output["features"]) == {"encoded", "fused"}
    assert output["features"]["encoded"].shape[:2] == (1, 3)
