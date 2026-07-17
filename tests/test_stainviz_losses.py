import torch

from models.stainviz_losses import CrossSliceConsistencyLoss, masked_robust_loss
from models.stainviz_warp import identity_grid


def _grids(batch, pairs, height, width):
    return identity_grid(height, width).expand(batch, pairs, height, width, 2).clone()


def test_masked_robust_loss_ignores_invalid_pixels():
    prediction = torch.tensor([[[[1.0, 20.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0, 0.0]]]])

    loss = masked_robust_loss(prediction, target, mask, kind="l1")

    assert torch.isclose(loss, torch.tensor(1.0))


def test_identical_outputs_have_zero_cross_slice_loss_and_flicker_increases_it():
    base = torch.zeros(1, 3, 1, 8, 8)
    flicker = base.clone()
    flicker[:, 1] = 1.0
    grids = _grids(1, 2, 8, 8)
    confidence = torch.ones(1, 2, 1, 8, 8)
    valid = torch.ones(1, 3, dtype=torch.bool)
    criterion = CrossSliceConsistencyLoss(space="rgb")

    smooth_loss = criterion(base, grids, confidence, valid)
    flicker_loss = criterion(flicker, grids, confidence, valid)

    assert smooth_loss.item() < 1e-5
    assert flicker_loss > smooth_loss + 0.1


def test_zero_confidence_returns_graph_connected_zero():
    prediction = torch.randn(1, 3, 1, 8, 8, requires_grad=True)
    loss = CrossSliceConsistencyLoss(space="sobel")(
        prediction,
        _grids(1, 2, 8, 8),
        torch.zeros(1, 2, 1, 8, 8),
        torch.ones(1, 3, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None


def test_source_change_gating_reduces_false_consistency_penalty():
    prediction = torch.zeros(1, 2, 1, 8, 8)
    prediction[:, 1, :, :, 4:] = 1.0
    source = torch.zeros_like(prediction)
    source[:, 1, :, :, 4:] = 1.0
    grids = _grids(1, 1, 8, 8)
    confidence = torch.ones(1, 1, 1, 8, 8)
    valid = torch.ones(1, 2, dtype=torch.bool)

    plain = CrossSliceConsistencyLoss(space="rgb", source_change_beta=0.0)(
        prediction, grids, confidence, valid, source=source
    )
    gated = CrossSliceConsistencyLoss(space="rgb", source_change_beta=5.0)(
        prediction, grids, confidence, valid, source=source
    )

    assert gated < plain * 0.1
