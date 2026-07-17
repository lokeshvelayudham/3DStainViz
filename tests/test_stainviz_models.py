from types import SimpleNamespace

import torch

import models


def _opt(tmp_path, model_name):
    return SimpleNamespace(
        model=model_name,
        isTrain=True,
        checkpoints_dir=str(tmp_path),
        name="smoke",
        device=torch.device("cpu"),
        preprocess="none",
        input_nc=3,
        output_nc=3,
        ngf=8,
        ndf=8,
        netD="pixel",
        n_layers_D=3,
        norm="instance",
        init_type="normal",
        init_gain=0.02,
        no_dropout=True,
        fusion_heads=2,
        lr=0.0002,
        beta1=0.5,
        gan_mode="lsgan",
        pool_size=0,
        direction="AtoB",
        lambda_L1=10.0,
        reconstruction_loss="charbonnier",
        lambda_ssim=1.0,
        lambda_cross_slice=1.0,
        cross_slice_space="sobel",
        source_change_beta=5.0,
        lambda_A=10.0,
        lambda_B=10.0,
        lambda_identity=0.0,
        lambda_cross_A=1.0,
        lambda_cross_B=0.0,
        mixed_precision=True,
        init_G_from="",
        init_G_A_from="",
        allow_partial_generator_load=False,
    )


def _paired_batch():
    batch, slices, channels, size = 1, 3, 3, 32
    return {
        "A": torch.randn(batch, slices, channels, size, size),
        "B": torch.randn(batch, slices, channels, size, size),
        "A_paths": [["a0"], ["a1"], ["a2"]],
        "z_um_A": torch.tensor([[-20.0, 0.0, 20.0]]),
        "neighbor_valid_A": torch.ones(batch, slices, dtype=torch.bool),
        "pair_valid": torch.ones(batch, slices, dtype=torch.bool),
        "tissue_mask_A": torch.ones(batch, slices, 1, size, size),
        "pair_confidence": torch.ones(batch, slices, 1, size, size),
        "warp_A_to_next": torch.stack(
            [
                torch.stack(torch.meshgrid(
                    (torch.arange(size) + 0.5) * 2 / size - 1,
                    (torch.arange(size) + 0.5) * 2 / size - 1,
                    indexing="ij",
                ), dim=-1).flip(-1)
                for _ in range(slices - 1)
            ]
        ).unsqueeze(0),
        "warp_conf_A_to_next": torch.ones(batch, slices - 1, 1, size, size),
    }


def test_paired_model_plugin_completes_cpu_optimization_and_saves(tmp_path):
    model = models.create_model(_opt(tmp_path, "stainviz_3d_pix2pix"))
    model.set_input(_paired_batch())

    model.optimize_parameters()
    model.save_dir.mkdir(parents=True, exist_ok=True)
    model.save_networks("latest")

    assert torch.isfinite(model.loss_G)
    assert model.fake_B_center.shape == (1, 3, 32, 32)
    assert (model.save_dir / "latest_net_G.pth").exists()


def test_cyclegan_model_plugin_handles_ordered_a_and_unordered_b(tmp_path):
    option = _opt(tmp_path, "stainviz_3d_cycle_gan")
    model = models.create_model(option)
    batch = _paired_batch()
    batch.update({
        "B": torch.randn(1, 1, 3, 32, 32),
        "B_paths": [["b0"]],
        "neighbor_valid_B": torch.ones(1, 1, dtype=torch.bool),
        "tissue_mask_B": torch.ones(1, 1, 1, 32, 32),
        "warp_B_to_next": torch.empty(1, 0, 32, 32, 2),
        "warp_conf_B_to_next": torch.empty(1, 0, 1, 32, 32),
    })

    model.set_input(batch)
    model.optimize_parameters()

    assert torch.isfinite(model.loss_G)
    assert model.fake_B_center.shape == (1, 3, 32, 32)
    assert model.fake_A_center.shape == (1, 3, 32, 32)
    assert model.loss_cross_B == 0.0

