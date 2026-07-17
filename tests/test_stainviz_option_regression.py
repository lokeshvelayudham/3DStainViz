import sys

from options.train_options import TrainOptions


def _parse_train(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    return TrainOptions().parse()


def test_original_pix2pix_option_parsing_is_unchanged(monkeypatch, tmp_path):
    opt = _parse_train(
        monkeypatch,
        [
            "--dataroot",
            str(tmp_path),
            "--name",
            "legacy_pix2pix",
            "--checkpoints_dir",
            str(tmp_path / "checkpoints"),
            "--model",
            "pix2pix",
            "--dataset_mode",
            "aligned",
            "--direction",
            "BtoA",
        ],
    )

    assert opt.model == "pix2pix"
    assert opt.dataset_mode == "aligned"
    assert opt.direction == "BtoA"


def test_stainviz_option_parsing_adds_volumetric_contract(monkeypatch, tmp_path):
    opt = _parse_train(
        monkeypatch,
        [
            "--dataroot",
            str(tmp_path),
            "--name",
            "stainviz",
            "--checkpoints_dir",
            str(tmp_path / "checkpoints"),
            "--model",
            "stainviz_3d_pix2pix",
            "--dataset_mode",
            "blockface_paired",
            "--manifest_path",
            str(tmp_path / "manifest.csv"),
            "--context_slices",
            "5",
            "--fusion_heads",
            "2",
        ],
    )

    assert opt.model == "stainviz_3d_pix2pix"
    assert opt.dataset_mode == "blockface_paired"
    assert opt.context_slices == 5
    assert opt.fusion_heads == 2
