"""Sparse-anchor paired blockface slab dataset."""

import torch

from .base_dataset import BaseDataset
from .stainviz_transforms import JointVolumeTransform
from .stainviz_volume_dataset import OrderedSlabLoader


class BlockfacePairedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--manifest_path", required=True)
        parser.add_argument("--manifest_root", default="")
        parser.add_argument("--source_domain", default="blockface")
        parser.add_argument("--target_domain", default="HE")
        parser.add_argument("--context_slices", type=int, default=3)
        parser.add_argument("--context_stride", type=int, default=1)
        parser.add_argument("--context_offsets", default="")
        parser.add_argument("--boundary_padding", choices=("replicate", "reflect", "invalid"), default="replicate")
        parser.add_argument("--assume_registered", action="store_true")
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loader = OrderedSlabLoader(opt, opt.source_domain)
        self.transform = JointVolumeTransform((opt.crop_size, opt.crop_size), random_flip=not opt.no_flip)

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):
        item = self.loader.load(index, self.opt.input_nc, load_targets=True)
        rows = item.pop("rows")
        item = self.transform(item)
        center = len(rows) // 2
        return {
            **item,
            "A_paths": [str(row.image_path) for row in rows],
            "B_paths": [str(row.paired_target_path or "") for row in rows],
            "specimen_id": rows[center].specimen_id,
            "volume_id": rows[center].volume_id,
            "center_index": torch.tensor(center),
            "slice_indices_A": torch.tensor([row.slice_index for row in rows]),
            "z_um_A": torch.tensor([row.z_um for row in rows], dtype=torch.float32),
            "qc_score_A": torch.tensor([row.qc_score for row in rows], dtype=torch.float32),
        }
