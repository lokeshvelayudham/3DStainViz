"""Unpaired ordered-blockface to ordered-or-unordered stain dataset."""

import random

import torch

from .base_dataset import BaseDataset
from .stainviz_transforms import JointVolumeTransform
from .stainviz_volume_dataset import OrderedSlabLoader


class BlockfaceUnalignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return __import__("data.blockface_paired_dataset", fromlist=["BlockfacePairedDataset"]).BlockfacePairedDataset.modify_commandline_options(parser, is_train)

    def __init__(self, opt):
        super().__init__(opt)
        self.source = OrderedSlabLoader(opt, opt.source_domain)
        target_opt = type("TargetOptions", (), vars(opt).copy())()
        target_opt.context_slices = 1
        target_opt.context_offsets = "0"
        self.target = OrderedSlabLoader(target_opt, opt.target_domain)
        self.transform = JointVolumeTransform((opt.crop_size, opt.crop_size), random_flip=not opt.no_flip)

    def __len__(self):
        return max(len(self.source), len(self.target))

    def __getitem__(self, index):
        source = self.source.load(index % len(self.source), self.opt.input_nc)
        target_index = index % len(self.target) if self.opt.serial_batches else random.randrange(len(self.target))
        target = self.target.load(target_index, self.opt.output_nc)
        source_rows = source.pop("rows")
        target_rows = target.pop("rows")
        source = self.transform(source)
        target = self.transform(target)
        source_center = len(source_rows) // 2
        return {
            "A": source["A"],
            "B": target["A"],
            "A_paths": [row.image_path for row in source_rows],
            "B_paths": [row.image_path for row in target_rows],
            "specimen_id": source_rows[source_center].specimen_id,
            "volume_id": source_rows[source_center].volume_id,
            "center_index": torch.tensor(source_center),
            "slice_indices_A": torch.tensor([row.slice_index for row in source_rows]),
            "z_um_A": torch.tensor([row.z_um for row in source_rows], dtype=torch.float32),
            "neighbor_valid_A": source["neighbor_valid_A"],
            "neighbor_valid_B": target["neighbor_valid_A"],
            "tissue_mask_A": source["tissue_mask_A"],
            "tissue_mask_B": target["tissue_mask_A"],
            "warp_A_to_next": source["warp_A_to_next"],
            "warp_B_to_next": target["warp_A_to_next"],
            "warp_conf_A_to_next": source["warp_conf_A_to_next"],
            "warp_conf_B_to_next": target["warp_conf_A_to_next"],
        }
