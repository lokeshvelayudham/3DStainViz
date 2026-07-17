"""Sparse-paired 2.5D pix2pix model for research-use virtual staining."""

from contextlib import nullcontext

import torch

from .base_model import BaseModel
from . import networks
from .stainviz_3d_networks import StainViz25DGenerator
from .stainviz_checkpoint import load_generator_checkpoint
from .stainviz_losses import CrossSliceConsistencyLoss, masked_robust_loss, masked_ssim_loss


class StainViz3DPix2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="instance", netG="stainviz_25d", dataset_mode="blockface_paired", pool_size=0)
        parser.add_argument("--fusion_heads", type=int, default=4)
        parser.add_argument("--mixed_precision", action="store_true")
        parser.add_argument("--init_G_from", default="")
        parser.add_argument("--allow_partial_generator_load", action="store_true")
        if is_train:
            parser.add_argument("--lambda_L1", type=float, default=100.0)
            parser.add_argument("--reconstruction_loss", choices=("l1", "charbonnier"), default="charbonnier")
            parser.add_argument("--lambda_ssim", type=float, default=0.0)
            parser.add_argument("--lambda_cross_slice", type=float, default=0.0)
            parser.add_argument("--cross_slice_space", choices=("rgb", "sobel", "generator"), default="sobel")
            parser.add_argument("--source_change_beta", type=float, default=5.0)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ["G_GAN", "G_reconstruction", "G_ssim", "G_cross", "D_real", "D_fake"]
        self.visual_names = ["real_A_center", "fake_B_center", "real_B_center"]
        self.model_names = ["G", "D"] if self.isTrain else ["G"]
        self.netG = StainViz25DGenerator(opt.input_nc, opt.output_nc, opt.ngf, opt.fusion_heads)
        if self.isTrain:
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.norm,
                opt.init_type,
                opt.init_gain,
            )
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.cross_slice = CrossSliceConsistencyLoss(opt.cross_slice_space, opt.source_change_beta)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.extend((self.optimizer_G, self.optimizer_D))
        self.amp_enabled = bool(opt.mixed_precision and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.initialization_report = None

    def setup(self, opt):
        super().setup(opt)
        if opt.init_G_from and not getattr(opt, "continue_train", False):
            self.initialization_report = load_generator_checkpoint(
                self.netG, opt.init_G_from, opt.allow_partial_generator_load
            )

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.amp_enabled else nullcontext()

    def set_input(self, inputs):
        if self.opt.direction != "AtoB":
            raise ValueError("StainViz paired volumes currently require --direction AtoB")
        self.real_A = inputs["A"].to(self.device)
        self.real_B = inputs["B"].to(self.device)
        self.z_offsets = inputs.get("z_um_A")
        self.z_offsets = self.z_offsets.to(self.device) if self.z_offsets is not None else None
        self.neighbor_valid = inputs["neighbor_valid_A"].to(self.device)
        self.pair_valid = inputs["pair_valid"].to(self.device)
        self.tissue_mask = inputs["tissue_mask_A"].to(self.device)
        self.pair_confidence = inputs["pair_confidence"].to(self.device)
        self.grids = inputs["warp_A_to_next"].to(self.device)
        self.warp_confidence = inputs["warp_conf_A_to_next"].to(self.device)
        self.image_paths = inputs["A_paths"]

    def forward(self):
        output = self.netG(
            self.real_A,
            z_offsets=self.z_offsets,
            neighbor_valid=self.neighbor_valid,
            registration_confidence=self.warp_confidence,
        )
        self.fake_B = output["prediction"]
        center = self.fake_B.shape[1] // 2
        self.fake_B_center = output["center_prediction"]
        self.real_A_center = self.real_A[:, center]
        self.real_B_center = self.real_B[:, center]

    def _center_valid(self):
        return self.pair_valid[:, self.pair_valid.shape[1] // 2]

    def backward_D(self):
        valid = self._center_valid()
        if valid.any():
            fake_pair = torch.cat((self.real_A_center[valid], self.fake_B_center[valid]), dim=1)
            real_pair = torch.cat((self.real_A_center[valid], self.real_B_center[valid]), dim=1)
            self.loss_D_fake = self.criterionGAN(self.netD(fake_pair.detach()), False)
            self.loss_D_real = self.criterionGAN(self.netD(real_pair), True)
            loss = (self.loss_D_fake + self.loss_D_real) * 0.5
        else:
            loss = sum(parameter.sum() for parameter in self.netD.parameters()) * 0.0
            self.loss_D_fake = loss
            self.loss_D_real = loss
        self.scaler.scale(loss).backward()
        return loss

    def backward_G(self):
        valid = self._center_valid()
        if valid.any():
            fake_pair = torch.cat((self.real_A_center[valid], self.fake_B_center[valid]), dim=1)
            self.loss_G_GAN = self.criterionGAN(self.netD(fake_pair), True)
        else:
            self.loss_G_GAN = self.fake_B.sum() * 0.0
        valid_mask = self.pair_valid[:, :, None, None, None] * self.pair_confidence * self.tissue_mask
        self.loss_G_reconstruction = masked_robust_loss(
            self.fake_B, self.real_B, valid_mask, self.opt.reconstruction_loss
        ) * self.opt.lambda_L1
        flat_fake = self.fake_B.reshape(-1, *self.fake_B.shape[2:])
        flat_real = self.real_B.reshape(-1, *self.real_B.shape[2:])
        flat_mask = valid_mask.reshape(-1, *valid_mask.shape[2:])
        self.loss_G_ssim = masked_ssim_loss(flat_fake, flat_real, flat_mask) * self.opt.lambda_ssim
        self.loss_G_cross = self.cross_slice(
            self.fake_B,
            self.grids,
            self.warp_confidence,
            self.neighbor_valid,
            tissue_mask=self.tissue_mask,
            source=self.real_A,
        ) * self.opt.lambda_cross_slice
        self.loss_G = self.loss_G_GAN + self.loss_G_reconstruction + self.loss_G_ssim + self.loss_G_cross
        self.scaler.scale(self.loss_G).backward()

    def optimize_parameters(self):
        with self._autocast():
            self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad(set_to_none=True)
        with self._autocast():
            self.loss_D = self.backward_D()
        self.scaler.step(self.optimizer_D)
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad(set_to_none=True)
        with self._autocast():
            self.backward_G()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

