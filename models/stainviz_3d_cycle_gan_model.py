"""Unpaired 2.5D CycleGAN with sequence-aware consistency constraints."""

from contextlib import nullcontext
import itertools

import torch

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .stainviz_3d_networks import StainViz25DGenerator
from .stainviz_checkpoint import load_generator_checkpoint
from .stainviz_losses import CrossSliceConsistencyLoss


class StainViz3DCycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True, netG="stainviz_25d", dataset_mode="blockface_unaligned")
        parser.add_argument("--fusion_heads", type=int, default=4)
        parser.add_argument("--mixed_precision", action="store_true")
        parser.add_argument("--init_G_A_from", default="")
        parser.add_argument("--allow_partial_generator_load", action="store_true")
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0)
            parser.add_argument("--lambda_B", type=float, default=10.0)
            parser.add_argument("--lambda_identity", type=float, default=0.5)
            parser.add_argument("--lambda_cross_A", type=float, default=0.0)
            parser.add_argument("--lambda_cross_B", type=float, default=0.0)
            parser.add_argument("--cross_slice_space", choices=("rgb", "sobel", "generator"), default="sobel")
            parser.add_argument("--source_change_beta", type=float, default=5.0)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = [
            "D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B", "cross_A", "cross_B"
        ]
        self.visual_names = [
            "real_A_center", "fake_B_center", "rec_A_center", "real_B_center", "fake_A_center", "rec_B_center"
        ]
        self.model_names = ["G_A", "G_B", "D_A", "D_B"] if self.isTrain else ["G_A", "G_B"]
        self.netG_A = StainViz25DGenerator(opt.input_nc, opt.output_nc, opt.ngf, opt.fusion_heads)
        self.netG_B = StainViz25DGenerator(opt.output_nc, opt.input_nc, opt.ngf, opt.fusion_heads)
        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            if opt.lambda_identity > 0 and opt.input_nc != opt.output_nc:
                raise ValueError("identity loss requires matching input and output channels")
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.cross_slice = CrossSliceConsistencyLoss(opt.cross_slice_space, opt.source_change_beta)
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.extend((self.optimizer_G, self.optimizer_D))
        self.amp_enabled = bool(opt.mixed_precision and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.initialization_report = None

    def setup(self, opt):
        super().setup(opt)
        if opt.init_G_A_from and not getattr(opt, "continue_train", False):
            self.initialization_report = load_generator_checkpoint(
                self.netG_A, opt.init_G_A_from, opt.allow_partial_generator_load
            )

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.amp_enabled else nullcontext()

    def set_input(self, inputs):
        self.real_A = inputs["A"].to(self.device)
        self.real_B = inputs["B"].to(self.device)
        self.valid_A = inputs["neighbor_valid_A"].to(self.device)
        self.valid_B = inputs["neighbor_valid_B"].to(self.device)
        self.mask_A = inputs["tissue_mask_A"].to(self.device)
        self.mask_B = inputs["tissue_mask_B"].to(self.device)
        self.grids_A = inputs["warp_A_to_next"].to(self.device)
        self.grids_B = inputs["warp_B_to_next"].to(self.device)
        self.conf_A = inputs["warp_conf_A_to_next"].to(self.device)
        self.conf_B = inputs["warp_conf_B_to_next"].to(self.device)
        z = inputs.get("z_um_A")
        self.z_A = z.to(self.device) if z is not None else None
        self.image_paths = inputs["A_paths"]

    @staticmethod
    def _center(volume):
        return volume[:, volume.shape[1] // 2]

    def forward(self):
        out_B = self.netG_A(self.real_A, self.z_A, self.valid_A, self.conf_A)
        self.fake_B = out_B["prediction"]
        self.rec_A = self.netG_B(self.fake_B, neighbor_valid=self.valid_A)["prediction"]
        out_A = self.netG_B(self.real_B, neighbor_valid=self.valid_B, registration_confidence=self.conf_B)
        self.fake_A = out_A["prediction"]
        self.rec_B = self.netG_A(self.fake_A, neighbor_valid=self.valid_B)["prediction"]
        self.real_A_center, self.real_B_center = self._center(self.real_A), self._center(self.real_B)
        self.fake_A_center, self.fake_B_center = self._center(self.fake_A), self._center(self.fake_B)
        self.rec_A_center, self.rec_B_center = self._center(self.rec_A), self._center(self.rec_B)

    def backward_D_basic(self, discriminator, real, fake):
        loss_real = self.criterionGAN(discriminator(real), True)
        loss_fake = self.criterionGAN(discriminator(fake.detach()), False)
        loss = (loss_real + loss_fake) * 0.5
        self.scaler.scale(loss).backward()
        return loss

    def backward_D(self):
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B_center, self.fake_B_pool.query(self.fake_B_center))
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A_center, self.fake_A_pool.query(self.fake_A_center))

    def backward_G(self):
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B_center), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A_center), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        if self.opt.lambda_identity > 0:
            self.idt_A = self.netG_A(self.real_B, neighbor_valid=self.valid_B)["prediction"]
            self.idt_B = self.netG_B(self.real_A, neighbor_valid=self.valid_A)["prediction"]
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_B * self.opt.lambda_identity
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_A * self.opt.lambda_identity
        else:
            self.loss_idt_A = self.fake_B.sum() * 0.0
            self.loss_idt_B = self.fake_A.sum() * 0.0
        self.loss_cross_A = self.cross_slice(
            self.fake_B, self.grids_A, self.conf_A, self.valid_A, self.mask_A, self.real_A
        ) * self.opt.lambda_cross_A
        if self.fake_A.shape[1] > 1 and self.opt.lambda_cross_B > 0:
            self.loss_cross_B = self.cross_slice(
                self.fake_A, self.grids_B, self.conf_B, self.valid_B, self.mask_B, self.real_B
            ) * self.opt.lambda_cross_B
        else:
            self.loss_cross_B = 0.0
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G = self.loss_G + self.loss_idt_A + self.loss_idt_B + self.loss_cross_A + self.loss_cross_B
        self.scaler.scale(self.loss_G).backward()

    def optimize_parameters(self):
        with self._autocast():
            self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad(set_to_none=True)
        with self._autocast():
            self.backward_G()
        self.scaler.step(self.optimizer_G)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad(set_to_none=True)
        with self._autocast():
            self.backward_D()
        self.scaler.step(self.optimizer_D)
        self.scaler.update()
