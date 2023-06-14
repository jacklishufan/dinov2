# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import torch
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss,MSLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead,MLPR2O
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

from dinov2.models.vision_transformer import BlockChunk

try:
    from xformers.ops import fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
assert XFORMERS_AVAILABLE, "xFormers is required for DINOv2 training"
import wandb
from dinov2.logging.wandb import wandb_dump_img
logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.do_multiscale = cfg.multiscale.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")
        if self.do_multiscale:
            self.multiscale_loss_weight = cfg.multiscale.loss_weight
            self.multiscale_loss = MSLoss(self.ibot_out_dim)
            self.multiscale_separate_head = cfg.multiscale.separate_head
            self.ms_debug = cfg.multiscale.debug_wandb
            if cfg.multiscale.predictor.enabled:
                predictor = MLPR2O(cfg.multiscale.head_n_prototypes,cfg.multiscale.predictor.hidden_size,cfg.multiscale.head_n_prototypes,pre_norm=False)
            else:
                predictor = nn.Identity()
            student_model_dict["predictor"] = predictor
            if self.multiscale_separate_head:
                logger.info(f"OPTIONS -- MS -- loss_weight: {cfg.multiscale.loss_weight}")
                logger.info(f"OPTIONS -- MS -- head_n_prototypes: {cfg.multiscale.head_n_prototypes}")
                logger.info(f"OPTIONS -- MS -- head_bottleneck_dim: {cfg.multiscale.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- MS -- head_hidden_dim: {cfg.multiscale.head_hidden_dim}")
                # ms_head = partial(
                #     DINOHead,
                #     in_dim=embed_dim,
                #     out_dim=cfg.multiscale.head_n_prototypes,
                #     hidden_dim=cfg.multiscale.head_hidden_dim,
                #     bottleneck_dim=cfg.multiscale.head_bottleneck_dim,
                #     nlayers=cfg.multiscale.head_nlayers,
                # )
                ms_head = partial(
                    MLPR2O,
                    input_dim=embed_dim,
                    output_dim=cfg.multiscale.head_n_prototypes,
                    hidden_dim=cfg.multiscale.head_hidden_dim,
                    # bottleneck_dim=cfg.multiscale.head_bottleneck_dim,
                    # nlayers=cfg.multiscale.head_nlayers,
                )
                student_model_dict["ms_head"] = ms_head()
                teacher_model_dict["ms_head"] = ms_head()
            else:
                logger.info("OPTIONS -- MS -- head shared with DINO")
        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def mask_pool(self,src,msk):
        '''
        src: N X HW X D
        msk: N X HW X M
        '''
        return torch.einsum('nld,nlm->nmd',src,msk)

    def forward_backward(self, images, teacher_temp,log_img=False):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        if self.do_multiscale:
            global_crops_msk = images["collated_global_crops_msk"].cuda(non_blocking=True)
            global_crops_msk_one_hot = [torch.nn.functional.one_hot(global_crops_msk[:,i],d**2).flatten(1,2) for i,d in enumerate(self.cfg.crops.spatial_dims)]
            global_crops_msk_one_hot = torch.cat(global_crops_msk_one_hot,dim=-1).to(global_crops.dtype)
            gloabal_crops_msk_area = global_crops_msk_one_hot.sum(1,keepdim=True)
            global_crops_msk_one_hot /= (global_crops_msk_one_hot.sum(1,keepdim=True)+1e-3)
            # mask select
            ms_msk_areas_indicator = (gloabal_crops_msk_area > 0).chunk(2)
            ms_msk_areas_indicator = ms_msk_areas_indicator[0] * ms_msk_areas_indicator[1]
            # ms_msk_areas_indicator *= torch.rand_like(ms_msk_areas_indicator.half()) < self.cfg.multiscale.sample_ratio
            ms_msk_areas_indicator = ms_msk_areas_indicator.repeat(2,1,1)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops
        ms_loss_scale = 1.0 / n_global_crops
        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]

            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]
            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None


            multiscale_teacher_softmaxed_centered = None
            if self.do_multiscale:
                ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
                multi_scale_teacher = self.mask_pool(ibot_teacher_patch_tokens,global_crops_msk_one_hot.to(ibot_teacher_patch_tokens.dtype))
                multi_scale_teacher = multi_scale_teacher.chunk(n_global_crops_teacher)
                multi_scale_teacher = torch.cat([multi_scale_teacher[1],multi_scale_teacher[0]])
                if self.ms_debug and log_img:
                    bs =  multi_scale_teacher.shape[0] // 2
                    dd = x.shape[-1]
                    view_x = x.permute(0,2,3,1)
                    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
                    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
                    ms_view1 = view_x[0].cpu().numpy() * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
                    ms_view2 = view_x[bs].cpu().numpy()* IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
                    wandb_dump_img([ms_view1.clip(0,1),ms_view2.clip(0,1)],"img")
                # LOGGGING
                if self.ms_debug and log_img:
                    bs =  multi_scale_teacher.shape[0] // 2
                    dd = self.cfg.crops.spatial_dims[0]
                    ms_view1 = multi_scale_teacher[0][:dd**2].view(dd,dd,-1).cpu().numpy()[...,0]
                    ms_view2 = multi_scale_teacher[bs][:dd**2].view(dd,dd,-1).cpu().numpy()[...,0]
                    wandb_dump_img([ms_view1,ms_view2],"ms_teacher")
                _dim = multi_scale_teacher.shape[-1]
                if self.multiscale_separate_head:
                    multi_scale_teacher = self.teacher.ms_head(multi_scale_teacher)
                else:
                    multi_scale_teacher = multi_scale_teacher.reshape(-1,_dim)#[ms_msk_areas_indicator.view(-1)]
                    multi_scale_teacher = self.teacher.dino_head(multi_scale_teacher)
            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])
                if self.do_multiscale:
                    multiscale_teacher_softmaxed_centered = self.multiscale_loss.softmax_center_teacher(
                        multi_scale_teacher.unsqueeze(0),teacher_temp=teacher_temp
                    )
                    self.multiscale_loss.update_center(multi_scale_teacher)
            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )
                if self.do_multiscale:
                    multiscale_teacher_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        multi_scale_teacher,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=multi_scale_teacher.shape[0],
                    )
            else:
                raise NotImplementedError
            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered,multiscale_teacher_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered,multiscale_teacher_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        if self.do_multiscale:
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            multi_scale_student = self.mask_pool(ibot_student_patch_tokens,global_crops_msk_one_hot)
            _dim = multi_scale_student.shape[-1]
            if self.ms_debug and log_img:
                    bs =  multi_scale_student.shape[0] // 2
                    dd = self.cfg.crops.spatial_dims[0]
                    ms_view1 = multi_scale_student[0][:dd**2].view(dd,dd,-1).detach().cpu().numpy()[...,0]
                    ms_view2 = multi_scale_student[bs][:dd**2].view(dd,dd,-1).detach().cpu().numpy()[...,0]
                    wandb_dump_img([ms_view1,ms_view2],"ms_student")
                    indicator_view1 = ms_msk_areas_indicator[0,0][:dd**2].view(dd,dd).detach().cpu().numpy()
                    indicator_view2 = ms_msk_areas_indicator[bs,0][:dd**2].view(dd,dd).detach().cpu().numpy()
                    wandb_dump_img([ms_view1,ms_view2],"ms_student")
                    wandb_dump_img([indicator_view1,indicator_view2],"indicator")
            
            if self.multiscale_separate_head:
                multi_scale_student_after_head = self.student.ms_head(multi_scale_student)
                multi_scale_student_after_head = self.student.predictor(multi_scale_student_after_head)
            else:
                multi_scale_student = multi_scale_student.reshape(-1,_dim)#[ms_msk_areas_indicator.view(-1)]
                inputs_for_student_head_list.append(multi_scale_student.unsqueeze(0))
            if self.ms_debug and log_img:
                bs =  multi_scale_student_after_head.shape[0] // 2
                dd = self.cfg.crops.spatial_dims[0]
                ms_view1 = multi_scale_student_after_head[0][:dd**2].view(dd,dd,-1).detach().cpu().numpy()[...,0]
                ms_view2 = multi_scale_student_after_head[bs][:dd**2].view(dd,dd,-1).detach().cpu().numpy()[...,0]
                wandb_dump_img([ms_view1,ms_view2],"student_after_head")
                ms_view1 = multiscale_teacher_softmaxed_centered[0][0][:dd**2].view(dd,dd,-1).detach().cpu().numpy()[...,0]
                ms_view2 = multiscale_teacher_softmaxed_centered[0][bs][:dd**2].view(dd,dd,-1).detach().cpu().numpy()[...,0]
                wandb_dump_img([ms_view1,ms_view2],"teacher_after_head")
        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]
        if self.do_multiscale and not self.multiscale_separate_head:
            multi_scale_student_after_head = outputs_list.pop(0).squeeze(0)
            multi_scale_student_after_head = self.student.predictor(multi_scale_student_after_head)
        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            if torch.isnan(student_global_masked_patch_tokens_after_head.sum()):
                print("student_global_masked_patch_tokens_after_head",
                      student_global_masked_patch_tokens_after_head)
            if torch.isnan(masked_teacher_ibot_softmaxed_centered.sum()):
                print("masked_teacher_ibot_softmaxed_centered",
                      masked_teacher_ibot_softmaxed_centered)
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss
        if self.do_multiscale:
            # compute loss
            ms_mask_weight = 0.5 / (ms_msk_areas_indicator.sum(-1)+1e-3)
            ms_mask_weight = ms_mask_weight.repeat(1,ms_msk_areas_indicator.shape[-1])
            ms_mask_weight = ms_mask_weight.view(-1)[ms_msk_areas_indicator.view(-1)]
            if torch.isnan(multi_scale_student_after_head.sum()):
                print("HERE")
                print(multi_scale_student_after_head)
                print(multi_scale_student_after_head.shape)
                print(multi_scale_student)
                print(multi_scale_student.shape)
                print(ms_msk_areas_indicator.sum(),multi_scale_student.shape)
            if wandb.run is not None and self.ms_debug:
                wandb.log(
                    dict(
                        student_max = multi_scale_student_after_head.max().item(),
                        student_min = multi_scale_student_after_head.min().item(),
                        patch_max = ibot_student_patch_tokens.max().item(),
                        patch_min = ibot_student_patch_tokens.min().item(),
                        teacher_max = multiscale_teacher_softmaxed_centered.max().item(),
                        teacher_min = multiscale_teacher_softmaxed_centered.min().item(),
                    )
                )
            ms_patch_loss = (
                self.multiscale_loss.forward_masked(
                    multi_scale_student_after_head,
                    multiscale_teacher_softmaxed_centered,
                    student_masks_flat=torch.ones_like(masks),
                    n_masked_patches=multi_scale_student_after_head.shape[0],
                    masks_weight=ms_msk_areas_indicator,
                )
                * loss_scales
                * ms_loss_scale
            )

            # store for display
            loss_dict["ms_loss"] = ms_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.multiscale_loss_weight * ms_patch_loss
        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = (
                self.teacher.dino_head._streams
            ) = self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                if k not in 'predictor':              
                    for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                        student_param_list += ms.params
                        teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            if k not in ['predictor']:
                self.teacher[k].load_state_dict(self.student[k].state_dict()) 
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            if k not in ['predictor']:
                teacher_model_cfg = self.cfg.compute_precision.teacher[k]
                self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
