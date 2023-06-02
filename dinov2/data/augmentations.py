# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any

from torchvision import transforms

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

import torchvision.transforms.functional as F
import torch

logger = logging.getLogger("dinov2")

class CustomCompose:

    def __init__(self,aug_list) -> None:
        self.aug_list = aug_list

    def __call__(self,img) -> Any:
        transfoms_lst = []
        for v in self.aug_list:
            if isinstance(v,transforms.RandomResizedCrop):
                i,j,h,w = v.get_params(img,v.scale,v.ratio)
                _, im_h, im_w = F.get_dimensions(img)
                transfoms_lst.append(dict(type="crop",parms=(i,j,h,w,im_h,im_w)))
                img = F.resized_crop(img, i, j, h, w, v.size, v.interpolation, antialias=v.antialias)
            elif isinstance(v,transforms.RandomHorizontalFlip):
                if torch.rand(1) < v.p:
                    img = F.hflip(img)
                    transfoms_lst.append(dict(type="hflip",parms=True))
            else:
                img = v(img)
        return img,transfoms_lst
    
def reapply(img,augs,size,interpolation=transforms.InterpolationMode.BICUBIC,antialias='warn'):
    for aug in augs:
        if aug['type'] == 'crop':
            i, j, h, w,im_h,im_w = aug['parms']
            _, hh, ww = F.get_dimensions(img)
            rh,rw = hh / im_h, ww / im_w
            i,j,h,w = int(i * rh), int (j * rw), int(h*rh),int(w*rw)
            img = F.resized_crop(img, i, j, h, w, size, interpolation, antialias=antialias)
        elif aug['type'] == 'hflip':
            img = F.hflip(img)
    return img

def get_spatial_mask(dimension,out_size=56,device='cpu'):
    coords = torch.stack(torch.meshgrid(torch.arange(dimension, device=device), torch.arange(dimension, device=device),indexing='ij'), 0)
    coords = coords[0] * dimension + coords[1]
    if out_size != dimension:
        coords = F.resize(coords.unsqueeze(0),(out_size,out_size),interpolation=transforms.InterpolationMode.NEAREST, antialias=None)[0]
    return coords

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        spatial_dims = [14,7,3],
        patch_size = 14,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.patch_size = patch_size
        self.global_msk_dim = self.global_crops_size // self.patch_size
        self.local_msk_dim = self.local_crops_size // self.patch_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = CustomCompose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = CustomCompose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])
        self.spatial_dims = spatial_dims

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base,tf_1 = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base,tf_2 = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        tf_locals = []
        local_crops = []
        for _ in range(self.local_crops_number):
            im_local_base,tf_local = self.geometric_augmentation_local(image)
            local_crop = self.local_transfo(im_local_base)
            local_crops.append(local_crop)
            tf_locals.append(tf_local)
        output["local_crops"] = local_crops
        output["offsets"] = ()
        output.update(
           dict( 
            global_transform_1=tf_1,
            global_transform_2=tf_2,
            local_transforms=tf_locals
            )
        )
        if len(self.spatial_dims) > 0:
            spatial_msk = torch.stack([get_spatial_mask(s) for s in self.spatial_dims]) # N X H X W
            global_crop_1_msk = reapply(spatial_msk,tf_1,(self.global_msk_dim,self.global_msk_dim),interpolation=transforms.InterpolationMode.NEAREST)
            global_crop_2_msk = reapply(spatial_msk,tf_2,(self.global_msk_dim,self.global_msk_dim),interpolation=transforms.InterpolationMode.NEAREST)
            local_crop_msks = [
                 reapply(spatial_msk,tf_local,(self.local_msk_dim,self.local_msk_dim),
                         interpolation=transforms.InterpolationMode.NEAREST) for tf_local in tf_locals
            ]
            output.update(
                dict( 
                    global_msks=[global_crop_1_msk,global_crop_2_msk],
                    local_crop_msks=local_crop_msks,
                    spatial_dims=self.spatial_dims,
                    #raw_msk=spatial_msk
                )
            )
        return output
