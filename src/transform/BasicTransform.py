#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/07/2022


import numpy as np
from monai.transforms import *
from .transform_zoo import ReadNumpyd, IntensityNormd, GaussianNoised, GaussianBlurd, BrightnessMultiplicatived, \
 ContrastAugmentationd, SimulateLowResolutiond, Gammad


class BasicTransform(object):
    def __init__(self, crop_size, num_samples, modality):
        self.__version__ = "0.1.1"
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.modality = modality

        self.train = Compose([
            ReadNumpyd(keys=['npz']), 
            AddChanneld(keys=["data_image", "data_mask"]),
            CastToTyped(keys=["data_image"], dtype=np.int16),

            # padding
            SpatialPadd(
                keys=["data_image", "data_mask"],
                spatial_size=self.crop_size,
                mode='constant'),

            # modality-specific intensity normalization
            IntensityNormd(
                keys=["data_image"],
                modality=self.modality),
            
            # rotating
            RandRotated(
                keys=['data_image', 'data_mask'], 
                range_x=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                range_y=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                range_z=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                prob=0.2, 
                mode=('bilinear', 'nearest'),
                keep_size=True),

            # scaling
            RandZoomd(
                keys=['data_image', 'data_mask'], 
                prob=0.2, 
                min_zoom=0.7, 
                max_zoom=1.4, 
                mode=('trilinear', 'nearest'), 
                keep_size=True),
            
            # cropping into patches
            RandCropByPosNegLabeld(
                keys=["data_image", "data_mask"],
                label_key="data_mask",
                spatial_size=self.crop_size,
                pos=2,
                neg=1,
                num_samples=self.num_samples),

            # Gaussian noise
            GaussianNoised(keys=["data_image"], prob=0.1),

            # Gaussian blur
            GaussianBlurd(
                keys=["data_image"], 
                prob=0.2, 
                blur_sigma=(0.5, 1), 
                different_sigma_per_channel=True, 
                p_per_channel=0.5),

            # brightness multiplicative
            BrightnessMultiplicatived(
                keys=["data_image"],
                prob=0.15,
                multiplier_range=(0.75, 1.25)),

            # contrast augmentation
            ContrastAugmentationd(keys=["data_image"], prob=0.15),

            # low resolution simulation
            SimulateLowResolutiond(
                keys=["data_image"], 
                prob=0.25,
                zoom_range=(0.5, 1),
                per_channel=True,
                p_per_channel=0.5, 
                order_downsample=0,
                order_upsample=3,
                ignore_axes=None),

            # inverted gamma
            Gammad(
                keys=["data_image"],
                prob=0.1, 
                gamma_range=(0.7, 1.5),
                invert_image=True, 
                per_channel=True, 
                retain_stats=True),

            # gamma 
            Gammad(
                keys=["data_image"],
                prob=0.3, 
                gamma_range=(0.7, 1.5),
                invert_image=False, 
                per_channel=True, 
                retain_stats=True),

            # mirroring: may introduce too much variability
            # RandFlipd(
            #     keys=["data_image", "data_mask"],
            #     spatial_axis=[0],
            #     prob=0.50),

            # RandFlipd(
            #     keys=["data_image", "data_mask"],
            #     spatial_axis=[1],
            #     prob=0.50),
            
            # RandFlipd(
            #     keys=["data_image", "data_mask"],
            #     spatial_axis=[2],
            #     prob=0.50),

            CastToTyped(
                keys=["data_image", "data_mask"], 
                dtype=[np.float32, np.uint8]),

            ToTensord(keys=["data_image", "data_mask"]),])

        self.infer = Compose([
            ReadNumpyd(keys=['npz']), 
            AddChanneld(
                keys=["data_image", "data_mask"],
                allow_missing_keys=True), 

            IntensityNormd(
                keys=["data_image"],
                modality=self.modality),

            CastToTyped(
                keys=["data_image", "data_mask"], 
                dtype=[np.float32, np.uint8],
                allow_missing_keys=True),

            ToTensord(
                keys=["data_image", "data_mask"], 
                allow_missing_keys=True),])
