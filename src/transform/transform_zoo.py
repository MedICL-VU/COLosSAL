#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 06/28/2022


import numpy as np
from typing import Tuple, Union, Callable, List
from monai.transforms import *
from batchgenerators.augmentations.color_augmentations import augment_contrast, augment_brightness_multiplicative, augment_gamma
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise, augment_gaussian_blur
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy


class ReadNumpyd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)

    def __call__(self, data):
        fname = data['npz']
        data = dict(np.load(data['npz'], allow_pickle=True))
        del data['data_specs']
        if 'real_label' in data:
            del data['real_label']
        data['subject'] = fname
        return data


def IntensityNormd(keys, modality):
    if modality == 'ct':
        return ScaleIntensityRanged(
            keys=keys,
            a_min=-1024,
            a_max=1024,
            b_min=0,
            b_max=1,
            clip=True)

    elif modality == 'mr':
        return Compose([
            NormalizeIntensityd(keys=keys), 
            ScaleIntensityRangePercentilesd(
                keys=keys, 
                lower=1, 
                upper=99, 
                b_min=0, 
                b_max=1, 
                clip=True)])


class GaussianNoised(MapTransform):
    def __init__(
        self, 
        keys: Tuple[str], 
        prob: float = 1, 
        noise_variance: Union[Tuple[float, float], Callable[[], float]] = (0, 0.1),
        p_per_channel: float = 1,
        per_channel: bool = False) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.prob = prob
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, data):
        if np.random.uniform() < self.prob:
            for k in self.keys:
                data[k] =  augment_gaussian_noise(
                    data[k], 
                    self.noise_variance,
                    self.p_per_channel,
                    self.per_channel)
        return data


class GaussianBlurd(MapTransform):
    def __init__(
        self, 
        keys: Tuple[str], 
        prob: float = 1, 
        blur_sigma: Tuple[float, float] = (1, 5),
        different_sigma_per_channel: bool = True,
        different_sigma_per_axis: bool = False,
        p_isotropic: float = 0,
        p_per_channel: float = 1) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.prob = prob
        self.blur_sigma = blur_sigma
        self.different_sigma_per_channel = different_sigma_per_channel
        self.different_sigma_per_axis = different_sigma_per_axis
        self.p_isotropic = p_isotropic
        self.p_per_channel = p_per_channel

    def __call__(self, data):
        if np.random.uniform() < self.prob:
            for k in self.keys:
                data[k] = augment_gaussian_blur(
                    data[k], 
                    self.blur_sigma,
                    self.different_sigma_per_channel,
                    self.p_per_channel,
                    different_sigma_per_axis=self.different_sigma_per_axis,
                    p_isotropic=self.p_isotropic)
        return data


class BrightnessMultiplicatived(MapTransform):
    def __init__(
        self, 
        keys: Tuple[str], 
        prob: float = 1, 
        multiplier_range: Tuple[float, float] = (0.5, 2),
        per_channel: bool = True,
        ) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.prob = prob
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel

    def __call__(self, data):
        if np.random.uniform() < self.prob:
            for k in self.keys:
                data[k] = augment_brightness_multiplicative(
                    data[k], 
                    self.multiplier_range,
                    self.per_channel)
        return data


class ContrastAugmentationd(MapTransform):
    def __init__(
        self, 
        keys: Tuple[str], 
        prob: float = 1, 
        contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
        preserve_range: bool = True,
        per_channel: bool = True,
        p_per_channel: float = 1) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.prob = prob
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, data):
        if np.random.uniform() < self.prob:
            for k in self.keys:
                data[k] = augment_contrast(
                    data[k], 
                    contrast_range=self.contrast_range,
                    preserve_range=self.preserve_range,
                    per_channel=self.per_channel,
                    p_per_channel=self.p_per_channel)
        return data


class SimulateLowResolutiond(MapTransform):
    def __init__(
        self, 
        keys: Tuple[str], 
        prob: float = 1, 
        zoom_range: Union[Tuple[float, float], Callable[[], float]] = (0.5, 1), 
        per_channel: bool = False, 
        p_per_channel=1,
        channels=None,
        order_downsample=1,
        order_upsample=0,
        ignore_axes=None) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.prob = prob
        self.zoom_range = zoom_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.channels = channels
        self.order_downsample = order_downsample
        self.order_upsample = order_upsample
        self.ignore_axes = ignore_axes

    def __call__(self, data):
        if np.random.uniform() < self.prob:
            for k in self.keys:
                data[k] = augment_linear_downsampling_scipy(
                    data[k], 
                    zoom_range=self.zoom_range,
                    per_channel=self.per_channel,
                    p_per_channel=self.p_per_channel,
                    channels=self.channels,
                    order_downsample=self.order_downsample,
                    order_upsample=self.order_upsample,
                    ignore_axes=self.ignore_axes)
        return data


class Gammad(MapTransform):
    def __init__(
        self, 
        keys: Tuple[str], 
        prob: float = 1, 
        gamma_range: Union[Tuple[float, float], Callable[[], float]] = (0.5, 2), 
        invert_image: bool = False, 
        per_channel: bool = False, 
        retain_stats: Union[bool, Callable[[], bool]] = False) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.prob = prob
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.per_channel = per_channel
        self.retain_stats = retain_stats

    def __call__(self, data):
        if np.random.uniform() < self.prob:
            for k in self.keys:
                data[k] = augment_gamma(
                    data[k], 
                    self.gamma_range,
                    self.invert_image,
                    per_channel=self.per_channel,
                    retain_stats=self.retain_stats)
        return data




# if __name__ == "__main__":

#     file = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/npz/Stomach/RS.NA045_rrCT1_UNK20190827.npz'

#     x = ReadNumpyd(keys=['npz'])({'npz': file})
    
#     # test_transform = GaussianNoised(keys=["data_image"], prob=1)
#     # test_transform = GaussianBlurd(keys=["data_image"], prob=1, blur_sigma=(0.5, 1), different_sigma_per_channel=True, p_per_channel=0.5)
#     # test_transform = SimulateLowResolutiond(keys=["data_image"], prob=1, zoom_range=(0.5, 1), per_channel=True,
#     #                                                     p_per_channel=0.5,
#     #                                                     order_downsample=0, order_upsample=3, ignore_axes=None)

#     test_transform = Gammad(
#         ["data_image"], 1, (1, 5), True, True, True)

#     out = test_transform(x)

#     print(out)
