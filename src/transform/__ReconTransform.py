#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/10/2022


import numpy as np
from typing import Tuple, Union, Callable, List
from monai.transforms import *
from .transform_zoo import IntensityNormd


class ReadImageOnlyd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)

    def __call__(self, data):
        fname = data['npz']
        data = dict(np.load(data['npz'], allow_pickle=True))
        del data['data_specs'], data["data_mask"]
        data['subject'] = fname
        return data


class ReconTransform_Patch(object):
    def __init__(self, crop_size, num_samples, modality):
        self.__version__ = "0.1.1"
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.modality = modality

        self.train = Compose([
            ReadImageOnlyd(keys=['npz']), 
            AddChanneld(keys=["data_image"]),
            CastToTyped(keys=["data_image"], dtype=np.int16),

            SpatialPadd(
                keys=["data_image"],
                spatial_size=self.crop_size,
                ),

            IntensityNormd(
                keys=["data_image"],
                modality=self.modality),

            RandSpatialCropd(
                keys=["data_image"],
                roi_size=self.crop_size,
                random_center=True,
                random_size=False),
           
            CastToTyped(
                keys=["data_image"], 
                dtype=np.float32),

            ToTensord(keys=["data_image"]),])

        self.infer = Compose([
            ReadImageOnlyd(keys=['npz']), 
            AddChanneld(
                keys=["data_image"],
                allow_missing_keys=True), 

            IntensityNormd(
                keys=["data_image"],
                modality=self.modality),

            CastToTyped(
                keys=["data_image"], 
                dtype=np.float32),

            ToTensord(
                keys=["data_image"]),])
