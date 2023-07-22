#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/08/2023


# Program description
# proxy task: generate a noisy binary segmentation mask (pseudo label) for supervision.


from monai.utils import set_determinism
from options.options import Options
from utils.util import parse_options
from data_loading.BasicLoader import BasicLoader
from transform.BasicTransform import BasicTransform
from models.NetworkLoader import NetworkLoader
from network_training.TrainerV2 import TrainerV2


def main() -> None:
    opt = parse_options(Options())

    # reproducibility
    set_determinism(seed=opt.seed)  

    transform = BasicTransform(
        crop_size=opt.crop_size, 
        num_samples=opt.num_samples,
        modality=opt.modality)

    data = BasicLoader(
        tr=transform, 
        opt=opt, 
        phase='train',
        folder_name='proxy_data')

    model = NetworkLoader(opt).load()

    TrainerV2(data=data, model=model, opt=opt).run()


if __name__ == "__main__":
    main()
