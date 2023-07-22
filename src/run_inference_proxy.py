#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/08/2023

# Program description
# Run the pipeline to measure the uncertainty with MC Dropout


from monai.utils import set_determinism
from options.options import Options
from utils.util import parse_options
from data_loading.AL_Loader import AL_Loader
from transform.ProxyTransform import ProxyTransform
from models.NetworkLoader import NetworkLoader
from inference.ProxyPredictor import ProxyPredictor


def main() -> None:
    opt = parse_options(Options(), save_config=False)

    # reproducibility
    set_determinism(seed=opt.seed)  

    transform = ProxyTransform(
        crop_size=opt.crop_size, 
        num_samples=opt.num_samples)

    data = ProxyLoader(
        tr=transform, 
        opt=opt, 
        phase='test')

    model = NetworkLoader(opt).load()

    ProxyPredictor(data=data, model=model, opt=opt).select_data(mc_number=opt.mc_number)


if __name__ == "__main__":
    main()
