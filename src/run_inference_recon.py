#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/11/2023

# Program description
# Run the pipeline to measure the representativeness via image reconstruction


from monai.utils import set_determinism
from options.options import Options
from utils.util import parse_options
from data_loading.ProxyLoader import ProxyLoader
from transform.ReconTransform import ReconTransform
from models.NetworkLoader import NetworkLoader
from inference.ReconPredictor import ReconPredictor


def main() -> None:
    opt = parse_options(Options(), save_config=False)

    # reproducibility
    set_determinism(seed=opt.seed)  

    transform = ReconTransform(
        crop_size=opt.crop_size, 
        num_samples=opt.num_samples)

    data = ProxyLoader(
        tr=transform, 
        opt=opt, 
        phase='test')

    model = NetworkLoader(opt).load()

    ReconPredictor(data=data, model=model, opt=opt).generate_feats()


if __name__ == "__main__":
    main()
