#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 07/06/2022

# Program description
# General inference pipeline for medical image segmentation 


from monai.utils import set_determinism
from options.options import Options
from utils.util import parse_options
from data_loading.BasicLoader import BasicLoader
# from data_loading.ListLoader import ListLoader
from transform.BasicTransform import BasicTransform
from models.NetworkLoader import NetworkLoader
from inference.BasicPredictor import BasicPredictor


# python -W ignore run_inference.py -n ST -o JA_CM -c 13 --load_ckpt --save_output --epoch best_model.pth
# python -W ignore run_inference.py -n PLT+_ft -o UT_PE -c 8 --load_ckpt --save_output --epoch best_model.pth -track


def main() -> None:
    opt = parse_options(Options(), save_config=False)

    # reproducibility
    set_determinism(seed=opt.seed)  

    transform = BasicTransform(
        crop_size=opt.crop_size, 
        num_samples=opt.num_samples)

    data = BasicLoader(
        tr=transform, 
        opt=opt, 
        phase='test')

    model = NetworkLoader(opt).load()

    BasicPredictor(data=data, model=model, opt=opt).make_inference()


if __name__ == "__main__":
    main()
