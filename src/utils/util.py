#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 06/02/2022


import os
import os.path as osp
import pdb
from typing import List, Tuple
import numpy as np
import pytz
import datetime
import SimpleITK as sitk
from cc3d import connected_components


def debug():
    pdb.set_trace()


def mkdir(save_dir: str) -> None:
    """Create a directory if it did not exist
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


def mkdirs(save_dirs: List[str]) -> None:
    """Create a directory if it did not exist
    """
    for save_dir in save_dirs:
        mkdir(save_dir)


def update_log(log_str, fname, verbose=True):
    with open(fname, 'a') as f:
        f.write(log_str + '\n')
    if verbose:
        print(log_str)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_time():
    return datetime.datetime.now(pytz.timezone('US/Eastern'))


def poly_lr(count, max_counts, initial_lr, exponent=0.9):
    return initial_lr * (1 - count / max_counts)**exponent


def parse_options(options, save_config=True):
    opt = options.get_options()
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = options.parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    opt.expr_dir = osp.join(opt.checkpoints_dir, opt.organ[0], opt.name)
    mkdir(opt.checkpoints_dir)
    mkdir(osp.join(opt.checkpoints_dir, opt.organ[0]))
    mkdir(opt.expr_dir)

    if save_config:
        file_name = osp.join(opt.expr_dir, 'Config.log')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    return opt


def create_sitkImage(
    img: np.array,
    spacing: Tuple[float]=(1, 1, 1),
    origin: Tuple[float]=(0, 0, 0),
    direction: Tuple[float]=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    verbose: bool=False,
):
    sitk_image = sitk.GetImageFromArray(img)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(direction)
    if verbose:
        print(sitk_image)
    return sitk_image


def get_cc3d_bin(msk: np.array, top_k: int=1, binary=True, verbose: bool=False) -> np.array:
    msk = connected_components(msk.astype('uint8'))
    indices, counts = np.unique(msk, return_counts=True)
    if verbose:
        print('** Connected components info **')
        print(f'indices: {indices}')
        print(f'counts:  {counts}')
    indices, counts = indices[1:], counts[1:]
    labels = indices[np.argpartition(counts, -top_k)[-top_k:]]
    msk[~np.isin(msk, labels)] = 0
    if binary:
        msk[msk != 0] = 1
    return msk


def get_cc3d(msk):
    fg_labels = np.unique(msk)[1:]
    output = np.zeros(msk.shape)
    for fg in fg_labels:
        tmp = msk.copy()
        tmp[tmp != fg] = 0
        tmp = get_cc3d_bin(msk=tmp, binary=True)
        output += tmp * fg
    return output