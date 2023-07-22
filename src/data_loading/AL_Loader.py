#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/08/2023


from glob import glob
import numpy as np
import os.path as osp
import random
from utils.util import update_log, get_time
from monai.data import Dataset, DataLoader


"""
ONLY used for inference. To analyze which training data are the 
ones to be annotated, we purposely take the training data as test data.
"""


class AL_Loader(object):
    def __init__(self, tr, opt, data_folder='data'):
        self.tr = tr
        self.data_folder = data_folder
        self.expr_dir = opt.expr_dir
        self.run_log = osp.join(self.expr_dir, 'run.log')
        self.num_workers = opt.num_workers
        self.test_ds = None

        # NOTE THAT: we take the training set here
        f = open(osp.join(opt.dataroot, f'{opt.organ[0]}', 'train.txt'), 'r')
        lines = [item.replace('\n', '').split(",")[0] for item in f.readlines()]
        self.paths = [osp.join(opt.dataroot, f'{opt.organ[0]}', f'{data_folder}', f'{fname}.npz') for fname in lines]
        test_ds = [{"npz": path} for path in self.paths]
        print(f'Number of unlabeled training images: {len(test_ds)}')
        test_ds = Dataset(data=test_ds, transform=self.tr.infer)
        
        self.test_ds = DataLoader(
            test_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.num_workers)

    def get_data(self):
        return self.test_ds
