#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/07/2023


from glob import glob
import numpy as np
import os.path as osp
import random
from utils.util import update_log, get_time
from monai.data import Dataset, DataLoader


class BasicLoader(object):
    def __init__(self, tr, opt, phase:str='train', folder_name='data'):
        self.tr = tr
        self.phase = phase
        self.folder_name = folder_name
        self.expr_dir = opt.expr_dir
        self.run_log = osp.join(self.expr_dir, 'run.log')
        self.batch_size = opt.batch_size
        self.num_workers = opt.num_workers
        self.train_ds, self.val_ds, self.test_ds = None, None, None

        self.all_paths = {}
        for split in ['train', 'val']:  # only validation set for AL algorithms
            f = open(osp.join(opt.dataroot, f'{opt.organ[0]}', f'{split}.txt'), 'r')
            lines = [item.replace('\n', '').split(",")[0] for item in f.readlines()]
            paths = [osp.join(opt.dataroot, f'{opt.organ[0]}', f'{folder_name}', f'{fname}.npz') for fname in lines]
            self.all_paths[f'{split}'] = paths
        train_paths = self.all_paths['train']
        
        # ---------------- active learning ----------------

        if opt.num_labeled < len(train_paths):

            # Strategy: random selection
            if opt.RS_seed >= 0:
                random.Random(opt.RS_seed).shuffle(train_paths)
                self.all_paths['train'] = train_paths[:opt.num_labeled]
                update_log(f'Strategy: RANDOM SELCETION, seed={opt.RS_seed}, num_labeled={opt.num_labeled}\n', self.run_log)            

            # Strategy: customized selection
            else:
                plan_path = osp.join(opt.dataroot, f'{opt.organ[0]}', 'plans', f'{opt.plan}.npz')
                assert osp.exists(plan_path)
                self.all_paths['train'] = dict(np.load(plan_path, allow_pickle=True))['paths'][:opt.num_labeled]
                assert len(self.all_paths['train']) == opt.num_labeled
                update_log(f'Strategy: CUSTOMIZED SELECTION, plan={opt.plan}, num_labeled={opt.num_labeled}\n', self.run_log)            

        # ------------------------------------------------

        if self.phase == 'train':
            update_log(f'Training data:\n{[osp.splitext(osp.basename(path))[0] for path in self.all_paths["train"]]}\n', self.run_log)            
            train_paths, val_paths = self.all_paths['train'], self.all_paths['val']
            train_ds = [{"npz": path} for path in train_paths]
            val_ds = [{"npz": path} for path in val_paths]
            self.train_ratio = len(train_ds)/(len(train_ds) + len(val_ds))

            update_log(
                (f'\n{get_time():%Y-%m-%d %H:%M:%S}: The total number'
                    ' of training data (training + validation):'
                    f' {len(train_ds) + len(val_ds)}'), self.run_log)

            update_log(
                (f'{get_time():%Y-%m-%d %H:%M:%S}: '
                    f'{len(train_ds)} and {len(val_ds)} are used for'
                    ' training and validation'),  self.run_log)

            train_ds = Dataset(data=train_ds, transform=self.tr.train)
            val_ds = Dataset(data=val_ds, transform=self.tr.infer)

            self.train_ds = DataLoader(
                train_ds, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers)

            self.val_ds = DataLoader(
                val_ds, 
                batch_size=1, 
                shuffle=False, 
                num_workers=self.num_workers)
        else:
            test_paths = self.all_paths['val']
            test_ds = [{"npz": path} for path in test_paths]
            print(f'Number of testing images: {len(test_ds)}')
            test_ds = Dataset(data=test_ds, transform=self.tr.infer)
            
            self.test_ds = DataLoader(
                test_ds, 
                batch_size=1, 
                shuffle=False, 
                num_workers=self.num_workers)

    def get_data(self):
        if self.phase == 'train':
            return self.train_ds, self.val_ds
        else:
            return self.test_ds
