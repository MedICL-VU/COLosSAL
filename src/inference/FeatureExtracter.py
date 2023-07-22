#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/11/2023


import os.path as osp
import torch
import numpy as np
from numpy import inf
from time import time
from tqdm import tqdm
from monai.transforms import *
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from utils.util import update_log, get_time, mkdir
import torch.nn.functional as F
from typing import Tuple
import pdb



class FeatureExtracter(object):
    def __init__(self, data, model, opt):
        self.opt = opt
        self.infer_log = osp.join(self.opt.expr_dir, 'recon.log')
        self.test_ds = data.get_data()
        self.model = model.cuda() if torch.cuda.is_available() else model

        if self.opt.multi_gpu and torch.cuda.device_count() > 1:
            print(f"multiple GPUs are used: {torch.cuda.device_count()}")
            self.model = torch.nn.DataParallel(self.model).cuda().module
            
        self.result_dir = osp.join(self.opt.expr_dir, f'results_epoch_{self.opt.epoch}')
        self.activation = {}

        if self.opt.save_output:
            mkdir(self.result_dir)

        if self.opt.load_ckpt:
            ckpt_path = osp.join(self.opt.expr_dir, f'{self.opt.epoch}')
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt['state_dict'])
            update_log(f"model and optimizer are initialized from {ckpt_path}", self.infer_log)

        data_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', f'{opt.mode}_data.tsv')
        metadata_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', f'{opt.mode}_metadata.tsv')

        self.f_data = open(data_path, 'w')
        self.f_metadata = open(metadata_path, 'w')


    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook


    def extract(self, mode='global'):
        feats, name_list = [], []
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=len(self.test_ds)) as pbar:
                for i, test_data in enumerate(self.test_ds):
                    img, target, sub = test_data["data_image"].cuda(), test_data["data_mask"].cuda(), test_data["subject"]

                    if mode == 'global':
                        img = Compose([SqueezeDim(dim=0), DivisiblePad(k=16), AddChannel()])(img)
                    
                    elif mode == 'local':
                        img = Compose([SqueezeDim(dim=0), BorderPad(spatial_border=16), AddChannel()])(img)
                        target = Compose([SqueezeDim(dim=0), BorderPad(spatial_border=16), AddChannel()])(target)
                        cropper = CropForeground(margin=5, k_divisible=16, return_coords=True)
                        _, coords_1, coords_2 = cropper(target[0, ...])
                        img = img[:, :, coords_1[0]:coords_2[0], coords_1[1]:coords_2[1], coords_1[2]:coords_2[2]]

                    pid = osp.splitext(osp.basename(sub[0]))[0]
                    start = time()

                    # extract the bottleneck feature
                    self.model.model[1].submodule[1].submodule[1].submodule[1].submodule.register_forward_hook(self.get_activation('residual'))
                    pred = self.model(img)                
                    feat = self.activation['residual']  # shape: [1, 128, 8, 8, 8]
                    feat = F.adaptive_avg_pool3d(feat, 1).squeeze(4).squeeze(3).squeeze(2).squeeze(0)
                    feat = feat.detach().cpu().numpy() # reformat for plotting
                    self.f_data.write(f'{feat.tolist()}\n'.replace('[', '').replace(']', '').replace(', ', '    '))
                    self.f_metadata.write(f'{pid}\n')
                    feats.append(feat)
                    name_list.append(pid)
                    pbar.update(1)

        np.savez(
            osp.join(self.opt.dataroot, f'{self.opt.organ[0]}','feats', f'{mode}.npz'), 
            feats=feats, 
            name_list=name_list)


