#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/08/2023


import os.path as osp
import torch
import numpy as np
from numpy import inf
from time import time
from tqdm import tqdm
import random
from monai.metrics import compute_meandice,compute_hausdorff_distance,compute_percent_hausdorff_distance,compute_average_surface_distance
from monai.transforms import AsDiscrete, CropForeground
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from utils.util import update_log, get_time, mkdir
import SimpleITK as sitk
from typing import Tuple
import pdb



def entropy_3d_volume(vol_input):
    vol_input = vol_input.astype(dtype='float32')
    dims = vol_input.shape
    reps = dims[0]
    entropy = np.zeros(dims[2:], dtype='float32')
    threshold = 0.00005
    vol_input[vol_input <= 0] = threshold

    if len(dims) == 5:
        for channel in range(dims[1]):
            t_vol = np.squeeze(vol_input[:, channel, :, :, :])
            t_sum = np.sum(t_vol, axis=0)
            t_avg = np.divide(t_sum, reps)
            t_log = np.log(t_avg)
            t_entropy = -np.multiply(t_avg, t_log)
            entropy = entropy + t_entropy
    else:
        t_vol = np.squeeze(vol_input)
        t_sum = np.sum(t_vol, axis=0)
        t_avg = np.divide(t_sum, reps)
        t_log = np.log(t_avg)
        t_entropy = -np.multiply(t_avg, t_log)
        entropy = entropy + t_entropy
    return entropy


def variance_3d_volume(vol_input):
    vol_input = vol_input.astype(dtype='float32')
    dims = vol_input.shape
    threshold = 0.0005
    vol_input[vol_input<=0] = threshold
    vari = np.nanvar(vol_input, axis=0)
    variance = np.sum(vari, axis=0)
    variance = np.expand_dims(variance, axis=0)
    variance = np.expand_dims(variance, axis=0)
    return variance


class UncertaintyEstimater(object):
    def __init__(self, data, model, opt):
        self.opt = opt
        self.paths = data.paths
        self.infer_log = osp.join(self.opt.expr_dir, 'proxy.log')
        self.test_ds = data.get_data()
        self.model = model.cuda() if torch.cuda.is_available() else model

        if self.opt.multi_gpu and torch.cuda.device_count() > 1:
            print(f"multiple GPUs are used: {torch.cuda.device_count()}")
            self.model = torch.nn.DataParallel(self.model).cuda().module
            
        self.result_dir = osp.join(self.opt.expr_dir, f'results_epoch_{self.opt.epoch}')
        self.include_background = False

        if self.opt.save_output:
            mkdir(self.result_dir)

        if self.opt.load_ckpt:
            ckpt_path = osp.join(self.opt.expr_dir, f'{self.opt.epoch}')
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt['state_dict'])
            update_log(f"model and optimizer are initialized from {ckpt_path}", self.infer_log)


    def generate_uncertainty_file(self, score_list, name_list, metric='entropy', mode='global'):
        unc_path = osp.join(self.opt.dataroot, f'{self.opt.organ[0]}', 'unc', f'{mode}_{metric}.npz')
        np.savez(unc_path, name_list=name_list, score_list=score_list)


    def estimate_from_proxy(self, mc_number=20, mode='global'):
        self.model.train() 
        self.mc_number = mc_number
        ent_list, var_list, name_list = [], [], []

        with torch.no_grad():
            for i, test_data in enumerate(self.test_ds):
                img, target, sub = test_data["data_image"].cuda(), test_data["data_mask"].cuda(), test_data["subject"]
                pid = osp.splitext(osp.basename(sub[0]))[0]
                start = time()
                accum_outputs = []

                for mc in range(self.mc_number):
                    pred = sliding_window_inference(
                        inputs=img, 
                        roi_size=self.opt.crop_size, 
                        sw_batch_size=self.opt.sw_batch_size, 
                        predictor=self.model,
                        overlap=self.opt.overlap,
                        mode=self.opt.blend_mode,
                        sigma_scale=self.opt.blend_sigma,
                        padding_mode=self.opt.padding_mode,
                        cval=self.opt.padding_val)

                    if mode == 'local':
                        cropper = CropForeground(margin=5, return_coords=True)
                        _, coords_1, coords_2 = cropper(target[0, ...])
                        pred = pred[:, :, coords_1[0]:coords_2[0], coords_1[1]:coords_2[1], coords_1[2]:coords_2[2]]

                    soft_pred = torch.softmax(pred, dim=1)
                    accum_outputs.append(soft_pred)

                accum_tensor = torch.stack(accum_outputs)
                accum_tensor = torch.squeeze(accum_tensor)
                accum_numpy = accum_tensor.to('cpu').numpy()
                accum_numpy = accum_numpy[:, 1:, :, :, :]

                # compute two metrics for uncertainty measure: entropy, variance
                ent_unc = np.nanmean(entropy_3d_volume(accum_numpy))  
                var_unc = np.nanmean(variance_3d_volume(accum_numpy))
                update_log(f"subject: {pid}, entropy: {ent_unc:.5f}, variance: {var_unc:.5f}", self.infer_log)

                ent_list.append(ent_unc)
                var_list.append(var_unc)
                name_list.append(sub[0])
        
        self.generate_uncertainty_file(ent_list, name_list, "entropy", mode)        
        self.generate_uncertainty_file(var_list, name_list, "variance", mode)        
    

    def estimate_from_recon(self, mc_number=20, mode='global'):
        self.model.train() 
        self.mc_number = mc_number

        # for regression task, we estimate uncertainty with variance
        var_list, name_list = [], []

        with torch.no_grad():
            for i, test_data in enumerate(self.test_ds):
                img, target, sub = test_data["data_image"].cuda(), test_data["data_mask"].cuda(), test_data["subject"]
                pid = osp.splitext(osp.basename(sub[0]))[0]
                start = time()
                accum_outputs = []

                for mc in range(self.mc_number):
                    pred = sliding_window_inference(
                        inputs=img, 
                        roi_size=self.opt.crop_size, 
                        sw_batch_size=self.opt.sw_batch_size, 
                        predictor=self.model,
                        overlap=self.opt.overlap,
                        mode=self.opt.blend_mode,
                        sigma_scale=self.opt.blend_sigma,
                        padding_mode=self.opt.padding_mode,
                        cval=self.opt.padding_val)

                    if mode == 'local':
                        cropper = CropForeground(margin=5, return_coords=True)
                        _, coords_1, coords_2 = cropper(target[0, ...])
                        pred = pred[:, :, coords_1[0]:coords_2[0], coords_1[1]:coords_2[1], coords_1[2]:coords_2[2]]

                    accum_outputs.append(pred)

                accum_tensor = torch.stack(accum_outputs)
                accum_tensor = torch.squeeze(accum_tensor)
                accum_numpy = accum_tensor.to('cpu').numpy()

                # compute variance to estimate the uncertainty for regression
                var_unc = np.nanmean(variance_3d_volume(accum_numpy))
                update_log(f"subject: {pid}, variance: {var_unc:.5f}", self.infer_log)

                var_list.append(var_unc)
                name_list.append(sub[0])
        
        self.generate_uncertainty_file(var_list, name_list, "ReconVar", mode)        



