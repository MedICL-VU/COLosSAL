#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 07/06/2022


import os.path as osp
import torch
import numpy as np
from numpy import inf
from time import time
from tqdm import tqdm
from monai.metrics import compute_meandice,compute_hausdorff_distance,compute_percent_hausdorff_distance,compute_average_surface_distance
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from utils.util import update_log, get_time, mkdir
import SimpleITK as sitk
from typing import Tuple



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


class BasicPredictor(object):
    def __init__(self, data, model, opt):
        self.opt = opt
        self.infer_log = osp.join(self.opt.expr_dir, 'inference.log')
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
            # ckpt_path = osp.join(self.opt.expr_dir, f'{self.opt.epoch}_model.pth')
            # ckpt_path = osp.join(self.opt.expr_dir, f'best.model')
            ckpt_path = osp.join(self.opt.expr_dir, f'{self.opt.epoch}')
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt['state_dict'])
            update_log(f"model and optimizer are initialized from {ckpt_path}", self.infer_log)

    def post_pred(self, pred):
        pred = decollate_batch(pred)[0]
        pred = AsDiscrete(argmax=True, to_onehot=self.opt.num_classes)(pred)
        pred = pred.unsqueeze(0)
        return pred

    def post_label(self, target):
        return one_hot(target, self.opt.num_classes, dim=1)

    def get_dice(self, pred, target, include_background):
        dice = np.array(compute_meandice(pred, target)[0].cpu())
        if not include_background:
            dice = dice[1:]
        return dice

    def get_HD(self, pred, target, include_background):
        hd = np.array(compute_hausdorff_distance(pred, target, include_background, percentile=95)[0].cpu())
        return hd

    def get_ASSD(self, pred, target, include_background):
        assd = np.array(compute_average_surface_distance(pred, target, include_background)[0].cpu())
        return assd

    def make_inference(self):
        self.model.eval()
        self.test_dice, self.test_hd, self.test_assd = [], [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_ds)) as pbar:
                for i, test_data in enumerate(self.test_ds):
                    img, target, sub = test_data["data_image"].cuda(), test_data["data_mask"].cuda(), test_data["subject"]
                    sub = osp.splitext(osp.basename(sub[0]))[0]

                    start = time()

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

                    pred = self.post_pred(pred)

                    target = self.post_label(target)
                    dice = self.get_dice(pred, target, self.include_background)
                    hd = self.get_HD(pred, target, self.include_background)
                    assd = self.get_ASSD(pred, target, self.include_background)
                    assd[assd == inf] = np.nan

                    # for pseudo label generation
                    # dice, hd, assd = 0, 0, 0 

                    self.test_dice.append(dice)
                    self.test_hd.append(hd)
                    self.test_assd.append(assd)

                    if self.opt.display_per_iter:
                        update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                            f"epoch={self.opt.epoch}, id={i+1}/{len(self.test_ds.dataset)}, "
                            f"subject={sub}, time={time()-start:.4f}, "
                            f"dsc={list(map('{:.4f}'.format, dice))}, "
                            f"hd={list(map('{:.2f}'.format, hd))}, "
                            f"assd={list(map('{:.2f}'.format, assd))}"), self.infer_log)

                    if self.opt.save_output:
                        pred = torch.argmax(pred.squeeze(0), dim=0).detach().cpu().numpy().astype('uint8')
                        sitk_image = create_sitkImage(pred)
                        sitk.WriteImage(sitk_image, osp.join(self.result_dir, f'{sub}_pred.nii.gz'))

                        # img = img.squeeze(0).squeeze(0).detach().cpu().numpy()
                        # sitk_image = create_sitkImage(img)
                        # sitk.WriteImage(sitk_image, osp.join(self.result_dir, f'{sub}_image.nii.gz'))

                    pbar.update(1)

                np.savez(osp.join(self.opt.expr_dir, 'results.npz'),
                    dice=self.test_dice,
                    hd=self.test_hd,
                    assd=self.test_assd)

                self.mean_dice = np.nanmean(self.test_dice, axis=0)  # report mean dice without nans
                self.mean_hd = np.nanmean(self.test_hd, axis=0)  
                self.mean_assd = np.nanmean(self.test_assd, axis=0)  
                
                self.std_dice = np.nanstd(self.test_dice, axis=0)  # report std dice without nans
                self.std_hd = np.nanstd(self.test_hd, axis=0)  
                self.std_assd = np.nanstd(self.test_assd, axis=0)  

                update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                    "Test foreground metrics: mean: (Dice|HD|ASSD): "
                    f"{list(map('{:.4f}'.format, self.mean_dice))}, "
                    f"{list(map('{:.2f}'.format, self.mean_hd))}, "
                    f"{list(map('{:.2f}'.format, self.mean_assd))}"), self.infer_log)

                update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                    "Test foreground metrics: std: (Dice|HD|ASSD): "
                    f"{list(map('{:.4f}'.format, self.std_dice))}, "
                    f"{list(map('{:.2f}'.format, self.std_hd))}, "
                    f"{list(map('{:.2f}'.format, self.std_assd))}"), self.infer_log)

                update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                    "Test mean foreground metrics(Dice|HD|ASSD): "
                    f"{np.nanmean(self.mean_dice):.4f}, "
                    f"{np.nanmean(self.mean_hd):.2f}, "
                    f"{np.nanmean(self.mean_assd):.2f}"), self.infer_log)

                update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                    "Test std foreground metrics(Dice|HD|ASSD): "
                    f"{np.nanmean(self.std_dice):.4f}, "
                    f"{np.nanmean(self.std_hd):.2f}, "
                    f"{np.nanmean(self.std_assd):.2f}"), self.infer_log)