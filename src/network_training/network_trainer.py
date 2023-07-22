#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 07/20/2022


import os.path as osp
import torch
import numpy as np
from time import time
from monai.losses import DiceLoss
from monai.metrics import compute_meandice
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
from loss_functions.deep_supervision import MultipleOutputLoss
from utils.util import update_log, get_lr, get_time, poly_lr, create_sitkImage


#------------------------------
#         basic trainer
#------------------------------
# Adam optimizer
# No learning rate scheduler
# Loss: Dice loss
# w/ or w/o gradient clipping


class NetworkTrainer(object):

    def __init__(self, data, model, opt):
        self.opt = opt
        self.train_ds, self.valid_ds = data.get_data()
        self.model = model.cuda() if torch.cuda.is_available() else model

        if self.opt.multi_gpu and torch.cuda.device_count() > 1:
            print(f"multiple GPUs are used: {torch.cuda.device_count()}")
            self.model = torch.nn.DataParallel(self.model).cuda().module

        self.optim = torch.optim.Adam(
            params=self.model.parameters(), 
            lr=self.opt.init_lr, 
            weight_decay=self.opt.weight_decay)

        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        self.include_background = False

        self.run_log = osp.join(self.opt.expr_dir, 'run.log')
        self.val_log = osp.join(self.opt.expr_dir, 'valid_dsc.log')
        self.max_epoch = self.opt.max_iterations // len(self.train_ds)
        self.val_interval = self.opt.val_iterations // len(self.train_ds)
        self.skip_val_epoch = self.opt.skip_val_iterations // len(self.train_ds)
        self.iter_num, self.epoch, self.best_metric, self.best_metric_epoch, self.epochs_no_improve = 0, 1, -1, -1, 0

        if self.opt.load_ckpt:
            self.load_ckpt()
            
    def train(self):
        update_log(f"\n{get_time():%Y-%m-%d %H:%M:%S}", self.run_log) 
        update_log(f'epoch:  {self.epoch}', self.run_log)

        self.train_loss = 0
        self.set_training_mode()
        
        for data in self.train_ds:
            self.iter_num += 1

            if self.opt.debug:
                if self.iter_num > 1:
                    break

            self.fit(data)
            self.lr_scheduler()

            if self.opt.display_per_iter:
                update_log((f'[Train]: epoch={self.epoch}, '
                    f'iteration={self.iter_num}/{self.opt.max_iterations}, '
                    f'loss={self.loss.item():.4f}'), self.run_log)

        update_log((f'{get_time():%Y-%m-%d %H:%M:%S}: '
            f'train loss: {(self.train_loss/len(self.train_ds)):.4f}'), self.run_log)

    def valid(self):
        torch.cuda.empty_cache()
        self.set_evaluation_mode()
        self.val_dice = []

        with torch.no_grad():
            for i, data in enumerate(self.valid_ds):
                start = time()
                image, target = data["data_image"], data["data_mask"]
                image, target = image.cuda(), target.cuda()
                target = self.post_label(target)
                sub = data["subject"]
                sub = osp.splitext(osp.basename(sub[0]))[0]
                
                pred = self.predict(image)
                pred = self.post_pred(pred)

                dice = self.get_dice(pred, target, self.include_background)
                self.val_dice.append(dice)

                if self.opt.display_per_iter:
                    update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                        f"epoch={self.epoch}, id={i+1}/{len(self.valid_ds.dataset)}, "
                        f"subject={sub}, time={time()-start:.4f}, "
                        f"dsc={list(map('{:.4f}'.format, dice))}"), self.run_log)
            
            self.val_dice = np.nanmean(self.val_dice, axis=0)  # report dice without nans
            
            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                "validation foreground dice: "
                f"{list(map('{:.4f}'.format, self.val_dice))}"), self.run_log)

            self.val_dice = np.nanmean(self.val_dice)

            update_log(f"{self.val_dice}", self.val_log, verbose=False)
            
            self.save_ckpt()
            
            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: current mean dice:"
                f" {self.val_dice:.4f}, best mean dice: {self.best_metric:.4f}"
                f" at epoch {self.best_metric_epoch}"), self.run_log)

    def set_training_mode(self):
        self.model.train()

    def set_evaluation_mode(self):
        self.model.eval()

    def fit(self, data):
        image, target = data["data_image"], data["data_mask"]
        image, target = image.cuda(), target.cuda()
        target = self.prep_label(target)
        pred = self.model(image)
        self.loss = self.loss_fn(pred, target)
        self.optim.zero_grad()
        self.loss.backward()
        self.gradient_clipping()
        self.optim.step()
        self.train_loss += self.loss.item()

    def warp_ds(self):
        ds_weights = np.array([1 / (2 ** i) for i in range(self.opt.num_pool)])
        self.loss_fn = MultipleOutputLoss(self.loss_fn, self.opt.num_classes, ds_weights)

    def prep_label(self, target):
        return target

    def gradient_clipping(self):
        pass

    def lr_scheduler(self):
        pass

    def predict(self, data, **kwargs):
        return sliding_window_inference(
            inputs=data, 
            roi_size=self.opt.crop_size, 
            sw_batch_size=self.opt.sw_batch_size, 
            predictor=self.model,
            overlap=self.opt.overlap,
            mode=self.opt.blend_mode,
            sigma_scale=self.opt.blend_sigma,
            padding_mode=self.opt.padding_mode,
            cval=self.opt.padding_val)

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
    
    def load_ckpt(self):
        ckpt_path = osp.join(self.opt.expr_dir, f'{self.opt.epoch}_model.pth')
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['state_dict'])      
      
        self.optim.load_state_dict(ckpt['optimizer_state_dict'])
        self.epoch = ckpt['epoch']
        self.best_metric = ckpt['best_metric']
        self.best_metric_epoch = ckpt['best_metric_epoch']
        self.optim.param_groups[0]['lr'] = poly_lr(self.epoch + 1, self.opt.max_epoch, self.opt.init_lr, 0.9)
        
        update_log(f"model and optimizer are initialized from {ckpt_path}", self.run_log)
        update_log((f"Epoch={self.epoch}, LR={np.round(get_lr(self.optim), decimals=6)}, "
            f"best_metric={self.best_metric}, best_epoch={self.best_metric_epoch} now"), self.run_log)
        
        self.epoch += 1

    def save_ckpt(self):
        if self.val_dice > self.best_metric:
            self.epochs_no_improve = 0
            self.best_metric = self.val_dice
            self.best_metric_epoch = self.epoch
            torch.save({'epoch': self.epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'best_metric': self.best_metric,
                        'best_metric_epoch': self.best_metric_epoch},
                        osp.join(self.opt.expr_dir, f'best_model.pth'))
        else:
            self.epochs_no_improve += 1

    def run(self):
        update_log(('\nexperiment timestamp: '
            f'{get_time():%Y-%m-%d %H:%M:%S}'), self.run_log)

        if self.opt.do_ds:
            self.warp_ds()

        while self.epoch <= self.max_epoch:
            start = time()
            self.train()

            if self.epoch > self.skip_val_epoch:
                if self.epoch % self.val_interval == 0:
                    self.valid()
                    if self.epochs_no_improve == self.opt.early_stop:
                        update_log('Early Stopping', self.run_log)
                        break

                    torch.save(
                        {'epoch': self.epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'best_metric': self.best_metric,
                        'best_metric_epoch': self.best_metric_epoch},
                        osp.join(self.opt.expr_dir, f'current_model.pth'))

            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                f"lr: {np.round(get_lr(self.optim), decimals=6)}"), self.run_log)        
            
            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                f"This epoch took {time()-start:.2f} s"), self.run_log)

            self.epoch += 1

        print(f"training completed, best_metric={self.best_metric:.4f} at epoch={self.best_metric_epoch}")
