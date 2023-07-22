#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 07/07/2022


import os.path as osp
import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt
from monai.losses import *
from monai.metrics import compute_meandice
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

from loss_functions.deep_supervision import MultipleOutputLoss2
from utils.util import update_log, poly_lr, get_lr, get_time


# Modified Dominik's training pipeline
#-----------------------------------
# Normalization (500, 1524) -> (-1, 1)
# PosNegSample scheme vs RandSample
# Adam 1e-3 instead of SGD 1e-2
# No learning rate scheduler
# No gradient clipping
# Model initialization: scratch/fine-tune


class TrainerV1(object):

    def __init__(self, data, model, opt):
        self.opt = opt
        self.fname = osp.join(self.opt.expr_dir, 'run.log')
        self.eval = osp.join(self.opt.expr_dir, 'valid_dsc.log')
        self.timestamp_start = get_time()
        self.train_ds, self.valid_ds = data.get_data()
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.optim = torch.optim.Adam(self.model.parameters(), self.opt.init_lr)
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        if self.opt.do_ds:
            ds_weights = np.array([1 / (2 ** i) for i in range(self.opt.num_pool)])
            self.loss_fn = MultipleOutputLoss2(self.loss_fn, self.opt.num_classes, ds_weights)

        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.opt.num_classes)
        self.post_label = AsDiscrete(to_onehot=self.opt.num_classes)
        self.epoch, self.best_metric, self.best_metric_epoch, self.epochs_no_improve = 1, -1, -1, 0
        self.epoch_loss_values, self.metric_values = [], []

    def train(self):
        update_log(f"\n{get_time():%Y-%m-%d %H:%M:%S}", self.fname) 
        update_log(f'epoch:  {self.epoch}', self.fname)
        self.model.train()
        self.epoch_loss = 0
        step = 0
        for train_data in self.train_ds:
            step += 1
            img, lab = train_data["data_image"].cuda(), train_data["data_mask"].cuda()
            self.optim.zero_grad()
            pred = self.model(img)
            del img
            loss = self.loss_fn(pred, lab)
            loss.backward()
            self.optim.step()
            self.epoch_loss += loss.item()
            if self.opt.display_per_iter:
                update_log(f"[Train]: epoch={self.epoch}, batch_idx={step}/{len(self.train_ds)}, loss={loss.item():.4f}", self.fname)

        self.epoch_loss /= step
        self.epoch_loss_values.append(self.epoch_loss)
        update_log(f"{get_time():%Y-%m-%d %H:%M:%S}: train loss: {self.epoch_loss:.4f}", self.fname)

    def valid(self):
        self.model.eval()
        self.val_dice = []
        with torch.no_grad():
            for i, valid_data in enumerate(self.valid_ds):
                img, lab, sub = valid_data["data_image"].cuda(), valid_data["data_mask"].cuda(), valid_data["subject"]
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

                pred = self.post_pred(decollate_batch(pred)[0]).unsqueeze(0)
                lab = self.post_label(decollate_batch(lab)[0]).unsqueeze(0)
                dice = np.array(compute_meandice(pred, lab)[0][1:].cpu())  # ignore background
                self.val_dice.append(dice)
                if self.opt.display_per_iter:
                    update_log(f"{get_time():%Y-%m-%d %H:%M:%S}: epoch={self.epoch}, id={i+1}/{len(self.valid_ds.dataset)}, subject={sub}, time={time()-start:.4f}, dsc={list(map('{:.4f}'.format, dice))}", self.fname)
            
            self.val_dice = np.mean(self.val_dice, axis=0)
            update_log(f"{get_time():%Y-%m-%d %H:%M:%S}: validation foreground dice: {list(map('{:.4f}'.format, self.val_dice))}", self.fname)
            self.val_dice = np.mean(self.val_dice)
            update_log(f"{self.val_dice}", self.eval, verbose=False)
            self.metric_values.append(self.val_dice)

            if self.val_dice > self.best_metric:
                self.epochs_no_improve = 0
                self.best_metric = self.val_dice
                self.best_metric_epoch = self.epoch
                torch.save({'epoch': self.epoch,
                            'state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optim.state_dict()},
                            osp.join(self.opt.expr_dir, f'best_model.pth'))
            else:
                self.epochs_no_improve += 1
            update_log(f"{get_time():%Y-%m-%d %H:%M:%S}: current mean dice: {self.val_dice:.4f}, best mean dice: {self.best_metric:.4f} at epoch {self.best_metric_epoch}", self.fname)

    def fit(self):
        update_log(f'\nexperiment timestamp: {self.timestamp_start:%Y-%m-%d %H:%M:%S}', self.fname)
        while self.epoch <= self.opt.max_epoch:
            start = time()
            self.train()

            if self.epoch > self.opt.skip_val_epoch:
                if self.epoch % self.opt.val_interval == 0:
                    self.valid()
                    if self.epochs_no_improve == self.opt.early_stop:
                        update_log('Early Stopping', self.fname)
                        break

            torch.save({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict()},
                osp.join(self.opt.expr_dir, f'current_model.pth'))

            update_log(f"{get_time():%Y-%m-%d %H:%M:%S}: lr: {np.round(get_lr(self.optim), decimals=6)}", self.fname)        
            update_log(f"{get_time():%Y-%m-%d %H:%M:%S}: This epoch took {time()-start:.2f} s", self.fname)        
            self.epoch += 1

        print(f"training completed, best_metric={self.best_metric:.4f} at epoch={self.best_metric_epoch}")
        self.training_summary()

    def training_summary(self):
        update_log('plotting training losses and validation metrics...', self.fname)
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Average Training Loss")
        x = [i + 1 for i in range(len(self.epoch_loss_values))]
        y = self.epoch_loss_values
        plt.xlabel("epoch")
        plt.ylabel("training loss")
        plt.plot(x, y)
        plt.subplot(1, 2, 2)
        plt.title("Validation Dice")
        x = [self.opt.val_interval * (i + 1) for i in range(len(self.metric_values))]
        y = self.metric_values
        plt.xlabel("epoch")
        plt.ylim(0.0, 1.0)
        plt.plot(x, y)
        plt.savefig(osp.join(self.opt.expr_dir, 'losses.png'))
          # if opt.continue_train:
        #     assert osp.exists(opt.checkpoint_path)
        #     checkpoint = torch.load(f'{opt.checkpoint_path}')
        #     self.model.load_state_dict(checkpoint['state_dict'])
        #     self.optim.load_state_dict(checkpoint['optimizer_states'])
        #     self.epoch = checkpoint['epoch']
        #     print(f"model and optimizer are initialized from {opt.checkpoint_path}")
        #     print(f'Epoch={self.epoch} now')