#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/10/2023


import torch
import numpy as np
from time import time
import os.path as osp
from network_training.TrainerV2 import TrainerV2
from utils.util import poly_lr, update_log, get_time, get_lr


#--------------------------------------
# Trainer for image reconstruction task
#--------------------------------------
# Loss: reconstruction loss, L2


class TrainerV2_Recon(TrainerV2):
    
    def __init__(self, data, model, opt):
        super().__init__(data, model, opt)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.best_metric = 1e9

    def fit(self, data):
        image, target = data["data_image"], data["data_image"]
        image, target = image.cuda(), target.cuda()
        pred = self.model(image)
        self.loss = self.loss_fn(pred, target)
        self.optim.zero_grad()
        self.loss.backward()
        self.gradient_clipping()
        self.optim.step()
        self.train_loss += self.loss.item()

    def valid(self):
        torch.cuda.empty_cache()
        self.set_evaluation_mode()
        self.val_loss = []

        with torch.no_grad():
            for i, data in enumerate(self.valid_ds):
                start = time()
                image, target = data["data_image"], data["data_image"]
                image, target = image.cuda(), target.cuda()
                sub = data["subject"]
                sub = osp.splitext(osp.basename(sub[0]))[0]
                pred = self.predict(image)
                assert pred.shape == target.shape
                mse = torch.mean((pred - target) ** 2).cpu().numpy()
                self.val_loss.append(mse)

                if self.opt.display_per_iter:
                    update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                        f"epoch={self.epoch}, id={i+1}/{len(self.valid_ds.dataset)}, "
                        f"subject={sub}, time={time()-start:.4f}, "
                        f"mse={mse:.4f}"), self.run_log)
            
            self.val_loss = np.nanmean(self.val_loss, axis=0)  # report mse without nans
            update_log(f"{self.val_loss}", self.val_log, verbose=False)
            self.save_ckpt()
            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: current mean mse:"
                f" {self.val_loss:.4f}, best mean mse: {self.best_metric:.4f}"
                f" at epoch {self.best_metric_epoch}"), self.run_log)

    def save_ckpt(self):
        if self.val_loss < self.best_metric:
            self.epochs_no_improve = 0
            self.best_metric = self.val_loss
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
        super().run()
    
