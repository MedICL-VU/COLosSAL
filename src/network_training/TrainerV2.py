#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 06/02/2022


import torch
from monai.losses import DiceCELoss
from monai.networks.utils import one_hot
from network_training.network_trainer import NetworkTrainer
from utils.util import poly_lr


#-------------------------------
#           Trainer V2
#-------------------------------
# SGD optimizer
# Learning rate scheduler: poly
# Gradient clipping 
# Loss: Dice + CrossEntropy


class TrainerV2(NetworkTrainer):
    
    def __init__(self, data, model, opt):
        super().__init__(data, model, opt)
        
        self.optim = torch.optim.SGD(
            params=self.model.parameters(), 
            lr=self.opt.init_lr, 
            weight_decay=self.opt.weight_decay, 
            momentum=0.99, 
            nesterov=True)

        self.loss_fn = DiceCELoss(to_onehot_y=False, softmax=True)

    def prep_label(self, target):
        """Manually convert to one-hot based on the number of classes
        This is useful when foreground labels may be absent for certain subjects
        e.g., male subjects with removed prostate; prostate channel will be all zeros 
        """
        return one_hot(target, self.opt.num_classes, dim=1)

    def gradient_clipping(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)

    def lr_scheduler(self):
        # self.optim.param_groups[0]['lr'] = poly_lr(self.epoch, self.opt.max_epoch, self.opt.init_lr, 0.9)
        self.optim.param_groups[0]['lr'] = poly_lr(self.iter_num, self.opt.max_iterations, self.opt.init_lr, 0.9)

    def run(self):
        super().run()
    
