#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 06/02/2022


import os.path as osp
import torch
import torch.nn as nn
from monai.networks.layers import Norm
from monai.networks.nets import *
from models.DI2IN import *
from models.nnFormer import nnFormer
from models.generic_UNet import Generic_UNet
from models.DoDNet import DoDNet
from models.DeeplabV3_plus import DeepLabV3_3D
from utils.util import update_log


class NetworkLoader(object):
    def __init__(self, opt):
        self.opt = opt
        self.fname = osp.join(self.opt.expr_dir, 'run.log')

    def load(self)-> nn.Module:
        if self.opt.nid == 0:
            update_log(f'model architecture: DI2IN (nbase={self.opt.nbase})', self.fname)
            m = DI2IN(num_classes=self.opt.num_classes, nbase=self.opt.nbase)

        elif self.opt.nid == 1:
            update_log('model architecture: UNet (MONAI)', self.fname)
            m = UNet(
                dimensions=self.opt.dim,
                in_channels=self.opt.input_nc,
                out_channels=self.opt.num_classes,
                channels=(16, 32, 48, 64, 128),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
                dropout=self.opt.dropout)

        elif self.opt.nid == 2:
            assert self.opt.dim==3, f'nnFormer requires input dim to be 3, but get {self.opt.dim}'
            update_log('model architecture: nnFormer', self.fname)
            m = nnFormer(
                num_classes=self.opt.num_classes, 
                crop_size=self.opt.crop_size,
                embedding_dim=self.opt.embedding_dim,
                input_channels=self.opt.input_nc, 
                conv_op=nn.Conv3d, 
                depths=self.opt.depths,
                num_heads=self.opt.num_heads,
                patch_size=self.opt.patch_size,
                window_size=self.opt.window_size,
                deep_supervision=self.opt.do_ds)

        elif self.opt.nid == 3:
            update_log('model architecture: generic UNet from nnUNet', self.fname)
            m = Generic_UNet(
                input_channels=self.opt.input_nc, 
                base_num_features=self.opt.nbase, 
                num_classes=self.opt.num_classes, 
                num_pool=self.opt.num_pool,
                deep_supervision=self.opt.do_ds)

        elif self.opt.nid == 4:
            update_log(f'model architecture: DI2IN_DS (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_DS(num_classes=self.opt.num_classes, nbase=self.opt.nbase)

        elif self.opt.nid == 5:
            update_log(f'model architecture: DoDNet (partial label learning))', self.fname)
            m = DoDNet(num_classes=self.opt.num_classes)

        elif self.opt.nid == 6:
            update_log('model architecture: generic UNet customized', self.fname)

            # Pancreas model
            # m = Generic_UNet(
            #     input_channels=1, 
            #     base_num_features=32, 
            #     num_classes=2, 
            #     num_pool=5,
            #     pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
            #     conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            #     deep_supervision=self.opt.do_ds)
            
            # Whole_Bowel/Constrictor_Muscles model
            m = Generic_UNet(
                input_channels=1, 
                base_num_features=32, 
                num_classes=4, 
                num_pool=5,
                pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
                conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                deep_supervision=self.opt.do_ds)

        elif self.opt.nid == 7:
            update_log(f'model architecture: DI2IN_KD (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_KD(num_classes=self.opt.num_classes, nbase=self.opt.nbase)

        elif self.opt.nid == 8:
            update_log(f'model architecture: DI2IN_BNF_ConvTrans (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_BNF_ConvTrans(num_classes=self.opt.num_classes, nbase=self.opt.nbase)
            torch.backends.cudnn.deterministic = True

        elif self.opt.nid == 9:
            update_log(f'model architecture: DI2IN_BNF_ConvTrans_KD (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_BNF_ConvTrans_KD(num_classes=self.opt.num_classes, nbase=self.opt.nbase)
            torch.backends.cudnn.deterministic = True

        elif self.opt.nid == 10:
            update_log(f'model architecture: DI2IN_BNF_ConvTrans_KD_V2 (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_BNF_ConvTrans_KD_V2(num_classes=self.opt.num_classes, nbase=self.opt.nbase)
            torch.backends.cudnn.deterministic = True

        elif self.opt.nid == 11:
            update_log(f'model architecture: DeeplabV3+ (3D)', self.fname)
            update_log('Adapted from: https://github.com/ChoiDM/pytorch-deeplabv3plus-3D', self.fname)
            m = DeepLabV3_3D(
                num_classes=self.opt.num_classes,
                input_channels=self.opt.input_nc,
                resnet='resnet18_os16',
                last_activation=None)

        if self.opt.nid == 12:
            update_log(f'model architecture: DI2IN_L (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_L(num_classes=self.opt.num_classes, nbase=self.opt.nbase)

        elif self.opt.nid == 999:
            update_log('model architecture: DI2IN-32 (pre-training)', self.fname)
            m = DI2IN(num_classes=82, nbase=32)       
            checkpoint = torch.load(
                '/model_pool/selfSupervised/ctmultiorgan/pretrained_models_2022-03-15/DI2IN_32/best-loss_valid=0.2610-epoch=0087.ckpt')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace("net.", "")
                new_state_dict[name] = v
            m.load_state_dict(new_state_dict, strict=False)
            print('successfully loaded the pretrained model weights')
            m.output = nn.Conv3d(m.output.in_channels, self.opt.num_classes, kernel_size=1, stride=1, padding=0, bias=True)

            if self.opt.ft_id == 0:
                update_log('we wil fine-tune all the model parameters', self.fname)
            elif self.opt.ft_id == 1:
                update_log('we wil fine-tune the parameters in the decoder', self.fname)
                for i, (name, param) in enumerate(m.named_parameters()):
                    if i < 40:
                        param.requires_grad = False
            elif self.opt.ft_id == 2:
                update_log('we wil fine-tune the parameters in the output layer', self.fname)
                for name, param in m.named_parameters():
                    if 'output' not in name:
                        param.requires_grad = False
            return m
                
        if not self.opt.load_ckpt:
            if self.opt.init == 'scratch':
                update_log('model initialization: Scratch', self.fname)
            elif self.opt.init == "kaiming":
                update_log('model initialization: Kaiming', self.fname)
                m.apply(InitWeights_He(1e-2))
            elif self.opt.init == 'xavier':
                update_log('model initialization: Xavier uniform', self.fname)
                m.apply(InitWeights_XavierUniform(1))
            else:
                raise NotImplementedError("This initialization method has not been implemented...")
        return m


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitWeights_XavierUniform(object):
    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
