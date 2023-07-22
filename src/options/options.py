#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 06/02/2022


import argparse
import os
import os.path as osp


class Options():
    """This class defines options used during both training and inference time.
    """
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Active learning 
        parser.add_argument('-l', '--num_labeled', type=int, default=1e9, help='number of training data')
        parser.add_argument('--RS_seed', type=int, default=-1, help='seed for randomly select labeled data')
        parser.add_argument('--mode', type=str, default='global', help='whether to use global/local uncertainty/diversity ["global" | "local"]')
        parser.add_argument('--plan', type=str, default='', help='name of the plan file for customized selection')

        # Experiment setup
        parser.add_argument('-n', '--name', type=str, required=True, help='experiment name')
        parser.add_argument('-o', '--organ', type=str, nargs='+', required=True, help='organ of interest')
        parser.add_argument('-c', '--num_classes', type=int, required=True, help='number of classes')
        parser.add_argument('-m', '--modality', type=str, required=True, help='The input imaging modality ["ct" | "mr"]')

        parser.add_argument('--dataroot', type=str, default='./data/', help='data directory')
        parser.add_argument('--resolution', type=str, default='2mm', help='data resolution')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='networks are saved here')

        # Load pre-trained weights
        parser.add_argument('--load_ckpt', action='store_true', help='whether to use the pre-trained model weights/checkpoint. Defaults to False')
        parser.add_argument('--ft_id', type=int, default=-1, help='how to fine-tune the network [0: all parameters | 1: decoder | 2: output layer]')
        parser.add_argument('--init', type=str, default='kaiming', help='model initialization ["scratch" | "kaiming" | "xavier"]')

        # CNN
        parser.add_argument('--nid', type=int, default=0, help="network id [0: DI2IN | 1: UNet (monai) | 2: nnFormer | 3: generic UNet | 999: fine-tune]")
        parser.add_argument('--dim', type=int, default=3, help='convolution dimension')
        parser.add_argument('--input_nc', type=int, default=1, help='number of input image channels')
        parser.add_argument('--num_pool', type=int, default=5, help='number of pooling layers')
        parser.add_argument('--nbase', type=int, default=8, help='number of base filters')
        parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
        parser.add_argument('--do_ds', action='store_true', help='whether to use deep supervision. Defaults to False')

        # Transformer 
        parser.add_argument('--embedding_dim', type=int, default=96, help='dimension of embedding feature vector in nnFormer')
        parser.add_argument('--depths', default=(2, 2, 2, 2), type=tuple, help='depths of nnFormer')
        parser.add_argument('--num_heads', default=(3, 6, 12, 24), type=tuple, help='number of attention heads')
        parser.add_argument('--swin_patch_size', default=(4, 4, 4), type=tuple, help='patch size of the swin block')
        parser.add_argument('--swin_window_size', default=(4, 4, 8, 4), type=tuple, help='window size of the swin block')

        # Train 
        parser.add_argument('--debug', action='store_true', help='whether to use debug mode (training step = 1). Defaults to False')
        parser.add_argument('--multi_gpu', action='store_true', help='whether to use multiple gpus. Defaults to False')
        parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
        parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
        parser.add_argument('--train_ratio', type=float, default=0.8, help='train / train + valid')
        parser.add_argument('--crop_size', default=(128, 128, 128), nargs=3, type=int, help='cropped patch size')
        parser.add_argument('--num_samples', default=1, type=int, help='number of patch samples cropped per volume')
        parser.add_argument('-bs', '--batch_size', type=int, default=2, help='training batch size')
        parser.add_argument('--init_lr', type=float, default=1e-2, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
        parser.add_argument('--max_iterations', type=int, default=30000, help='maximum number of iterations')
        parser.add_argument('--val_iterations', type=int, default=200, help='number of iterations per validation phase')
        parser.add_argument('--skip_val_iterations', type=int, default=10000, help='number of iterations to skip validation phases')
        parser.add_argument('--early_stop', type=int, default=1e9, help='number of epochs without improvements for early stop')
        parser.add_argument('-track','--display_per_iter', action='store_true', help='whether to display training loss and validation dsc per iteration')
        
        # Infer
        parser.add_argument('--epoch', type=str, default='best_model.pth', help='load the model weights at specific epoch')
        parser.add_argument('--save_output', action='store_true', help='whether to save predictions in an output directory. Defaults to False')
        parser.add_argument('--sw_batch_size', type=int, default=8, help='max number of patches per network inference iteration')
        parser.add_argument('--overlap', type=float, default=0.5, help='amount of overlap between patches')
        parser.add_argument('--blend_mode', type=str, default='gaussian', help='how to blend output of overlapping patches [constant | gaussian]')
        parser.add_argument('--blend_sigma', type=float, default=0.125, help='if using gaussian blend mode, std of gaussian window')
        parser.add_argument('--padding_mode', type=str, default='constant', help='how to pad when crop_size is larger than inputs [constant | reflect | replicate | circular]')
        parser.add_argument('--padding_val', type=float, default=0, help='fill this value for padding')

        self.parser = parser


    def get_options(self):
        return self.parser.parse_args()
        