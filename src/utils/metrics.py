#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 09/16/2021


import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance


def compute_metrics(pred, gt):
    def dice_score(pred, gt, eps: float = 1e-7):
        intersection = np.logical_and(pred, gt)
        return 2. * intersection.sum() / (pred.sum() + gt.sum() + eps)

    def IoU_score(pred, gt, eps: float = 1e-7):
        intersection = np.logical_and(pred, gt)
        union = np.logical_or(pred, gt)
        return np.sum(intersection) / (np.sum(union) + eps)

    def Hausdorff_distance(pred, gt):
        pred = torch.Tensor(pred).unsqueeze(0).unsqueeze(0)
        gt = torch.Tensor(gt).unsqueeze(0).unsqueeze(0)
        hd = compute_hausdorff_distance(pred, gt, include_background=True, percentile=95, directed=False)
        hd = hd.squeeze(0).squeeze(0).numpy()
        return hd

    def average_surface_distance(pred, gt):
        pred = torch.Tensor(pred).unsqueeze(0).unsqueeze(0)
        gt = torch.Tensor(gt).unsqueeze(0).unsqueeze(0)
        asd = compute_average_surface_distance(pred, gt, include_background=True, symmetric=True)
        asd = asd.squeeze(0).squeeze(0).numpy()
        return asd

    metric = []
    assert pred.shape == gt.shape, "Shape mismatch: pred and gt must have the same shape."
    num_classes = gt.shape[0]
    # order: oc_dice, oc_IoU, oc_HD, oc_ASSD, od_dice, od_IoU, od_HD, od_ASSD
    for i in range(num_classes):
        metric.append(dice_score(pred[i, :, :], gt[i, :, :]))
        metric.append(IoU_score(pred[i, :, :], gt[i, :, :]))
        metric.append(Hausdorff_distance(pred[i, :, :], gt[i, :, :]))
        metric.append(average_surface_distance(pred[i, :, :], gt[i, :, :]))
    return metric


def print_results(metrics):
    """order: dice, IoU, HD, ASSD
    """
    names = ['Dice', 'IoU ', 'HD  ', 'ASSD']
    print('** optic cup  **')
    for i in range(4):
        print(f'{names[i]}: {np.mean(metrics[:, i]):.3f} ({np.std(metrics[:, i]):.3f})')
    print('\n** optic disc **')
    for i in range(4):
        print(f'{names[i]}: {np.mean(metrics[:, i+4]):.3f} ({np.std(metrics[:, i+4]):.3f})')