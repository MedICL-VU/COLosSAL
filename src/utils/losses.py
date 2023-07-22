#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 08/27/2021


def compute_mean_dice(pred, gt):
    score = []
    for i in range(pred.size()[0] - 1):
        score.append(dice_loss(pred[i, :, :, :], gt[i, :, :, :]))
    return score


def dice_loss(pred, gt, eps: float = 1e-7):
    """soft dice loss"""
    pred = pred.view(-1)
    gt = gt.view(-1)
    intersection = (pred * gt).sum()
    return 1 - 2. * intersection / ((pred ** 2).sum() + (gt ** 2).sum() + eps)
