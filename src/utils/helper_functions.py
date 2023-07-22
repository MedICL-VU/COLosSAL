#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 06/13/2022


import os
import os.path as osp
from typing import List
import numpy as np
import pickle


def find_best_val_dsc(valid_dsc_path: str):
    """Search the epoch with the best validation dice
    """
    with open(valid_dsc_path) as f:
        dsc_scores = [line.rstrip()[4:].split(',') for line in f]
    best_classes, best_epoch, best_mean = [], 0, 0
    for i, score in enumerate(dsc_scores):
        score = [float(s) for s in score]
        mean_score = np.mean(score)
        if mean_score > best_mean:
            best_classes = score
            best_epoch = i
    print(f'best epoch: {i+1}, best dice scores: {best_classes}')


def nnUNet_read_plan(plan_file: str):
    assert osp.splitext(osp.basename(plan_file))[1] == '.pkl'
    with open(plan_file, 'rb') as f:
        data = pickle.load(f)

    print('\nnnUNet plans:\n--------------------------------------')
    print(f"modalities: {data['modalities']}")
    print(f"normalization_schemes: {data['normalization_schemes']}")
    print(f"preprocesser name: {data['preprocessor_name']}")

    print(f"dataset mean: {data['dataset_properties']['intensityproperties'][0]['mean']}")
    print(f"dataset std : {data['dataset_properties']['intensityproperties'][0]['sd']}")
    print(f"dataset min : {data['dataset_properties']['intensityproperties'][0]['mn']}")
    print(f"dataset max : {data['dataset_properties']['intensityproperties'][0]['mx']}")
    print(f"dataset 0.05: {data['dataset_properties']['intensityproperties'][0]['percentile_00_5']}")
    print(f"dataset 99.5: {data['dataset_properties']['intensityproperties'][0]['percentile_99_5']}")

    print(f"base_num_features: {data['base_num_features']}")
    print(f"use_mask_for_norm: {data['use_mask_for_norm']}")
    print(f"keep_only_largest_region: {data['keep_only_largest_region']}")
    print(f"min_region_size_per_class: {data['min_region_size_per_class']}")
    print(f"batch_size: {data['plans_per_stage'][0]['batch_size']}")
    print(f"patch_size: {data['plans_per_stage'][0]['patch_size']}")
    print(f"num_pool_per_axis: {data['plans_per_stage'][0]['num_pool_per_axis']}")
    print(f"median_patient_size_in_voxels: {data['plans_per_stage'][0]['median_patient_size_in_voxels']}")
    print(f"current_spacing: {data['plans_per_stage'][0]['current_spacing']}")
    print(f"original_spacing: {data['plans_per_stage'][0]['original_spacing']}")
    print(f"do_dummy_2D_data_aug: {data['plans_per_stage'][0]['do_dummy_2D_data_aug']}")
    print(f"pool_op_kernel_sizes: {data['plans_per_stage'][0]['pool_op_kernel_sizes']}")
    print(f"conv_kernel_sizes: {data['plans_per_stage'][0]['conv_kernel_sizes']}")
    print(f"conv_per_stage: {data['conv_per_stage']}")


if __name__ == "__main__":

    # pkl_file = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/nii/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/3d_fullres/Task003_PA/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl'
    pkl_file = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/nii/nnUNet_raw_data_base/nnUNet_trained_models/nnUNet/3d_fullres/Task003_PA/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl'
    nnUNet_read_plan(pkl_file)
    
    # f1 = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/nii/nnUNet_raw_data_base/nnUNet_cropped_data/Task001_WB/WB_001.npz'
    # f2 = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/nii/nnUNet_raw_data_base/nnUNet_preprocessed/Task001_WB/nnUNetData_plans_v2.1_stage0/WB_001.npz'
    
    # data1 = dict(np.load(f1, allow_pickle=True))
    # data2 = dict(np.load(f2, allow_pickle=True))

    # x1 = data1['data'][0,...]
    # x2 = data2['data'][0,...]

    # x1 = np.clip(x1, 46, 1426)
    # x1 = (x1 - 946.6099243164062) / 259.2779235839844


    # print(data)
    # print(data['data'])
    
    # x = data['data']
    # from util import create_sitkImage

    # img = x[0,...]

    # print(img.shape)
    # out = create_sitkImage(img)
    # import SimpleITK as sitk

    # sitk.WriteImage(out, 'unet_out.nii.gz')




    # pkl_dir = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/nii/nnUNet_raw_data_base/nnUNet_cropped_data/Task001_WB'

    # from glob import glob
    # pkls = glob(pkl_dir + '/*.pkl')
    
    # z, y, x = [], [], []

    # for pkl in pkls:
    #     with open(pkl, 'rb') as f:
    #         data = pickle.load(f)
    #         # ori = data['original_size_of_raw_data']
    #         try:
    #             new = data['size_after_cropping']
    #         except:
    #             continue

    #         z.append(new[0])
    #         y.append(new[1])
    #         x.append(new[2])


    # print(np.median(z), np.median(y), np.median(x))

            # if ori[0]!=new[0] or ori[1]!=new[1] or ori[2]!=new[2]:
            #     print(pkl, ori, new)


    
    







