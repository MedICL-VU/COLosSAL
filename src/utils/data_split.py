#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 07/22/2022


import os
import os.path as osp
import argparse
from glob import glob
import numpy as np
from util import mkdirs, mkdir, get_cc3d
import shutil
import random
from typing import List
from tqdm import tqdm
import SimpleITK as sitk
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--split_dir', type=str, default='/pct_wbo2/home/han.l/RO/Data/splits', help='where to save split files')
parser.add_argument('--npz_dir', type=str, default='/pct_wbo2/home/han.l/RO/Data/npz', help='npz directory')
parser.add_argument('--data_dir', type=str, default='/pct_wbo2/projects/RO/RO_Contouring/Data_Processed', help='root directory of processed data')
args = parser.parse_args()


random.seed(0)


def read_split(split_path, split_name) -> List[str]:
    assert split_name in ['train', 'val', 'test']
    split = np.load(split_path, allow_pickle=True)
    return split[split_name]


def read_list(organ, data_list, res='2mm'):
    f = open(data_list, 'r')
    return [f'{args.data_dir}/{res}/npz/{organ}/{line.rstrip()}.npz' for line in f]


def split_list(all_paths: List[str]):
    """ Hard coding for 8:1:1 split ratio
    """
    random.shuffle(all_paths)
    num_paths = len(all_paths)
    num_train = int(num_paths*0.5)
    num_val = int((num_paths *0.25))
    num_test = num_paths - num_train - num_val

    # num_train = 600
    # num_val = int((num_paths - num_train))
    # num_test = num_paths - num_train - num_val    

    print(f'data splits (train:val:test) = {num_train}:{num_val}:{num_test}')
    train_paths = all_paths[:num_train]
    val_paths = all_paths[num_train:num_train+num_val]
    test_paths = all_paths[num_train+num_val:]
    return train_paths, val_paths, test_paths


def save_split_uterus():
    all_paths = []
    train_list = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/train__Uterus_list.txt'
    test_list = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/test__Uterus_list.txt'
    all_paths = read_list('_Uterus', train_list) + read_list('_Uterus', test_list)
    train_paths, val_paths, test_paths = split_list(all_paths)
    output_file = osp.join(args.split_dir, '_Uterus.npz')
    np.savez(output_file, train=train_paths, val=val_paths, test=test_paths)


def save_split_pelvic6():
    all_paths = []
    train_list = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/train_Pelvic6_list.txt'
    test_list = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/test_Pelvic6_list.txt'
    all_paths = read_list('Pelvic6', train_list) + read_list('Pelvic6', test_list)

    # split data based on different situations
    f = open('/pct_wbo2/home/han.l/MultiOrganSegWithCuda/pseudo_pelvic.log')
    lines = [line.strip('\n').strip(' ') for line in f]

    male, female = [], []

    for line in lines:
        gender = line[123:125]    
        sub = line[175:]
        sub = sub[:sub.find('/')]
        if gender == 'PB':
            male.append(sub)
        else:
            female.append(sub)

    male_label_types = {}
    with tqdm(total=len(all_paths)) as pbar:
        for path in all_paths:
            sub = osp.basename(path)[:-4]
            if sub in male:
                data = np.load(path, allow_pickle=True)
                msk = data['data_mask']
                label = str(np.unique(msk).tolist())
                if label not in male_label_types:
                    male_label_types[f'{label}'] = [path]
                else:
                    male_label_types[f'{label}'].append(path)
            pbar.update(1)

    train_paths, val_paths, test_paths = [], [], []

    # collect male subjects
    for key in male_label_types.keys():
        print(key)
        current_train_paths, current_val_paths, current_test_paths = split_list(male_label_types[key])
        train_paths += current_train_paths
        val_paths += current_val_paths
        test_paths += current_test_paths

    # collect female subjects
    female = [f'{args.data_dir}/2mm/npz/Pelvic6/{sub}.npz' for sub in female]
    current_train_paths, current_val_paths, current_test_paths = split_list(female)
    train_paths += current_train_paths
    val_paths += current_val_paths
    test_paths += current_test_paths 

    # save npz
    output_file = osp.join(args.split_dir, 'Pelvic6.npz')
    np.savez(output_file, train=train_paths, val=val_paths, test=test_paths)


#-----------------------------------------------------------------------------------------------------

def save_split(organ:str, res:str='2mm'):
    all_paths = []
    train_list = f'/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/train_{organ}_list.txt'
    test_list = f'/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/test_{organ}_list.txt'
    all_paths = read_list(f'{organ}', train_list, f'{res}') + read_list(f'{organ}', test_list, f'{res}')
    train_paths, val_paths, test_paths = split_list(all_paths)
    output_file = osp.join(args.split_dir, f'{organ}.npz')
    np.savez(output_file, train=train_paths, val=val_paths, test=test_paths)


def generate_list_from_npz(split_path, output_dir):
    for split_name in ['train', 'val', 'test']:
        splits = read_split(split_path, split_name)
        f = open(osp.join(output_dir, f'{split_name}.txt'), 'w')
        for sub in splits:
            sub = osp.basename(sub)[:-4]
            f.write(f'{sub}\n') 


def merge_split(split_paths: List[str], save_path:str):
    train_paths, val_paths, test_paths = [], [], []

    for split_path in split_paths:
        train_paths += read_split(split_path, 'train').tolist()
        val_paths += read_split(split_path, 'val').tolist()
        test_paths += read_split(split_path, 'test').tolist()

    np.savez(save_path, train=train_paths, val=val_paths, test=test_paths)


def assign_split(paths: List[str], split_path, split_name, overwrite=False):
    assert split_name in ['train', 'val', 'test']

    if overwrite:
        save_path = split_path
    else:
        save_path = split_path.replace('.npz', '_new.npz')

    split = dict(np.load(split_path, allow_pickle=True))
    split[split_name] = paths

    np.savez(save_path, train=split['train'], val=split['val'], test=split['test'])


def save_split_whole_bowel():
    all_paths = []
    train_list = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/train_Whole_Bowel_list.txt'
    test_list = '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/test_Whole_Bowel_list.txt'
    all_paths = read_list('Whole_Bowel', train_list) + read_list('Whole_Bowel', test_list)
    train_paths, val_paths, test_paths = split_list(all_paths)
    output_file = osp.join(args.split_dir, 'Whole_Bowel.npz')
    np.savez(output_file, train=train_paths, val=val_paths, test=test_paths)


def create_pancreas_duodenum_dataset():
    PA_train_list = read_list('Pancreas', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/train_Pancreas_list.txt')
    PA_val_list = read_list('Pancreas', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/test_Pancreas_list.txt')
    DU_train_list = read_list('Whole_Bowel', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/train_Whole_Bowel_list.txt')
    DU_val_list = read_list('Whole_Bowel', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/test_Whole_Bowel_list.txt')

    PA_list = PA_train_list + PA_val_list
    DU_list = DU_train_list + DU_val_list

    # copy files to my folder
    PA_dst = '/pct_wbo2/home/han.l/RO/Data/npz/Pancreas'
    DU_dst = '/pct_wbo2/home/han.l/RO/Data/npz/Duodenum'

    mkdir(PA_dst)
    mkdir(DU_dst)

    final_PA_list, final_DU_list = [], []

    with tqdm(total=len(PA_list)) as pbar:
        for path in PA_list:
            pbar.update(1)
            shutil.copyfile(path, osp.join(PA_dst, osp.basename(path)))
            final_PA_list.append(osp.join(PA_dst, osp.basename(path)))
    
    with tqdm(total=len(DU_list)) as pbar:
        for path in DU_list:
            pbar.update(1)
            shutil.copyfile(path, osp.join(DU_dst, osp.basename(path)))
            final_DU_list.append(osp.join(DU_dst, osp.basename(path)))

    PA_train, PA_val, PA_test = split_list(final_PA_list)
    DU_train, DU_val, DU_test = split_list(final_DU_list)
    np.savez(osp.join(args.split_dir, f'Pancreas.npz'), train=PA_train, val=PA_val, test=PA_test)
    np.savez(osp.join(args.split_dir, f'Duodenum.npz'), train=DU_train, val=DU_val, test=DU_test)


def from_WB_to_DU():
    # keep only duodenum from whole bowel structures
    for path in glob('/pct_wbo2/home/han.l/RO/Data/npz/Duodenum' + '/*.npz'):
        # pdb.set_trace()
        print(path)
        data = dict(np.load(path, allow_pickle=True))
        mask = data['data_mask']
        mask[mask != 1] = 0
        np.savez(
            path, 
            data_image=data['data_image'],
            data_mask=mask,
            data_specs=data['data_specs'])


def create_whole_bowel_dataset():
    WB_train_list = read_list('Whole_Bowel', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/train_Whole_Bowel_list.txt')
    WB_val_list = read_list('Whole_Bowel', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/test_Whole_Bowel_list.txt')
    WB_list = WB_train_list + WB_val_list

    # copy files to my folder
    WB1_dst = '/pct_wbo2/home/han.l/RO/Data/npz/WB1'
    WB2_dst = '/pct_wbo2/home/han.l/RO/Data/npz/WB2'

    mkdir(WB1_dst)
    mkdir(WB2_dst)

    WB1_list = WB_list[:int(len(WB_list)/2)]
    WB2_list = WB_list[int(len(WB_list)/2):]

    for path in WB1_list:
        print(path)
        shutil.copyfile(path, osp.join(WB1_dst, osp.basename(path)))

    WB1_list = [osp.join(WB1_dst, osp.basename(path)) for path in WB1_list]

    for path in WB2_list:
        print(path)
        shutil.copyfile(path, osp.join(WB2_dst, osp.basename(path)))

    WB2_list = [osp.join(WB2_dst, osp.basename(path)) for path in WB2_list]

    WB1_train, WB1_val, WB1_test = split_list(WB1_list)
    WB2_train, WB2_val, WB2_test = split_list(WB2_list)
    np.savez(osp.join(args.split_dir, f'WB1.npz'), train=WB1_train, val=WB1_val, test=WB1_test)
    np.savez(osp.join(args.split_dir, f'WB2.npz'), train=WB2_train, val=WB2_val, test=WB2_test)


def from_WB_to_WB12():
    for path in glob('/pct_wbo2/home/han.l/RO/Data/npz/WB1' + '/*.npz'):
        print(path)
        data = dict(np.load(path, allow_pickle=True))
        mask = data['data_mask']
        mask[mask > 2] = 0
        np.savez(
            path, 
            data_image=data['data_image'],
            data_mask=mask,
            data_specs=data['data_specs'])

    for path in glob('/pct_wbo2/home/han.l/RO/Data/npz/WB2' + '/*.npz'):
        print(path)
        data = dict(np.load(path, allow_pickle=True))
        mask = data['data_mask']
        mask[mask < 3] = 0
        mask = mask - 2  # label: 3,4,5->1,2,3
        mask[mask < 0] = 0
        np.savez(
            path, 
            data_image=data['data_image'],
            data_mask=mask,
            data_specs=data['data_specs'])


def create_PA_DU_PseudoDataSet():
    # Pancreas: 1, Duodenum: 2
    # process pancreas data first
    PA_train = read_split(osp.join(args.split_dir, 'Pancreas.npz'), 'train')
    PA_val = read_split(osp.join(args.split_dir, 'Pancreas.npz'), 'val')
    PA_test = read_split(osp.join(args.split_dir, 'Pancreas.npz'), 'test')

    DU_train = read_split(osp.join(args.split_dir, 'Duodenum.npz'), 'train')
    DU_val = read_split(osp.join(args.split_dir, 'Duodenum.npz'), 'val')
    DU_test = read_split(osp.join(args.split_dir, 'Duodenum.npz'), 'test')

    # Duodenum -> 2
    DU_val_dir = '/pct_wbo2/home/han.l/RO/Data/npz/DU_val'
    DU_test_dir = '/pct_wbo2/home/han.l/RO/Data/npz/DU_test'
    mkdir(DU_val_dir)
    mkdir(DU_test_dir)

    print('creating testing set for Duodenum (label from 1 to 2)...')
    DU_paths = glob('/pct_wbo2/home/han.l/RO/Data/npz/Duodenum' + '/*.npz')
    DU_train_subs = [osp.basename(tmp) for tmp in DU_train]
    DU_val_subs = [osp.basename(tmp) for tmp in DU_val]
    DU_test_subs = [osp.basename(tmp) for tmp in DU_test]

    with tqdm(total=len(DU_paths)) as pbar:
        for path in DU_paths:
            pbar.update(1)
            sub = osp.basename(path)
            if sub in DU_train_subs:
                continue
            
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']
            mask[mask != 0] = 2

            if sub in DU_val_subs:
                np.savez(
                osp.join(DU_val_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    
            elif sub in DU_test_subs:
                np.savez(
                osp.join(DU_test_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    
            

    R_PA_F_DU_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_PA_F_DU'
    F_DU_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/Duodenum/D811/F_DU'
    mkdir(R_PA_F_DU_dir)
    print('creating R_PA_F_DU training set...')
    with tqdm(total=len(PA_train)) as pbar:
        for path in PA_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # PA=1
            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_DU_dir, f'{sub}_pred.nii.gz')))
            fake_msk = get_cc3d(fake_msk)
            fake_msk[fake_msk == 1] = 2
            real_label = [1]
            mask[(mask==0) & (fake_msk==2)] = 2

            np.savez(osp.join(R_PA_F_DU_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)
    

    R_DU_F_PA_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_DU_F_PA'
    F_PA_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/Pancreas/D811/F_PA'
    mkdir(R_DU_F_PA_dir)
    print('creating R_DU_F_PA training set...')
    with tqdm(total=len(DU_train)) as pbar:
        for path in DU_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # DU=1
            mask[mask == 1] = 2  # DU=2
            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_PA_dir, f'{sub}_pred.nii.gz')))  # PA=1
            fake_msk = get_cc3d(fake_msk)
            mask[(mask==0) & (fake_msk==1)] = 1
            real_label = [2]
            np.savez(osp.join(R_DU_F_PA_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)

    train_paths = glob(R_DU_F_PA_dir + '/*.npz') + glob(R_PA_F_DU_dir + '/*.npz')
    val_paths = glob(DU_val_dir + '/*.npz') + PA_val.tolist()
    test_paths = glob(DU_test_dir + '/*.npz') + PA_test.tolist()
    np.savez(osp.join(args.split_dir, f'PA_DU.npz'), train=train_paths, val=val_paths, test=test_paths)




def create_WB1_WB2_PseudoDataSet():
    # WB1: 1, 2. WB2: 3, 4, 5
    # process pancreas data first
    WB1_train = read_split(osp.join(args.split_dir, 'WB1.npz'), 'train')
    WB1_val = read_split(osp.join(args.split_dir, 'WB1.npz'), 'val')
    WB1_test = read_split(osp.join(args.split_dir, 'WB1.npz'), 'test')

    WB2_train = read_split(osp.join(args.split_dir, 'WB2.npz'), 'train')
    WB2_val = read_split(osp.join(args.split_dir, 'WB2.npz'), 'val')
    WB2_test = read_split(osp.join(args.split_dir, 'WB2.npz'), 'test')

    # WB1_train = WB1_train[:int(len(WB1_train)/4)]  # *********
    # WB2_train = WB2_train[:int(len(WB2_train)/4)]  # *********

    # Duodenum -> 1, 2, 3 -> 3, 4, 5
    WB2_val_dir = '/pct_wbo2/home/han.l/RO/Data/npz/WB2_val'
    WB2_test_dir = '/pct_wbo2/home/han.l/RO/Data/npz/WB2_test'
    mkdir(WB2_val_dir)
    mkdir(WB2_test_dir)

    print('creating testing set for WB2 (label from 1,2,3 to 3,4,5)...')
    WB2_paths = glob('/pct_wbo2/home/han.l/RO/Data/npz/WB2' + '/*.npz')
    WB2_train_subs = [osp.basename(tmp) for tmp in WB2_train]
    WB2_val_subs = [osp.basename(tmp) for tmp in WB2_val]
    WB2_test_subs = [osp.basename(tmp) for tmp in WB2_test]

    with tqdm(total=len(WB2_paths)) as pbar:
        for path in WB2_paths:
            pbar.update(1)
            sub = osp.basename(path)
            if sub in WB2_train_subs:
                continue
            if sub == 'RS.XB031_CT1_CHM20210423.npz':
                continue
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']
            mask[mask == 3] = 5
            mask[mask == 2] = 4
            mask[mask == 1] = 3

            # assert np.unique(mask).tolist() == [0,3,4,5], f'{np.unique(mask).tolist()}'

            if sub in WB2_val_subs:
                np.savez(
                osp.join(WB2_val_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    
            elif sub in WB2_test_subs:
                np.savez(
                osp.join(WB2_test_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    
            

    R_WB1_F_WB2_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_WB1_F_WB2_G32'   # *********
    F_WB2_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/WB2/G32_1gpu/F_WB2'  # *********
    mkdir(R_WB1_F_WB2_dir)
    print('creating R_WB1_F_WB2 training set...')
    with tqdm(total=len(WB1_train)) as pbar:
        for path in WB1_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # 1,2
            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_WB2_dir, f'{sub}_pred.nii.gz')))
            
            fake_msk[fake_msk == 3] = 5
            fake_msk[fake_msk == 2] = 4
            fake_msk[fake_msk == 1] = 3
            
            real_label = [1, 2]

            mask[(mask==0) & (fake_msk==3)] = 3
            mask[(mask==0) & (fake_msk==4)] = 4
            mask[(mask==0) & (fake_msk==5)] = 5

            # assert np.unique(mask).tolist() == [0,1,2,3,4,5], f'{np.unique(mask).tolist()}'

            np.savez(osp.join(R_WB1_F_WB2_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)
    

    R_WB2_F_WB1_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_WB2_F_WB1_G32'  # *********
    F_WB1_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/WB1/G32_1gpu/F_WB1'  # *********
    mkdir(R_WB2_F_WB1_dir)
    print('creating R_WB2_F_WB1 training set...')
    with tqdm(total=len(WB2_train)) as pbar:
        for path in WB2_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # 1,2,3
            
            mask[mask == 3] = 5
            mask[mask == 2] = 4
            mask[mask == 1] = 3
            
            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_WB1_dir, f'{sub}_pred.nii.gz')))  
            mask[(mask==0) & (fake_msk==1)] = 1
            mask[(mask==0) & (fake_msk==2)] = 2

            # assert np.unique(mask).tolist() == [0,1,2,3,4,5], f'{np.unique(mask).tolist()}'

            real_label = [3,4,5]
            np.savez(osp.join(R_WB2_F_WB1_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)

    train_paths = glob(R_WB1_F_WB2_dir + '/*.npz') + glob(R_WB2_F_WB1_dir + '/*.npz')
    val_paths = glob(WB2_val_dir + '/*.npz') + WB1_val.tolist()
    test_paths = glob(WB2_test_dir + '/*.npz') + WB1_test.tolist()
    np.savez(osp.join(args.split_dir, f'WB1_WB2_G32.npz'), train=train_paths, val=val_paths, test=test_paths)   # *********


    # merge_split(split_paths, osp.join(args.split_dir, 'JA_CM.npz'))

    # save_dir1 = '/pct_wbo2/home/han.l/RO/Data/npz/R_JA_F_CM'
    # save_dir2 = '/pct_wbo2/home/han.l/RO/Data/npz/R_CM_F_JA'

    # train_paths = glob(save_dir1 + '/*.npz') + glob(save_dir2 + '/*.npz')
    # assign_split(train_paths, osp.join(args.split_dir, 'JA_CM.npz'), 'train', overwrite=True)

    # val_paths = read_split(osp.join(args.split_dir, 'JA_CM.npz'), 'val')
    # test_paths = read_split(osp.join(args.split_dir, 'JA_CM.npz'), 'test')

    # paths1 = []
    # for test_path in test_paths:
    #     if 'Constrictor_Muscles' in test_path:
    #         test_path = test_path.replace('/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/1mm/npz/Constrictor_Muscles', '/pct_wbo2/home/han.l/RO/Data/npz/CM_infer')
    #     paths1.append(test_path)

    # paths2 = []
    # for val_path in val_paths:
    #     if 'Constrictor_Muscles' in val_path:
    #         val_path = val_path.replace('/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/1mm/npz/Constrictor_Muscles', '/pct_wbo2/home/han.l/RO/Data/npz/CM_infer')
    #     paths2.append(val_path)

    # assign_split(paths1, osp.join(args.split_dir, 'JA_CM.npz'), 'test', overwrite=True)
    # assign_split(paths2, osp.join(args.split_dir, 'JA_CM.npz'), 'val', overwrite=True)
    
    
    # # # inspect data
    # x = read_split(osp.join(args.split_dir, 'JA_CM.npz'), 'test')

    # for path in x:
    #     data = dict(np.load(path, allow_pickle=True))
    #     # img = data['data_image']
    #     msk = data['data_mask']
    #     labels = np.unique(msk)
    #     print(labels)
    #     # if 10 not in labels or 11 not in labels or 12 not in labels:
    #     #     print(path)


def create_pelvic_dataset():
    PE_train_list = read_list('Pelvic6', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/train_Pelvic6_list.txt')
    PE_val_list = read_list('Pelvic6', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/test_Pelvic6_list.txt')
    PE_list = PE_train_list + PE_val_list

    # copy files to my folder
    PE1_dst = '/pct_wbo2/home/han.l/RO/Data/npz/PE1'
    PE2_dst = '/pct_wbo2/home/han.l/RO/Data/npz/PE2'

    mkdir(PE1_dst)
    mkdir(PE2_dst)

    PE1_list = PE_list[:int(len(PE_list)/2)]
    PE2_list = PE_list[int(len(PE_list)/2):]

    for path in PE1_list:
        print(path)
        shutil.copyfile(path, osp.join(PE1_dst, osp.basename(path)))

    PE1_list = [osp.join(PE1_dst, osp.basename(path)) for path in PE1_list]

    for path in PE2_list:
        print(path)
        shutil.copyfile(path, osp.join(PE2_dst, osp.basename(path)))

    PE2_list = [osp.join(PE2_dst, osp.basename(path)) for path in PE2_list]

    PE1_train, PE1_val, PE1_test = split_list(PE1_list)
    PE2_train, PE2_val, PE2_test = split_list(PE2_list)
    np.savez(osp.join(args.split_dir, f'PE1.npz'), train=PE1_train, val=PE1_val, test=PE1_test)
    np.savez(osp.join(args.split_dir, f'PE2.npz'), train=PE2_train, val=PE2_val, test=PE2_test)


def from_PE_to_PE12():
    """ 1,2,3,4: bladder, prostate, rectum, seminal
        5,6: femur left, femur right: 5,6 -> 1,2
    """
    for path in glob('/pct_wbo2/home/han.l/RO/Data/npz/PE1' + '/*.npz'):
        print(path)
        data = dict(np.load(path, allow_pickle=True))
        mask = data['data_mask']
        mask[mask ==4] = 0  # left femur
        mask[mask ==5] = 0  # right femur
        if 6 in mask:
            mask[mask==6]=4

        np.savez(
            path, 
            data_image=data['data_image'],
            data_mask=mask,
            data_specs=data['data_specs'])

    for path in glob('/pct_wbo2/home/han.l/RO/Data/npz/PE2' + '/*.npz'):
        print(path)
        data = dict(np.load(path, allow_pickle=True))
        mask = data['data_mask']
        mask[mask < 4] = 0
        mask[mask > 5] = 0
        mask[mask == 4] = 1
        mask[mask == 5] = 2
        np.savez(
            path, 
            data_image=data['data_image'],
            data_mask=mask,
            data_specs=data['data_specs'])


def create_PE1_PE2_PseudoDataSet():
    # PE1: 1, 2, 3, 4. PE2: 5,6
    # process PE1 data first
    PE1_train = read_split(osp.join(args.split_dir, 'PE1.npz'), 'train')
    PE1_val = read_split(osp.join(args.split_dir, 'PE1.npz'), 'val')
    PE1_test = read_split(osp.join(args.split_dir, 'PE1.npz'), 'test')

    PE2_train = read_split(osp.join(args.split_dir, 'PE2.npz'), 'train')
    PE2_val = read_split(osp.join(args.split_dir, 'PE2.npz'), 'val')
    PE2_test = read_split(osp.join(args.split_dir, 'PE2.npz'), 'test')

    PE2_val_dir = '/pct_wbo2/home/han.l/RO/Data/npz/PE2_val'
    PE2_test_dir = '/pct_wbo2/home/han.l/RO/Data/npz/PE2_test'
    mkdir(PE2_val_dir)
    mkdir(PE2_test_dir)

    print('creating testing set for PE2 (label from 1,2 to 5,6)...')
    PE2_paths = glob('/pct_wbo2/home/han.l/RO/Data/npz/PE2' + '/*.npz')
    PE2_train_subs = [osp.basename(tmp) for tmp in PE2_train]
    PE2_val_subs = [osp.basename(tmp) for tmp in PE2_val]
    PE2_test_subs = [osp.basename(tmp) for tmp in PE2_test]

    with tqdm(total=len(PE2_paths)) as pbar:
        for path in PE2_paths:
            pbar.update(1)
            sub = osp.basename(path)
            if sub in PE2_train_subs:
                continue
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']
            mask[mask == 1] = 5
            mask[mask == 2] = 6

            if sub in PE2_val_subs:
                np.savez(
                osp.join(PE2_val_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    

            elif sub in PE2_test_subs:
                np.savez(
                osp.join(PE2_test_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    
            
    R_PE1_F_PE2_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_PE1_F_PE2'
    F_PE2_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/PE2/D811/F_PE2'
    mkdir(R_PE1_F_PE2_dir)
    print('creating R_PE1_F_PE2 training set...')

    with tqdm(total=len(PE1_train)) as pbar:
        for path in PE1_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # 1,2,3,4
            real_label = [1, 2, 3, 4]

            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_PE2_dir, f'{sub}_pred.nii.gz')))
            fake_msk[fake_msk == 1] = 5
            fake_msk[fake_msk == 2] = 6
            
            mask[(mask==0) & (fake_msk==5)] = 5
            mask[(mask==0) & (fake_msk==6)] = 6

            np.savez(osp.join(R_PE1_F_PE2_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)
    
    R_PE2_F_PE1_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_PE2_F_PE1'
    F_PE1_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/PE1/D811/F_PE1'
    mkdir(R_PE2_F_PE1_dir)
    print('creating R_PE2_F_PE1 training set...')
    
    with tqdm(total=len(PE2_train)) as pbar:
        for path in PE2_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # 1,2
            mask[mask == 1] = 5
            mask[mask == 2] = 6
            real_label = [5,6]
            
            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_PE1_dir, f'{sub}_pred.nii.gz')))  
            mask[(mask==0) & (fake_msk==1)] = 1
            mask[(mask==0) & (fake_msk==2)] = 2
            mask[(mask==0) & (fake_msk==3)] = 3
            mask[(mask==0) & (fake_msk==4)] = 4
        
            np.savez(osp.join(R_PE2_F_PE1_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)

    train_paths = glob(R_PE1_F_PE2_dir + '/*.npz') + glob(R_PE2_F_PE1_dir + '/*.npz')
    val_paths = glob(PE2_val_dir + '/*.npz') + PE1_val.tolist()
    test_paths = glob(PE2_test_dir + '/*.npz') + PE1_test.tolist()
    np.savez(osp.join(args.split_dir, f'PE1_PE2.npz'), train=train_paths, val=val_paths, test=test_paths)


#--------------------------- Head & Neck ----------------------------------
def create_HN_dataset():
    # JA_list = sorted(read_list('Jaws', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/all_HN_list.txt', res='1mm'))
    EY_list = sorted(read_list('Eyes_LR', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/all_HN_list.txt', res='1mm'))
    ON_list = sorted(read_list('ON_LR', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/all_HN_list.txt', res='1mm'))

    half_len = int(len(EY_list)/2)
    
    EY_list = EY_list[:half_len]
    ON_list = ON_list[half_len:]

    # copy files to my folder
    # JA_dst = '/pct_wbo2/home/han.l/RO/Data/npz/JA'
    EY_dst = '/pct_wbo2/home/han.l/RO/Data/npz/EY'
    ON_dst = '/pct_wbo2/home/han.l/RO/Data/npz/ON'

    # mkdir(JA_dst)
    mkdir(EY_dst)
    mkdir(ON_dst)

    for path in EY_list:
        print(path)
        shutil.copyfile(path, osp.join(EY_dst, osp.basename(path)))

        # my_data = np.load(osp.join(EY_dst, osp.basename(path)), allow_pickle=True)
        # pdb.set_trace()

    EY_list = [osp.join(EY_dst, osp.basename(path)) for path in EY_list]

    for path in ON_list:
        print(path)
        shutil.copyfile(path, osp.join(ON_dst, osp.basename(path)))

    ON_list = [osp.join(ON_dst, osp.basename(path)) for path in ON_list]

    EY_train, EY_val, EY_test = split_list(EY_list)
    ON_train, ON_val, ON_test = split_list(ON_list)
    np.savez(osp.join(args.split_dir, f'EY.npz'), train=EY_train, val=EY_val, test=EY_test)
    np.savez(osp.join(args.split_dir, f'ON.npz'), train=ON_train, val=ON_val, test=ON_test)


def create_HN1_HN2_PseudoDataSet():
    # EY: 1, 2, 3, 4. ON: 5,6,7
    # process EY data first
    EY_train = read_split(osp.join(args.split_dir, 'EY.npz'), 'train')
    EY_val = read_split(osp.join(args.split_dir, 'EY.npz'), 'val')
    EY_test = read_split(osp.join(args.split_dir, 'EY.npz'), 'test')

    ON_train = read_split(osp.join(args.split_dir, 'ON.npz'), 'train')
    ON_val = read_split(osp.join(args.split_dir, 'ON.npz'), 'val')
    ON_test = read_split(osp.join(args.split_dir, 'ON.npz'), 'test')

    EY_train = EY_train[:int(len(EY_train)/4)]  # *********
    ON_train = ON_train[:int(len(ON_train)/4)]  # *********

    ON_val_dir = '/pct_wbo2/home/han.l/RO/Data/npz/ON_val'
    ON_test_dir = '/pct_wbo2/home/han.l/RO/Data/npz/ON_test'
    mkdir(ON_val_dir)
    mkdir(ON_test_dir)

    print('creating testing set for ON (label from 1,2,3 to 5,6,7)...')
    ON_paths = glob('/pct_wbo2/home/han.l/RO/Data/npz/ON' + '/*.npz')
    ON_train_subs = [osp.basename(tmp) for tmp in ON_train]
    ON_val_subs = [osp.basename(tmp) for tmp in ON_val]
    ON_test_subs = [osp.basename(tmp) for tmp in ON_test]

    with tqdm(total=len(ON_paths)) as pbar:
        for path in ON_paths:
            pbar.update(1)
            sub = osp.basename(path)
            if sub in ON_train_subs:
                continue
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']
            mask[mask == 1] = 5
            mask[mask == 2] = 6
            mask[mask == 3] = 7

            if sub in ON_val_subs:
                np.savez(
                osp.join(ON_val_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    

            elif sub in ON_test_subs:
                np.savez(
                osp.join(ON_test_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    
            
    R_EY_F_ON_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_EY_F_ON_0.25'                # *********
    F_ON_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/ON/0.25/F_ON'      # *********
    mkdir(R_EY_F_ON_dir)
    print('creating R_EY_F_ON training set...')

    with tqdm(total=len(EY_train)) as pbar:
        for path in EY_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # 1,2,3,4
            real_label = [1, 2, 3, 4]

            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_ON_dir, f'{sub}_pred.nii.gz')))
            fake_msk[fake_msk == 1] = 5
            fake_msk[fake_msk == 2] = 6
            fake_msk[fake_msk == 3] = 7
            
            mask[(mask==0) & (fake_msk==5)] = 5
            mask[(mask==0) & (fake_msk==6)] = 6
            mask[(mask==0) & (fake_msk==7)] = 7

            np.savez(osp.join(R_EY_F_ON_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)
    
    R_ON_F_EY_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_ON_F_EY_0.25'                 # *********
    F_EY_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/EY/0.25/F_EY'       # *********
    mkdir(R_ON_F_EY_dir)
    print('creating R_ON_F_EY training set...')
    
    with tqdm(total=len(ON_train)) as pbar:
        for path in ON_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # 1,2,3
            mask[mask == 1] = 5
            mask[mask == 2] = 6
            mask[mask == 3] = 7
            real_label = [5,6,7]
            
            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_EY_dir, f'{sub}_pred.nii.gz')))  
            mask[(mask==0) & (fake_msk==1)] = 1
            mask[(mask==0) & (fake_msk==2)] = 2
            mask[(mask==0) & (fake_msk==3)] = 3
            mask[(mask==0) & (fake_msk==4)] = 4
        
            np.savez(osp.join(R_ON_F_EY_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)

    train_paths = glob(R_EY_F_ON_dir + '/*.npz') + glob(R_ON_F_EY_dir + '/*.npz')
    val_paths = glob(ON_val_dir + '/*.npz') + EY_val.tolist()
    test_paths = glob(ON_test_dir + '/*.npz') + EY_test.tolist()
    np.savez(osp.join(args.split_dir, f'EY_ON_0.25.npz'), train=train_paths, val=val_paths, test=test_paths)  # *********

#--------------------------- Jaws1 & Jaws2 ----------------------------------

def create_Jaws_dataset():
    JA_list = sorted(read_list('Jaws', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/all_HN_list.txt', res='1mm'))
    
    JA1_list = JA_list[:int(len(JA_list)/2)]
    JA2_list = JA_list[int(len(JA_list)/2):]

    # copy files to my folder
    JA1_dst = '/pct_wbo2/home/han.l/RO/Data/npz/JA1'
    JA2_dst = '/pct_wbo2/home/han.l/RO/Data/npz/JA2'

    mkdir(JA1_dst)
    mkdir(JA2_dst)

    for path in JA1_list:
        print(path)
        shutil.copyfile(path, osp.join(JA1_dst, osp.basename(path)))

    JA1_list = [osp.join(JA1_dst, osp.basename(path)) for path in JA1_list]

    for path in JA2_list:
        print(path)
        shutil.copyfile(path, osp.join(JA2_dst, osp.basename(path)))

    JA2_list = [osp.join(JA2_dst, osp.basename(path)) for path in JA2_list]

    JA1_train, JA1_val, JA1_test = split_list(JA1_list)
    JA2_train, JA2_val, JA2_test = split_list(JA2_list)
    np.savez(osp.join(args.split_dir, f'JA1.npz'), train=JA1_train, val=JA1_val, test=JA1_test)
    np.savez(osp.join(args.split_dir, f'JA2.npz'), train=JA2_train, val=JA2_val, test=JA2_test)


def from_JA_to_JA12():
    """ 1,2,3,4,5 (was 1,4,5,8,9): Mandible Supraglottic_Larynx Glottis  Oral_Cavity Lips
        6,7,8,9 -> 1,2,3,4 (was 2,3,6,7): Parotid_Left Parotid_Right Submandibular_Left Submandibular_Right
    """
    for path in glob('/pct_wbo2/home/han.l/RO/Data/npz/JA1' + '/*.npz'):
        print(path)
        data = dict(np.load(path, allow_pickle=True))
        mask = data['data_mask']
        mask[mask ==2] = 0  
        mask[mask ==3] = 0  
        mask[mask ==6] = 0  
        mask[mask ==7] = 0  
        
        mask[mask==4]=2
        mask[mask==5]=3
        mask[mask==8]=4
        mask[mask==9]=5

        np.savez(
            path, 
            data_image=data['data_image'],
            data_mask=mask,
            data_specs=data['data_specs'])

    for path in glob('/pct_wbo2/home/han.l/RO/Data/npz/JA2' + '/*.npz'):
        print(path)
        data = dict(np.load(path, allow_pickle=True))
        mask = data['data_mask']
        
        mask[mask ==1] = 0  
        mask[mask ==4] = 0  
        mask[mask ==5] = 0  
        mask[mask ==8] = 0  
        mask[mask ==9] = 0  

        mask[mask==2]=1
        mask[mask==3]=2
        mask[mask==6]=3
        mask[mask==7]=4

        np.savez(
            path, 
            data_image=data['data_image'],
            data_mask=mask,
            data_specs=data['data_specs'])


def create_JA1_JA2_PseudoDataSet():
    # JA1: 1, 2, 3, 4, 5. JA2: 6,7,8,9
    # process PE1 data first
    JA1_train = read_split(osp.join(args.split_dir, 'JA1.npz'), 'train')
    JA1_val = read_split(osp.join(args.split_dir, 'JA1.npz'), 'val')
    JA1_test = read_split(osp.join(args.split_dir, 'JA1.npz'), 'test')

    JA2_train = read_split(osp.join(args.split_dir, 'JA2.npz'), 'train')
    JA2_val = read_split(osp.join(args.split_dir, 'JA2.npz'), 'val')
    JA2_test = read_split(osp.join(args.split_dir, 'JA2.npz'), 'test')

    JA1_train = JA1_train[:int(len(JA1_train)/4)]  # *********
    JA2_train = JA2_train[:int(len(JA2_train)/4)]  # *********

    JA2_val_dir = '/pct_wbo2/home/han.l/RO/Data/npz/JA2_val'
    JA2_test_dir = '/pct_wbo2/home/han.l/RO/Data/npz/JA2_test'
    mkdir(JA2_val_dir)
    mkdir(JA2_test_dir)

    print('creating testing set for JA2 (label from 1,2,3,4 to 6,7,8,9)...')
    JA2_paths = glob('/pct_wbo2/home/han.l/RO/Data/npz/JA2' + '/*.npz')
    JA2_train_subs = [osp.basename(tmp) for tmp in JA2_train]
    JA2_val_subs = [osp.basename(tmp) for tmp in JA2_val]
    JA2_test_subs = [osp.basename(tmp) for tmp in JA2_test]

    with tqdm(total=len(JA2_paths)) as pbar:
        for path in JA2_paths:
            pbar.update(1)
            sub = osp.basename(path)
            if sub in JA2_train_subs:
                continue
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']
            mask[mask == 1] = 6
            mask[mask == 2] = 7
            mask[mask == 3] = 8
            mask[mask == 4] = 9

            if sub in JA2_val_subs:
                np.savez(
                osp.join(JA2_val_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    

            elif sub in JA2_test_subs:
                np.savez(
                osp.join(JA2_test_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    
            
    R_JA1_F_JA2_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_JA1_F_JA2_0.25'            # *********
    F_JA2_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/JA2/0.25/F_JA2'   # *********
    mkdir(R_JA1_F_JA2_dir)
    print('creating R_JA1_F_JA2 training set...')

    with tqdm(total=len(JA1_train)) as pbar:
        for path in JA1_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # 1,2,3,4,5
            real_label = [1, 2, 3, 4, 5]

            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_JA2_dir, f'{sub}_pred.nii.gz')))
            fake_msk[fake_msk == 1] = 6
            fake_msk[fake_msk == 2] = 7
            fake_msk[fake_msk == 3] = 8
            fake_msk[fake_msk == 4] = 9
            
            mask[(mask==0) & (fake_msk==6)] = 6
            mask[(mask==0) & (fake_msk==7)] = 7
            mask[(mask==0) & (fake_msk==8)] = 8
            mask[(mask==0) & (fake_msk==9)] = 9

            np.savez(osp.join(R_JA1_F_JA2_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)
    
    R_JA2_F_JA1_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_JA2_F_JA1_0.25'            # *********
    F_JA1_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/JA1/0.25/F_JA1'   # *********
    mkdir(R_JA2_F_JA1_dir)
    print('creating R_JA2_F_JA1 training set...')
    
    with tqdm(total=len(JA2_train)) as pbar:
        for path in JA2_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # 1,2,3,4
            mask[mask == 1] = 6
            mask[mask == 2] = 7
            mask[mask == 3] = 8
            mask[mask == 4] = 9
            real_label = [6,7,8,9]
            
            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_JA1_dir, f'{sub}_pred.nii.gz')))  
            mask[(mask==0) & (fake_msk==1)] = 1
            mask[(mask==0) & (fake_msk==2)] = 2
            mask[(mask==0) & (fake_msk==3)] = 3
            mask[(mask==0) & (fake_msk==4)] = 4
            mask[(mask==0) & (fake_msk==5)] = 5
        
            np.savez(osp.join(R_JA2_F_JA1_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)

    train_paths = glob(R_JA1_F_JA2_dir + '/*.npz') + glob(R_JA2_F_JA1_dir + '/*.npz')
    val_paths = glob(JA2_val_dir + '/*.npz') + JA1_val.tolist()
    test_paths = glob(JA2_test_dir + '/*.npz') + JA1_test.tolist()
    np.savez(osp.join(args.split_dir, f'JA1_JA2_0.25.npz'), train=train_paths, val=val_paths, test=test_paths)  # *********



#--------------------------- Eye & Optic Nerve ----------------------------------
def create_OE_dataset():
    # ON: 1,2,3  EY: 4,5,6,7
    OE_list = sorted(read_list('Orbit', '/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/all_Orbit_list.txt', res='1mm'))
    half = int(len(OE_list)/2)
    
    OE1_list = OE_list[:half]
    OE2_list = OE_list[half:]

    # copy files to my folder
    OE1_dst = '/pct_wbo2/home/han.l/RO/Data/npz/OE1'
    OE2_dst = '/pct_wbo2/home/han.l/RO/Data/npz/OE2'

    mkdir(OE1_dst)
    mkdir(OE2_dst)

    for path in OE1_list:
        print(path)
        shutil.copyfile(path, osp.join(OE1_dst, osp.basename(path)))

    OE1_list = [osp.join(OE1_dst, osp.basename(path)) for path in OE1_list]

    for path in OE2_list:
        print(path)
        shutil.copyfile(path, osp.join(OE2_dst, osp.basename(path)))

    OE2_list = [osp.join(OE2_dst, osp.basename(path)) for path in OE2_list]

    OE1_train, OE1_val, OE1_test = split_list(OE1_list)
    OE2_train, OE2_val, OE2_test = split_list(OE2_list)
    np.savez(osp.join(args.split_dir, f'OE1.npz'), train=OE1_train, val=OE1_val, test=OE1_test)
    np.savez(osp.join(args.split_dir, f'OE2.npz'), train=OE2_train, val=OE2_val, test=OE2_test)

    for path in glob('/pct_wbo2/home/han.l/RO/Data/npz/OE1' + '/*.npz'):
        print(path)
        data = dict(np.load(path, allow_pickle=True))
        mask = data['data_mask']
        mask[mask > 3] = 0
        np.savez(
            path, 
            data_image=data['data_image'],
            data_mask=mask,
            data_specs=data['data_specs'])

    for path in glob('/pct_wbo2/home/han.l/RO/Data/npz/OE2' + '/*.npz'):
        print(path)
        data = dict(np.load(path, allow_pickle=True))
        mask = data['data_mask']
        mask[mask < 4] = 0
        mask[mask != 0] -= 3  # label: 4,5,6,7 -> 1,2,3,4

        np.savez(
            path, 
            data_image=data['data_image'],
            data_mask=mask,
            data_specs=data['data_specs'])


def create_OE_PseudoDataSet():
    # ON: 1,2,3  EY: 4,5,6,7

    OE1_train = read_split(osp.join(args.split_dir, 'OE1.npz'), 'train')
    OE1_val = read_split(osp.join(args.split_dir, 'OE1.npz'), 'val')
    OE1_test = read_split(osp.join(args.split_dir, 'OE1.npz'), 'test')

    OE2_train = read_split(osp.join(args.split_dir, 'OE2.npz'), 'train')
    OE2_val = read_split(osp.join(args.split_dir, 'OE2.npz'), 'val')
    OE2_test = read_split(osp.join(args.split_dir, 'OE2.npz'), 'test')

    OE2_val_dir = '/pct_wbo2/home/han.l/RO/Data/npz/OE2_val'
    OE2_test_dir = '/pct_wbo2/home/han.l/RO/Data/npz/OE2_test'
    mkdir(OE2_val_dir)
    mkdir(OE2_test_dir)

    print('creating testing set for OE2 (label from 1,2,3,4 to 4,5,6,7)...')
    OE2_paths = glob('/pct_wbo2/home/han.l/RO/Data/npz/OE2' + '/*.npz')
    OE2_train_subs = [osp.basename(tmp) for tmp in OE2_train]
    OE2_val_subs = [osp.basename(tmp) for tmp in OE2_val]
    OE2_test_subs = [osp.basename(tmp) for tmp in OE2_test]

    with tqdm(total=len(OE2_paths)) as pbar:
        for path in OE2_paths:
            pbar.update(1)
            sub = osp.basename(path)
            if sub in OE2_train_subs:
                continue
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']
            mask[mask == 4] = 7
            mask[mask == 3] = 6
            mask[mask == 2] = 5
            mask[mask == 1] = 4

            if sub in OE2_val_subs:
                np.savez(
                osp.join(OE2_val_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    

            elif sub in OE2_test_subs:
                np.savez(
                osp.join(OE2_test_dir, osp.basename(path)), 
                data_image=data['data_image'],
                data_mask=mask,
                data_specs=data['data_specs'])    
            
    R_OE1_F_OE2_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_OE1_F_OE2'                    # *********
    F_OE2_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/OE2/D811/F_OE2'       # *********
    mkdir(R_OE1_F_OE2_dir)
    print('creating R_OE1_F_OE2 training set...')

    with tqdm(total=len(OE1_train)) as pbar:
        for path in OE1_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # 1,2,3
            real_label = [1, 2, 3]

            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_OE2_dir, f'{sub}_pred.nii.gz')))
            fake_msk[fake_msk == 4] = 7
            fake_msk[fake_msk == 3] = 6
            fake_msk[fake_msk == 2] = 5
            fake_msk[fake_msk == 1] = 4
            
            mask[(mask==0) & (fake_msk==4)] = 4
            mask[(mask==0) & (fake_msk==5)] = 5
            mask[(mask==0) & (fake_msk==6)] = 6
            mask[(mask==0) & (fake_msk==7)] = 7

            np.savez(osp.join(R_OE1_F_OE2_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)
    
    R_OE2_F_OE1_dir = '/pct_wbo2/home/han.l/RO/Data/npz/R_OE2_F_OE1'                 # *********
    F_OE1_dir = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/OE1/D811/F_OE1'       # *********
    mkdir(R_OE2_F_OE1_dir)
    print('creating R_OE2_F_OE1 training set...')
    
    with tqdm(total=len(OE2_train)) as pbar:
        for path in OE2_train:
            pbar.update(1)
            sub = osp.basename(path)[:-4]
            data = dict(np.load(path, allow_pickle=True))
            mask = data['data_mask']  # 1,2,3,4 

            mask[mask == 4] = 7
            mask[mask == 3] = 6
            mask[mask == 2] = 5
            mask[mask == 1] = 4

            real_label = [4,5,6,7]
            
            fake_msk = sitk.GetArrayFromImage(sitk.ReadImage(osp.join(F_OE1_dir, f'{sub}_pred.nii.gz')))  
            mask[(mask==0) & (fake_msk==1)] = 1
            mask[(mask==0) & (fake_msk==2)] = 2
            mask[(mask==0) & (fake_msk==3)] = 3
        
            np.savez(osp.join(R_OE2_F_OE1_dir, osp.basename(path)),
                 data_image=data['data_image'],
                 data_mask=mask,
                 data_specs=data['data_specs'],
                 real_label=real_label)

    train_paths = glob(R_OE1_F_OE2_dir + '/*.npz') + glob(R_OE2_F_OE1_dir + '/*.npz')
    val_paths = glob(OE2_val_dir + '/*.npz') + OE1_val.tolist()
    test_paths = glob(OE2_test_dir + '/*.npz') + OE1_test.tolist()
    np.savez(osp.join(args.split_dir, f'OE1_OE2.npz'), train=train_paths, val=val_paths, test=test_paths)  # *********


def significant_test():

    npz1 = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/OE1_OE2/PLT/results.npz'
    npz2 = '/pct_wbo2/home/han.l/RO/SegFrame/src/checkpoints/OE1_OE2/TAL/results.npz'

    result1 = dict(np.load(npz1, allow_pickle=True))
    result2 = dict(np.load(npz2, allow_pickle=True))

    dice1 = result1['dice']
    dice2 = result2['dice']

    
    
    import pdb
    pdb.set_trace()





if __name__ == "__main__":


    significant_test()

    # create_OE_dataset()
    # create_OE_PseudoDataSet()

    # create_pancreas_duodenum_dataset()
    # from_WB_to_DU()

    # create_whole_bowel_dataset()
    # from_WB_to_WB12()


    # create_pelvic_dataset()
    # from_PE_to_PE12()

    # create_HN_dataset()   # 124:62:63, 125:62:63


    # create_Jaws_dataset()   # 124:62:63, 125:62:63
    # from_JA_to_JA12()

    # create_PA_DU_PseudoDataSet()
    # create_WB1_WB2_PseudoDataSet()

    # create_PE1_PE2_PseudoDataSet()
    # create_HN1_HN2_PseudoDataSet()
    # create_JA1_JA2_PseudoDataSet()


    # xx = read_split(args.split_dir + '/Pancreas.npz', 'train')
    # yy = read_split(args.split_dir + '/Duodenum.npz', 'train')
    # pdb.set_trace()



    # save_split_uterus()
    # save_split_pelvic6()

    # save_split_whole_bowel()

    # train_paths = read_split(args.split_dir + '/Whole_Bowel.npz', 'train')
    # valid_paths = read_split(args.split_dir + '/Whole_Bowel.npz', 'val')
    # test_paths = read_split(args.split_dir + '/Whole_Bowel.npz', 'test')

    # import shutil

    # new_train_paths, new_val_paths, new_test_paths = [], [], []

    # for path in train_paths:
    #     new_path = osp.join('/pct_wbo2/home/han.l/RO/Data', 'Whole_Bowel', osp.basename(path))
    #     shutil.copyfile(path, new_path)
    #     new_train_paths.append(new_path)
    
    # for path in valid_paths:
    #     new_path = osp.join('/pct_wbo2/home/han.l/RO/Data', 'Whole_Bowel', osp.basename(path))
    #     shutil.copyfile(path, new_path)
    #     new_val_paths.append(new_path)

    # for path in test_paths:
    #     new_path = osp.join('/pct_wbo2/home/han.l/RO/Data', 'Whole_Bowel', osp.basename(path))
    #     shutil.copyfile(path, new_path)
    #     new_test_paths.append(new_path)

    # output_file = osp.join(args.split_dir, 'Whole_Bowel_new.npz')
    # np.savez(output_file, train=new_train_paths, val=new_val_paths, test=new_test_paths)









    # print(read_split(osp.join(args.split_dir, 'Pelvic6.npz'), 'test'))
    # print(read_split(osp.join(args.split_dir, '_Uterus.npz'), 'val'))

    # split_paths = [osp.join(args.split_dir, 'Pelvic6.npz'), osp.join(args.split_dir, '_Uterus.npz')]
    # merge_split(split_paths, osp.join(args.split_dir, 'UT_PE.npz'))

    # save_dir1 = '/pct_wbo2/home/han.l/RO/Data/npz/R_PE_F_UT'
    # save_dir2 = '/pct_wbo2/home/han.l/RO/Data/npz/R_UT_F_PE'

    # paths = glob(save_dir1 + '/*.npz') + glob(save_dir2 + '/*.npz')
    # print(len(paths))

    # assign_split(paths, osp.join(args.split_dir, 'UT_PE.npz'), 'train')

    # UT_infer = '/pct_wbo2/home/han.l/RO/Data/npz/UT_infer'
    # # val_paths = read_split(osp.join(args.split_dir, 'UT_PE_ST.npz'), 'val')
    # test_paths = read_split(osp.join(args.split_dir, 'UT_PE_ST.npz'), 'test')
    
    # paths = []
    # for test_path in test_paths:
    #     if '_Uterus' in test_path:
    #         test_path = test_path.replace('/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/2mm/npz/_Uterus', '/pct_wbo2/home/han.l/RO/Data/npz/UT_infer')
    #     paths.append(test_path)

    # assign_split(paths, osp.join(args.split_dir, 'UT_PE_ST.npz'), 'test')
    # # print(read_split(osp.join(args.split_dir, 'UT_PE_ST_new.npz'), 'val'))
    
    # inspect data
    # x = read_split(osp.join(args.split_dir, 'UT_PE_ST.npz'), 'train')

    # for path in x:
    #     data = dict(np.load(path, allow_pickle=True))
    #     img = data['data_image']
    #     msk = data['data_mask']
    #     print(np.unique(msk))
    
    
    # split_path = '/pct_wbo2/home/han.l/RO/Data/splits/Pelvic6.npz'
    # output_dir = '/pct_wbo2/home/han.l/RO/Data/splits'

    # generate_list_from_npz(split_path, output_dir)


    # save_split('Jaws', '1mm')
    # save_split('Constrictor_Muscles', '1mm')

    # save_split('Whole_Bowel', '2mm')
    # save_split('Stomach', '2mm')
    # save_split('Pancreas', '2mm')
    
    # save_split('A_Pulmonary', '2mm')
    # save_split('V_Venacava_I', '2mm')
    # save_split('V_Venacava_S', '2mm')

    # merge_split(['/pct_wbo2/home/han.l/RO/Data/splits/Jaws.npz', '/pct_wbo2/home/han.l/RO/Data/splits/Constrictor_Muscles.npz'], '/pct_wbo2/home/han.l/RO/Data/splits/JA_CM.npz')
    # merge_split(['/pct_wbo2/home/han.l/RO/Data/splits/Whole_Bowel.npz', '/pct_wbo2/home/han.l/RO/Data/splits/Stomach.npz', '/pct_wbo2/home/han.l/RO/Data/splits/Pancreas.npz'], '/pct_wbo2/home/han.l/RO/Data/splits/WB_ST_PA.npz')
    

    #----------------------
    # split_paths = [osp.join(args.split_dir, 'Constrictor_Muscles.npz'), osp.join(args.split_dir, 'Jaws.npz')]
    # merge_split(split_paths, osp.join(args.split_dir, 'JA_CM.npz'))

    # save_dir1 = '/pct_wbo2/home/han.l/RO/Data/npz/R_JA_F_CM'
    # save_dir2 = '/pct_wbo2/home/han.l/RO/Data/npz/R_CM_F_JA'

    # train_paths = glob(save_dir1 + '/*.npz') + glob(save_dir2 + '/*.npz')
    # assign_split(train_paths, osp.join(args.split_dir, 'JA_CM.npz'), 'train', overwrite=True)

    # val_paths = read_split(osp.join(args.split_dir, 'JA_CM.npz'), 'val')
    # test_paths = read_split(osp.join(args.split_dir, 'JA_CM.npz'), 'test')

    # paths1 = []
    # for test_path in test_paths:
    #     if 'Constrictor_Muscles' in test_path:
    #         test_path = test_path.replace('/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/1mm/npz/Constrictor_Muscles', '/pct_wbo2/home/han.l/RO/Data/npz/CM_infer')
    #     paths1.append(test_path)

    # paths2 = []
    # for val_path in val_paths:
    #     if 'Constrictor_Muscles' in val_path:
    #         val_path = val_path.replace('/pct_wbo2/projects/RO/RO_Contouring/Data_Processed/1mm/npz/Constrictor_Muscles', '/pct_wbo2/home/han.l/RO/Data/npz/CM_infer')
    #     paths2.append(val_path)

    # assign_split(paths1, osp.join(args.split_dir, 'JA_CM.npz'), 'test', overwrite=True)
    # assign_split(paths2, osp.join(args.split_dir, 'JA_CM.npz'), 'val', overwrite=True)
    
    
    # # # inspect data
    # x = read_split(osp.join(args.split_dir, 'JA_CM.npz'), 'test')

    # for path in x:
    #     data = dict(np.load(path, allow_pickle=True))
    #     # img = data['data_image']
    #     msk = data['data_mask']
    #     labels = np.unique(msk)
    #     print(labels)
    #     # if 10 not in labels or 11 not in labels or 12 not in labels:
    #     #     print(path)