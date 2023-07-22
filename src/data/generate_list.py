import os
import os.path as osp
from glob import glob
import numpy as np
import random


# def generate_datasplit(data_dir):
#     os.mkdir(osp.join(data_dir, 'plans'))
#     paths = sorted(glob(data_dir + '/data/*.*'))
#     data_num = len(paths)
#     train_num = np.round(data_num * 0.6)
#     val_num = np.round(data_num * 0.2)
#     test_num = data_num - train_num - val_num

#     f1 = open(osp.join(data_dir, 'train.txt'), 'w')
#     f2 = open(osp.join(data_dir, 'val.txt'), 'w')
#     f3 = open(osp.join(data_dir, 'test.txt'), 'w')

#     for i, path in enumerate(paths):
#         if i < train_num:
#             f1.write(f'{osp.basename(path).split(".")[0]}\n')
#         elif train_num < i and i < train_num+val_num:
#             f2.write(f'{osp.basename(path).split(".")[0]}\n')
#         else:
#             f3.write(f'{osp.basename(path).split(".")[0]}\n')


def generate_datasplit(data_dir):
    if not osp.exists(osp.join(data_dir, 'plans')):
        os.mkdir(osp.join(data_dir, 'plans'))
        
    paths = sorted(glob(data_dir + '/data/*.*'))
    data_num = len(paths)
    train_num = np.round(data_num * 0.8)
    val_num = data_num - train_num

    f1 = open(osp.join(data_dir, 'train.txt'), 'w')
    f2 = open(osp.join(data_dir, 'val.txt'), 'w')

    for i, path in enumerate(paths):
        if i < train_num:
            f1.write(f'{osp.basename(path).split(".")[0]}\n')
        else:
            f2.write(f'{osp.basename(path).split(".")[0]}\n')


def random_select_labeled(data_dir, times=5):
    for seed in range(times):
        f = open(osp.join(data_dir, 'train.txt'), 'r')
        image_list = f.readlines()
        image_list = [item.replace('\n', '').split(",")[0] for item in image_list]
        random.Random(seed).shuffle(image_list)
        
        f_out = open(osp.join(data_dir, f'random_{seed}.txt'), 'w')
        for name in image_list:
            f_out.write(f'{name}\n')




if __name__ == "__main__":

    generate_datasplit(f'./Liver')
    generate_datasplit(f'./Pancreas')
    generate_datasplit(f'./HepaticVessel')




