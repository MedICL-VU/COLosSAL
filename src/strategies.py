#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/11/2023

# Program description
# Implementation of different data selection strategies, including
# (1) uncertainty-based (2) diversity-based (3) combined


import pdb
import os
import os.path as osp
import numpy as np
import random
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import seaborn as sns



def calculate_uncertainty_from_queries(organ, unc_file, seeds, num_labeled):
    def calculate_uncertainty_from_query(organ, unc_file, RS_seed, num_labeled):
        f = open(f'./data/{organ}/train.txt', 'r')
        lines = [item.replace('\n', '').split(",")[0] for item in f.readlines()]
        paths = [f'./data/{organ}/data/{fname}.npz' for fname in lines]
        random.Random(RS_seed).shuffle(paths)
        query = np.array(paths[:num_labeled])
        file = dict(np.load(unc_file, allow_pickle=True))
        scores, names = np.array(file['score_list']), np.array(file['name_list'])
        return scores[np.where(np.isin(names, query))].sum()

    uncs = []
    for seed in seeds:
        unc = calculate_uncertainty_from_query(organ, unc_file, seed, num_labeled)
        uncs.append(unc)
    print(uncs)


def generate_diversity_plan(organ, feats_file, num_labeled, method='kmeans', order='diverse'):
    random.seed(0)
    assert order in ['diverse', 'similar']
    scope = osp.basename(feats_file)[:-4]
    data_dict = dict(np.load(feats_file, allow_pickle=True))
    feats = np.array(data_dict['feats'])
    name_list = np.array(data_dict['name_list'])
    paths = []

    if method == 'kmeans':
        if order == 'diverse':
            km = KMeans(n_clusters=num_labeled, random_state=0).fit(feats)
            labs = km.labels_
            clost, _ = pairwise_distances_argmin_min(km.cluster_centers_, feats)
            pids = name_list[clost]
            paths = [osp.join(f'./data/{organ}/data/{pid}.npz') for pid in pids]

        elif order == 'similar':
            # Silhouette analysis to compute the optimal number of clusters
            range_n_clusters = list(range(2, num_labeled+1))
            best_score, best_n_clusters = -1, 0
            for n_clusters in range_n_clusters:
                km = KMeans(n_clusters=n_clusters, random_state=0)
                labs = km.fit_predict(feats)
                score = silhouette_score(feats, labs)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters

            print(f'Optimal number of clusters: {best_n_clusters}')
            km = KMeans(n_clusters=best_n_clusters, random_state=0)
            labs = km.fit_predict(feats)

            valid_labs = []
            for i in range(best_n_clusters):
                if (labs==i).sum() > num_labeled:
                    valid_labs.append(i)

            # randomly select one cluster 
            lab = random.choice(valid_labs)
            pids = np.random.choice(name_list[labs==lab], size=num_labeled, replace=False)
            paths = [osp.join(f'./data/{organ}/data/{pid}.npz') for pid in pids]

    elif method == 'agglomerative':
        if order == 'diverse':
            paths = []
            agg = AgglomerativeClustering(n_clusters=num_labeled).fit(feats)
            labs = agg.labels_
            for i in range(num_labeled):
                pid = random.choice(name_list[labs==i].tolist())
                path = osp.join(f'./data/{organ}/data/{pid}.npz')
                paths.append(path)

        elif order == 'similar':
            # Silhouette analysis to compute the optimal number of clusters
            range_n_clusters = list(range(2, num_labeled+1))
            best_score, best_n_clusters = -1, 0
            for n_clusters in range_n_clusters:
                agg = AgglomerativeClustering(n_clusters=n_clusters).fit(feats)
                labs = agg.labels_
                score = silhouette_score(feats, labs)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters

            print(f'Optimal number of clusters: {best_n_clusters}')
            agg = AgglomerativeClustering(n_clusters=best_n_clusters).fit(feats)
            labs = agg.labels_

            valid_labs = []
            for i in range(best_n_clusters):
                if (labs==i).sum() > num_labeled:
                    valid_labs.append(i)

            # randomly select one cluster (in practice we do not have info for the cluster of testing data)
            lab = random.choice(valid_labs)
            pids = np.random.choice(name_list[labs==lab], size=num_labeled, replace=False)
            paths = [osp.join(f'./data/{organ}/data/{pid}.npz') for pid in pids]

    print(f'Select:\n{paths}')
    save_path = f'./data/{organ}/plans/{scope}_{method}_{order}_{num_labeled}.npz'
    np.savez(save_path, paths=paths)
    print(f'Saved {num_labeled} most {order} samples into {save_path} using {method}')


def calculate_diversity_from_queries(organ, feats_file, seeds, num_labeled, method='kmeans'):
    def calculate_diversity_from_query(organ, feats_file, RS_seed, num_labeled, method):
        f = open(f'./data/{organ}/train.txt', 'r')
        lines = [item.replace('\n', '').split(",")[0] for item in f.readlines()]
        paths = [f'./data/{organ}/data/{fname}.npz' for fname in lines]
        random.Random(RS_seed).shuffle(paths)
        query = paths[:num_labeled]
        query = np.array([osp.basename(q)[:-4] for q in query])
        data_dict = dict(np.load(feats_file, allow_pickle=True))
        feats, name_list = np.array(data_dict['feats']), np.array(data_dict['name_list'])

        if method == 'kmeans':
            labs = KMeans(n_clusters=num_labeled, random_state=0).fit(feats).labels_
        elif method == 'agglomerative':
            labs = AgglomerativeClustering(n_clusters=num_labeled).fit(feats).labels_
        return len(np.unique(labs[np.where(np.isin(name_list, query))]))

    divs = []
    for seed in seeds:
        div = calculate_diversity_from_query(organ, feats_file, seed, num_labeled, method)
        divs.append(div)
    print(divs)


def generate_uncertainty_plan(organ, unc_path, order='uncertain'):
    assert order in ['uncertain', 'certain']
    unc_file = dict(np.load(unc_path, allow_pickle=True))
    name_list, score_list = unc_file['name_list'], unc_file['score_list']
    prefix = osp.basename(unc_path)[:-4]

    if order == 'uncertain':  # uncertainty: [5,4,3, ...]
        paths = [x for _, x in sorted(zip(score_list, name_list))][::-1]
    elif order == 'certain': # uncertainty: [1,2,3, ...]
        paths = [x for _, x in sorted(zip(score_list, name_list))]

    save_path = f'./data/{organ}/plans/{prefix}_{order}.npz'
    np.savez(save_path, paths=paths)
    print(f"Saved the paths of the top {prefix} training data into {save_path}")


def generate_diversity_plans():
    for organ in ["Spleen", "Liver", "Pancreas", "Heart", "Hippocampus"]:
        for scope in ["global", "local"]:
            for method in ["agglomerative", "kmeans"]:
                for order in ['diverse', 'similar']:
                    for num_labeled in [5]:
                        generate_diversity_plan(
                            organ=f'{organ}', 
                            feats_file=f'./data/{organ}/feats/{scope}.npz', 
                            num_labeled=num_labeled, 
                            method=method,
                            order=order)


def generate_uncertainty_plans():
    for organ in ["Spleen", "Liver", "Pancreas", "Heart", "Hippocampus"]:
        for scope in ["global", "local"]:
            for metric in ['entropy', 'variance', 'ReconVar']:
                for order in ['uncertain', 'certain']:
                    generate_uncertainty_plan(
                        organ=f'{organ}', 
                        unc_path=f'./data/{organ}/unc/{scope}_{metric}.npz', 
                        order=order)



if __name__ == "__main__":
    

    generate_diversity_plans()        
    # generate_uncertainty_plans()    
    
    # generate_uncertainty_plan('Spleen', f'/data/MIC23/src/data/Spleen/unc/global_variance.npz', order='uncertain')
    # generate_diversity_plan('Spleen', './data/Spleen/feats/global.npz', 5, method='kmeans', order='similar')

    # calculate_uncertainty_from_queries(
    #     organ=f'{organ}', 
    #     unc_file=f'/data/MIC23/src/data/{organ}/unc/local_mc_20_entropy.npz', 
    #     seeds=list(range(15)), 
    #     num_labeled=5)

    # calculate_diversity_from_queries(
    #     organ=f'{organ}', 
    #     feats_file=f'/data/MIC23/src/data/{organ}/feats/local.npz', 
    #     seeds=list(range(15)), 
    #     num_labeled=5,
    #     method='agglomerative')

