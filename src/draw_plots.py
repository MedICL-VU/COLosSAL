#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/11/2023

# Program description
# draw plots


import pdb
import os
import os.path as osp
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def spleen_uncertainty_analysis():
    dice = np.array([0.8585, 0.9028, 0.8939, 0.8672, 0.8525, 0.9053, 0.9147, 0.9242, 0.8943, 0.9042, 0.8658, 0.9051, 0.9129, 0.9080, 0.9122])
    g_entropy = np.array([0.09129692, 0.092948675, 0.10464831, 0.085223794, 0.09491223, 0.10974261, 0.106435776, 0.11157892, 0.111518525, 0.11274911, 0.09513488, 0.11098941, 0.11038907, 0.081416234, 0.096581414])
    g_variance = np.array([0.0011433993, 0.0009036509, 0.001144201, 0.0011357416, 0.0014791131, 0.0017007098, 0.0014950231, 0.0014704141, 0.0015946077, 0.0015199102, 0.0010681866, 0.0012240122, 0.0017462429, 0.0008680421, 0.0010016527])
    l_entropy = np.array([0.27785942, 0.38116935, 0.29105747, 0.2639644, 0.37536252, 0.3485272, 0.3518476, 0.33097348, 0.36697182, 0.31430963, 0.34322503, 0.34715474, 0.31931064, 0.25827584, 0.29686764])
    l_variance = np.array([0.0040166723, 0.004612474, 0.0039592516, 0.0047980826, 0.0073705777, 0.0070586703, 0.006207192, 0.005713237, 0.0073833787, 0.005069229, 0.0059525287, 0.0046999147, 0.0067926166, 0.0034153324, 0.003733403])
    
    sns.set_style("whitegrid")
    plt.rc('axes', labelsize=20)
    fig, axes = plt.subplots(1, 4, figsize=(40, 10))

    fig1 = sns.regplot(ax=axes[0], x=g_entropy, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_entropy, dice*100)
    axes[0].set_title(f"Global (Entropy) PCC={cor:.3f}", fontsize='30')
    axes[0].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig1.tick_params(axis='both', which='major', labelsize=18)
    
    fig2 = sns.regplot(ax=axes[1], x=g_variance, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_variance, dice*100)
    axes[1].set_title(f"Global (Variance) PCC={cor:.3f}", fontsize='30')
    axes[1].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig2.tick_params(axis='both', which='major', labelsize=18)
    
    fig3 = sns.regplot(ax=axes[2], x=l_entropy, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_entropy, dice*100)
    axes[2].set_title(f"Local (entropy) PCC={cor:.3f}", fontsize='30')
    axes[2].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig3.tick_params(axis='both', which='major', labelsize=18)

    fig4 = sns.regplot(ax=axes[3], x=l_variance, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_variance, dice*100)
    axes[3].set_title(f"Local (Variance) PCC={cor:.3f}", fontsize='30')
    axes[3].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig4.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(f"spleen_uncertainty_analysis.png") 


def liver_uncertainty_analysis():
    dice = np.array([0.4655, 0.5733, 0.6034, 0.5847, 0.5782, 0.5899, 0.4522, 0.5528, 0.5349, 0.5675, 0.6632, 0.4567, 0.6193, 0.6460, 0.5778])    
    g_entropy = np.array([0.16994461, 0.15326834, 0.16377884, 0.15394807, 0.14721793, 0.14507201, 0.17599979, 0.16518459, 0.15730733, 0.16524895, 0.15816024, 0.16242193, 0.17220479, 0.17166844, 0.1458245])
    g_variance = np.array([0.0012386818, 0.00075027754, 0.0012140225, 0.00075824896, 0.00080946466, 0.0011104695, 0.0014301207, 0.0012324725, 0.0009025319, 0.0012344951, 0.0007510452, 0.0010274216, 0.0012420744, 0.0012571291, 0.0006841867])
    l_entropy = np.array([0.5625321, 0.5682389, 0.5913079, 0.55045825, 0.5540843, 0.5386117, 0.5476116, 0.512255, 0.55345, 0.5034675, 0.5480553, 0.5010783, 0.53800046, 0.51829106, 0.5506566])
    l_variance = np.array([0.003701068, 0.002513959, 0.004155157, 0.0025396622, 0.0032074356, 0.0044763014, 0.0046175933, 0.0037679854, 0.0034001789, 0.0034806677, 0.0024169087, 0.003074429, 0.0037383283, 0.0035324525, 0.0027293658])
    
    sns.set_style("whitegrid")
    plt.rc('axes', labelsize=20)
    fig, axes = plt.subplots(1, 4, figsize=(40, 10))

    fig1 = sns.regplot(ax=axes[0], x=g_entropy, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_entropy, dice*100)
    axes[0].set_title(f"Global (Entropy) PCC={cor:.3f}", fontsize='30')
    axes[0].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig1.tick_params(axis='both', which='major', labelsize=18)
    
    fig2 = sns.regplot(ax=axes[1], x=g_variance, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_variance, dice*100)
    axes[1].set_title(f"Global (Variance) PCC={cor:.3f}", fontsize='30')
    axes[1].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig2.tick_params(axis='both', which='major', labelsize=18)
    
    fig3 = sns.regplot(ax=axes[2], x=l_entropy, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_entropy, dice*100)
    axes[2].set_title(f"Local (entropy) PCC={cor:.3f}", fontsize='30')
    axes[2].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig3.tick_params(axis='both', which='major', labelsize=18)

    fig4 = sns.regplot(ax=axes[3], x=l_variance, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_variance, dice*100)
    axes[3].set_title(f"Local (Variance) PCC={cor:.3f}", fontsize='30')
    axes[3].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig4.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(f"liver_uncertainty_analysis.png") 


def heart_uncertainty_analysis():

    dice = np.array([91.78, 93.02, 91.66, 91.53, 91.76, 92.68, 91.52, 92.05, 92.21, 91.53, 92.40, 92.57, 91.39, 92.44, 92.71])
    # criteria
    g_entropy = np.array([0.09624864, 0.10438952, 0.112696625, 0.107479796, 0.09908596, 0.10323209, 0.10586094, 0.10880481, 0.102086835, 0.10478308, 0.10547292, 0.09755077, 0.09970844, 0.09978076, 0.0993084])
    g_variance = np.array([0.0007410991, 0.0007787661, 0.0007852277, 0.0006981299, 0.0007505929, 0.0007901678, 0.00080709014, 0.0008880724, 0.00085439533, 0.0007861404, 0.0007766993, 0.00073064637, 0.0007763258, 0.0008860531, 0.0008081362])
    l_entropy = np.array([0.1499221, 0.15049969, 0.15507613, 0.17475882, 0.15753883, 0.15868041, 0.13786823, 0.1530365, 0.14532569, 0.13774836, 0.14960463, 0.16460678, 0.14865454, 0.14650092, 0.14159192])
    l_variance = np.array([0.0012267901, 0.0012823908, 0.001209656, 0.0012824732, 0.001314869, 0.0012151236, 0.0011785227, 0.0012883326, 0.0012991482, 0.0011873981, 0.0012311919, 0.0012872189, 0.00123623, 0.0012977166, 0.0012387078])

    sns.set_style("whitegrid")
    plt.rc('axes', labelsize=20)
    fig, axes = plt.subplots(1, 4, figsize=(40, 10))
    fig1 = sns.regplot(ax=axes[0], x=g_entropy, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_entropy, dice)
    axes[0].set_title(f"Global (Entropy) PCC={cor:.3f}", fontsize='30')
    axes[0].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig1.tick_params(axis='both', which='major', labelsize=18)

    fig2 = sns.regplot(ax=axes[1], x=g_variance, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_variance, dice)
    axes[1].set_title(f"Global (Variance) PCC={cor:.3f}", fontsize='30')
    axes[1].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig2.tick_params(axis='both', which='major', labelsize=18)

    fig3 = sns.regplot(ax=axes[2], x=l_entropy, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_entropy, dice)
    axes[2].set_title(f"Local (entropy) PCC={cor:.3f}", fontsize='30')
    axes[2].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig3.tick_params(axis='both', which='major', labelsize=18)

    fig4 = sns.regplot(ax=axes[3], x=l_variance, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_variance, dice)
    axes[3].set_title(f"Local (Variance) PCC={cor:.3f}", fontsize='30')
    axes[3].set(xlabel='Uncertainty Score', ylabel='Dice Score (%)')
    fig4.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(f"heart_uncertainty_analysis.png") 


def spleen_diveristy_analysis():
    dice = np.array([0.8585, 0.9028, 0.8939, 0.8672, 0.8525, 0.9053, 0.9147, 0.9242, 0.8943, 0.9042, 0.8658, 0.9051, 0.9129, 0.9080, 0.9122])
    g_km = np.array([3, 3, 3, 2, 3, 3, 4, 3, 3, 3, 4, 2, 3, 3, 3])
    g_agg = np.array([4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 4])
    l_km = np.array([3, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 5, 4, 4, 3])
    l_agg = np.array([2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 3])

    sns.set_style("whitegrid")
    plt.rc('axes', labelsize=20)
    fig, axes = plt.subplots(1, 4, figsize=(40, 10))

    fig1 = sns.regplot(ax=axes[0], x=g_km, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_km, dice*100)
    axes[0].set_title(f"Global (Kmeans) PCC={cor:.3f}", fontsize='30')
    axes[0].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig1.tick_params(axis='both', which='major', labelsize=18)
    
    fig2 = sns.regplot(ax=axes[1], x=g_agg, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_agg, dice*100)
    axes[1].set_title(f"Global (Agglo) PCC={cor:.3f}", fontsize='30')
    axes[1].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig2.tick_params(axis='both', which='major', labelsize=18)
    
    fig3 = sns.regplot(ax=axes[2], x=l_km, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_km, dice*100)
    axes[2].set_title(f"Local (Kmeans) PCC={cor:.3f}", fontsize='30')
    axes[2].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig3.tick_params(axis='both', which='major', labelsize=18)

    fig4 = sns.regplot(ax=axes[3], x=l_agg, y=dice*100, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_agg, dice*100)
    axes[3].set_title(f"Local (Agglo) PCC={cor:.3f}", fontsize='30')
    axes[3].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig4.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(f"spleen_diveristy_analysis.png") 


def heart_diveristy_analysis():
    dice = np.array([91.78, 93.02, 91.66, 91.53, 91.76, 92.68, 91.52, 92.05, 92.21, 91.53, 92.40, 92.57, 91.39, 92.44, 92.71])
    g_km = np.array([3, 3, 3, 4, 3, 4, 4, 4, 2, 4, 3, 3, 4, 4, 5])
    g_agg = np.array([3, 3, 3, 4, 3, 4, 4, 4, 2, 4, 3, 3, 4, 4, 5])
    l_km = np.array([4, 3, 3, 3, 3, 4, 3, 4, 3, 3, 4, 4, 2, 2, 3])
    l_agg = np.array([3, 4, 3, 3, 4, 5, 3, 4, 4, 3, 3, 4, 2, 2, 3])

    sns.set_style("whitegrid")
    plt.rc('axes', labelsize=20)
    fig, axes = plt.subplots(1, 4, figsize=(40, 10))

    fig1 = sns.regplot(ax=axes[0], x=g_km, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_km, dice)
    axes[0].set_title(f"Global (Kmeans) PCC={cor:.3f}", fontsize='30')
    axes[0].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig1.tick_params(axis='both', which='major', labelsize=18)
    
    fig2 = sns.regplot(ax=axes[1], x=g_agg, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_agg, dice)
    axes[1].set_title(f"Global (Agglo) PCC={cor:.3f}", fontsize='30')
    axes[1].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig2.tick_params(axis='both', which='major', labelsize=18)
    
    fig3 = sns.regplot(ax=axes[2], x=l_km, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_km, dice)
    axes[2].set_title(f"Local (Kmeans) PCC={cor:.3f}", fontsize='30')
    axes[2].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig3.tick_params(axis='both', which='major', labelsize=18)

    fig4 = sns.regplot(ax=axes[3], x=l_agg, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_agg, dice)
    axes[3].set_title(f"Local (Agglo) PCC={cor:.3f}", fontsize='30')
    axes[3].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig4.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(f"heart_diveristy_analysis.png") 


def liver_diveristy_analysis():
    dice = np.array([0.4655, 0.5733, 0.6034, 0.5847, 0.5782, 0.5899, 0.4522, 0.5528, 0.5349, 0.5675, 0.6632, 0.4567, 0.6193, 0.6460, 0.5778])
    g_km = np.array([3, 4, 3, 4, 3, 2, 4, 3, 2, 3, 3, 2, 3, 3, 3])
    g_agg = np.array([3, 3, 3, 4, 2, 2, 3, 2, 2, 3, 4, 2, 3, 2, 3])
    l_km = np.array([2, 3, 3, 2, 3, 4, 3, 5, 2, 3, 3, 3, 3, 3, 4])
    l_agg = np.array([2, 3, 2, 2, 3, 3, 3, 4, 2, 3, 3, 3, 3, 3, 3])

    sns.set_style("whitegrid")
    plt.rc('axes', labelsize=20)
    fig, axes = plt.subplots(1, 4, figsize=(40, 10))

    fig1 = sns.regplot(ax=axes[0], x=g_km, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_km, dice)
    axes[0].set_title(f"Global (Kmeans) PCC={cor:.3f}", fontsize='30')
    axes[0].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig1.tick_params(axis='both', which='major', labelsize=18)
    
    fig2 = sns.regplot(ax=axes[1], x=g_agg, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(g_agg, dice)
    axes[1].set_title(f"Global (Agglo) PCC={cor:.3f}", fontsize='30')
    axes[1].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig2.tick_params(axis='both', which='major', labelsize=18)
    
    fig3 = sns.regplot(ax=axes[2], x=l_km, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_km, dice)
    axes[2].set_title(f"Local (Kmeans) PCC={cor:.3f}", fontsize='30')
    axes[2].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig3.tick_params(axis='both', which='major', labelsize=18)

    fig4 = sns.regplot(ax=axes[3], x=l_agg, y=dice, scatter_kws={'s':120})
    cor, pval = stats.pearsonr(l_agg, dice)
    axes[3].set_title(f"Local (Agglo) PCC={cor:.3f}", fontsize='30')
    axes[3].set(xlabel='Diversity Score', ylabel='Dice Score (%)')
    fig4.tick_params(axis='both', which='major', labelsize=18)

    plt.savefig(f"liver_diveristy_analysis.png") 


def selection_comparison():
    plt.rc('axes', labelsize=12)
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    df = pd.DataFrame()
    my_palette = 'muted'
    
    df["organ"] = np.array(["Spleen"]*23)
    df["color"] = np.array(["Random"]*15 + ['Global entropy', 'Global variance', 'Local entropy', 'Local variance', 'Global K-means', 'Global Agglo', 'Local K-means', 'Local Agglo'])
    df['Dice Score (%)'] = np.array([0.8585, 0.9028, 0.8939, 0.8672, 0.8525, 0.9053, 0.9147, 0.9242, 0.8943, 0.9042, 0.8658, 0.9051, 0.9129, 0.9080, 0.9122,        0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90])
    fig0 = sns.stripplot(ax=axes[0], data=df, x="organ", y="Dice Score (%)", alpha=0.85, s=9, hue="color", palette=my_palette, jitter=True, linewidth=1, edgecolor='gray')
    fig0 = sns.boxplot(ax=axes[0], data=df, x="organ", y="Dice Score (%)", hue="color", palette=my_palette,fliersize=0)
    fig0.set(xlabel=None)
    axes[0].set_title(None)
    fig0.tick_params(axis='both', which='major', labelsize=10)
    fig0.get_legend().remove()
    handles, _ = fig0.get_legend_handles_labels()
    labels = ['Random', 'Global entropy', 'Global variance', 'Local entropy', 'Local variance', 'Global K-means', 'Global Agglo', 'Local K-means', 'Local Agglo']
    ylabels = ['{:,.2f}'.format(x) for x in fig0.get_yticks()]
    fig0.set_yticklabels(ylabels)
    l = plt.legend(handles[0:9], labels, bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0.)
    plt.tight_layout()

    df["organ"] = np.array(["Liver"]*23)
    df["color"] = np.array(["Random"]*15 + ['Global entropy', 'Global variance', 'Local entropy', 'Local variance', 'Global K-means', 'Global Agglo', 'Local K-means', 'Local Agglo'])
    df['Dice Score (%)'] = np.array([0.4655, 0.5733, 0.6034, 0.5847, 0.5782, 0.5899, 0.4522, 0.5528, 0.5349, 0.5675, 0.6632, 0.4567, 0.6193, 0.6460, 0.5778,         0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])
    fig1 = sns.stripplot(ax=axes[1], data=df, x="organ", y="Dice Score (%)", alpha=0.85, s=9, hue="color", palette=my_palette, jitter=True, linewidth=1, edgecolor='gray')
    fig1 = sns.boxplot(ax=axes[1], data=df, x="organ", y="Dice Score (%)", hue="color", palette=my_palette,fliersize=0)
    fig1.set(xlabel=None, ylabel=None)
    axes[1].set_title(None)
    fig1.tick_params(axis='both', which='major', labelsize=10)
    fig1.get_legend().remove()
    handles, _ = fig1.get_legend_handles_labels()
    labels = ['Random', 'Global entropy', 'Global variance', 'Local entropy', 'Local variance', 'Global K-means', 'Global Agglo', 'Local K-means', 'Local Agglo']
    ylabels = ['{:,.2f}'.format(x) for x in fig1.get_yticks()]
    fig1.set_yticklabels(ylabels)
    l = plt.legend(handles[0:9], labels, bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0.)
    plt.tight_layout()

    df["organ"] = np.array(["Heart"]*23)
    df["color"] = np.array(["Random"]*15 + ['Global entropy', 'Global variance', 'Local entropy', 'Local variance', 'Global K-means', 'Global Agglo', 'Local K-means', 'Local Agglo'])
    df['Dice Score (%)'] = np.array([0.9178, 0.9302, 0.9166, 0.9153, 0.9176, 0.9268, 0.9152, 0.9205, 0.9221, 0.9153, 0.9240, 0.9257, 0.9139, 0.9244, 0.9271,         0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92])
    fig2 = sns.stripplot(ax=axes[2], data=df, x="organ", y="Dice Score (%)", alpha=0.85, s=9, hue="color", palette=my_palette, jitter=True, linewidth=1, edgecolor='gray')
    fig2 = sns.boxplot(ax=axes[2], data=df, x="organ", y="Dice Score (%)", hue="color", palette=my_palette, fliersize=0)
    fig2.set(xlabel=None, ylabel=None)
    axes[2].set_title(None)
    fig2.tick_params(axis='both', which='major', labelsize=10)
    fig2.get_legend().remove()
    handles, _ = fig2.get_legend_handles_labels()
    ylabels = ['{:,.3f}'.format(x) for x in fig2.get_yticks()]
    fig2.set_yticklabels(ylabels)
    
    labels = ['Random', 'Global entropy', 'Global variance', 'Local entropy', 'Local variance', 'Global K-means', 'Global Agglo', 'Local K-means', 'Local Agglo']
    l = plt.legend(handles[0:9], labels, bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0.)
    plt.tight_layout()

    plt.savefig(f"selection_comparison.png", bbox_inches='tight') 


if __name__ == "__main__":

    selection_comparison()

    # spleen_uncertainty_analysis()
    # liver_uncertainty_analysis()
    # heart_uncertainty_analysis()
    # spleen_diveristy_analysis()
    # heart_diveristy_analysis()
    # liver_diveristy_analysis()