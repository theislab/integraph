#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 14:01
# @Author  : Li Xiao
# @File    : SNF.py
import snf
import pandas as pd
import numpy as np
import argparse
import seaborn as sns
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True,
                        help='Location of input files')
    parser.add_argument('--metric', '-m', type=str, choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
                        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'], default='sqeuclidean',
                        help='Distance metric to compute. Must be one of available metrics in :py:func scipy.spatial.distance.pdist.')
    parser.add_argument('--K', '-k', type=int, default=20,
                        help='(0, N) int, number of neighbors to consider when creating affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 20.')
    parser.add_argument('--mu', '-mu', type=float, default=0.5,
                        help='(0, 1) float, Normalization factor to scale similarity kernel when constructing affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 0.5.')
    parser.add_argument('--transform', '-tr', type=str, default='', help='Where to save the output (name of the transformation applied).')
    parser.add_argument('--level', '-lev', type=str, default='', help='Level.')
    args = parser.parse_args()

    print('Load data files...')
    omics_data = []
    names = {}
    i = 0
    read_path = os.path.join(args.path, args.transform)
    for filename in os.listdir(read_path):
        if not filename == ".DS_Store":
            data_path = os.path.join(read_path, filename)
            omics_data.append(pd.read_csv(data_path, header=0, index_col=None))
            names[i] = filename
            i+=1
      

    print(omics_data[1].shape, omics_data[2].shape, omics_data[3].shape, omics_data[-1].shape, omics_data[-2].shape, omics_data[-3].shape)

    #if omics_data_1.shape[0] != omics_data_2.shape[0] or omics_data_1.shape[0] != omics_data_3.shape[0]:
    #    print('Input files must have same samples.')
    #    exit(1)

    omics_data = [data.rename(columns={data.columns.tolist()[0]: 'Sample'}) for data in omics_data]
    # align samples of different data
    omics_data = [data.sort_values(by='Sample', ascending=True) for data in omics_data]

    print('Start similarity network fusion...')

    affinity_nets = snf.make_affinity([omic.iloc[:, 1:].values.astype("float64") for omic in omics_data],
                                      metric=args.metric, K=args.K, mu=args.mu, normalize=False)

    fused_net =snf.snf(affinity_nets, K=args.K, t= 40)

    print('Save fused adjacency matrix...')
    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = omics_data[1]['Sample'].tolist()
    fused_df.index = omics_data[1]['Sample'].tolist()
    fused_df.to_csv('../result-level' + args.level + '/' + args.transform + '/SNF_fused_matrix.csv', header=True, index=True)



    np.fill_diagonal(fused_df.values, 0)
    fig = sns.clustermap(fused_df.iloc[:, :], cmap='vlag', figsize=(8,8),)
    fig.savefig('../result-level' + args.level + '/' + args.transform  + '/SNF_fused_clustermap.png', dpi=300)
    print('Success! Results can be seen in result file')