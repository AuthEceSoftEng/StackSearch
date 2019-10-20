#!/usr/bin/env python
"""
Used on k-means clustered fastText embeddings.
Samples and outputs a list of n_tokens from each cluster.
It's used to get a grasp of how meaningful the produced clusters are.
"""

import sys
from collections import OrderedDict

import pandas as pd


def print_sampled_clusters(dataframe, labels, n_clusters, n_tokens=20):
    token_index = list(pd.read_pickle(dataframe).index)
    with open(labels, 'r') as l:
        c_labels = []
        for row in l:
            c_labels.append(int(row.strip()))
    groups = OrderedDict()
    for i in range(n_clusters):
        groups[i] = []
    for ii, cluster in enumerate(c_labels):
        groups[cluster].append(token_index[ii])

    for i in range(n_clusters):
        print(i, '_', groups[i][:n_tokens])


if __name__ == '__main__':
    dataframe = sys.argv[1]
    labels = sys.argv[2]
    n_clusters = int(sys.argv[3])

    print_sampled_clusters(dataframe, labels, n_clusters)