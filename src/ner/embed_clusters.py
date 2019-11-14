#!/usr/bin/env python

import os
import sys
import json
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

from fasttext import load_model


class EmbeddingClusterer:
    def __init__(self, c_path=None):
        if c_path:
            self.load_clusterer(c_path)

    def load_embeddings_from_pickle(self, filepath):
        with open(filepath, 'rb') as _in:
            self.embed_index = pickle.load(_in)

    def load_embeddings_from_text(self, filepath):
        print('loading embeddings from text file')
        with open(filepath, 'r') as embed_file:
            dimensions = [int(dim) for dim in embed_file.readline().split()]
            print('wordvec matrix dimensions:', dimensions)
            vec_matrix = np.zeros(dimensions, dtype=np.float32)
            index_tokens = []
            for idx, row in enumerate(embed_file):
                token, vector = row.split(' ', 1)
                vec_matrix[idx] = np.fromstring(vector, sep=' ')
                index_tokens.append(token)
        self.embed_index = pd.DataFrame(data=vec_matrix, index=index_tokens)
        self.embed_index.to_pickle('embed_index.pkl')

    def load_clusterer(self, path):
        with open(path, 'rb') as _in:
            self.c = pickle.load(_in)

    def save_clusterer(self, path):
        self.c.set_params(verbose=0)
        with open(path, 'wb') as out:
            pickle.dump(self.c, out)
        print('Clusterer model saved at', path)

    def train(self,
              n_clusters,
              max_iter,
              alg='simple',
              batch_size=100,
              init='k-means++',
              n_init=10,
              v=1):

        print('Training {} k-means'.format(alg))
        if alg == 'simple':
            self.c = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                n_jobs=-1,
                verbose=v)
        elif alg == 'mini-batch':
            self.c = MiniBatchKMeans(
                n_clusters=n_clusters,
                init=init,
                max_iter=max_iter,
                batch_size=batch_size,
                n_init=n_init,
                max_no_improvement=1000,
                reassignment_ratio=0.01,
                verbose=v)
        else:
            raise ValueError(
                'Clusterer can either be "simple" or "mini-batch" k-means.')

        self.c.fit_predict(self.embed_index.values)
        with open('labels_' + str(n_clusters), 'w') as lout:
            for l in self.c.labels_:
                lout.write(str(l) + '\n')

    def predict(self, vector_matrix):
        return self.c.predict(vector_matrix)


def _load_clusterers(mbkm_dir):
    mbkm_names = []
    mbkm_models = []
    for _file in os.listdir(mbkm_dir):
        if _file.endswith('.pkl'):
            name = _file.split('_')[-1][:-4]
            mbkm_names.append(name)
            with open(os.path.join(mbkm_dir, _file), 'rb') as c:
                mbkm_models.append(pickle.load(c))
    return mbkm_models


def token_ce_dict(token_dict, ft_model, mbkm_dir, min_freq):
    """Calculate the compound embeddings of each token in the given dict.

    Args:
        token_dict: A dictionary of tokens with their appearance frequency.
        ft_model: Path to the fastText model.
        mbkm_dir: Path to the MiniBatchKMeans models.
        min_freq: The minimum frequency a token needs to have for it to be
                  included in the output token-clusters dict.

    Returns:
        A dict with a list of compound embeddings (MiniBatchKMeans clusters)
        for each token.
    """
    with open(token_dict, 'r') as _in:
        token_dict = json.load(_in)

    fasttext = load_model(ft_model)
    mbkm_models = _load_clusterers(mbkm_dir)

    token_clusters = {}
    for token in token_dict.keys():
        if token_dict[token] >= min_freq:
            cluster_list = []
            token_vec = fasttext.get_word_vector(token).reshape(1, -1)
            for mbkm in mbkm_models:
                cluster_list.append(str(mbkm.predict(token_vec)[0]))
            token_clusters[token] = cluster_list
    return token_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        '''Module for training MiniBatchKMeans (or KMeans) models on word embeddings.
        Provides utility functions for calculating compound embedding features.'''
    )
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    train_parser = subparsers.add_parser(
        'train', help='Train MiniBatchKMeans/KMeans clusterers.')
    train_parser.add_argument(
        'embed_file',
        help=
        'Path to the fastText embeddings file. (".vec": text in GloVe format, ".pkl": pickled dict)'
    )
    train_parser.add_argument(
        'num_clusters', type=int, help='Number of clusters for the model.')
    train_parser.add_argument(
        'cluster_type', help='Type of cluster (mini-batch/simple k-means)')
    train_parser.add_argument(
        'output_dir', help='Output directory for the trained model.')
    ce_parser = subparsers.add_parser(
        'ce', help='Calculate compound embedding features.')
    ce_parser.add_argument(
        'token_dict',
        help='Dictionary of tokens with their appearance frequencies.')
    ce_parser.add_argument(
        'fasttext', help='Path to a trained fastText model.')
    ce_parser.add_argument(
        'mbkm_dir',
        help='Path to the directory of the MiniBatchKMeans models.')
    ce_parser.add_argument(
        'min_freq',
        type=int,
        help='Min appeqrance frequency for the token to be considered.')
    ce_parser.add_argument(
        'output_dir',
        help='Output directory for the compound embeddings dict.')

    args = parser.parse_args()

    if args.command == 'train':
        ec = EmbeddingClusterer()

        if args.embed_file[-3:] == 'vec':
            ec.load_embeddings_from_text(args.embed_file)
        elif args.embed_file[-3:] == 'pkl':
            ec.load_embeddings_from_pickle(args.embed_file)

        ec.train(int(args.num_clusters), 100, args.cluster_type)
        ec.save_clusterer(args.output_dir)
    elif args.command == 'ce':
        token_ce = token_ce_dict(args.token_dict, args.fasttext, args.mbkm_dir,
                                 args.min_freq)

        output_file = 'ce_feat_min' + str(args.min_freq) + '.json'
        with open(os.path.join(args.output_dir, output_file), 'w') as out:
            json.dump(token_ce, out)
