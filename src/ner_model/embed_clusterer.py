#!/usr/bin/env python

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans


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


if __name__ == '__main__':
    embed_file = sys.argv[1]
    clusters = sys.argv[2]
    c_type = sys.argv[3]
    output_dir = sys.argv[4]

    c_name = c_type + '_k_means_' + clusters + '.pkl'
    c_output = os.path.join(output_dir, c_name)

    print('args:', embed_file, clusters, c_type, c_name)

    ec = EmbeddingClusterer()

    if embed_file[-3:] == 'vec':
        ec.load_embeddings_from_text(embed_file)
    elif embed_file[-3:] == 'pkl':
        ec.load_embeddings_from_pickle(embed_file)

    ec.train(int(clusters), 100, c_type)
    ec.save_clusterer(c_output)
