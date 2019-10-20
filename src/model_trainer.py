#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
from wordvec_models.tfidf_model import load_text_list, train_tfidf_model
from wordvec_models.glove_model import build_glove_model
fastText_params = {}


def train_fasttext_model(corpus_path, params_path):
    pass


def main(corpus_path, export_dir, model_type, params_path=None):
    if model_type == 'ft':
        assert params_path
        train_fasttext_model(corpus_path, params_path)
    elif model_type == 'tfidf':
        model_path = os.path.join(export_dir, 'tfidf_v0.4.pkl')
        train_matrix_path = os.path.join(export_dir,
                                         'tfidf_train_matrix_v0.4.pkl')
        train_tfidf_model(
            load_text_list(corpus_path), model_path, train_matrix_path)
    elif model_type == 'glove':
        model_path = os.path.join(export_dir, 'glove_v0.1.1.pkl')
        index_path = os.path.join(export_dir, 'glove_v0.1.1_wordvec_index.pkl')
        build_glove_model(corpus_path, index_path, model_path)
    else:
        raise ValueError('Model type "{}" not recognized.'.format(model_type))


if __name__ == '__main__':
    main('wordvec_models/train_data/corpus_lemma_nsw', 'wordvec_models/tfidf_archive', 'tfidf')
