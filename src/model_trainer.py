#!/usr/bin/env python

import os
import sys
import json
import argparse
import subprocess
import multiprocessing

import fasttext
from wordvec_models.tfidf_model import load_text_list, train_tfidf_model
from wordvec_models.glove_model import build_glove_model


def main(corpus_path, export_path, model_type, training_params):
    if model_type == 'fasttext':
        model = fasttext.train_unsupervised(corpus_path, **training_params)
        model.save_model(export_path[0])
    elif model_type == 'tfidf':
        train_tfidf_model(load_text_list(corpus_path), export_path[0],
                          export_path[1])
    elif model_type == 'glove':
        # Uses a glove word - vector dump file to build and save a glove search model
        build_glove_model(corpus_path, export_path[1], export_path[0])


def param_parser(params_filepath, model_type):
    params = {
        'corpus_path': None,
        'export_path': None,
        'model_type': model_type,
        'training_params': None,
    }

    with open(params_filepath, 'r') as _in:
        params_dict = json.load(_in)

    if model_type not in ['fasttext', 'tfidf', 'glove']:
        print('Valid model types: [fasttext, tfidf, glove]')
        exit()
    m = params_dict['training'][model_type]
    params['corpus_path'] = params_dict['training']['corpus']

    if model_type == 'fasttext':
        model = m['export_name'] + '_' + m['export_ver'] + '.bin'
        params['export_path'] = [os.path.join(m['export_dir'], model)]
        params['training_params'] = m['model_params']
    elif model_type == 'tfidf':
        model = m['export_name'] + '_' + m['export_ver'] + '.pkl'
        matrix = m['export_name'] + '_' + m['export_ver'] + '_train_matrix.pkl'
        model = os.path.join(m['export_dir'], model)
        matrix = os.path.join(m['export_dir'], matrix)
        params['export_path'] = [model, matrix]
    else:
        model = m['export_name'] + '_' + m['export_ver'] + '.pkl'
        index = m['export_name'] + '_' + m['export_ver'] + '_wordvec_index.pkl'
        model = os.path.join(m['export_dir'], model)
        index = os.path.join(m['export_dir'], index)
        params['export_path'] = [model, index]

    return params


def validate_file(filepath):
    if not os.path.exists(filepath):
        print('File "{}" does not exist.'.format(filepath))
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model trainer.')
    parser.add_argument(
        'model_type',
        help='Model type to train. Types: [fasttext, tfidf, glove]')
    parser.add_argument(
        '-p',
        '--params',
        default='params.json',
        help='Path to a valid params file. (default: params.json)')

    args = parser.parse_args()
    validate_file(args.params)

    p = param_parser(args.params, args.model_type)
    print(p)
    exit()
    main(**p)
