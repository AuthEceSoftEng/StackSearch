#!/usr/bin/env python

import os
import argparse

from wordvec_models.utils import print_linked_posts
from wordvec_models.tfidf_model import TfIdfSearch
from wordvec_models.fasttext_model import FastTextSearch
from wordvec_models.hybrid_model import HybridSearch

# versions
ft_version = 'v0.6.1'
tfidf_version = 'v0.3'

## Paths
# Models
tfidf_model = 'wordvec_models/tfidf_archive/tfidf_' + tfidf_version + '.pkl'
fasttext_model = 'wordvec_models/fasttext_archive/ft_' + ft_version + '.bin'

# Indexes
fasttext_index = 'wordvec_models/index/ft_' + ft_version + '_post_index.pkl'
tfidf_index = os.path.realpath('wordvec_models/index/tfidf_' + tfidf_version +
                               '_post_index.pkl')

# WordVec Index
fasttext_wordvec_index = 'wordvec_models/index/ft_' + ft_version + '_wordvec_index.pkl'

# Metadata
metadata_path = os.path.realpath('wordvec_models/index/metadata.json')

# API Dict
api_dict = os.path.realpath('data/api_dict.pkl')


## Demo Functions
def fasttext_demo(api_labels=False):
    ft = None
    index_keys = ['BodyV', 'TitleV']

    if api_labels:
        ft = FastTextSearch(
            model_path=fasttext_model,
            index_path=fasttext_index,
            index_keys=index_keys,
            metadata_path=metadata_path,
            wordvec_index_path=fasttext_wordvec_index,
            api_dict_path=api_dict)
    else:
        ft = FastTextSearch(
            model_path=fasttext_model,
            index_path=fasttext_index,
            index_keys=index_keys,
            metadata_path=metadata_path)

    ft.search()  #postid_fn=print_linked_posts, api_labels=True)


def hybrid_demo():
    index_keys = ['BodyV', 'TitleV']

    hy = HybridSearch(
        ft_model_path=fasttext_model,
        ft_index_path=fasttext_index,
        tfidf_model_path=tfidf_model,
        tfidf_index_path=tfidf_index,
        index_keys=index_keys,
        metadata_path=metadata_path)

    hy.search()


def tfidf_demo():
    index_keys = ['BodyV', 'TitleV']
    tfidf = TfIdfSearch(
        model_path=tfidf_model,
        index_path=tfidf_index,
        index_keys=index_keys,
        metadata_path=metadata_path)

    tfidf.search()  #postid_fn=print_linked_posts)


class _HelpAction(argparse._HelpAction):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()

        # retrieve subparsers from parser
        subparsers_actions = [
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        for subparsers_action in subparsers_actions:
            # get all subparsers and print help
            for choice, subparser in subparsers_action.choices.items():
                print('\nSubparser \'{}\''.format(choice))
                print(subparser.format_help())

        parser.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='StackSearch demo script', add_help=False)
    parser.add_argument(
        '-h',
        '--help',
        action=_HelpAction,
        help='show this help message and exit')
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True
    ft_parser = subparsers.add_parser(
        'fasttext', help='Use a FastText model for searching.')
    ft_parser.add_argument('model_path', help='Path to the FastText model.')
    ft_parser.add_argument(
        'index_path', help='Path to the FastText search index.')
    ft_parser.add_argument('metadata_path', help='Path to the metadata index.')
    ft_parser.add_argument('num_res', help='Number of results for each query.')
    tfidf_parser = subparsers.add_parser(
        'tfidf', help='Use a TF-IDF model for searching.')
    tfidf_parser.add_argument('model_path', help='Path to the TF-IDF model.')
    tfidf_parser.add_argument(
        'index_path', help='Path to the TF-IDF search index.')
    tfidf_parser.add_argument(
        'metadata_path', help='Path to the metadata index.')
    tfidf_parser.add_argument(
        'num_res', help='Number of results for each query.')
    hybrid_parser = subparsers.add_parser(
        'hybrid', help='Use a Hybrid model (FastText & TF-IDF) for searching.')
    hybrid_parser.add_argument(
        'ft_model_path', help='Path to the FastText model.')
    hybrid_parser.add_argument(
        'tfidf_model_path', help='Path to the TF-IDF model.')
    hybrid_parser.add_argument(
        'ft_index_path', help='Path to the FastText search index.')
    hybrid_parser.add_argument(
        'tfidf_index_path', help='Path to the TF-IDF search index.')
    hybrid_parser.add_argument(
        'metadata_path', help='Path to the metadata index.')
    hybrid_parser.add_argument(
        'num_res', help='Number of results for each query.')

    args = parser.parse_args()

    if args.model == 'fasttext':
        print('FastText model')
        fasttext_demo()
    elif args.model == 'tfidf':
        print('TF-IDF model')
        tfidf_demo()
    elif args.model == 'hybrid':
        print('Hybrid model')
        hybrid_demo()
