#!/usr/bin/env python

import os
import argparse

from wordvec_models.utils import print_linked_posts
from wordvec_models.tfidf_model import TfIdfSearch
from wordvec_models.fasttext_model import FastTextSearch
from wordvec_models.hybrid_model import HybridSearch

## Default Index Keys
# Valid keys depend on the index_builder output
# Possible keys could include BodyV, TitleV, TagV
index_keys = ['BodyV', 'TitleV']


## Terminal print colors
class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


## Messages
DESC_MSG = colors.BOLD + colors.HEADER + 'StackSearch Demo' + colors.ENDC
PARSER_HEADER = colors.BLUE + colors.BOLD + '\nSearch model \'{}\'' + colors.ENDC


## Demo Functions
def fasttext_demo(args):
    ft = FastTextSearch(model_path=args.model_path,
                        index_path=args.index_path,
                        index_keys=index_keys,
                        metadata_path=args.metadata_path)

    ft.search()


def hybrid_demo(args):
    hy = HybridSearch(ft_model_path=args.ft_model_path,
                      ft_index_path=args.ft_index_path,
                      tfidf_model_path=args.tfidf_model_path,
                      tfidf_index_path=args.tfidf_index_path,
                      index_keys=index_keys,
                      metadata_path=args.metadata_path)

    hy.search()


def tfidf_demo(args):
    tfidf = TfIdfSearch(model_path=args.model_path,
                        index_path=args.index_path,
                        index_keys=index_keys,
                        metadata_path=args.metadata_path)

    tfidf.search()


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
                print(PARSER_HEADER.format(choice))
                print(subparser.format_help())

        parser.exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESC_MSG, add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action=_HelpAction,
                        help='show this help message and exit')
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True
    ft_parser = subparsers.add_parser(
        'fasttext', help='Use a FastText model for searching.')
    ft_parser.add_argument('model_path',
                           metavar='MODEL',
                           help='Path to the FastText model.')
    ft_parser.add_argument('index_path',
                           metavar='INDEX',
                           help='Path to the FastText search index.')
    ft_parser.add_argument('metadata_path',
                           metavar='METADATA',
                           help='Path to the metadata index.')
    ft_parser.add_argument('num_res',
                           metavar='RESULTS',
                           help='Number of results for each query.')
    tfidf_parser = subparsers.add_parser(
        'tfidf', help='Use a TF-IDF model for searching.')
    tfidf_parser.add_argument('model_path',
                              metavar='MODEL',
                              help='Path to the TF-IDF model.')
    tfidf_parser.add_argument('index_path',
                              metavar='INDEX',
                              help='Path to the TF-IDF search index.')
    tfidf_parser.add_argument('metadata_path',
                              metavar='METADATA',
                              help='Path to the metadata index.')
    tfidf_parser.add_argument('num_res',
                              metavar='RESULTS',
                              help='Number of results for each query.')
    hybrid_parser = subparsers.add_parser(
        'hybrid', help='Use a Hybrid model (FastText & TF-IDF) for searching.')
    hybrid_parser.add_argument('ft_model_path',
                               metavar='FASTTEXT MODEL',
                               help='Path to the FastText model.')
    hybrid_parser.add_argument('tfidf_model_path',
                               metavar='TFIDF MODEL',
                               help='Path to the TF-IDF model.')
    hybrid_parser.add_argument('ft_index_path',
                               metavar='FASTTEXT INDEX',
                               help='Path to the FastText search index.')
    hybrid_parser.add_argument('tfidf_index_path',
                               metavar='TFIDF INDEX',
                               help='Path to the TF-IDF search index.')
    hybrid_parser.add_argument('metadata_path',
                               metavar='METADATA',
                               help='Path to the metadata index.')
    hybrid_parser.add_argument('num_res',
                               metavar='RESULTS',
                               help='Number of results for each query.')

    args = parser.parse_args()

    if args.model == 'fasttext':
        print('FastText model')
        fasttext_demo(args)
    elif args.model == 'tfidf':
        print('TF-IDF model')
        tfidf_demo(args)
    elif args.model == 'hybrid':
        print('Hybrid model')
        hybrid_demo(args)
