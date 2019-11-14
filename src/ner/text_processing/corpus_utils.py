#!/usr/bin/env python

import os
import re
import sys
import argparse

from spacy.lang.en import STOP_WORDS

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path)

from tokenizer import get_custom_tokenizer
import lib.twokenize

## RegEx for text cleaning
# 1. Large Binary Strings
# 2. Large Package Names
# 3. Very large strings
BIN_STR = re.compile(r'([01]{4,20}\.){3,30}[01]{4,20}')
PKG_NMS = re.compile(r'([a-zA-Z0-9_$]{2,50}\.){5,}[a-zA-Z0-9_$]{2,50}')
LRG_STR = re.compile(r'\S{30,}')


class CorpusUtils:
    """A class containing various functions for corpus pre processing."""

    def __init__(self, sent_split):
        """
        Args:
            sent_split: A boolean variable/flag denoting whether the spaCy
            tokenizer will use the parser module.
        """

        if sent_split:
            # keep parser: used for accurate sentence splitting
            self.nlp = get_custom_tokenizer(disable=['tagger', 'ner'])
        else:
            # default: disables parser, tagger, ner
            self.nlp = get_custom_tokenizer()

    def filter_line(self, line):
        line = BIN_STR.sub('', line)
        line = PKG_NMS.sub('', line)
        line = LRG_STR.sub('', line)
        return line

    def sentence_list(self, text):
        """
        """

        tok_text = ' '.join(lib.twokenize.tokenize(text[:]))
        tok_text = self.filter_line(tok_text)
        doc = self.nlp(tok_text)
        return [sent.text for sent in doc.sents]

    def sentence_feed(self, text):
        """
        A sentence iterator that yields tokenized sentences.
        
        Tokenizes text with lib.twokenize and splits it into sentences using
        a custom spaCy tokenizer.
        """

        tok_text = ' '.join(lib.twokenize.tokenize(text[:]))
        tok_text = self.filter_line(tok_text)
        doc = self.nlp(tok_text)
        for sent in doc.sents:
            yield sent.text

    def crf_sequence_feed(self, text):
        """Utility function preparing text data for the crfsuite python module.
        """

        for sent_str in self.sentence_feed(text):
            X = [{
                'F': [],
                'token': t,
                'entity': 'O'
            } for t in sent_str.split()]
            yield X

    def corpus_line_feed(self, corpus):
        with open(corpus, 'r') as c:
            for ii, line in enumerate(c):
                print('\r@line', ii + 1, end='')
                yield line.strip()
            print()

    def remove_corpus_stop_words(self, corpus, export_path):
        def is_int(string):
            try:
                int(string)
                return True
            except ValueError:
                return False

        stop_words = frozenset(STOP_WORDS)
        output_corpus = []
        with open(export_path, 'w') as out:
            for line in self.corpus_line_feed(corpus):
                new_line = []
                for token in line.strip().split():
                    if token not in stop_words and not is_int(
                            token) and len(token) > 1:
                        new_line.append(token)
                out.write(' '.join(new_line) + '\n')
                output_corpus.append()

    def normalize_corpus(self, corpus, export_path):
        def keep_token(token):
            return not (token.is_punct or token.is_bracket or token.is_quote or
                        token._.is_symbol or token.like_num or token.like_url
                        or token.like_email or token.is_space)

        with open(export_path, 'w') as out:
            for line in self.corpus_line_feed(corpus):
                new_line = ' '.join(
                    token.norm_ for token in self.nlp(line)
                    if keep_token(token))
                new_line = ' '.join(
                    token.norm_ for token in self.nlp(new_line)
                    if keep_token(token))
                out.write(new_line + '\n')

    def split_corpus_sentences(self, corpus, export_path):
        if export_path:  # write sentences to file
            with open(export_path, 'w') as out:
                for line in self.corpus_line_feed(corpus):
                    doc = self.nlp(line)
                    for sent in doc.sents:
                        out.write(sent.text + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Utility tools for named entity recognition.')
    parser.add_argument(
        'input', metavar='INPUT_PATH', help='The input corpus file path.')
    parser.add_argument(
        'output', metavar='OUTPUT_PATH', help='The output corpus file path.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-sw',
        '--stop-words',
        action='store_true',
        help='Remove stop words from the given corpus.')
    group.add_argument(
        '-n',
        '--norm',
        action='store_true',
        help=
        'Corpus normalization (remove punct/symbols, transform to base form words).'
    )
    group.add_argument(
        '-ss',
        '--sent-split',
        action='store_true',
        help='Split the given corpus into sentences using the spaCy module.')

    args = parser.parse_args()

    cu = CorpusUtils(sent_split=False)
    if args.stop_words:
        cu.remove_corpus_stop_words(args.input, args.output)
    elif args.norm:
        cu.normalize_corpus(args.input, args.output)
    elif args.sent_split:
        cu.split_corpus_sentences(args.input, args.output)
