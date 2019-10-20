#!/usr/bin/env python

import re
import json
import numpy as np
from keras.preprocessing.text import Tokenizer

# RegEx patterns
ERRLINE_RE = re.compile(r':\d*\)')
keepPunct_RE = r'|\||\&|\#|\@|\~|\_|\'|\"|\=|\\|\/|\-|\:|\;|\*|\.|\$|\(|\)|\[|\]|\{|\}'
TOKEN_PATTERN = re.compile(r'(?u)\b\w\w+\b' + keepPunct_RE)

# Lambda functions
tokenize = lambda line: TOKEN_PATTERN.findall(line)
errline = lambda line: ERRLINE_RE.sub(':_xerrx_)', line)

# Out Of Vocabulary default token value
#oov = 11999


class Vectorizer:
    def __init__(self, oov_val=11999, dictionary_path=None):
        self.oov = oov_val
        self.token_dict = None
        if dictionary_path:
            self.token_dict = self.read_dict(dictionary_path)

    def read_dict(self, dict_path):
        with open(dict_path, 'r') as _in:
            return json.load(_in)

    def dump_dict(self, output_path):
        with open(output_path, 'w') as out:
            json.dump(self.token_dict, out, indent=2)

    def build_dict(self, doc_path):
        token = Tokenizer(filters='')
        posts = []
        with open(doc_path, 'r') as f:
            for line in f:
                posts.append(' '.join(tokenize(errline(line.strip().lower()))))
        # fit tokenizer on posts and create token index
        token.fit_on_texts(posts)
        self.token_dict = token.word_index

    def vectorize_doc(self, doc_path):
        # encode post tokens using the provided dictionary
        enc_doc = []
        with open(doc_path, 'r') as f:
            for line in f:
                enc_doc.append(self.vectorize_string(line))
        return enc_doc

    def vectorize_list(self, doc_list):
        enc_doc = []
        for line in doc_list:
            enc_doc.append(self.vectorize_string(line))
        return enc_doc

    def vectorize_string(self, string):
        string = ' '.join(tokenize(errline(string.strip().lower())))
        return np.array(
            [self.token_dict.get(t, self.oov) for t in string.split()])


if __name__ == '__main__':
    # build token dictionary on the selected and labeled posts
    vectorizer = Vectorizer()
    vectorizer.build_dict('training_data/raw_data/labeled_posts')
    vectorizer.dump_dict('data/token_dictionary.json')
