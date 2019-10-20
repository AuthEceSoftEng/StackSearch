#!/usr/bin/env python

import os
import sys

from spacy.lang.en import STOP_WORDS

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path)

from tokenizer import get_custom_tokenizer
import lib.twokenize


def corpus_line_feed(corpus):
    with open(corpus, 'r') as c:
        for ii, line in enumerate(c):
            print('\rline:', ii + 1, end='')
            yield line.strip()
        print()


def remove_stop_words(corpus, export_path):
    def is_int(string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    stop_words = frozenset(STOP_WORDS)
    output_corpus = []
    with open(export_path, 'w') as out:
        for line in corpus_line_feed(corpus):
            new_line = []
            for token in line.strip().split():
                if token not in stop_words and not is_int(
                        token) and len(token) > 1:
                    new_line.append(token)
            out.write(' '.join(new_line) + '\n')
            output_corpus.append()


def normalize_corpus(corpus, export_path):
    def keep_token(token):
        return not (token.is_punct or token.is_bracket or token.is_quote
                    or token._.is_symbol or token.like_num or token.like_url
                    or token.like_email or token.is_space)

    nlp = get_custom_tokenizer()

    with open(export_path, 'w') as out:
        for line in corpus_line_feed(corpus):
            new_line = ' '.join(
                token.norm_ for token in nlp(line) if keep_token(token))
            new_line = ' '.join(
                token.norm_ for token in nlp(new_line) if keep_token(token))
            out.write(new_line + '\n')


def ner_sentence_feed(text_data):
    nlp = get_custom_tokenizer(disable=['tagger', 'ner'])
    tok_data = ' '.join(lib.twokenize.tokenize(text_data[:]))
    doc = nlp(tok_data)
    for sent in doc.sents:
        yield str(sent)


def crf_sequence_feed(text_data):
    nlp = get_custom_tokenizer(disable=['tagger', 'ner'])
    tok_data = ' '.join(lib.twokenize.tokenize(text_data[:]))
    doc = nlp(tok_data)
    for sent in doc.sents:
        X = [{'F': [], 'token': t, 'entity': 'O'} for t in sent.text.split()]
        yield X


def ner_prepare_file(filename, output):
    corpus = []
    with open(filename, 'r') as _in:
        for line in _in:
            corpus.append(line.strip())

    with open(output, 'w') as out:
        for line in corpus:
            for sent in ner_sentence_feed(line):
                for token in sent.split():
                    out.write(token + ' O\n')
                out.write('\n')


def sentence_splitting(corpus, export_path=None):
    # keep parser disable tagger and ner
    # parser is needed for more accurate sentence splitting.
    nlp = get_custom_tokenizer(disable=['tagger', 'ner'])

    if export_path:  # write sentences to file
        with open(export_path, 'w') as out:
            for line in corpus_line_feed(corpus):
                doc = nlp(line)
                for sent in doc.sents:
                    out.write(sent.text + '\n')
    else:  # return in memory result sentences
        for line in corpus_line_feed(corpus):
            doc = nlp(line)
            for sent in doc.sents:
                yield sent.text


if __name__ == '__main__':
    option = sys.argv[1]
    corpus_path = os.path.realpath(sys.argv[2])
    export_path = os.path.realpath(sys.argv[3])

    print('Export Path: ' + export_path)

    if option == 'stop-words':
        remove_stop_words(corpus_path, export_path)
    elif option == 'normalize':
        normalize_corpus(corpus_path, export_path)
    elif option == 'sent-split':
        sentence_splitting(corpus_path, export_path)
