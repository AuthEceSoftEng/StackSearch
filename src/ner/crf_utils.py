#!/usr/bin/env python

import argparse
import subprocess
from subprocess import Popen, PIPE, STDOUT

import crfsuite

from feature_extractor import FeatureExtractor
from text_processing.corpus_utils import CorpusUtils


def sequence_feed(filename, field_names, separator=' '):
    X = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':  # on empty line yield sequence/sentence
                yield X
                X = []
            else:
                fields = line.split(separator)
                if len(fields) < len(field_names):
                    raise ValueError('Expected {} fields: {}'.format(
                        len(field_names), field_names))
                item = {'F': []}  # item features
                for ii, field_name in enumerate(field_names):
                    item[field_name] = fields[ii]
                X.append(item)


def output_features(out, X, f):
    for item in X:
        out.write('%s' % item[f])
        for field in item['F']:
            out.write('\t%s' % field)
        out.write('\n')
    out.write('\n')


def features_string(X, f):
    fstr = ''
    for item in X:
        fstr += '%s' % item[f]
        for field in item['F']:
            fstr += '\t%s' % field
        fstr += '\n'
    fstr += '\n'
    return fstr


def to_crfsuite(X):
    """
    Convert an item sequence into an object compatible with crfsuite
    Python module.

    @type   X:      list of mapping objects
    @param  X:      The sequence.
    @rtype          crfsuite.ItemSequence
    @return        The same sequence in crfsuite.ItemSequence type.
    """
    xseq = crfsuite.ItemSequence()
    for x in X:
        item = crfsuite.Item()
        for f in x['F']:
            item.append(crfsuite.Attribute(f))
        xseq.append(item)
    return xseq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRFsuite utilities')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    train_parser = subparsers.add_parser(
        'train', help='Train a CRF NER model.')
    train_parser.add_argument(
        'bin',
        help='Path to the CRF-Suite binary. (Usually in ~/local/bin/crfsuite)')
    train_parser.add_argument(
        'input', help='Training data prepared with the \'features\' command.')
    train_parser.add_argument('model', help='Name of the output CRF model.')
    tag_parser = subparsers.add_parser('tag', help='tag file')
    tag_parser.add_argument('model')
    tag_parser.add_argument('input')
    tag_parser.add_argument('output')

    args = parser.parse_args()

    conll_fields = ['token', 'entity']

    if args.command == 'train':
        cmd = [args.bin, 'learn', '-m', args.model, args.input]
        subprocess.call(cmd)
    elif args.command == 'tag':
        fextractor = FeatureExtractor()
        print('feature extractor loaded')

        tagger = crfsuite.Tagger()
        tagger.open(args.model)

        cu = CorpusUtils(sent_split=True)
        with open(args.input, 'r') as _in:
            text = _in.readlines()

        tags = {}
        with open(args.output, 'w') as out:
            for X in cu.crf_sequence_feed(text):
                fextractor.sequence_features(X)
                xseq = to_crfsuite(X)
                yseq = tagger.tag(xseq)
                for ii, v in enumerate(X):
                    if yseq[ii] != 'O':
                        tags[v['token']] = yseq[ii]
                for ii, v in enumerate(X):
                    out.write('\t'.join(v[f] for f in conll_fields))
                    out.write('\t%s\n' % yseq[ii])
                out.write('\n')
        print(tags)