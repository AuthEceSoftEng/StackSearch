#!/usr/bin/env python

import json
from collections import OrderedDict

import numpy as np


def token_frequencies(post_dict):
    freqs = {}
    for sent_list in post_dict.values():
        for sent in sent_list:
            tokens = sent.split()
            for t in tokens:
                if t not in freqs:
                    freqs[t] = 0
                freqs[t] += 1
    return freqs


def sort_frequencies(freqs):
    tokens = list(freqs.keys())
    tfreqs = list(freqs.values())

    sorted_idx = np.argsort(tfreqs)
    sorted_tfreqs = [tfreqs[idx] for idx in sorted_idx[::-1]]
    sorted_tokens = [tokens[idx] for idx in sorted_idx[::-1]]

    sorted_freqs = OrderedDict()
    for idx, token in enumerate(sorted_tokens):
        sorted_freqs[token] = sorted_tfreqs[idx]
    return sorted_freqs


if __name__ == '__main__':
    with open('temp/post_sent_dict.json', 'r') as _in:
        post_dict = json.load(_in, object_pairs_hook=OrderedDict)

    sorted_frequencies = sort_frequencies(token_frequencies(post_dict))

    with open('token_frequencies.json', 'w') as out:
        json.dump(sorted_frequencies, out, indent=2)
