import os
import re
import sys
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from fastText import load_model

## Template Features
orth = [
    'w', 'wl', 'dot', 'brac', 'uscr', 'init_cap', 'ccase', 'api_ptrn', 'p1',
    'p2', 'p3', 'p4', 's1', 's2', 's3', 's4', 'is_api', 'is_plat', 'is_pl',
    'is_org', 'is_stan', 'is_fram'
]
unigram = ['w', 'wl', 'type']
#, 'p1', 'p2', 'p3', 'p4', 's1', 's2', 's3', 's4']
bigram = ['w', 'wl']
word_rep = [
    'brown', 'brown-p2', 'brown-p4', 'brown-p6', 'brown-p8', 'brown-p10',
    'brown-p12', 'brown-p14', 'ce500', 'ce1000', 'ce1500', 'ce2000', 'ce3000'
]

## API patterns
api1 = r'[a-zA-Z_$][a-zA-Z0-9_$]*\(\)$'
api2 = r'([a-zA-Z_$][a-zA-Z0-9_$]*\.)+[a-zA-Z_$][a-zA-Z0-9_$]*(\(\))?'

## Default filepaths and directories
ft_model = 'data/ft_archive/fasttext_v0.1.bin'
ft_cluster_dir = 'data/mbkm_clusters'
brown_paths = 'data/brown/paths.pkl'
gazetteer_dir = 'data/gazetteers'


class FeatureExtractor:
    def __init__(self,
                 ft_model=ft_model,
                 ft_cluster_dir=ft_cluster_dir,
                 brown_paths=brown_paths,
                 gazetteer_dir=gazetteer_dir):
        self.fasttext = load_model(ft_model)
        self.brown_paths = pd.read_pickle(brown_paths)
        self._load_clusterers(ft_cluster_dir)
        self._load_gazetteers(gazetteer_dir)
        self._build_template()

    def _load_clusterers(self, c_dir):
        self.c_names = []
        self.c_models = []
        for _file in os.listdir(c_dir):
            if _file.endswith('.pkl'):
                name = _file.split('_')[-1][:-4]
                self.c_names.append(name)
                with open(os.path.join(c_dir, _file), 'rb') as c:
                    self.c_models.append(pickle.load(c))

    def _load_gazetteers(self, g_dir):
        def load_set(filename):
            ll = []
            with open(filename, 'r') as f:
                for line in f:
                    ll.append(line.strip().lower())
            return frozenset(ll)

        self.api_g = load_set(os.path.join(g_dir, 'ApiList.txt'))
        self.plat_g = load_set(os.path.join(g_dir, 'PlatformList.txt'))
        self.pl_g = load_set(
            os.path.join(g_dir, 'ProgrammingLanguageList.txt'))
        self.org_g = load_set(os.path.join(g_dir, 'SoftwareOrgList.txt'))
        self.stan_g = load_set(os.path.join(g_dir, 'SoftwareStandardList.txt'))
        self.fram_g = load_set(
            os.path.join(g_dir, 'ToolLibraryFrameworkList.txt'))

    def _type(self, token):
        T = (
            'AllUpper',
            'AllDigit',
            'AllSymbol',
            'AllUpperDigit',
            'AllUpperSymbol',
            'AllDigitSymbol',
            'AllUpperDigitSymbol',
            'InitUpper',
            'AllLetter',
            'AllAlnum',
        )

        R = set(T)
        for ii, char in enumerate(token):
            if char.isupper():
                R.discard('AllDigit')
                R.discard('AllSymbol')
                R.discard('AllDigitSymbol')
            elif char.isdigit() or char in (',', '.'):
                R.discard('AllUpper')
                R.discard('AllSymbol')
                R.discard('AllUpperSymbol')
                R.discard('AllLetter')
            elif char.islower():
                R.discard('AllUpper')
                R.discard('AllDigit')
                R.discard('AllSymbol')
                R.discard('AllUpperDigit')
                R.discard('AllUpperSymbol')
                R.discard('AllDigitSymbol')
                R.discard('AllUpperDigitSymbol')
            else:
                R.discard('AllUpper')
                R.discard('AllDigit')
                R.discard('AllUpperDigit')
                R.discard('AllLetter')
                R.discard('AllAlnum')

            if ii == 0 and not char.isupper():
                R.discard('InitUpper')

        for tag in T:
            if tag in R:
                return tag
        return 'NO'

    def _shape(self, token):
        shape = ''
        for char in token:
            if char.isupper():
                shape += 'U'
            elif char.islower():
                shape += 'L'
            elif char.isdigit():
                shape += 'D'
            elif char in ('.', '_'):
                shape += '.'
            elif char in ('+', '-', '*', '/', '=', '|', ',', ';', ':', '?',
                          '!'):
                shape += '-'
            elif char in ('(', '{', '[', '<'):
                shape += '('
            elif char in (')', '}', ']', '>'):
                shape += ')'
            else:
                shape += '-'
        return shape

    def _brackets(self, token):
        if token.endswith('()'):
            return 'T'
        return 'F'

    def _dot(self, token):
        if '.' in token:
            return 'T'
        return 'F'

    def _underscore(self, token):
        if '_' in token:
            return 'T'
        return 'F'

    def _init_capital(self, token):
        if token[0].isupper():
            return 'T'
        return 'F'

    def _camel_case(self, token):
        # starts with lowercase then matches any capitals
        if re.match(r'([a-z]+[A-Z]+\\w+)+', token):
            return 'T'
        return 'F'

    def _api_pattern(self, token):
        if re.match(api1, token) or re.match(api2, token):
            return 'T'
        return 'F'

    def _orthographic_features(self, token):
        # exact token
        self.feat['w'] = token
        # token lowercase
        self.feat['wl'] = token.lower()
        # token shape
        #self.feat['shape'] = self._shape(token)
        # token type
        self.feat['type'] = self._type(token)
        # contains dot
        self.feat['dot'] = self._dot(token)
        # ends with brackets
        self.feat['brac'] = self._brackets(token)
        # contains underscore
        self.feat['uscr'] = self._underscore(token)
        # begins with capital letter
        self.feat['init_cap'] = self._init_capital(token)
        # is camel cased
        self.feat['ccase'] = self._camel_case(token)
        # matches api pattern
        self.feat['api_ptrn'] = self._api_pattern(token)

        # token prefixes
        for p in range(1, 5):
            self.feat['p%d' % p] = token[:p]

        # token suffixes
        for s in range(1, 5):
            self.feat['s%d' % s] = token[-s:]

    def _gazetteer_features(self, token):
        if token.endswith('()'):
            token = token[:-2]
        token = token.lower()
        if token in self.api_g:
            self.feat['is_api'] = 'T'
        else:
            self.feat['is_api'] = 'F'
        if token in self.plat_g:
            self.feat['is_plat'] = 'T'
        else:
            self.feat['is_plat'] = 'F'
        if token in self.pl_g:
            self.feat['is_pl'] = 'T'
        else:
            self.feat['is_pl'] = 'F'
        if token in self.org_g:
            self.feat['is_org'] = 'T'
        else:
            self.feat['is_org'] = 'F'
        if token in self.stan_g:
            self.feat['is_stan'] = 'T'
        else:
            self.feat['is_stan'] = 'F'
        if token in self.fram_g:
            self.feat['is_fram'] = 'T'
        else:
            self.feat['is_fram'] = 'F'

    def _brown_cluster_features(self, token):
        def get_prefix(path, prefix):
            return path[:prefix]  #.ljust(prefix, '0')

        if token in self.brown_paths:
            path = self.brown_paths[token]
            self.feat['brown'] = path
            for prefix in range(2, 16, 2):
                self.feat['brown-p%d' % prefix] = get_prefix(path, prefix)
        else:
            self.feat['brown'] = ''
            for prefix in range(2, 16, 2):
                self.feat['brown-p%d' % prefix] = ''

    def _compound_embedding_features(self, token):
        token_vec = self.fasttext.get_word_vector(token).reshape(1, -1)
        for ii, name in enumerate(self.c_names):
            self.feat['ce' + name] = str(
                self.c_models[ii].predict(token_vec)[0])

    def _build_template(self):
        ablist = [-1, 0]
        blist = [-1, 0, 1]
        self.template = []
        for tag in orth:
            self.template += [(((tag, 0), ))]
        '''
        for tag in unigram:
            # Unigram info for tokens in range [-1, 1] where 0 is the focus token
            self.template += [((tag, i), )
                              for i in [-2, -1, 0, 1, 2]]  # -1,0,1
        '''
        for tag in bigram:
            # Bigram info for combinations (wᵢ, wᵢ₊₁) for i in [-1, 0]
            self.template += [((tag, i), (tag, i + 1)) for i in ablist]
        for tag in word_rep:
            if tag == 'brown':
                self.template += [((tag, i), ) for i in blist]
                self.template += [((tag, i), (tag, i + 1)) for i in ablist]
                self.template += [((tag, -1), (tag, 1))]
            elif tag.startswith('brown'):
                self.template += [((tag, i), ) for i in blist]
            elif tag.startswith('ce'):
                self.template += [((tag, i), ) for i in blist]
                self.template += [((tag, i), (tag, i + 1)) for i in ablist]
                self.template += [((tag, -1), (tag, 1))]

    def _apply_template(self, X):
        for obj in self.template:
            field_name = '|'.join(['%s[%d]' % (f, o) for f, o in obj])
            for ii in range(len(X)):
                values = []
                for field, offset in obj:
                    p = ii + offset
                    if p not in range(len(X)):
                        values = []
                        break
                    values.append(X[p][field])

                if values:  # non empty list
                    X[ii]['F'].append('%s=%s' % (field_name, '|'.join(values)))

    def token_features(self, item):
        token = item['token']
        self.feat = OrderedDict()
        self._orthographic_features(token)
        self._brown_cluster_features(token)
        self._compound_embedding_features(token)
        self._gazetteer_features(token)
        item.update(self.feat)

    def sequence_features(self, X):
        for item in X:
            self.token_features(item)

        self._apply_template(X)

        if X:
            X[0]['F'].append('__BOS__')
            X[-1]['F'].append('__EOS__')


if __name__ == '__main__':
    fe = FeatureExtractor()
    print(fe.template)