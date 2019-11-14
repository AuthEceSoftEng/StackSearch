import os
import pickle

import numpy as np
import pandas as pd
from numpy.linalg import norm

from wordvec_models.search_model import BaseSearchModel

## Vector building error strings
doc_path_error = 'Provided document path doesn\'t exist.'
doc_type_error = 'Invalid "doc" variable type {}. Expected str(path) or list.'


class GloVeModel(BaseSearchModel):
    def __init__(self, wordvec_index, build_index, export_path=None):
        if build_index:
            if export_path:
                print('Building wordvec index...')
                self.build_wordvec_index(wordvec_index, export_path)
            else:
                raise Exception('Export file path is required.')
        else:
            self.wordvec_index = pd.read_pickle(wordvec_index)
        self.dim = self.wordvec_index.shape[1]
        self.vocab = frozenset(list(self.wordvec_index.index))
        self.unk_vec = self._normalize_vector(
            self.wordvec_index.loc['<unk>'].values)

    def _normalize_vector(self, vector):
        return vector / norm(vector)

    def build_wordvec_index(self, vec_filepath, export_path):
        """Given a GloVe vector file, output word vectors into a DataFrame where
        key: word token (string), value: word vector (numpy array).
        NOTE: First row in a GloVe vector file holds the number of tokens and 
        vector length.
        """
        with open(vec_filepath, 'r') as vec_file:
            dimensions = [int(dim) for dim in vec_file.readline().split()]
            dimensions[0] += 1  ## include <unk> token
            print('wordvec matrix dimensions:', dimensions)
            vec_matrix = np.zeros(dimensions, dtype=np.float32)
            index_tokens = []
            for idx, row in enumerate(vec_file):
                token, vector = row.split(' ', 1)
                vec_matrix[idx] = np.fromstring(vector, sep=' ')
                index_tokens.append(token)
        self.wordvec_index = pd.DataFrame(data=vec_matrix, index=index_tokens)
        self.wordvec_index.to_pickle(export_path)
        print('GloVe wordvec index saved in', os.path.realpath(export_path))

    def infer_vector(self, text):
        """Calculates sentence vectors by mimiking the fastText algorithm.
        Average of the unit norm vectors of every token.
        """
        count = 0
        svec = np.zeros(self.dim, dtype=np.float32)
        for token in text.split():
            count += 1
            if token in self.vocab:
                svec += self._normalize_vector(
                    self.wordvec_index.loc[token].values)
            else:
                svec += self.unk_vec
        if count == 0:
            return self.unk_vec
        return svec / count

    def search(self, num_results=20, field_weights=None, postid_fn=None):
        super().search(num_results=num_results,
                       field_weights=field_weights,
                       ranking_fn=self.ranking,
                       postid_fn=postid_fn)


def load_glove_model(model_path):
    with open(model_path, 'rb') as _in:
        return pickle.load(_in)


def build_glove_model(wordvec_index, wv_export_path, model_export_path):
    glove = GloVeModel(wordvec_index=wordvec_index,
                       build_index=True,
                       export_path=wv_export_path)
    with open(model_export_path, 'wb') as out:
        pickle.dump(glove, out)
    return glove


def build_doc_vectors(model, doc, export_path=None):
    """Expected input is a preprocessed document.
    Calculates sentence vectors as the average of the unit norm vectors
    of every token, like fastText.
    """
    def calc_vectors(doc, vector_matrix):
        for idx, line in enumerate(doc):
            print('\rcalculating vector for line #', idx, end='')
            vector_matrix.append(model.infer_vector(str(line.strip())))
        print()

    vector_matrix = []
    if isinstance(model, str):
        model = load_glove_model(model)
    if isinstance(doc, str):
        if os.path.exists(doc):
            with open(doc, 'r') as doc_file:
                calc_vectors(doc_file, vector_matrix)
        else:
            raise ValueError(doc_path_error)
    elif isinstance(doc, list):
        calc_vectors(doc, vector_matrix)
    else:
        raise TypeError(doc_type_error.format(type(doc)))
    vector_matrix = np.array(vector_matrix)
    if export_path:
        np.save(export_path, vector_matrix)
        print('\nGloVe doc vectors saved in', os.path.realpath(export_path))
    return vector_matrix