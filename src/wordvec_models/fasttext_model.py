import os

import numpy as np
import pandas as pd
from fasttext import load_model

from wordvec_models.search_model import BaseSearchModel

## Vector building error strings
doc_path_error = 'Provided document path doesn\'t exist.'
doc_type_error = 'Invalid "doc" variable type {}. Expected str(path) or list.'


class FastTextSearch(BaseSearchModel):
    def __init__(self, model_path, index_path, index_keys, metadata_path):

        self.model = load_model(model_path)
        print('FastText model {} loaded.'.format(os.path.basename(model_path)))
        super().__init__(index_path, index_keys, metadata_path)

    def infer_vector(self, text):
        return {
            'query_vec':
            self.model.get_sentence_vector(text.lower().strip()).reshape(
                1, -1)
        }

    def search(self, num_results=20, field_weights=None, postid_fn=None):
        super().search(num_results=num_results,
                       field_weights=field_weights,
                       ranking_fn=self.ranking,
                       postid_fn=postid_fn)


def build_doc_vectors(model, doc, export_path=None):
    """Expected input is a preprocessed document.
    Calculates sentence vectors using the built-in fastText function which
    averages the word-vector norms of all the words in the given sentence.
    """
    def calc_vectors(doc, vector_matrix):
        for idx, line in enumerate(doc):
            print('\rcalculating vector for line #', idx, end='')
            vector_matrix.append(model.get_sentence_vector(str(line.strip())))
        print()

    vector_matrix = []
    if isinstance(model, str):
        model = load_model(model)
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
        print('\nfasttext doc vectors saved in', os.path.realpath(export_path))
    return vector_matrix


def build_wordvec_index(vec_filename, export_path):
    """Given a fastText vector file, output word vectors into a DataFrame where
    key: word token (string), value: word vector (numpy array).
    NOTE: First row in a fastText vector file holds the number of tokens and 
    vector length.
    """
    with open(vec_filename, 'r') as vec_file:
        dimensions = [int(dim) for dim in vec_file.readline().split()]
        print('wordvec matrix dimensions:', dimensions)
        vec_matrix = np.zeros(dimensions, dtype=np.float32)
        index_tokens = []
        for idx, row in enumerate(vec_file):
            token, vector = row.split(' ', 1)
            vec_matrix[idx] = np.fromstring(vector, sep=' ')
            index_tokens.append(token)
    pd.DataFrame(data=vec_matrix, index=index_tokens).to_pickle(export_path)
    print('wordvec index saved in', os.path.realpath(export_path))
