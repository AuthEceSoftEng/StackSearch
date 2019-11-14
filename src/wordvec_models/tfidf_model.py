import os
import pickle

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from wordvec_models.search_model import BaseSearchModel

# every token consists of two or more non whitespace characters
TOKEN_RE = r'\S\S+'


class TfIdfSearch(BaseSearchModel):
    def __init__(self, model_path, index_path, index_keys, metadata_path):
        self.model = self._read_pickle(model_path)
        print('TFIDF model {} loaded.'.format(os.path.basename(model_path)))
        super().__init__(index_path, index_keys, metadata_path)

    def infer_vector(self, text):
        return {'query_vec': self.model.transform([text.lower().strip()])}

    def search(self, num_results=20, field_weights=None, postid_fn=None):
        super().search(num_results=num_results,
                       field_weights=field_weights,
                       ranking_fn=self.ranking,
                       postid_fn=postid_fn)


def load_text_list(filename):
    """Returns a list of strings."""
    text_list = []
    with open(filename, 'r') as f:
        for line in f:
            text_list.append(line.strip())
    return text_list


def load_tfidf_model(model_path):
    with open(model_path, 'rb') as _in:
        return pickle.load(_in)


def train_tfidf_model(post_list, model_export_path, vec_export_path):
    tfidf = TfidfVectorizer(token_pattern=TOKEN_RE,
                            preprocessor=None,
                            tokenizer=None,
                            stop_words='english',
                            smooth_idf=True)
    tfidf_matrix = tfidf.fit_transform(post_list)
    sparse.save_npz(vec_export_path, tfidf_matrix)
    with open(model_export_path, 'wb') as out:
        pickle.dump(tfidf, out)
    return tfidf


def build_doc_vectors(model, doc, export_path=None):
    """Calculates sentence vectors using the provided pre-trained tf-idf model
    """
    vec_matrix = None
    if isinstance(model, str):
        model = load_tfidf_model(model)
    if isinstance(doc, str):
        if os.path.exists(doc):
            doc = load_text_list(doc)
            vec_matrix = model.transform(doc)
        else:
            raise ValueError(
                'Provided document path {} doesn\'t exist.'.format(doc))
    elif isinstance(doc, list):
        vec_matrix = model.transform(doc)
    else:
        raise ValueError('Invalid "doc" variable type {}.'.format(
            str(type(doc))))
    if export_path:
        sparse.save_npz(export_path, vec_matrix)
        print('\ntfidf doc vectors saved in', os.path.realpath(export_path))
    return vec_matrix
