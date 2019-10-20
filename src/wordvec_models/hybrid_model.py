import os

import numpy as np
import pandas as pd
from fastText import load_model

from wordvec_models.search_model import BaseSearchModel

## Vector building error strings
doc_path_error = 'Provided document path doesn\'t exist.'
doc_type_error = 'Invalid "doc" variable type {}. Expected str(path) or list.'


class HybridSearch(BaseSearchModel):
    def __init__(self, ft_model_path, tfidf_model_path, ft_index_path,
                 tfidf_index_path, index_keys, metadata_path):

        self.ft_model = load_model(ft_model_path)
        print('FastText model {} loaded.'.format(
            os.path.basename(ft_model_path)))

        self.ft_index = self._read_pickle(ft_index_path)
        for key in list(self.ft_index.keys()):
            if key not in index_keys:
                del self.ft_index[key]

        self.tfidf_model = self._read_pickle(tfidf_model_path)
        print('TF-IDF model {} loaded.'.format(
            os.path.basename(tfidf_model_path)))
        self.tfidf_index = self._read_pickle(tfidf_index_path)
        for key in list(self.tfidf_index.keys()):
            if key not in index_keys:
                del self.tfidf_index[key]

        self.num_index_keys = len(self.ft_index)
        self.index_size = len(next(iter(self.ft_index.values())))
        print('Index keys used:', ', '.join(self.ft_index.keys()), end='\n\n')

        self.metadata = self._read_json(metadata_path)

    def infer_vector(self, text):
        text = text.lower().strip()
        ft_vec = self.ft_model.get_sentence_vector(text).reshape(1, -1)
        tfidf_vec = self.tfidf_model.transform([text])
        return ft_vec, tfidf_vec

    def hybrid_ranking(self, ft_query_vec, tfidf_query_vec, num_results):
        """
        """
        sims = np.zeros([self.index_size], dtype=np.float32)
        # fastText sims
        for index_matrix in self.ft_index.values():
            sims -= self._calc_cossims(ft_query_vec, index_matrix)
        sims = sims / self.num_index_keys
        # tfidf sims
        for index_matrix in self.tfidf_index.values():
            sims -= self._calc_cossims(tfidf_query_vec, index_matrix)
        sims = sims / self.num_index_keys

        indices = np.argsort(sims)[:num_results]
        sim_values = [(-sims[i]) for i in indices]
        return indices, sim_values

    def search(self,
               num_results=20,
               custom_weights=None,
               postid_fn=None,
               api_labels=False,
               vector_fn_kwargs={}):

        while (True):
            query = input('New Query: ')
            if query == 'exit':
                break

            ent_labels = input('Labels: ')
            ent_labels = ent_labels.replace(' ', '').split(',')

            ft_vec, tfidf_vec = self.infer_vector(query)
            indices, sim_values = self.hybrid_ranking(ft_vec, tfidf_vec,
                                                      num_results)
            meta_df = self.metadata_frame(indices, sim_values)
            self.presenter(meta_df, num_results)
