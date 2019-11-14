import os

import numpy as np
import pandas as pd
from fasttext import load_model

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
        self.index_size = (next(iter(self.ft_index.values()))).shape[0]
        print('Index keys used:', ', '.join(self.ft_index.keys()), end='\n\n')

        mtdt = self._read_pickle(metadata_path)
        self.metadata = mtdt['metadata']
        self.etag_lookup = mtdt['etag_lookup']
        mtdt = None

    def infer_vector(self, text):
        text = text.lower().strip()
        ft_vec = self.ft_model.get_sentence_vector(text).reshape(1, -1)
        tfidf_vec = self.tfidf_model.transform([text])
        return {'ft_query_vec': ft_vec, 'tfidf_query_vec': tfidf_vec}

    def hybrid_ranking(self,
                       ft_query_vec,
                       tfidf_query_vec,
                       num_results,
                       field_weights=None,
                       tags=None):
        """Given a query vector, calculate the ranking of posts using cossine
        similarities. In case `field_weights` are given, apply weights in the formula.
        e.g. sims = 0.4*BodyMatrixSims + 0.6*TitleMatrixSims.
        In case `tags` are provided, filter the resulting list to include posts
        containing these tags.

        Args:
            ft_query_vec: A numpy array containing the fastText infered query vector.
            tfidf_query_vec: A numpy array containing the TFIDF infered query vector.
            num_results: The final number of results (post indices) returned.
            field_weights: Field weights (Title, Body, Tags) for the calculation of sims.
            tags: A list of tags to filter the final results.

        Returns:
            An index of PostIds and their similarity calues to the given query.
        """
        sims = np.zeros([self.index_size], dtype=np.float32)
        if field_weights:
            # fastText sims
            for idx, index_matrix in enumerate(self.ft_index.values()):
                sims -= self._calc_cossims(ft_query_vec,
                                           index_matrix) * field_weights[idx]
            # tfidf sims
            for idx, index_matrix in enumerate(self.tfidf_index.values()):
                sims -= self._calc_cossims(tfidf_query_vec,
                                           index_matrix) * field_weights[idx]
            #sims = sims / self.num_index_keys ##Field weights 0.5 each
        else:
            # fastText sims
            for index_matrix in self.ft_index.values():
                sims -= self._calc_cossims(ft_query_vec, index_matrix)
            # tfidf sims
            for index_matrix in self.tfidf_index.values():
                sims -= self._calc_cossims(tfidf_query_vec, index_matrix)
            #sims = sims / self.num_index_keys ##Field weights 0.5 each
        #sims = sims / 2  ##Model weights 0.5 each

        indices = np.argsort(sims)
        if tags:
            indices = self._index_filter(indices, tags)
        indices = indices[:num_results]
        sim_values = [(-sims[i]) for i in indices]
        return indices, sim_values

    def search(self, num_results=20, field_weights=None, postid_fn=None):
        super().search(num_results=num_results,
                       field_weights=field_weights,
                       ranking_fn=self.hybrid_ranking,
                       postid_fn=postid_fn)
