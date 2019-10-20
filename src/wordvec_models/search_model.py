import os
import re
import json
import pickle
import subprocess

import scipy
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity

## StackOverflow Base URL
base_url = 'https://stackoverflow.com/questions/'

## Error Strings
cw_sum_error = '"custom_weights" array elements must have a sum of 1.'
cw_num_el_error = '"custom_weights" array must be of length {}.'
cw_type_error = '"custom_weights" variable must be of type ndarray.'

## Presenter Strings
code_div = '################################# CODE #################################'
clear_screen_seq = subprocess.check_output('clear')


class BaseSearchModel:
    """Base model for searching a precomputed vector index for similar 
    documents.

    Given a user text query the corresponding vector is inferred using the 
    vector space model each subclass utilizes (FastText, TFIDF etc.)
    """

    def __init__(self, index_path, index_keys, metadata_path):
        self.index = self._read_pickle(index_path)
        for key in list(self.index.keys()):
            if key not in index_keys:
                del self.index[key]

        self.num_index_keys = len(self.index)
        self.index_size = (next(iter(self.index.values()))).shape[0]
        self.metadata = self._read_json(metadata_path)

    def infer_vector(self, text):
        """Function used to infer sentence vectors."""
        raise NotImplementedError

    def _read_pickle(self, filepath):
        """Utility function for loading pickled files.

        Args:
            filepath: The path to the pickled file.

        Returns:
            The unpickled file from disk.
        """

        with open(filepath, 'rb') as _in:
            return pickle.load(_in)

    def _read_json(self, filepath):
        """Utility function for loading json files.

        Args:
            filepath: The path to the actual file.

        Returns:
            The json object loaded from disk.
        """

        with open(filepath, 'rb') as f:
            return json.load(f)

    def _calc_cossims(self,
                      vector,
                      matrix,
                      batch_calc=False,
                      batch_size=100000):
        """Given a query vector, compute the cosine similarities between the given 
        vector and each row of the provided matrix of document vectors.
        Cosine similarity is computed as the normalized dot product of two vectors 
        X and Y as follows: K(X, Y) = <X, Y> / (||X||*||Y||).

        Args:
            vector: A numpy array containing the query vector.
            matrix: A numpy matrix containing the index vectors. 
            batch_calc: Reduces memory usage by calculating similarities in batches.
            batch_size: The size of each batch used when batch_calc is True.

        Returns:
            A numpy array containing the values of the computed cosine similarities. 
        """

        def batch_cossims(vector, matrix, batch_size):
            mat_len = len(matrix)
            if mat_len > batch_size:
                n_batches = round(mat_len / batch_size)
                matrix = np.array_split(matrix, n_batches)
            for batch in matrix:
                yield cosine_similarity(vector, batch).reshape(-1)

        if batch_calc and not scipy.sparse.issparse(vector):
            istart = 0
            cossims = np.zeros(len(matrix), dtype=np.float32)
            for batch_vec in batch_cossims(vector, matrix, batch_size):
                cossims[istart:(istart + len(batch_vec))] = batch_vec
                istart = istart + len(batch_vec)
            return cossims
        else:
            return cosine_similarity(vector, matrix).reshape(-1)

    def ranking(self, query_vec, num_results, custom_weights=None, ents=None):
        """
        """
        sims = np.zeros([self.index_size], dtype=np.float32)
        if custom_weights:
            for idx, index_matrix in enumerate(self.index.values()):
                sims -= (self._calc_cossims(query_vec, index_matrix) *
                         custom_weights[idx])
        else:
            for index_matrix in self.index.values():
                sims -= self._calc_cossims(query_vec, index_matrix)
            sims = sims / self.num_index_keys
        indices = np.argsort(sims)[:num_results]
        sim_values = [(-sims[i]) for i in indices]
        return indices, sim_values

    def metadata_frame(self, indices, sim_values):
        """
        """

        def sdict(snippet_list, postid):
            # highest scored answer
            snippet_str = snippet_list[0]
            #print(postid)
            #print(snippet_str)
            info_str = re.findall(r'Post: .*\n##Score -?[0-9]{1,5}',
                                  snippet_str)[0]
            info = info_str.split('\n')
            score = int(info[1][8:])
            answer_link = info[0][6:]
            snippet_str = re.sub(r'Post: .*\n##Score -?[0-9]{1,5}', '',
                                 snippet_str).strip()
            snippet = '\n' + '\n\n'.join(snippet_str.split('<_code_>')) + '\n'
            d = {'anslink': answer_link, 'snippet': snippet, 'score': score}
            return d

        postids = [self.metadata[i]['PostId'] for i in indices]
        df_dict = {
            'Title': [self.metadata[i]['Title'] for i in indices],
            'SnippetCount':
            [self.metadata[i]['SnippetCount'] for i in indices],
            'sdict': [
                sdict(self.metadata[i]['Snippets'], postids[jj])
                for jj, i in enumerate(indices)
            ],
            'PostId':
            postids,
            'Sim': [round(s, 4) for s in sim_values],
            'Link': [(base_url + str(_id)) for _id in postids]
        }
        return pd.DataFrame(df_dict)

    def presenter(self, df, num_results):
        def clear():
            print(chr(27) + '[2J')
            print(chr(27) + "[1;1f")

        def print_item(item, ii, max_ii):
            print('\n%d/%d' % (ii + 1, max_ii))
            print(code_div)
            print(item['sdict']['snippet'])
            print(code_div)
            print('\nTitle: %s' % item['Title'])
            print('Post: %s' % item['Link'])
            print('Answer: %s' % item['sdict']['anslink'], end='\n\n')
            print('Answer score: %d' % item['sdict']['score'])
            print('Snippets for this post: %d' % item['SnippetCount'])

        clear()
        print_item(df.iloc[0], 0, num_results)

        for ii, row in enumerate(df.iterrows()):
            if ii > 0:
                item = row[1]
                print()
                action = input(
                    'Action (enter for next snippet, or \'exit\' for new query): '
                )
                clear()
                if action == 'exit':
                    break

                print_item(item, ii, num_results)

        clear()

    def search(self,
               num_results=20,
               custom_weights=None,
               ranking_fn=None,
               postid_fn=None,
               vector_fn=None,
               **vector_fn_kwargs):
        """
        """

        def check_custom_weights(custom_weights):
            if isinstance(custom_weights, np.ndarray):
                if len(custom_weights) == self.num_index_keys:
                    if custom_weights.sum() == 1:
                        return custom_weights
                    else:
                        raise ValueError(cw_sum_error)
                else:
                    raise ValueError(
                        cw_num_el_error.format(self.num_index_keys))
            else:
                raise TypeError(cw_type_error)

        if custom_weights:
            custom_weights = check_custom_weights(custom_weights)
        while (True):
            query = input('New Query: ')
            if query == 'exit':
                break

            ent_labels = input('Labels: ')
            ent_labels = ent_labels.replace(' ', '').split(',')

            query_vec = self.infer_vector(query)
            indices, sim_values = ranking_fn(query_vec, num_results,
                                             custom_weights)
            meta_df = self.metadata_frame(indices, sim_values)
            self.presenter(meta_df, num_results)
            #str_out = tabulate(
            #    meta_df, showindex=False, headers=meta_df.columns)
            #print(str_out, end='\n\n')
            #if vector_fn:
            #vector_fn(query_vec, **vector_fn_kwargs)
            #if postid_fn:
            #postid_fn(list(meta_df['PostId']))
