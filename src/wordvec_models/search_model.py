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
no_metadata_error = 'A metadata file extended (etags) or otherwise must be provided'
cw_sum_error = '"field_weights" array elements must have a sum of 1.'
cw_num_el_error = '"field_weights" array must be of length {}.'
cw_type_error = '"field_weights" variable must be of type ndarray.'

## Presenter Strings
code_div = '################################# CODE #################################'


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
        print('Index keys used:', ', '.join(self.index.keys()), end='\n\n')

        mtdt = self._read_pickle(metadata_path)
        self.metadata = mtdt['metadata']
        self.etag_lookup = mtdt['etag_lookup']
        mtdt = None

    def infer_vector(self, text):
        """Function used to infer sentence vectors with the use of
        word vector model.
        """
        raise NotImplementedError

    def _read_pickle(self, filepath):
        """Utility function for loading pickled objects.

        Args:
            filepath: The path to the pickled object.

        Returns:
            The unpickled object from disk.
        """
        with open(filepath, 'rb') as _in:
            return pickle.load(_in)

    def _read_json(self, filepath):
        """Utility function for loading json files.

        Args:
            filepath: The path to the json file.

        Returns:
            The json object loaded from disk.
        """
        with open(filepath, 'rb') as f:
            return json.load(f)

    def _index_filter(self, indices, tags):
        """Given a list of tags, filter the index list by retaining indices
        of posts that include at least one of the tags. Tags that are no
        present in the etag lookup table are ignored.

        Args:
            indices: A list of indices corresponding to posts and their metadata.
            tags: A list of tags given by the user to filter results.

        Returns:
            A filtered list of indices containing posts that include the given tags.
        """
        tag_index_filter = []
        for tag in tags:
            tag_index_filter.extend(self.etag_lookup.get(tag, []))
        tag_index_filter = frozenset(tag_index_filter)

        return [i for i in indices if i in tag_index_filter]

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

    def ranking(self, query_vec, num_results, field_weights=None, tags=None):
        """Given a query vector, calculate the ranking of posts using cossine
        similarities. In case `field_weights` are given, apply weights in the formula.
        e.g. sims = 0.4*BodyMatrixSims + 0.6*TitleMatrixSims.
        In case `tags` are provided, filter the resulting list to include posts
        containing these tags.

        Args:
            query_vec: A numpy array containing the infered query vector.
            num_results: The final number of results (post indices) returned.
            field_weights: Field weights (Title, Body, Tags) for the calculation of sims.
            tags: A list of tags to filter the final results.

        Returns:
            An index of PostIds and their similarity calues to the given query.
        """
        sims = np.zeros([self.index_size], dtype=np.float32)
        if field_weights:
            for idx, index_matrix in enumerate(self.index.values()):
                sims -= (self._calc_cossims(query_vec, index_matrix) *
                         field_weights[idx])
        else:
            for index_matrix in self.index.values():
                sims -= self._calc_cossims(query_vec, index_matrix)
            #sims = sims / self.num_index_keys ##Field weights 0.5 each
        indices = np.argsort(sims)
        if tags:
            indices = self._index_filter(indices, tags)
        indices = indices[:num_results]
        sim_values = [(-sims[i]) for i in indices]
        return indices, sim_values

    def metadata_frame(self, indices, sim_values):
        """Given a ranked list of indices and their similarity values build
        a dataframe containing the corresponding metadata (title, code snippets etc.)
        and retrieve the most frequent tags appearing in the top results.

        Args:
            indices: A list of PostIds ranked by the ranking algorithm.
            sim_values: A list of cosine similarity values corresponding to the indices.

        Returns:
            A metadata dataframe and a list of the 8 most frequently observed tags.
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
            'Sim': [round(s, 4) for s in sim_values],
            'Link': [(base_url + str(_id)) for _id in postids]
        }

        # Calculate tag frequency and sort in descenting order
        # to retrieve the 8 most frequent tags
        tag_freq = {}
        for i in indices:
            tag_list = self.metadata[i]['ETags']
            for t in tag_list:
                if t in tag_freq:
                    tag_freq[t] += 1
                else:
                    tag_freq[t] = 1
        top_tags = sorted(tag_freq, key=tag_freq.get, reverse=True)[:8]

        return pd.DataFrame(data=df_dict, index=postids), top_tags

    def presenter(self, df, num_results, top_tags):
        """Given a dataframe containing the results and their metadata present
        them in a useful way.

        Args:
            df: The dataframe containing the results and the metadata.
            num_results: The number of results to be presented.
            top_tags: A list of the 8 most frequent tags found in the results.
        """
        def clear_screen():
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
            print('Top 8 tags for this query: %s' % ', '.join(top_tags))

        clear_screen()
        print_item(df.iloc[0], 0, num_results)

        for ii, row in enumerate(df.iterrows()):
            if ii > 0:
                item = row[1]
                print()
                action = input(
                    'Next code snippet [enter], new query [\'q\' + enter]: ')
                clear_screen()
                if action == 'q':
                    break

                print_item(item, ii, num_results)

        clear_screen()

    def search(self,
               num_results=20,
               field_weights=None,
               ranking_fn=None,
               postid_fn=None):
        """Provides the main search function, and the entry point of the search
        model.

        Args:
            num_results: An integer used to limit the presented results.
            field_weights: A list of floats used in the similarity calculation formula
                           as weights for the index fields (Title, Body, Tags).
            ranking_fn: The function that is used to rank the indices based on the 
                        similarities it calculates.
            postid_fn: A function that can be used to manipulate and use the PostIds of
                       the results.
        """
        def check_custom_weights(field_weights):
            if isinstance(field_weights, np.ndarray):
                if len(field_weights) == self.num_index_keys:
                    if field_weights.sum() == 1:
                        return field_weights
                    else:
                        raise ValueError(cw_sum_error)
                else:
                    raise ValueError(
                        cw_num_el_error.format(self.num_index_keys))
            else:
                raise TypeError(cw_type_error)

        if field_weights:
            field_weights = check_custom_weights(field_weights)

        while (True):
            query = input('Query [query + enter], quit [\'q\' + enter]: ')
            if query == 'q':
                break

            tags = input('Tags (e.g. java, android): ')
            tags = tags.replace(' ', '').replace(',', ' ').strip()
            if tags == '':
                tags = None
            else:
                tags = list(filter(bool, tags.split()))
                if len(tags) == 0:
                    tags = None

            query_vec = self.infer_vector(query)
            indices, sim_values = ranking_fn(**query_vec,
                                             num_results=num_results,
                                             field_weights=field_weights,
                                             tags=tags)
            meta_df, top_tags = self.metadata_frame(indices, sim_values)
            self.presenter(meta_df, len(meta_df.index), top_tags)

            if postid_fn:
                postid_fn(list(meta_df.index))
