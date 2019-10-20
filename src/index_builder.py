#!/usr/bin/env python

import os
import sys
import json
import pickle
import sqlite3

import numpy as np
import pandas as pd
from fastText import load_model as load_ft_model

from text_processing.utils import process_corpus
from wordvec_models.fasttext_model import build_doc_vectors as build_ft_vecs
from wordvec_models.fasttext_model import build_wordvec_index
from wordvec_models.tfidf_model import build_doc_vectors as build_tfidf_vecs
from wordvec_models.tfidf_model import load_tfidf_model
import wordvec_models.glove_model
from wordvec_models.glove_model import build_doc_vectors as build_glove_vecs
from wordvec_models.glove_model import GloVeModel, load_glove_model


class IndexBuilder:
    def __init__(self, qdataframe_path, database_path, fasttext_path,
                 tfidf_path, glove_path, temp_dir, export_dir):
        ## File/Model Paths
        self.qdataframe_path = qdataframe_path
        self.database_path = database_path
        self.fasttext_path = fasttext_path
        self.tfidf_path = tfidf_path
        self.wordvec_path = fasttext_path[:-4] + '.vec'
        self.glove_path = glove_path

        ## Folders
        self.temp_dir = temp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.export_dir = export_dir
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def _load_text_list(self, filename):
        text_list = []
        with open(filename, 'r') as f:
            for line in f:
                text_list.append(line.strip())
        return text_list

    def _dump_text_list(self, filename, text_list):
        with open(filename, 'w') as out:
            for line in text_list:
                out.write(' '.join(line.split()) + '\n')

    def _fetch_qids(self, query):
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = lambda cursor, row: row[0]
        c = conn.cursor()
        qids = c.execute(query).fetchall()
        return qids

    def _build_index_dataset(self, qids, keys=['Title', 'Body']):
        index_ids = []
        index_dataset = {key: [] for key in keys}
        qdf = pd.read_pickle(self.qdataframe_path)
        for index, row in qdf.iterrows():
            if index in qids:
                index_ids.append(index)
                for key in keys:
                    index_dataset[key].append(row[key])
        print('Index contains {} questions.'.format(len(index_ids)))
        return index_ids, index_dataset

    def build_metadata_index(self, qids, query):
        # TODO: run snippet_index_builder on new javaposts.db and remove
        # case of first post being empty string after split
        def split_snippets(snippet_str):
            if snippet_str == '':
                return []
            snippet_list = snippet_str.split('<_post_>')
            if len(snippet_list) == 1:
                return snippet_list
            else:
                return snippet_list[1:]

        def progress(iterable, max_items, max_n=30):
            n = max_items
            for index, element in enumerate(iterable):
                j = (index + 1) / n
                print(
                    '\r[{:{}s}] {}%'.format('=' * int(max_n * j), max_n,
                                            int(100 * j)),
                    end='')
                yield index, element
            print()

        metadata = []
        db_conn = sqlite3.connect(self.database_path)
        c = db_conn.cursor()
        c.execute(query.format(id_list=str(tuple(qids))))
        max_items = len(qids)
        for _, row in progress(c, max_items):
            str_out = {
                'PostId': row[0],
                'Score': row[1],
                'Title': row[2],
                'Tags': row[3][1:-1].replace('><', ' ').split(),
                'SnippetCount': row[4],
                'Snippets': split_snippets(row[5])
            }
            metadata.append(str_out)

        with open(os.path.join(self.export_dir, 'metadata.json'), 'w') as out:
            json.dump(metadata, out, indent=2)

    def build_search_index(self, index_dataset, model, keys=['Title', 'Body']):
        """Placeholder"""

        def split_tags(tagstring_list):
            taglist_list = []
            for row in list(tagstring_list):
                taglist_list.append(' '.join(
                    [tag for tag in row if tag != 'java']))
            if len(taglist_list) == 0:
                taglist_list.append('unk')
            return taglist_list

        def build_dict(index_dataset, model_path, load_model_fn, build_vecs_fn,
                       keys):
            model = load_model_fn(model_path)
            search_index = {key + 'V': [] for key in keys}
            for key in keys:
                key_text_list = []
                if key == 'Tags':
                    key_text_list = split_tags(index_dataset[key])
                else:
                    key_text_list = list(index_dataset[key])
                search_index[key + 'V'] = build_vecs_fn(model, key_text_list)
            output_path = os.path.join(
                self.export_dir,
                os.path.basename(model_path)[:-4] + '_post_index.pkl')
            return output_path, search_index

        output_path = None
        output_dict = None
        if model == 'ft':
            output_path, output_dict = build_dict(
                index_dataset, self.fasttext_path, load_ft_model,
                build_ft_vecs, keys)
        elif model == 'tfidf':
            output_path, output_dict = build_dict(
                index_dataset, self.tfidf_path, load_tfidf_model,
                build_tfidf_vecs, keys)
        elif model == 'glove':
            output_path, output_dict = build_dict(
                index_dataset, self.glove_path, load_glove_model,
                build_glove_vecs, keys)
        else:
            raise ValueError('Unknown model type {}.'.format(model))

        with open(output_path, 'wb') as out:
            pickle.dump(output_dict, out)

    def build_index(self,
                    index_query,
                    metadata_query,
                    processed_dataset=None,
                    build_metadata=True,
                    build_dataset=True,
                    build_ft_index=True,
                    build_tfidf_index=True,
                    build_glove_index=True,
                    build_wv_index=True):
        def build_init_index_dataset(index_query):
            qids = frozenset(self._fetch_qids(index_query))
            return self._build_index_dataset(qids)

        def load_processed_index_dataset(index_dataset,
                                         body_corpus=None,
                                         title_corpus=None):
            if isinstance(index_dataset, str):
                if os.path.exists(index_dataset):
                    df = pd.read_pickle(index_dataset)
                    keys = list(df.columns)
                    return {key: list(df[key]) for key in keys}
            else:
                # Tags are not processed until the search index building process
                index_dataset['Body'] = self._load_text_list(body_corpus)
                index_dataset['Title'] = self._load_text_list(title_corpus)

        # File Paths
        bcorpus = os.path.join(self.temp_dir, 'body_corpus')
        final_bcorpus = os.path.join(self.temp_dir, 'final_body_corpus')
        tcorpus = os.path.join(self.temp_dir, 'title_corpus')
        final_tcorpus = os.path.join(self.temp_dir, 'final_title_corpus')
        wordvec_output_path = os.path.join(
            self.export_dir,
            os.path.basename(self.fasttext_path)[:-4] + '_wordvec_index.pkl')

        # Index Dataset
        index_ids = None
        index_dataset = None
        dataset_processed = False

        if build_metadata:
            if index_dataset is None:
                index_ids, index_dataset = build_init_index_dataset(
                    index_query)
            # Build unprocessed index dataset and metadata index
            print('Building search index dataset and metadata lookup...')
            self.build_metadata_index(index_ids, metadata_query)

        if build_dataset:
            if index_dataset is None:
                index_ids, index_dataset = build_init_index_dataset(
                    index_query)
            # Dump question bodies & titles to disk and process each corpus
            print('Processing body & title corpora...')
            self._dump_text_list(bcorpus, index_dataset['Body'])
            process_corpus(bcorpus, final_bcorpus, True, 'norm')
            self._dump_text_list(tcorpus, index_dataset['Title'])
            process_corpus(tcorpus, final_tcorpus, False, 'norm')

            # Load processed bodies & titles from disk
            load_processed_index_dataset(index_dataset, final_bcorpus,
                                         final_tcorpus)
            dataset_processed = True

            # Sanity check after text processing
            if len(index_ids) != len(index_dataset['Body']):
                raise Exception('Length mismatch on Id & Body lists.')
            if len(index_ids) != len(index_dataset['Title']):
                raise Exception('Length mismatch on Id & Title lists.')

            # Save index dataset dataframe to export folder
            idataset_df = os.path.join(self.export_dir,
                                       'data/index_dataset.pkl')
            if not os.path.exists(os.path.dirname(idataset_df)):
                os.makedirs(os.path.dirname(idataset_df))
            pd.DataFrame(
                data=index_dataset, index=index_ids).to_pickle(idataset_df)

        # Load processed index from disk to build ft or tfidf search index
        if index_dataset is None or not dataset_processed:
            if build_ft_index or build_tfidf_index or build_glove_index:
                if processed_dataset:
                    index_dataset = load_processed_index_dataset(
                        processed_dataset)
                else:
                    raise Exception('Index dataset required')

        if build_ft_index:
            print('Building fasttext search index...')
            self.build_search_index(index_dataset, 'ft')

        if build_tfidf_index:
            print('Building tfidf search index...')
            self.build_search_index(index_dataset, 'tfidf')

        if build_glove_index:
            print('Building GloVe search index...')
            self.build_search_index(index_dataset, 'glove')

        if build_wv_index:
            print('Exporting word vector index...')
            build_wordvec_index(self.wordvec_path, wordvec_output_path)


def main(question_dataframe, database_path, fasttext_model_path,
         tfidf_model_path, glove_index_path, index_dataset, temp_dir,
         export_dir, index_qids_query, metadata_query):

    indexbuilder = IndexBuilder(
        qdataframe_path=question_dataframe,
        database_path=database_path,
        fasttext_path=fasttext_model_path,
        tfidf_path=tfidf_model_path,
        glove_path=glove_index_path,
        temp_dir=temp_dir,
        export_dir=export_dir)

    indexbuilder.build_index(
        index_query=index_qids_query,
        metadata_query=metadata_query,
        processed_dataset=index_dataset,
        build_metadata=False,
        build_dataset=False,
        build_ft_index=True,
        build_tfidf_index=False,
        build_glove_index=False,
        build_wv_index=False)


if __name__ == '__main__':
    QID_QUERY = "SELECT Id FROM questions WHERE AcceptedAnswerId NOT NULL AND Score>=1 AND AnswerCount>=1 AND SnippetCount>=1 ORDER BY Id"
    METADATA_QUERY = "SELECT Id, Score, Title, Tags, SnippetCount, Snippets FROM questions WHERE Id IN {id_list} ORDER BY Id"
    main('data/final_q_posts', 'database/javaposts.db',
         'wordvec_models/fasttext_archive/ft_v0.6.1.bin',
         'wordvec_models/tfidf_archive/tfidf_v0.3.pkl',
         'wordvec_models/glove_archive/glove_v0.1.1.pkl',
         'wordvec_models/index/data/index_dataset.pkl', 'temp_files',
         'wordvec_models/index', QID_QUERY, METADATA_QUERY)
