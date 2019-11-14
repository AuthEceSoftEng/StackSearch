#!/usr/bin/env python

import os
import gc
import json
import pickle
import sqlite3
import argparse

import pandas as pd

from code_parser.codeparser_stdin import CodeParserStdin
from post_classifier.classifier import PostClassifier
from post_classifier.utils import (list_to_disk, load_number_list,
                                   load_text_list, remove_rows)
from post_classifier.vectorizer import Vectorizer
from text_processing.text_eval import eval_text
from text_processing.utils import process_corpus

# Question query default settings
score_threshold = -3
ans_count_threshold = 1

# Database Queries
INIT_QUESTION_QUERY = '''SELECT Body, Id FROM questions
    WHERE AnswerCount>={ans_count} AND Score>={score} ORDER BY Id ASC'''
INIT_ANSWER_QUERY = '''SELECT Body, Id FROM answers
    WHERE ParentId IN {id_list} ORDER BY ParentId ASC'''
INIT_COMMENT_QUERY = '''SELECT Text AS Body, Id FROM comments
    WHERE PostId IN {id_list} ORDER BY PostId ASC'''

FINAL_QUESTION_QUERY = '''SELECT Id, Title, Tags, Entities, SnippetCount, Score FROM questions
    WHERE Id IN {id_list} ORDER BY Id ASC'''
FINAL_ANSWER_QUERY = '''SELECT Id, ParentId, Score FROM answers
    WHERE Id IN {id_list} ORDER BY ParentId ASC'''
FINAL_COMMENT_QUERY = '''SELECT Id, PostId FROM comments
    WHERE Id IN {id_list} ORDER BY PostId ASC'''


class CorpusBuilder:
    def __init__(self,
                 classifier_path,
                 vectorizer_dict_path,
                 database_path,
                 export_dir,
                 text_eval_fn,
                 qparams=None):

        self.classifier = PostClassifier(classifier_path)
        self.vectorizer = Vectorizer(dictionary_path=vectorizer_dict_path)
        self.db_conn = sqlite3.connect(database_path)
        self.text_eval_fn = text_eval_fn
        self.qparams = qparams

        # Create paths
        self.temp_dir = 'temp_files'
        self.export_dir = export_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        self.init_dfs = {
            'q': os.path.join(export_dir, 'init_q_posts'),
            'a': os.path.join(export_dir, 'init_a_posts'),
            'c': os.path.join(export_dir, 'init_c_posts')
        }
        self.final_dfs = {
            'q': os.path.join(export_dir, 'final_q_posts'),
            'a': os.path.join(export_dir, 'final_a_posts'),
            'c': os.path.join(export_dir, 'final_c_posts')
        }

    def _retrieve_db_data(self, query, post_type, eval_posts=True):
        c = self.db_conn.cursor()
        c.execute(query)
        cols = [d[0] for d in c.description]
        output_dict = {key: [] for key in cols}

        if eval_posts:
            codeparser = CodeParserStdin(index_path=os.path.join(
                self.export_dir, 'api_index'),
                                         extract_sequence=True,
                                         keep_imports=False,
                                         keep_comments=True,
                                         keep_literals=False,
                                         keep_method_calls=True,
                                         keep_unsolved_method_calls=False)
            # Format posts and discard low quality posts (excess punctuation)
            for idx, row in enumerate(c):
                print('\rpost:', idx, end='')
                body = row[0]
                if post_type == 'com':  # replace quote char from comments
                    body = body.replace('`', ' ')
                eval_res = self.text_eval_fn(body, row[1], codeparser)
                if eval_res != -1:
                    output_dict[cols[0]].append(eval_res)
                    for ii, val in enumerate(row[1:], start=1):
                        output_dict[cols[ii]].append(val)
            print()
            codeparser.close()
        else:
            for idx, row in enumerate(c):
                print('\rpost:', idx, end='')
                for ii, val in enumerate(row):
                    output_dict[cols[ii]].append(val)
            print()
        return output_dict

    def _filter_posts(self, posts, post_ids, post_type):
        # Load lists from disk if given a path
        if isinstance(posts, str):
            posts = load_text_list(posts)
        if isinstance(post_ids, str):
            post_ids = load_text_list(post_ids)

        # Create dump paths
        predictions_path = self.init_dfs[post_type] + '_predictions'

        # Vectorize posts and get classifier predictions
        vectorized_doc = self.vectorizer.vectorize_list(posts)
        with open(predictions_path, 'a') as pred_out:
            for batch in self.classifier.feed_data(vectorized_doc, verbose=1):
                self.classifier.save_predictions(
                    pred_out, self.classifier.make_prediction(batch, 0))

        # Filter out 'unclean' posts using the predictions
        labels = load_number_list(predictions_path, mode='bool')
        df = pd.DataFrame(data={'Body': remove_rows(posts, labels)},
                          index=remove_rows(post_ids, labels))

        if post_type == 'q':
            self.qid_list = list(df.index)
        else:
            self.ansid_list = list(df.index)

        # Save dataframe to disk
        df.to_pickle(self.init_dfs[post_type])

    def _build_initial_dataframe(self, query, post_type, keep_raw_data=False):
        db_data = self._retrieve_db_data(query, post_type)
        if keep_raw_data:
            export_path = os.path.join(self.temp_dir, 'raw_data_' + post_type)
            with open(export_path, 'wb') as out:
                pickle.dump(db_data, out)
            print('Raw intermediate file saved at {}.'.format(export_path))
        if post_type != 'c':  # skip classifier stage for comments
            self._filter_posts(db_data['Body'], db_data['Id'], post_type)
        else:
            pd.DataFrame(data={
                'Body': db_data['Body']
            }, index=db_data['Id']).to_pickle(self.init_dfs['c'])

    def _build_final_dataframe(self, query, post_type):
        def validate_data(original_ids, db_data):
            if original_ids != db_data['Id']:
                raise ValueError('Validation failed. Id mismatch.')
            del db_data['Id']  # discard ids from dict

        # Load initial dataframe (df_index: Ids)
        init_df = pd.read_pickle(self.init_dfs[post_type])
        df_index = list(init_df.index)
        df_dict = {'Body': list(init_df['Body'])}

        # Retrieve extra database info
        db_data = self._retrieve_db_data(
            query.format(id_list=str(tuple(df_index))), post_type, False)

        # Ensure Id matching
        validate_data(df_index, db_data)

        # Update dataframe dict and save final dataframe to disk
        df_dict.update(db_data)
        final_df = pd.DataFrame(data=df_dict, index=df_index)
        final_df.to_pickle(self.final_dfs[post_type])

    def build_initial_dataframes(self, qid_list=None, ansid_list=None):
        print('Building initial dataframes.')
        query = INIT_QUESTION_QUERY.format(ans_count=ans_count_threshold,
                                           score=score_threshold)
        if self.qparams:
            query = INIT_QUESTION_QUERY.format(**self.qparams)
        self._build_initial_dataframe(query, 'q')

        query = INIT_ANSWER_QUERY.format(id_list=str(tuple(self.qid_list)))
        self._build_initial_dataframe(query, 'a')

        com_postids = []
        if qid_list and ansid_list:
            com_postids = qid_list + ansid_list
        else:
            com_postids = self.qid_list + self.ansid_list

        query = INIT_COMMENT_QUERY.format(id_list=str(tuple(com_postids)))
        self._build_initial_dataframe(query, 'c')

    def build_final_dataframes(self):
        print('Building final dataframes.')
        self._build_final_dataframe(FINAL_QUESTION_QUERY, 'q')
        self._build_final_dataframe(FINAL_ANSWER_QUERY, 'a')
        self._build_final_dataframe(FINAL_COMMENT_QUERY, 'c')

    def _build_init_corpus(self):
        def progress(iterable, max_n=30):
            n = len(iterable)
            for index, element in enumerate(iterable):
                j = (index + 1) / n
                print('\r[{:{}s}] {}%'.format('=' * int(max_n * j), max_n,
                                              int(100 * j)),
                      end='')
                yield index, element
            print()

        text_list = []
        qdf = pd.read_pickle(self.final_dfs['q'])
        qids = list(qdf.index)
        qposts = list(qdf['Body'])
        qtitles = list(qdf['Title'])

        print('Building initial text corpus...')
        ansdf = pd.read_pickle(self.final_dfs['a'])
        for idx, qid in progress(qids):
            text_list.append(qtitles[idx])
            text_list.append(qposts[idx])
            text_list.extend(list(ansdf.loc[ansdf['ParentId'] == qid, 'Body']))

        print('Saving initial text corpus to disk...')
        init_corpus = os.path.join(self.export_dir, 'init_corpus')
        list_to_disk(init_corpus, text_list)
        return init_corpus

    ## TODO: include_comments=False
    def build_corpus(self,
                     init_corpus=None,
                     filter_corpus=True,
                     token_fn='norm'):
        """
        """

        if not init_corpus:
            init_corpus = self._build_init_corpus()

        final_corpus = os.path.join(self.export_dir, 'final_corpus_')
        final_corpus = final_corpus + token_fn
        if token_fn == 'norm':
            process_corpus(init_corpus, final_corpus, filter_corpus, token_fn)
        elif token_fn == 'lemma':
            process_corpus(init_corpus, final_corpus, filter_corpus, token_fn)


def main(classifier_path,
         vectorizer_dict_path,
         database_path,
         export_dir,
         text_eval_fn,
         qparams=None):

    corpus_builder = CorpusBuilder(classifier_path, vectorizer_dict_path,
                                   database_path, export_dir, text_eval_fn,
                                   qparams)

    corpus_builder.build_initial_dataframes()
    corpus_builder.build_final_dataframes()
    corpus_builder.build_corpus()


def param_parser(params_filepath, text_eval_fn):
    params = {
        'classifier_path': None,
        'vectorizer_dict_path': None,
        'database_path': None,
        'export_dir': None,
        'text_eval_fn': text_eval_fn,
        'qparams': None,
    }

    with open(params_filepath, 'r') as _in:
        params_dict = json.load(_in)

    params['classifier_path'] = params_dict['corpus']['classifier_path']
    params['vectorizer_dict_path'] = params_dict['corpus'][
        'vectorizer_dict_path']
    params['database_path'] = params_dict['database_path']
    params['export_dir'] = params_dict['corpus']['export_dir']
    params['qparams'] = params_dict['corpus']['qparams']

    return params


def validate_file(filepath):
    if not os.path.exists(filepath):
        print('File "{}" does not exist.'.format(filepath))
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Corpus builder.')
    parser.add_argument(
        '-p',
        '--params',
        default='params.json',
        help='Path to a valid params file. (default: params.json)')

    args = parser.parse_args()
    validate_file(args.params)

    p = param_parser(args.params, eval_text)
    main(**p)
