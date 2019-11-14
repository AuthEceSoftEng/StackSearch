#!/usr/bin/env python

##
# 1. Extracts entities from question & answer post bodies 
# using a trained crf model.
# 2. Rebuilds the sqlite database inserting the 'Entities' column in
# 'questions' table.
##

import os
import re
import sys
import glob
import json
import sqlite3
from collections import OrderedDict
from multiprocessing import Process, Manager, current_process

import crfsuite
import numpy as np
from bs4 import BeautifulSoup

sys.path.append('..')
from ner.feature_extractor import FeatureExtractor
from ner.text_processing.corpus_utils import CorpusUtils

## Script Path
script_path = os.path.dirname(os.path.realpath(__file__))

## Queries
q_query = 'SELECT Id, Body FROM questions ORDER BY Id'
ans_query = 'SELECT ParentId, Body FROM answers ORDER BY ParentId'
new_qtable = '''CREATE TABLE IF NOT EXISTS "questions" 
    (Id INTEGER, AcceptedAnswerId INTEGER, Title TEXT, Body TEXT, Tags TEXT, 
    Score INTEGER, Entities TEXT, SnippetCount INTEGER, Snippets TEXT, 
    FavoriteCount INTEGER, ViewCount INTEGER, AnswerCount INTEGER, 
    CommentCount INTEGER, OwnerUserId INTEGER, CreationDate DATETIME, 
    LastEditDate DATETIME)'''
old_cols = '''Id, AcceptedAnswerId, Title, Body, Tags, Score, SnippetCount, 
Snippets, FavoriteCount, ViewCount, AnswerCount, CommentCount, OwnerUserId, 
CreationDate, LastEditDate'''
new_cols = '''Id, AcceptedAnswerId, Title, Body, Tags, Score, Entities,
SnippetCount, Snippets, FavoriteCount, ViewCount, AnswerCount, CommentCount, 
OwnerUserId, CreationDate, LastEditDate'''

## CRF model path
crf_model_path = '../ner/model_archive/ner_v0.1.crf'

## Database path
database_path = 'javaposts.db'

## Separator regex
SEP_PATTERN = re.compile(r'(=|-|\+|\*|#|_){4,}')


def dump_dict(dict_name, _dict, index=''):
    output_name = dict_name + str(index) + '.json'
    with open(os.path.join(script_path, 'temp', output_name), 'w') as out:
        json.dump(_dict, out, indent=2)


def to_crfsuite(X):
    """
    Convert an item sequence into an object compatible with crfsuite
    Python module.

    @type   X:      list of mapping objects
    @param  X:      The sequence.
    @rtype          crfsuite.ItemSequence
    @return         The same sequence in crfsuite.ItemSequence type.
    """
    xseq = crfsuite.ItemSequence()
    for x in X:
        item = crfsuite.Item()
        for f in x['F']:
            item.append(crfsuite.Attribute(f))
        xseq.append(item)
    return xseq


def extract_entities(tagger, feat_extractor, sent):
    X = [{'F': [], 'token': t, 'entity': 'O'} for t in sent.split()]
    feat_extractor.sequence_features(X)
    yseq = tagger.tag(to_crfsuite(X))

    entities = []
    prev_ent = ''
    for ii, v in enumerate(X):
        if yseq[ii] != 'O':
            cur_ent = yseq[ii][2:]
            if yseq[ii].startswith('I-') and prev_ent == cur_ent:
                ## Check if same entity with previous (e.g. B-Plat, I-Plat).
                # If so combine them.
                entities[-1] = entities[-1] + ' ' + v['token']
            else:
                prev_ent = yseq[ii][2:]
                entities.append(prev_ent + '##' + v['token'])
    return entities


def build_entity_dict_proc(shared_dict, keys):
    proc_name = current_process().name
    print(proc_name + ': initializing')

    tagger = crfsuite.Tagger()
    tagger.open(crf_model_path)
    feat_extractor = FeatureExtractor(use_models=False)
    print(proc_name + ': tagger and feature extractor loaded')

    ents = OrderedDict()

    for ii, key in enumerate(keys):
        sents = shared_dict[key]
        if key not in ents:
            ents[key] = []
        for sent in sents:
            ents[key].extend(extract_entities(tagger, feat_extractor, sent))
        ents[key] = list(OrderedDict.fromkeys(ents[key]))
        # sampling/backup
        if ii % 50000 == 0:
            print(proc_name + ':', ii)
            dump_dict(proc_name + '_', ents, ii)

    dump_dict(proc_name + '_end', ents, len(keys))
    print(proc_name + ':', len(keys))
    print(proc_name + ': terminated')


def build_post_dict(db_name, query):
    def prep_text(soup):
        # Remove code fragments
        for code in soup.find_all('pre'):
            code.decompose()

        text = soup.get_text()
        # Strip separators
        text = SEP_PATTERN.sub(' ', text)
        # Strip whitespace
        text = text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
        return ' '.join(text.split())

    post_dict = OrderedDict()
    db = sqlite3.connect(db_name)
    c = db.cursor()
    c.execute(query)
    for ii, row in enumerate(c):
        print('\r@post ', ii, end='')
        soup = BeautifulSoup(row[1], 'lxml')
        text = prep_text(soup)
        if row[0] in post_dict:
            post_dict[row[0]] = ' '.join([post_dict[row[0]], text])
        else:
            post_dict[row[0]] = text
    print()
    return post_dict


def combine_dicts(q_dict, ans_dict):
    post_dict = OrderedDict()
    keys = list(q_dict.keys())
    dict_len = len(keys)

    for ii, key in enumerate(keys):
        print('\r', ii + 1, '/', dict_len, end='')
        post_dict[key] = ' '.join([q_dict[key], ans_dict.get(key, '')])
    print()
    return post_dict


def split_post_sentences(post_dict):
    cu = CorpusUtils(sent_split=True)

    output_dict = OrderedDict()
    keys = list(post_dict.keys())
    dict_len = len(post_dict)

    for ii, key in enumerate(keys):
        print('\r', ii + 1, '/', dict_len, end='')
        output_dict[key] = cu.sentence_list(post_dict[key])
    print()
    return output_dict


def main_post_preprocessing(db_name):
    print('Extracting question bodies...')
    q_dict = build_post_dict(db_name, q_query)
    dump_dict('question_dict', q_dict)

    print('Extracting answer bodies...')
    ans_dict = build_post_dict(db_name, ans_query)
    dump_dict('answer_dict', ans_dict)

    print('Combining question and answer bodies...')
    post_dict = combine_dicts(q_dict, ans_dict)
    dump_dict('post_dict', post_dict)

    # Unbind dicts to free memory
    del q_dict
    del ans_dict

    print('Splitting post bodies into sentences...')
    post_sentence_dict = split_post_sentences(post_dict)
    dump_dict('post_sent_dict', post_sentence_dict)


def main_entity_extraction(dict_name, dict_start, num_processes):
    if not os.path.exists(os.path.join(script_path, 'temp')):
        os.mkdir(os.path.join(script_path, 'temp'))

    manager = Manager()

    with open(dict_name, 'r') as _in:
        sd = json.load(_in, object_pairs_hook=OrderedDict)
        for key in list(sd.keys())[:dict_start]:
            del sd[key]
    print('Number of posts in dictionary:', len(sd))

    total_keys = list(sd.keys())
    total_keys = np.array_split(np.array(total_keys), num_processes)
    shared_dict = manager.dict(sd)

    # Unbind dict to free up memory
    del sd
    print('Shared dict and keys in memory...')

    procs = [
        Process(target=build_entity_dict_proc, args=(shared_dict, key_list))
        for key_list in total_keys
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    print('Combining temp entity-dict files...')
    # Get temp entity-dict files exported by the processes
    temp_files = glob.glob(os.path.join(script_path, 'temp', '*json'))
    temp_files = [f for f in temp_files if 'end' in os.path.basename(f)]
    temp_files = sorted(temp_files)

    # Combine and export new json
    ent_dict = OrderedDict()
    for fname in temp_files:
        with open(fname, 'r') as _in:
            ent_dict.update(json.load(_in, object_pairs_hook=OrderedDict))
    dump_dict('entity_dict', ent_dict)


def main_database_update(db_name, dict_name):
    def entity_str(ent_list):
        # Remove leading identifier (Fram, API etc.) and make strings lowercase
        ent_list = [e.split('##')[1].lower() for e in ent_list]
        # Remove trailing brackets and text artifacts in brackets
        ent_list = [re.sub(r'[\(\[]\S*$', '', e) for e in ent_list]
        # Remove 's from tokens
        ent_list = [re.sub(r'\'s', '', e) for e in ent_list]
        # Remove duplicates
        return '<_ent_>'.join(set(ent_list))

    with open(dict_name, 'r') as _in:
        ent_dict = json.load(_in,
            object_hook=lambda d: {int(k): v for k, v in d.items()})

    ent_dict = OrderedDict(sorted(ent_dict.items()))

    db = sqlite3.connect(db_name)
    src_c = db.cursor()
    des_c = db.cursor()

    # Alter questions table name
    src_c.execute('ALTER TABLE questions RENAME TO old_questions')
    # Create new questions table with an extra column {Entities}
    src_c.execute(new_qtable)
    db.commit()
    src_c.execute(
        'SELECT {} FROM old_questions ORDER BY Id ASC'.format(old_cols))

    vals_str = ','.join(('?', ) * len(new_cols.split(',')))
    for idx, row in enumerate(src_c):
        print('\rInserting row:', idx, end='')
        new_row = []
        _id = row[0]
        if _id in ent_dict:
            ent_str = entity_str(ent_dict[_id])
            new_row = row[:6] + (ent_str, ) + row[6:]
        else:
            new_row = row[:6] + ('', ) + row[6:]
        query = 'INSERT OR REPLACE INTO questions ({}) VALUES ({})'.format(
            new_cols, vals_str)
        des_c.execute(query, new_row)
    des_c.execute('DROP TABLE old_questions')
    db.commit()
    des_c.execute('VACUUM')
    db.commit()
    db.close()
    print('\nEntities inserted...')


if __name__ == '__main__':
    main_post_preprocessing('javaposts.db')
    main_entity_extraction('post_sent_dict.json', 0, os.cpu_count() - 2)
    main_database_update('javaposts.db', 'entity_dict.json')
