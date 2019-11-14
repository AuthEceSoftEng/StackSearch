#!/usr/bin/env python

#
# 1. Extracts code snippets from question & answer post bodies.
# 2. Rebuilds the sqlite database inserting the new 'SnippetCount' 'Snippets'
# columns.
#

import os
import re
import sys
import sqlite3
from collections import OrderedDict

import pandas as pd
from bs4 import BeautifulSoup

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('..')

from code_parser.codeparser_socket import CodeParserSocket
from code_parser.codeparser import ERROR_MESSAGE, EMPTY_MESSAGE

# Stack Overflow Attribution
so_attr = 'Code extracted from Stack Overflow'
user_attr = 'User: https://stackoverflow.com/users/'
post_attr = '\nPost: https://stackoverflow.com/questions/'

# Query
QID_INDEX = 0
ANSID_INDEX = 1
BODY_INDEX = 2
USERID_INDEX = 3
SCORE_INDEX = 4

ans_query = 'SELECT ParentId, Id, Body, OwnerUserId, Score FROM answers ORDER BY Score DESC'

new_qtable = '''CREATE TABLE IF NOT EXISTS "questions" 
    (Id INTEGER, AcceptedAnswerId INTEGER, Title TEXT, Body TEXT, Tags TEXT, Score INTEGER, Entities TEXT, 
    SnippetCount INTEGER, Snippets TEXT, FavoriteCount INTEGER, ViewCount INTEGER, 
    AnswerCount INTEGER, CommentCount INTEGER, OwnerUserId INTEGER, 
    CreationDate DATETIME, LastEditDate DATETIME)'''

old_cols = '''Id, AcceptedAnswerId, Title, Body, Tags, Score, Entities, 
FavoriteCount, ViewCount, AnswerCount, CommentCount, OwnerUserId, 
CreationDate, LastEditDate'''

new_cols = '''Id, AcceptedAnswerId, Title, Body, Tags, Score, Entities, 
SnippetCount, Snippets, FavoriteCount, ViewCount, AnswerCount, 
CommentCount, OwnerUserId, CreationDate, LastEditDate'''


def extract_code_snippets(row, codeparser, tag_name='code'):
    snippet_list = []
    soup = BeautifulSoup(row[BODY_INDEX], 'lxml')
    for tag_html in soup.find_all(tag_name):
        tag_text = tag_html.get_text().strip()
        if tag_text != '':
            code_snippet = codeparser.parse_code(tag_text,
                                                 row[QID_INDEX]).strip()
            if code_snippet != EMPTY_MESSAGE and code_snippet != ERROR_MESSAGE:
                snippet_list.append(tag_text)
    if len(snippet_list) > 0:
        attr = ''.join([post_attr, str(row[ANSID_INDEX])])
        snippet_str = ''.join([
            attr, '\n##Score {}\n'.format(row[SCORE_INDEX]),
            '<_code_>'.join(snippet_list)
        ])
        return len(snippet_list), snippet_str
    return 0, ''


def build_snippet_index(db_path):
    codeparser = CodeParserSocket(
        extract_sequence=False,
        keep_imports=False,
        keep_comments=False,
        keep_literals=True)

    c = sqlite3.connect(db_path).cursor()
    max_rows = c.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
    c.execute(ans_query)
    question_dict = OrderedDict()
    for idx, row in enumerate(c):
        print('\rrow:', idx, '/', max_rows, end='')
        qid = row[QID_INDEX]
        snippet_count, snippet_str = extract_code_snippets(row, codeparser)
        if qid not in question_dict:
            question_dict[qid] = {
                'SnippetCount': snippet_count,
                'Snippets': snippet_str
            }
        else:
            if snippet_count > 0:
                new_str = None
                old_count = question_dict[qid]['SnippetCount']
                new_count = old_count + snippet_count
                # avoid joining empty string (no snippets initially) with non-empty
                # leaves leading <_post_> tag which later complicates splitting
                # new_str = re.sub(r'^<_post_>', r'', new_str) or alternatively code bellow
                if old_count > 0:
                    new_str = '<_post_>'.join(
                        [question_dict[qid]['Snippets'], snippet_str])
                else:
                    new_str = snippet_str
                question_dict[qid] = {
                    'SnippetCount': new_count,
                    'Snippets': new_str
                }
    codeparser.close()

    return OrderedDict(sorted(question_dict.items()))


def insert_snippet_data(db_path, snippet_df):
    db = sqlite3.connect(db_path)
    src_c = db.cursor()
    des_c = db.cursor()
    # Alter questions table name
    src_c.execute('ALTER TABLE questions RENAME TO old_questions')
    # Create new questions table with two extra columns {SnippetCount, Snippets}
    src_c.execute(new_qtable)
    db.commit()
    src_c.execute(
        'SELECT {} FROM old_questions ORDER BY Id ASC'.format(old_cols))
    df_index = frozenset(snippet_df.index)
    vals_str = ','.join(('?', ) * len(new_cols.split(',')))
    for idx, row in enumerate(src_c):
        print('\rInserting row:', idx, end='')
        new_row = []
        _id = row[0]
        if _id in df_index:
            snippet_info = snippet_df.loc[_id]
            snippet_count = int(snippet_info['SnippetCount'])
            snippets = snippet_info['Snippets'].strip()
            new_row = row[:7] + tuple([snippet_count, snippets]) + row[7:]
        else:
            new_row = row[:7] + (0, '') + row[7:]
        query = 'INSERT INTO questions ({columns}) VALUES ({values})'.format(
            columns=new_cols, values=vals_str)
        des_c.execute(query, new_row)
    des_c.execute('DROP TABLE old_questions')
    db.commit()
    des_c.execute('VACUUM')
    db.commit()
    db.close()
    print('\nValues inserted...')


def main(db_path, snippet_df_path=None):
    snippet_df = None
    if not snippet_df_path:
        print('Creating snippet index...')
        snippet_df = pd.DataFrame.from_dict(
            build_snippet_index(db_path), orient='index')
        snippet_df.to_pickle('snippet_index.pkl')
    else:
        print('Loading snippet index...')
        snippet_df = pd.read_pickle(snippet_df_path)
    print('Inserting snippet data into database...')
    insert_snippet_data(db_path, snippet_df)


if __name__ == '__main__':
    main('javaposts.db')#, 'snippet_index.pkl')
