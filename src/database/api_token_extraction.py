#!/usr/bin/env python

#
# Extracts API calls from code snippets found in answer post bodies.
#

import os
import re
import sys
import sqlite3

from bs4 import BeautifulSoup

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('..')

from code_parser.codeparser_socket import CodeParserSocket
from code_parser.codeparser import ERROR_MESSAGE, EMPTY_MESSAGE

QID_INDEX = 0
BODY_INDEX = 1

ans_query = 'SELECT ParentId, Body FROM answers ORDER BY Score DESC'


def extract_api_tokens(row, codeparser, api_index, tag_name='code'):
    soup = BeautifulSoup(row[BODY_INDEX], 'lxml')
    for tag_html in soup.find_all(tag_name):
        tag_text = tag_html.get_text().strip()
        if tag_text != '':
            code = codeparser.parse_code(tag_text, row[QID_INDEX]).strip()
            if code != EMPTY_MESSAGE and code != ERROR_MESSAGE:
                code_seq = code.split(', ')
                for t in code_seq:
                    api_index.write(re.sub(r'^_(IM|OC|MC)_', r'', t) + '\n')


def build_api_list(db_path, export_path):
    codeparser = CodeParserSocket(
        extract_sequence=True,
        keep_imports=True,
        keep_comments=False,
        keep_literals=False,
        keep_method_calls=True,
        keep_unsolved_method_calls=False)

    c = sqlite3.connect(db_path).cursor()
    max_rows = c.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
    c.execute(ans_query)
    with open(export_path, 'w') as api_index:
        for idx, row in enumerate(c):
            print('\rrow:', idx, '/', max_rows, end='')
            extract_api_tokens(row, codeparser, api_index)

if __name__ == '__main__':
    build_api_list('javaposts.db', 'api_list.txt')
