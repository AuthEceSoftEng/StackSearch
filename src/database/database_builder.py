#!/usr/bin/env python
"""
Converts StackOverflow data-dump xml files into separate sqlite databases.
Required files: Posts.xml, Comments.xml, Tags.xml, Postlinks.xml
                Users.xml
Output: posts.db, comments.db, tags.db, postlinks.db, users.db

In case some of those dbs aren't required simply remove the corresponding 
names from the tables variable.

The build_java_db function requires the created posts.db and comments.db files
to build a database containing only java related posts with certain selected fields.
    
Java DB Tables: answers comments questions.
"""

import os
import sys
import logging
import sqlite3
from lxml import etree

tables = {
    'Posts': {
        'Id': 'INTEGER',
        'PostTypeId': 'INTEGER',  # 1: Question, 2: Answer
        'ParentId': 'INTEGER',  # (only present if PostTypeId is 2)
        'AcceptedAnswerId': 'INTEGER',  # (only present if PostTypeId is 1)
        'CreationDate': 'DATETIME',
        'Score': 'INTEGER',
        'ViewCount': 'INTEGER',
        'Body': 'TEXT',
        'OwnerUserId': 'INTEGER',  # (present only if user has not been deleted)
        'OwnerDisplayName': 'TEXT',
        'LastEditorUserId': 'INTEGER',
        'LastEditorDisplayName': 'TEXT',  # ="Rich B"
        'LastEditDate': 'DATETIME',  #="2009-03-05T22:28:34.823"
        'LastActivityDate': 'DATETIME',  #="2009-03-11T12:51:01.480"
        'CommunityOwnedDate': 'DATETIME',  #(present only if post is community wikied)
        'Title': 'TEXT',
        'Tags': 'TEXT',
        'AnswerCount': 'INTEGER',
        'CommentCount': 'INTEGER',
        'FavoriteCount': 'INTEGER',
        'ClosedDate': 'DATETIME'
    },
    'Comments': {
        'Id': 'INTEGER',
        'PostId': 'INTEGER',
        'Score': 'INTEGER',
        'Text': 'TEXT',
        'CreationDate': 'DATETIME',
        'UserId': 'INTEGER',
        'UserDisplayName': 'TEXT'
    },
    'Tags': {
        'Id': 'INTEGER',
        'TagName': 'TEXT',
        'Count': 'INTEGER',
        'ExcerptPostId': 'INTEGER',
        'WikiPostId': 'INTEGER'
    },
    'PostLinks': {
        'Id': 'INTEGER',
        'CreationDate': 'DATETIME',
        'PostId': 'INTEGER',
        'RelatedPostId': 'INTEGER',
        'LinkTypeId': 'INTEGER'
    },
    'Users': {
        'Id': 'INTEGER',
        'Reputation': 'INTEGER',
        'CreationDate': 'DATETIME',
        'DisplayName': 'TEXT',
        'EmailHash': 'TEXT',
        'LastAccessDate': 'DATETIME',
        'WebsiteUrl': 'TEXT',
        'Location': 'TEXT',
        'Age': 'INTEGER',
        'AboutMe': 'TEXT',
        'Views': 'INTEGER',
        'UpVotes': 'INTEGER',
        'DownVotes': 'INTEGER'
    }
}


def xml_to_sqlite(file_name, structure, dump_path='.', dump_database_name='posts.db',
                create_query='CREATE TABLE IF NOT EXISTS {table} ({fields})',
                insert_query='INSERT INTO {table} ({columns}) VALUES ({values})',
                log_filename='so-parser.log'):
    logging.basicConfig(filename=os.path.join(dump_path, log_filename), level=logging.INFO)
    db = sqlite3.connect(os.path.join(dump_path, dump_database_name))
    print("Opening {0}.xml".format(file_name))
    with open(os.path.join(dump_path, file_name + '.xml'), 'rb') as xml_file:
        tree = etree.iterparse(xml_file)
        table_name = file_name
        sql_create = create_query.format(
                table=table_name,
                fields=", ".join(['{0} {1}'.format(name, _type) for name, _type in list(structure.items())]))
        print('Creating table {0}'.format(table_name))
        try:
            logging.info(sql_create)
            db.execute(sql_create)
        except Exception as e:
            logging.warning(e)

        count = 0
        for events, row in tree:
            try:
                if row.attrib.values():
                    logging.debug(row.attrib.keys())
                    query = insert_query.format(
                        table=table_name,
                        columns=', '.join(row.attrib.keys()),
                        values=('?, ' * len(row.attrib.keys()))[:-2])
                    vals_str = []
                    for key, val in row.attrib.items():
                        if structure[key] == 'INTEGER':
                            vals_str.append(int(val))
                        elif structure[key] == 'BOOLEAN':
                            vals_str.append(1 if val=="TRUE" else 0)
                        else:
                            vals_str.append(val)
                    db.execute(query, vals_str)
                    count += 1
                    if (count % 1000 == 0):
                        print("\r{}".format(count), end='')
            except Exception as e:
                logging.warning(e)
                print("x", end="")
            finally:
                row.clear()
                while row.getprevious() is not None:
                    del row.getparent()[0]
        print("\n")
        db.commit()
        del (tree)

def build_java_db(posts_db_name, comments_db_name, export_path='.', 
                                            export_database_name='javaposts.db'):
    # Get question Ids Tagged as Java, OracleJDK or Swing (avoid Javascript)
    print('---START---', end='\n\n')
    posts_db = sqlite3.connect(posts_db_name)
    sc = posts_db.cursor()
    print('Fetching java question ids...')
    sc.execute('''SELECT Id from Posts WHERE PostTypeId=1 AND 
    (Tags LIKE '%java%' OR Tags LIKE '%swing%') AND 
    Tags NOT LIKE '%javascript%' ORDER BY Id ASC''')
    java_qids = []
    for idx, row in enumerate(sc):
        print('\rIds:', idx, end='')
        java_qids.append(row[0])
    qids_str = str(tuple(java_qids))

    # Create database and tables
    print('\n\nCreating javaposts database...')
    javaposts_db = sqlite3.connect(os.path.join(export_path, export_database_name))
    dc = javaposts_db.cursor()
    dc.execute(
    '''CREATE TABLE IF NOT EXISTS "questions" (Id INTEGER, AcceptedAnswerId INTEGER, 
    Title TEXT, Body TEXT, Tags TEXT, Score INTEGER, FavoriteCount INTEGER, 
    ViewCount INTEGER, AnswerCount INTEGER, CommentCount INTEGER, OwnerUserId INTEGER, 
    CreationDate DATETIME, LastEditDate DATETIME)''')
    dc.execute(
    '''CREATE TABLE IF NOT EXISTS "answers" (Id INTEGER, ParentId INTEGER, Body TEXT, 
    Score INTEGER, CommentCount INTEGER, OwnerUserId INTEGER, CreationDate DATETIME, 
    LastEditDate DATETIME)''')
    dc.execute(
    '''CREATE TABLE IF NOT EXISTS "comments" (Id INTEGER, PostId INTEGER, Text TEXT, 
    Score INTEGER, UserId INTEGER, CreationDate DATETIME)''')
    javaposts_db.commit()

    # Insert java questions using qids
    print('Fetching question rows...')
    sc.execute('''SELECT Id, AcceptedAnswerId, Title, Body, Tags, Score, 
    FavoriteCount, ViewCount, AnswerCount, CommentCount, OwnerUserId, CreationDate, 
    LastEditDate FROM Posts WHERE PostTypeId=1 AND Id IN ''' + qids_str + ' ORDER BY Id')
    cols = [d[0] for d in sc.description]
    cols_str = ','.join(cols)
    vals_str = ','.join(('?',) * len(cols))
    for idx, row in enumerate(sc):
        print('\rInserting row:', idx, end='')
        query = 'INSERT INTO questions ({columns}) VALUES ({values})'.format(
                columns=cols_str,
                values=vals_str)
        dc.execute(query, row)

    # Insert answers using qids on ParentId
    print('\n\nFetching answer rows...')
    sc.execute('''SELECT Id, ParentId, Body, Score, CommentCount, OwnerUserId, 
    CreationDate, LastEditDate FROM Posts WHERE PostTypeId=2 AND ParentId IN ''' + qids_str +
    ' ORDER BY Id')
    cols = [d[0] for d in sc.description]
    cols_str = ','.join(cols)
    vals_str = ','.join(('?',) * len(cols))
    java_ansids = []
    for idx, row in enumerate(sc):
        print('\rInserting row:', idx, end='')
        java_ansids.append(row[0])
        query = 'INSERT INTO answers ({columns}) VALUES ({values})'.format(
                columns=cols_str,
                values=vals_str)
        dc.execute(query, row)
    
    posts_db.close()
    all_ids = java_qids + java_ansids
    all_ids_str = str(tuple(all_ids))

    # Insert comments using qids and ansids
    comments_db = sqlite3.connect(comments_db_name)
    sc = comments_db.cursor()
    print('\n\nFetching comment rows...')
    sc.execute('''SELECT Id, PostId, Text, Score, UserId, CreationDate FROM Comments 
    WHERE PostId IN ''' + all_ids_str + ' ORDER BY Id')
    cols = [d[0] for d in sc.description]
    cols_str = ','.join(cols)
    vals_str = ','.join(('?',) * len(cols))
    for idx, row in enumerate(sc):
        print('\rInserting row:', idx, end='')
        query = 'INSERT OR REPLACE INTO comments ({columns}) VALUES ({values})'.format(
                columns=cols_str,
                values=vals_str)
        dc.execute(query, row)
    
    javaposts_db.commit()
    javaposts_db.close()
    print('\n\n---END---')

def build_java_postlinks(java_db_path, postlinks_db_path):
    java_db = sqlite3.connect(java_db_path)
    jc = java_db.cursor()
    jc.execute('''SELECT Id FROM questions ORDER BY Id ASC''')
    qids = []
    for row in jc:
        qids.append(row[0])
    java_db.close()
    qids_str = str(tuple(qids))

    # Create new postlink database where both PostId and RelatedPostId are in the question ids
    print('Creating new postlinks database...')
    new_postlinks_db = sqlite3.connect('new_postlinks.db')
    new_plc = new_postlinks_db.cursor()
    new_plc.execute('''CREATE TABLE IF NOT EXISTS "postlinks" (Id INTEGER, CreationDate DATETIME, 
    PostId INTEGER, RelatedPostId INTEGER, LinkTypeId INTEGER)''')
    new_postlinks_db.commit()

    #  Fetch rows from old postlinks database
    postlinks_db = sqlite3.connect(postlinks_db_path)
    plc = postlinks_db.cursor()
    print('\nFetching postlink rows...')
    plc.execute('''SELECT Id, CreationDate, PostId, RelatedPostId, LinkTypeId FROM PostLinks 
    WHERE PostId IN {} AND RelatedPostId IN {}'''.format(qids_str, qids_str))
    cols = [d[0] for d in plc.description]
    cols_str = ','.join(cols)
    vals_str = ','.join(('?',) * len(cols))
    for idx, row in enumerate(plc):
        print('\rInserting row:', idx, end='')
        query = 'INSERT OR REPLACE INTO postlinks ({columns}) VALUES ({values})'.format(
                columns=cols_str,
                values=vals_str)
        new_plc.execute(query, row)
    new_postlinks_db.commit()
    new_postlinks_db.close()
    postlinks_db.close()
    print('\n\n---END---')


if __name__ == '__main__':
    """
    for key, value in tables.items():
        xml_to_sqlite(file_name=key, dump_database_name=key.lower() + '.db', structure=value)
    
    # Manipulate created database files and create a java-posts/comments database
    build_java_db(posts_db_name='posts.db', comments_db_name='comments.db')
    """
    build_java_postlinks('javaposts.db', 'javapostlinks.db')
