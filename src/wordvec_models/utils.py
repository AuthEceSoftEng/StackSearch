import os
import sys
import sqlite3

file_path = os.path.dirname(os.path.abspath(__file__))

## Database Path
POSTLINKS = os.path.join(file_path, '..', 'database/javapostlinks.db')

## PostLink Query
linked_posts_query = '''SELECT DISTINCT PostId, RelatedPostId FROM postlinks 
WHERE PostId IN {} AND RelatedPostId IN {}'''


def get_linked_posts(post_ids):
    """Given a list of PostIds, retrieves the linked pairs of that list."""
    src_ids, tgt_ids = [], []
    postids_str = str(tuple(post_ids))
    pldb = sqlite3.connect(POSTLINKS)
    c = pldb.cursor()
    c.execute(linked_posts_query.format(postids_str, postids_str))
    for row in c:
        if row[0] not in src_ids and row[0] not in tgt_ids:
            src_ids.append(row[0])
            tgt_ids.append(row[1])
    pldb.close()
    return src_ids, tgt_ids


def print_linked_posts(post_ids):
    """Given a list of PostIds, prints the linked pairs of that list."""
    src_ids, tgt_ids = get_linked_posts(post_ids)
    for idx, sid in enumerate(src_ids):
        print(sid, 'linked with', tgt_ids[idx])
    print()