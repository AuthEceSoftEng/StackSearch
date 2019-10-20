import re
import sys
import sqlite3
from bs4 import BeautifulSoup

QUERY = """SELECT Id, Body FROM questions WHERE (
            Title LIKE '%exception%' OR 
            Title LIKE '%debug%' OR 
            Title LIKE '%error%' OR 
            Title LIKE '%fail%' OR 
            Title LIKE '%warning%'
            ) AND 
            Body NOT LIKE '%</pre>%' AND 
            ((length(Body) - length(replace(Body," ", "")))+1)>=150
        """

def strip_html_tags(text):
    soup = BeautifulSoup(text, 'lxml')
    with open('found_pre', 'a') as pre_out:
        for p in soup.find_all('pre'):
            pre_out.write(p.getText() + '\n')
    return soup.getText()

def strip_whitespace(text):
    text = text.replace('\n|\r\n', ' ')
    return ' '.join(text.split())

def row_feed(db_filename):
    conn = sqlite3.connect(db_filename)
    c = conn.cursor()
    c.execute(QUERY)
    for row in c:
        yield row


if __name__ == '__main__':
    db_path = sys.argv[1]
    with open('initial_posts', 'w') as out:
        with open('initial_post_ids', 'w') as id_out:
            for post in row_feed(db_path):
                post_out = strip_whitespace(strip_html_tags(post[1]))
                out.write(post_out + '\n')
                id_out.write(str(post[0]) + '\n')
