import re
import string
import warnings
from bs4 import BeautifulSoup

# Suppress BeautifulSoup warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# regular expressions
split_regex = r""" |!|"|\#|\$|%|&|'|\(|\)|\*|\+|,|-|\.|/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|`|{|}|~"""
SPLIT_PATTERN = re.compile(split_regex)
SEP_PATTERN = re.compile(r'(=|-|\+|\*|#|_){4,}')

line_quality_threshold = 1.6


def eval_text(text, identifier, codeparser):
    text = format_post(text, identifier, codeparser)
    return eval_line(text)


def eval_line(text):
    text_len = len(text)
    if text_len > 1000:
        if line_quality(text) >= line_quality_threshold:
            return text
    else:
        return text
    return -1


def line_quality(text):
    punct = 0
    for c in text:
        if c in string.punctuation:
            punct += 1
    if punct == 0:
        punct = 1
    words = len(list(filter(None, SPLIT_PATTERN.split(text))))
    quality = words / punct
    return quality


def format_post(post, identifier, codeparser, tag_param='pre'):
    soup = BeautifulSoup(post, 'lxml')
    for tag in soup.find_all(tag_param):
        sequence = codeparser.tokenize_sequence(
            tag.get_text(), identifier, unique_tokens=True)
        tag.string = ' '.join(sequence)
    return strip_whitespace(strip_separators(soup.get_text()))


def strip_separators(text):
    return SEP_PATTERN.sub(' ', text)


def strip_whitespace(text):
    text_out = text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
    return ' '.join(text_out.split())
