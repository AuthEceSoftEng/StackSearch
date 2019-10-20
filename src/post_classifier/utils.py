from itertools import compress


def load_text_list(filename):
    with open(filename, 'r') as f:
        posts = []
        for line in f:
            posts.append(line.strip())
    return posts


def load_number_list(filename, mode='int'):
    with open(filename, 'r') as f:
        _list = []
        for line in f:
            _list.append(int(line))
    if mode == 'bool':
        # classifier uses 1: unclean post and 0: clean post
        # reverse label values to filter list of documents
        _list = [not bool(el) for el in _list]
    return _list


def list_to_disk(filename, _list):
    with open(filename, 'w') as f:
        for row in _list:
            f.write(str(row).rstrip() + '\n')


def remove_rows(_list, labels):
    if isinstance(_list, str):
        _list = load_text_list(_list)
    if isinstance(labels, str):
        labels = load_number_list(labels, mode='bool')

    return list(compress(_list, labels))
