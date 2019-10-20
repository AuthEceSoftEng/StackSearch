#!/usr/bin/env python

import os
import sys


def iterate_files(input_dir, output_dir, tmp_suffix):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        filename = os.path.join(input_dir, filename)
        if filename.endswith('.conll'):
            annotated_data_conversion(filename, output_dir, tmp_suffix)


def annotated_data_conversion(input_file, output_dir, tmp_suffix):
    output_file = os.path.basename(input_file)[:-6] + tmp_suffix
    output_file = os.path.join(output_dir, output_file)
    with open(input_file, 'r') as _in:
        with open(output_file, 'w') as out:
            for line in _in:
                tokens = line.split()
                if len(tokens) == 0:
                    out.write('\n')
                else:
                    out.write(tokens[3] + ' ' + tokens[0] + '\n')


def merge_files(input_dir, tmp_suffix):
    merged_file = os.path.join(input_dir, 'final.conll')
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    files = os.listdir(input_dir)
    files = [fname for fname in files if fname.endswith(tmp_suffix)]
    with open(merged_file, 'w') as out:
        for filename in files:
            filename = os.path.join(input_dir, filename)
            with open(filename, 'r') as _in:
                for line in _in:
                    out.write(line.strip() + '\n')


def clean_up(output_dir, tmp_suffix):
    files = os.listdir(output_dir)
    del_files = [fname for fname in files if fname.endswith(tmp_suffix)]
    for filename in del_files:
        os.remove(os.path.join(output_dir, filename))


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    temp_file_suffix = '.temp'
    iterate_files(input_dir, output_dir, temp_file_suffix)
    merge_files(output_dir, temp_file_suffix)
    clean_up(output_dir, temp_file_suffix)