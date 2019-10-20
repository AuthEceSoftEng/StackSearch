#!/usr/bin/env python

import os
import sys
import shutil

usage = '''Usage: ./test_codeparser.py {instance_type}
instance_type: \t1. {stdin} \n\t\t2. {socket}'''

if len(sys.argv) != 2:
    print(usage)
    exit()

if sys.argv[1] == 'stdin':
    from codeparser_stdin import CodeParserStdin as CodeParser
elif sys.argv[1] == 'socket':
    from codeparser_socket import CodeParserSocket as CodeParser
else:
    print('Options: \'stdin\' or \'socket\'')

temp_folder = 'temp_files'
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

testfiles = []
for filename in os.listdir("tests"):
    if filename.startswith("test"):
        testfiles.append(os.path.join('tests', filename))

print('Testing {} files...'.format(len(testfiles)))

# Test CodeParsing
parser = CodeParser(
    log_path=os.path.join(temp_folder, 'test.log'),
    extract_sequence=False,
    keep_imports=False,
    keep_comments=False,
    keep_literals=False)

for ii, test in enumerate(testfiles):
    with open(test, 'r') as _in:
        code_snippet = _in.read()

    parsed_code = parser.parse_code(code_snippet, '00' + str(ii))
    print(parsed_code)

parser.close()

# Test SequenceExtraction
extractor = CodeParser(
    log_path=os.path.join(temp_folder, 'test.log'),
    index_path=os.path.join(temp_folder, 'api_index'),
    extract_sequence=True,
    keep_imports=True,
    keep_comments=True,
    keep_literals=True,
    keep_method_calls=True,
    keep_unsolved_method_calls=True)

for ii, test in enumerate(testfiles):
    with open(test, 'r') as in_:
        code_snippet = in_.read()

    sequence = extractor.tokenize_sequence(code_snippet, '00' + str(ii), False)
    print(sequence)

extractor.close()

with open('temp_files/test.log', 'r') as in_:
    log_output = in_.read()

print('\n LOG OUTPUT\n============\n' + log_output)
shutil.rmtree(temp_folder)
