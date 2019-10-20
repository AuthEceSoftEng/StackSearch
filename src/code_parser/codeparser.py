import os
import re
import sys
import time
import logging
from collections import OrderedDict

from javalang import tokenizer

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)
codeparser_jar = os.path.join(file_path, 'lib', 'CodeParser-0.4.jar')

## CodeParser Signals
INIT_SIGNAL = '__INIT__'
STOP_SIGNAL = '__STOP__'
ERROR_MESSAGE = '__ERROR__'
EMPTY_MESSAGE = '__EMPTY__'


class CodeParser:
    def __init__(self,
                 codeparser_jar=codeparser_jar,
                 connection_type='PyStdin',
                 log_path=None,
                 index_path=None,
                 extract_sequence=True,
                 keep_imports=False,
                 keep_comments=False,
                 keep_literals=False,
                 keep_method_calls=True,
                 keep_unsolved_method_calls=False):

        self._init_logger(log_path)
        self.index = None
        self.num_messages = 0

        if index_path:
            self.index = open(index_path, 'a')

        self.args = [
            'java', '-cp', codeparser_jar, connection_type,
            'true' if extract_sequence else 'false',
            'true' if keep_imports else 'false',
            'true' if keep_comments else 'false',
            'true' if keep_literals else 'false',
            'true' if keep_method_calls else 'false',
            'true' if keep_unsolved_method_calls else 'false'
        ]

    def _init_logger(self, log_path):
        if not log_path:
            log_path = os.path.join(file_path, str(int(time.time())) + '.log')
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s %(name)s %(levelname)s %(message)s')
        self.logger = logging.getLogger(__name__)

    def _print_error(self, message, end='\n'):
        self.logger.error(message)
        sys.stdout.write('\x1b[1;31m' + '[CodeParser]: ' + '\x1b[0m')
        print(message.strip(), end=end)

    def _print_info(self, message, end='\n'):
        self.logger.info(message)
        sys.stdout.write('\x1b[1;33m' + '[CodeParser]: ' + '\x1b[0m')
        print(message.strip(), end=end)

    def _init_connection(self):
        raise NotImplementedError

    def _restart_connection(self):
        raise NotImplementedError

    def _close_connection(self):
        raise NotImplementedError

    def _send_message(self, message, message_id):
        raise NotImplementedError

    def restart(self, force=False):
        if force or self._send_message(STOP_SIGNAL, 0) == STOP_SIGNAL:
            self._restart_connection()
        else:
            self._print_error('Error restarting CodeParser service...')
            exit()

    def close(self):
        if self.index:
            self.index.close()
            self.index = None
        self._close_connection()
        logging.shutdown()

    def parse_code(self, code_snippet, identifier):
        self.num_messages += 1
        if self.num_messages == 50000:
            self.restart()
        return self._send_message(code_snippet, identifier)

    def tokenize_sequence(self, code_snippet, identifier, unique_tokens=False):
        sequence = self.parse_code(code_snippet, identifier)
        if sequence == ERROR_MESSAGE or sequence == EMPTY_MESSAGE:
            return []

        api_tokens = []
        sequence = sequence.split(', ')
        for idx, token in enumerate(sequence):
            if not token.startswith('_COM_'):
                token = re.sub(r'^_(IM|VAR|OC|MC|UMC)_', r'', token)
                api_tokens.append(token)
            else:
                token = re.sub(r'^_COM_', r'', token)
            sequence[idx] = token

        if unique_tokens:
            sequence = list(OrderedDict.fromkeys(sequence))
            api_tokens = list(OrderedDict.fromkeys(api_tokens))

        if self.index and len(api_tokens) > 0:
            for token in api_tokens:
                self.index.write(token + '\n')

        return sequence

    def tokenize_code(self, code_snippet, identifier, verbose=0):
        code = self.parse_code(code_snippet, identifier)
        if code == ERROR_MESSAGE or code == EMPTY_MESSAGE:
            return []
        try:
            return [t.value for t in tokenizer.tokenize(code)]
        except Exception as e:
            if verbose == 1:
                print('\n'.join([code, e]))
            return []
