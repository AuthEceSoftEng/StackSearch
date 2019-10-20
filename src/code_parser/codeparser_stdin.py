import os
import sys
import base64
from subprocess import Popen, PIPE, STDOUT

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)

from codeparser import CodeParser, INIT_SIGNAL, STOP_SIGNAL, ERROR_MESSAGE


class CodeParserStdin(CodeParser):
    def __init__(self,
                 log_path=None,
                 index_path=None,
                 extract_sequence=True,
                 keep_imports=False,
                 keep_comments=False,
                 keep_literals=False,
                 keep_method_calls=True,
                 keep_unsolved_method_calls=False):

        super().__init__(
            connection_type='PyStdin',
            log_path=log_path,
            index_path=index_path,
            extract_sequence=extract_sequence,
            keep_imports=keep_imports,
            keep_comments=keep_comments,
            keep_literals=keep_literals,
            keep_method_calls=keep_method_calls,
            keep_unsolved_method_calls=keep_unsolved_method_calls)

        if self._init_connection() == INIT_SIGNAL:
            self._print_info('Connection established...')

    def _init_connection(self):
        self.proc = Popen(self.args, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        # Send init message
        recv_message = self._send_message(INIT_SIGNAL, 1)
        if recv_message != INIT_SIGNAL:
            self._print_error('Error connecting to CodeParser...')
            exit()
        return recv_message

    def _restart_connection(self):
        self.proc.kill()
        self.num_messages = 0
        self._init_connection()

    def _close_connection(self):
        if self._send_message(STOP_SIGNAL, 0) != STOP_SIGNAL:
            self.proc.kill()
            self._print_error('Killed CodeParser service...')
        self._print_info('Connection terminated...')

    def _send_message(self, message, message_id):
        def check_errors(message):
            if message == '__StackOverflowError__':
                raise Exception('JDT Compiler StackOverflowError')

        message_bytes = message.encode()
        b64encodedbytes = base64.b64encode(message_bytes)
        self.proc.stdin.write(b64encodedbytes + b'\n')
        self.proc.stdin.flush()
        recv_message = self.proc.stdout.readline()
        try:
            b64decodedbytes = base64.b64decode(recv_message)
            decoded_recv_message = b64decodedbytes.decode()
            check_errors(decoded_recv_message)
        except Exception as e:
            self.logger.warn(e)
            self.logger.warn(message_id)
            decoded_recv_message = ERROR_MESSAGE
            self.restart(force=True)
        return decoded_recv_message
