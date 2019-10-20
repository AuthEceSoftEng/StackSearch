import os
import sys
import socket as S
from subprocess import Popen, PIPE, STDOUT

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path)

from codeparser import CodeParser, INIT_SIGNAL, STOP_SIGNAL, ERROR_MESSAGE

ACK_SIGNAL = "__ACK__"


class CodeParserSocket(CodeParser):
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
            connection_type='PySocket',
            log_path=log_path,
            index_path=index_path,
            extract_sequence=extract_sequence,
            keep_imports=keep_imports,
            keep_comments=keep_comments,
            keep_literals=keep_literals,
            keep_method_calls=keep_method_calls,
            keep_unsolved_method_calls=keep_unsolved_method_calls)

        self._init_socket()
        if self._init_connection() == INIT_SIGNAL:
            self._print_info('Connection established...')

    def _init_socket(self):
        self.socket = S.socket(S.AF_INET, S.SOCK_STREAM)
        self.socket.setsockopt(S.SOL_SOCKET, S.SO_REUSEADDR, 1)
        self.socket.setsockopt(S.IPPROTO_TCP, S.TCP_NODELAY, 1)
        self.socket.bind(("localhost", 0))
        self.socket.listen(1)

    def _init_connection(self):
        # Initialize CodeParser process
        self.proc = Popen(self.args, stdin=PIPE)
        # Send socket port
        self.proc.stdin.write(
            str(self.socket.getsockname()[1]).encode() + b'\n')
        self.proc.stdin.flush()
        # Initialize connection
        self.connection, _ = self.socket.accept()
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
        if self._send_message(STOP_SIGNAL, -1) != STOP_SIGNAL:
            self.proc.kill()
            self._print_error('Killed CodeParser service...')
        self._print_info('Connection_terminated...')
        self.connection.close()
        self.socket.close()

    def _send_message(self, message, message_id):
        def check_errors(message, message_len):
            # 0-length message means that the Java process exited
            if message_len == 0:
                raise Exception('0-length message received')
            elif message == '__StackOverflowError__':
                raise Exception('JDT Compiler StackOverflowError')

        try:
            message_bytes = message.encode()
            self.connection.send(
                len(message_bytes).to_bytes(2, byteorder='big'))
            self.connection.send(message_bytes)
            recv_message_len = int.from_bytes(
                self.connection.recv(2), byteorder='big')
            recv_message = self.connection.recv(recv_message_len).decode()
            check_errors(recv_message, recv_message_len)
        except Exception as e:
            self.logger.warn(e)
            self.logger.warn(message_id)
            recv_message = ERROR_MESSAGE
            self._restart_connection()
        return recv_message
