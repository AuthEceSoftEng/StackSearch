import java.util.Scanner;

import java.net.Socket;
import java.net.InetAddress;
import java.io.IOException;
import java.io.DataInputStream;
import java.io.DataOutputStream;

/**
 * Python binder/entry point for the CodeParser and SequenceExtractor classes.
 * Communication between the python and java services is facilitated using
 * sockets.
 * 
 * @author ncode
 */
public class PySocket {

	public static void main(String args[]) {
		boolean extractSequence = args.length > 0 ? Boolean.parseBoolean(args[0]) : false;
		boolean keepImports = args.length > 1 ? Boolean.parseBoolean(args[1]) : false;
		boolean keepComments = args.length > 2 ? Boolean.parseBoolean(args[2]) : false;
		boolean keepLiterals = args.length > 3 ? Boolean.parseBoolean(args[3]) : false;
		boolean keepMethodCalls = args.length > 4 ? Boolean.parseBoolean(args[4]) : false;
		boolean keepUnsolvedMethodCalls = args.length > 5 ? Boolean.parseBoolean(args[5]) : false;

		// Receive communication port
		Scanner scanner = new Scanner(System.in);
		int socketPort = scanner.nextInt();
		scanner.close();

		try {
			InetAddress localhost = InetAddress.getByName("localhost");
			Socket socket = new Socket(localhost, socketPort);
			DataOutputStream out = new DataOutputStream(socket.getOutputStream());
			DataInputStream in = new DataInputStream(socket.getInputStream());

			while (true) {
				String inputString = in.readUTF().trim();

				if (inputString.equals("__INIT__")) {
					out.writeUTF(inputString);
					out.flush();
					continue;
				} else if (inputString.equals("__STOP__")) {
					out.writeUTF(inputString);
					out.flush();
					break;
				} else {
					if (extractSequence) {
						String outputString = SequenceExtractor.extractCodeInfo(inputString, keepImports, keepComments,
								keepLiterals, keepMethodCalls, keepUnsolvedMethodCalls);
						out.writeUTF(outputString);
						out.flush();
					} else {
						String outputString = CodeParser.parseCode(inputString, keepImports, keepComments,
								keepLiterals);
						out.writeUTF(outputString);
						out.flush();
					}
				}
			}
			socket.close();
		} catch (Exception e) {
			// Service terminates
		}
	}
}