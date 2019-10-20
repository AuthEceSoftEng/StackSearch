import java.util.Base64;
import java.util.Scanner;
import java.nio.charset.StandardCharsets;

/**
 * Python binder/entry point for the CodeParser and SequenceExtractor classes.
 * Communication between the python and java services is facilitated using the standard input pipe.
 * Messages in transit are base64 encoded. 
 * 
 * @author nikos
 */
public class PyStdin {

	public static void main(String[] args) {
		boolean extractSequence = args.length > 0 ? Boolean.parseBoolean(args[0]) : false;
		boolean keepImports = args.length > 1 ? Boolean.parseBoolean(args[1]) : false;
		boolean keepComments = args.length > 2 ? Boolean.parseBoolean(args[2]) : false;
		boolean keepLiterals = args.length > 3 ? Boolean.parseBoolean(args[3]) : false;
		boolean keepMethodCalls = args.length > 4 ? Boolean.parseBoolean(args[4]) : false;
		boolean keepUnsolvedMethodCalls = args.length > 5 ? Boolean.parseBoolean(args[5]) : false;

		Scanner scanner = new Scanner(System.in);
		while (scanner.hasNextLine()) {
			String b64inputString = scanner.nextLine();
			String inputString = new String(Base64.getDecoder().decode(b64inputString.getBytes()),
					StandardCharsets.UTF_8);

			if (inputString.equals("__INIT__")) {
				sendEncodedString(inputString);
				continue;
			} else if (inputString.equals("__STOP__")) {
				sendEncodedString(inputString);
				break;
			} else {
				if (extractSequence) {
					String outputString = SequenceExtractor.extractCodeInfo(inputString, keepImports, keepComments,
							keepLiterals, keepMethodCalls, keepUnsolvedMethodCalls);
					sendEncodedString(outputString);
				} else {
					String outputString = CodeParser.parseCode(inputString, keepImports, keepComments, keepLiterals);
					sendEncodedString(outputString);
				}
			}
		}
		scanner.close();
	}

	/**
	 * Encodes the given string in base64 format and sends the message using the standard input pipe.
	 * 
	 * @param outputString The string to be send.
	 */
	private static void sendEncodedString(String outputString) {
		String b64output = Base64.getEncoder().encodeToString(outputString.getBytes(StandardCharsets.UTF_8));
		System.out.println(b64output);
		System.out.flush();
	}
}