import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Java entry point for the CodeParser and SequenceExtractor classes.
 * 
 * @author nikos
 */
public class JCodeParser {

	public static void main(String[] args) {
		boolean extractSequence = args.length > 0 ? Boolean.parseBoolean(args[0]) : false;
		boolean keepImports = args.length > 1 ? Boolean.parseBoolean(args[1]) : false;
		boolean keepComments = args.length > 2 ? Boolean.parseBoolean(args[2]) : false;
		boolean keepLiterals = args.length > 3 ? Boolean.parseBoolean(args[3]) : false;
		boolean keepMethodCalls = args.length > 4 ? Boolean.parseBoolean(args[4]) : false;
		boolean keepUnsolvedMethodCalls = args.length > 5 ? Boolean.parseBoolean(args[5]) : false;
		String sourceFile = args.length > 6 ? args[6] : null;

		if (sourceFile != null) {
			try {
				String fileContents = new String(Files.readAllBytes(Paths.get(sourceFile)));
				if (extractSequence) {
					String entries = SequenceExtractor.extractCodeInfo(fileContents, keepImports, keepComments,
							keepLiterals, keepMethodCalls, keepUnsolvedMethodCalls);
					System.out.println(entries);
				} else {
					String outputString = CodeParser.parseCode(fileContents, keepImports, keepComments, keepLiterals);
					System.out.println(outputString);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			return;
		}
	}
}