import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseException;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;

// LiteralStringValueExpr, DoubleLiteralExpr, IntegerLiteralExpr, CharLiteralExpr
// StringLiteralExpr, BooleanLiteralExpr, NullLiteralExpr, MethodCallExpr
import com.github.javaparser.ast.expr.*;

/**
 * Class encapsulating the logic for parsing code snippets.
 * 
 * @author nikos
 */
public class CodeParser {

	private static final String[] wrapStart = { "", "class Generic$Class {",
			"class Generic$Class {\nvoid generic$Method() {" };
	private static final String[] wrapEnd = { "", "\n}", "\n}\n}" };

	/**
	 * Container for the parsed code snippet.
	 */
	public static class WrappedCompilationUnit {
		private final int wrap;
		private final String code;
		private final CompilationUnit cu;

		WrappedCompilationUnit(int wrap, String code, CompilationUnit cu) {
			this.wrap = wrap;
			this.code = code;
			this.cu = cu;
		}

		public int getWrap() {
			return wrap;
		}

		public String getCode() {
			return code;
		}

		public CompilationUnit getCu() {
			return cu;
		}
	}

	/**
	 * Attempts to parse the given code snippet and formats the output according to
	 * the provided flags.
	 * 
	 * @param inputCode    String representation of the code that is to be parsed.
	 * @param keepImports  Boolean variable denoting whether to keep import
	 *                     declarations in the output string.
	 * @param keepComments Boolean variable denoting whether to keep comments in the
	 *                     output string.
	 * @param keepLiterals Boolean variable denoting whether to keep literals or
	 *                     replace them with generic string representations.
	 * @return A string containing the parsed input code or an error code if parsing
	 *         was unsuccessful.
	 */
	public static String parseCode(String inputCode, boolean keepImports, boolean keepComments, boolean keepLiterals) {
		WrappedCompilationUnit wrappedCU = parseCodeSnippet(inputCode);
		CompilationUnit cu = wrappedCU.getCu();
		int wrapUsed = wrappedCU.getWrap();

		if (cu == null) {
			return "__ERROR__";
		}
		if (!keepImports) {
			// Remove PackageDeclaration
			cu.getPackageDeclaration().ifPresent(p -> p.remove());
			// Remove ImportDeclarations
			cu.getImports().removeIf(i -> true);
		}
		if (!keepComments) {
			cu.getComments().forEach(c -> c.remove());
		}
		if (!keepLiterals) {
			cu.findAll(LiteralExpr.class).forEach(l -> l.replace(replaceLiteralType(l)));
		}

		String parsedCode = cu.toString().trim();
		switch (wrapUsed) {
		case 1:
			parsedCode = parsedCode.replace("class Generic$Class {", "");
			parsedCode = parsedCode.substring(0, parsedCode.length() - 1).trim();
			parsedCode = parsedCode.replaceAll("(?m)^ {1,4}", "");
			break;
		case 2:
			parsedCode = parsedCode.replace("class Generic$Class {", "");
			parsedCode = parsedCode.replace("void generic$Method() {", "");
			parsedCode = parsedCode.substring(0, parsedCode.length() - 3).trim();
			parsedCode = parsedCode.replaceAll("(?m)^ {1,}", "");
			break;
		}

		if (parsedCode.trim().equals(""))
			return "__EMPTY__";

		return parsedCode;
	}

	/**
	 * Attempts to parse code snippets with JavaParser's parser. Code snippets are
	 * wrapped with class or method declarations, in an effort to make them to be
	 * compilable.
	 * 
	 * @param codeSnippet A string representation of the code snippet.
	 * @return A WrappedCompilationUnit containing the necessary information.
	 */
	public static WrappedCompilationUnit parseCodeSnippet(String codeSnippet) {
		for (int i = 0; i < wrapStart.length; i++) {
			try {
				String wrappedCodeSnippet = wrapStart[i] + codeSnippet + wrapEnd[i];
				CompilationUnit cu = JavaParser.parse(wrappedCodeSnippet);
				return new WrappedCompilationUnit(i, wrappedCodeSnippet, cu);
			} catch (Exception e) {
				continue;
			}
		}
		return new WrappedCompilationUnit(0, "", null);
	}

	/**
	 * Replaces literals with a simple string representation of the type.
	 * 
	 * @param literal The literal that is to be replaced.
	 * @return The string representation of the type of the passed literal.
	 */
	private static NameExpr replaceLiteralType(LiteralExpr literal) {
		if (literal instanceof LiteralStringValueExpr) {
			if (literal instanceof CharLiteralExpr) {
				return new NameExpr("__char__");
			} else if (literal instanceof DoubleLiteralExpr) {
				return new NameExpr("__double__");
			} else if (literal instanceof IntegerLiteralExpr) {
				return new NameExpr("__integer__");
			} else if (literal instanceof LongLiteralExpr) {
				return new NameExpr("__long__");
			} else if (literal instanceof StringLiteralExpr) {
				return new NameExpr("__string__");
			}
		} else if (literal instanceof BooleanLiteralExpr) {
			return new NameExpr("__boolean__");
		} else if (literal instanceof NullLiteralExpr) {
			return new NameExpr("null");
		}
		return new NameExpr("unk");
	}
}