import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.CompilationUnit;

import org.eclipse.jdt.core.dom.Comment;
import org.eclipse.jdt.core.dom.LineComment;
import org.eclipse.jdt.core.dom.BlockComment;
import org.eclipse.jdt.core.dom.Expression;
import org.eclipse.jdt.core.dom.ITypeBinding;
import org.eclipse.jdt.core.dom.IMethodBinding;
import org.eclipse.jdt.core.dom.IVariableBinding;
import org.eclipse.jdt.core.dom.MethodInvocation;
import org.eclipse.jdt.core.dom.ImportDeclaration;
import org.eclipse.jdt.core.dom.ClassInstanceCreation;
import org.eclipse.jdt.core.dom.VariableDeclarationFragment;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.lang.StackOverflowError;

/**
 * Class responsible for extracting a sequence of code elements that describe
 * the underlying functionality of the input code snippet.
 * 
 * @author nikos
 */
public class SequenceExtractor {

	/**
	 * Container for the extracted APIs, MethodInvocations and other code elements.
	 */
	private static class Entry implements Comparable<Entry> {
		private final String name;
		private final int position;

		Entry(String name, int position) {
			this.name = name;
			this.position = position;
		}

		public String getName() {
			return name;
		}

		public int getPosition() {
			return position;
		}

		@Override
		public int compareTo(Entry otherEntry) {
			int otherPosition = otherEntry.getPosition();
			return (int) (position - otherPosition);
		}

		@Override
		public String toString() {
			return "[" + position + "]: " + name;
		}
	}

	/**
	 * Parses the input code snippet and extracts a sequence of features.
	 * 
	 * @param inputCode               A string representation of the input code
	 *                                snippet.
	 * @param keepImports             Boolean variable denoting whether to extract
	 *                                import declarations.
	 * @param keepComments            Boolean variable denoting whether to extract
	 *                                comment bodies.
	 * @param keepLiterals            Boolean variable denoting whether to extract
	 *                                literals and variable declaration types.
	 * @param keepMethodCalls         Boolean variable denoting whether to extract
	 *                                method invocations.
	 * @param keepUnsolvedMethodCalls Boolean variable denoting whether to keep
	 *                                method invocations that cannot be resolved
	 *                                (type of the calling object).
	 * @return A string containing the list of extracted information.
	 */
	public static String extractCodeInfo(String inputCode, boolean keepImports, boolean keepComments,
			boolean keepLiterals, boolean keepMethodCalls, boolean keepUnsolvedMethodCalls) {
		ArrayList<String> entries = new ArrayList<String>();
		ArrayList<Entry> entries_list = new ArrayList<Entry>();

		// Parse code
		CodeParser.WrappedCompilationUnit wcu = CodeParser.parseCodeSnippet(inputCode);

		CompilationUnit cu = null;
		String sourceCode = wcu.getCode();

		// Check JavaParser parsing result
		if (sourceCode.equals(""))
			return "__ERROR__";

		// JDT Parsing
		try {
			cu = getCompilationUnit(sourceCode);
		} catch (StackOverflowError err) {
			// Handle a rare JDT compiler StackOverflowError
			// Keep the process running and return the following error string instead
			return "__StackOverflowError__";
		}

		// Extract imports
		if (keepImports) {
			List<ImportDeclaration> imports = cu.imports();
			for (ImportDeclaration i : imports) {
				String name = i.getName().toString();
				int position = i.getStartPosition();
				entries_list.add(new Entry("_IM_" + name, position));
			}
		}

		// Extract comments
		if (keepComments) {
			List<Comment> comments = cu.getCommentList();
			for (Comment c : comments) {
				c.accept(new ASTVisitor() {
					private void validateComment(Comment node) {
						int startPosition = node.getStartPosition();
						int endPosition = startPosition + node.getLength();
						String comment = sourceCode.substring(startPosition, endPosition);
						comment = comment.replaceAll("^(//|/\\*{1,2})", "").replaceAll("(\\*/)$", "");
						comment = comment.replaceAll("(?m)(^ {0,1}\\* {0,1})", "");
						comment = comment.trim().replaceAll(",", " ").replaceAll("\\s+", " ");
						if (!comment.startsWith("TODO Auto-generated") && !comment.equals("")) {
							entries_list.add(new Entry("_COM_" + comment, startPosition));
						}
					}

					@Override
					public boolean visit(LineComment node) {
						validateComment(node);
						return true;
					}

					@Override
					public boolean visit(BlockComment node) {
						validateComment(node);
						return true;
					}
				});
			}
		}

		// Extract Object Creations, Literals, Method Calls
		cu.accept(new ASTVisitor() {
			@Override
			public boolean visit(ClassInstanceCreation node) {
				String name = node.getType().toString();
				int position = node.getStartPosition();
				entries_list.add(new Entry("_OC_" + name, position));
				return true;
			}

			@Override
			public boolean visit(VariableDeclarationFragment node) {
				if (keepLiterals) {
					int position = node.getStartPosition();
					String typeSimpleName = "";
					IVariableBinding varBinding = node.resolveBinding();
					if (varBinding != null) {
						ITypeBinding typeBinding = varBinding.getType();
						if (typeBinding != null) {
							typeSimpleName = typeBinding.getName();
							entries_list.add(new Entry("_VAR_" + typeSimpleName, position));
						}
					}
				}
				return true;
			}

			@Override
			public boolean visit(MethodInvocation node) {
				if (keepMethodCalls) {
					String methodName = node.getName().toString();
					if (!(methodName.equals("print") || methodName.equals("println")
							|| methodName.equals("printStackTrace"))) {
						String objectName = "";
						int position = node.getStartPosition();
						IMethodBinding methodBinding = null;
						Expression expr = node.getExpression();
						if (expr != null) {
							// When the expression equals to "this"
							// the resolved object type is the wrapper class
							// It's preferred to use "this" as objectName
							if (expr.toString().equals("this")) {
								objectName = "this";
							} else {
								ITypeBinding typeBinding = expr.resolveTypeBinding();
								if (typeBinding != null) {
									objectName = typeBinding.getName();
								}
							}
						} else if ((methodBinding = node.resolveMethodBinding()) != null) {
							objectName = methodBinding.getDeclaringClass().getName();
						}
						if (!objectName.equals("") && !objectName.equals("Generic$Class")) {
							entries_list.add(new Entry("_MC_" + objectName + "." + methodName, position));
						} else if (keepUnsolvedMethodCalls) {
							entries_list.add(new Entry("_UMC_" + methodName, position));
						}
					}
				}
				return true;
			}
		});

		if (entries_list.size() == 0)
			return "__EMPTY__";

		// Sort entries on their position in the source code
		Collections.sort(entries_list);

		// Get entry names
		entries = entries_list.stream().map(obj -> new String(obj.getName())).collect(ArrayList::new, ArrayList::add,
				ArrayList::addAll);

		return String.join(", ", entries);
	}

	/**
	 * Utilizes the jdt compiler to retrieve an AST of the given source code.
	 * 
	 * @param inputCode A string representation of the input source code.
	 * @return A CompilationUnit containing the AST of the input source code.
	 */
	private static CompilationUnit getCompilationUnit(String inputCode) {
		ASTParser astParser = ASTParser.newParser(AST.JLS10);
		astParser.setSource(inputCode.toCharArray());
		astParser.setResolveBindings(true);
		astParser.setBindingsRecovery(true);
		astParser.setUnitName("sampleUnit");
		astParser.setKind(ASTParser.K_COMPILATION_UNIT);
		astParser.setEnvironment(null, null, null, true);
		return (CompilationUnit) astParser.createAST(null);
	}
}
