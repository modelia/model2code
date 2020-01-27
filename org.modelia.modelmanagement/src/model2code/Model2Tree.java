package model2code;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.common.util.TreeIterator;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.eclipse.gmt.modisco.java.FieldDeclaration;
import org.eclipse.gmt.modisco.java.MethodDeclaration;
import org.eclipse.gmt.modisco.java.Model;
import org.eclipse.gmt.modisco.java.Modifier;
import org.eclipse.gmt.modisco.java.PrimitiveType;
import org.eclipse.gmt.modisco.java.SingleVariableDeclaration;
import org.eclipse.gmt.modisco.java.Type;
import org.eclipse.gmt.modisco.java.TypeAccess;
import org.eclipse.gmt.modisco.java.emf.JavaPackage;
import org.eclipse.gmt.modisco.java.emf.impl.ClassDeclarationImpl;

import tree.TreeNode;

public class Model2Tree {
	
	public static void main(String[] args) throws IOException {
		String javaModelPath = args[0]; //"C:\\Users\\Lola\\Dropbox\\UOC\\papers\\2019\\model2vec\\model2vec_ws\\java2uml\\java_models\\eclipseModel-all.xmi";	
		String datasetpath = args[1]; //"C:\\Users\\Lola\\Dropbox\\UOC\\papers\\2019\\model2vec\\model2vec_ws\\org.modelia.modelmanagement\\jsonTrees\\dataset_eclipse_largemodels.json";
		
 		generateDatasetFromJavaModel(javaModelPath, datasetpath);
		
		System.out.println("Done!");
		
	}

	private static String generateVariableName(int n) {
		int c = n+'A';
		if ((c>='A' && c<='Z')) {
			return Character.toString((char) (c));
		} else {
			return generateVariableName(n/('Z'-'A'))+generateVariableName(n%('Z'-'A'));
		}
		
	}
	
	private static void generateDatasetFromJavaModel(String javaModelPath, String javaTreePath) throws IOException {
		
		// Initialize the model
		JavaPackage.eINSTANCE.eClass();

		// Register the XMI resource factory for the .xmi extension
		Resource.Factory.Registry reg = Resource.Factory.Registry.INSTANCE;
		reg.getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

		// Obtain a new resource set
		ResourceSet resourceSet = new ResourceSetImpl();

		// Get the resource
		URI uri = URI.createFileURI(javaModelPath);
		Resource resource = resourceSet.getResource(uri, true);
		
		FileWriter fw = new FileWriter(javaTreePath);
		BufferedWriter bw = new BufferedWriter(fw);
		bw.write("[");
		
		for (EObject m : resource.getContents()) {
			Model javaModel = (Model) m;
			// System.out.println(javaModel);
		
			
		TreeIterator<EObject> it = javaModel.eAllContents();
		
		
		while ( it.hasNext() ) {
			EObject obj = it.next();
			if (obj instanceof ClassDeclarationImpl && !((ClassDeclarationImpl) obj).getName().contains("org/eclipse/core/")
					&& ((ClassDeclarationImpl) obj).getSuperClass()==null) {
				
				TreeNode javaRoot = new TreeNode("PROGRAM");
				TreeNode umlRoot = new TreeNode("MODEL");
				
				Map<String, String> dictionary = new HashMap<String, String>();
				
				ClassDeclarationImpl clazz = (ClassDeclarationImpl) obj;
				
//				System.out.println("-----------------------------------");
				
				// Class
				TreeNode javaClassVisibilityNode = null;
				TreeNode umlClassVisibilityNode = null;
				if (clazz.getModifier()!=null) {
					if (clazz.getModifier().getVisibility()!=null) { // if no visiblity specified in java, it means it is visibility=package
	//					System.out.print(clazz.getModifier().getVisibility()+" ");
						javaClassVisibilityNode = new TreeNode(clazz.getModifier().getVisibility().toString());
						umlClassVisibilityNode = new TreeNode(clazz.getModifier().getVisibility().toString());
					}
					else {
						javaClassVisibilityNode = null;
						umlClassVisibilityNode = new TreeNode("package");
					}
				}
//				System.out.println(clazz.getName());
				
				TreeNode javaClassNode = new TreeNode("CLASS");
				TreeNode javaClassNameNode = new TreeNode(assignName(clazz.getName(), dictionary));
				javaRoot.addChild(javaClassNode);
				javaClassNode.addChild(javaClassNameNode);
				if (javaClassVisibilityNode!=null) { javaClassNode.addChild(javaClassVisibilityNode); }
				
				TreeNode umlClassNode = new TreeNode("CLASS");
				TreeNode umlClassNameNode = new TreeNode(assignName(clazz.getName(), dictionary));
				umlRoot.addChild(umlClassNode);
				umlClassNode.addChild(umlClassNameNode);
				if (umlClassVisibilityNode!=null) { umlClassNode.addChild(javaClassVisibilityNode); }
				
				// Attributes
				List<String> attributeNames = new LinkedList<String>();
				EList<EObject> clazzContents = clazz.eContents();
				for (EObject clazzContent : clazzContents) {
					if (clazzContent instanceof FieldDeclaration) {
						FieldDeclaration fd = (FieldDeclaration) clazzContent;
						if (fd.getType()!=null && fd.getName()!=null) {
							
							TreeNode javaAttNode = new TreeNode("ATT");
							TreeNode javaAttVisibilityNode = null;
							TreeNode javaAttNameNode = new TreeNode(assignName(fd.getName(), dictionary));
							TreeNode javaAttTypeNode = null;
//							
							TreeNode umlAttNode = new TreeNode("PROP");
							TreeNode umlAttVisibilityNode = null;
							TreeNode umlAttNameNode = new TreeNode(assignName(fd.getName(), dictionary));
							TreeNode umlAttTypeNode = null;
							
							attributeNames.add(fd.getName());
							
							System.out.print("\t");
							Modifier attModifier = fd.getModifier();
							if (attModifier!=null) {
								if (attModifier.getVisibility()!=null) {
	//								System.out.print(attModifier.getVisibility()+ " ");
									javaAttVisibilityNode = new TreeNode(attModifier.getVisibility().toString());
									umlAttVisibilityNode = new TreeNode(attModifier.getVisibility().toString());
								} else {
									javaAttVisibilityNode = null;
									umlAttVisibilityNode = new TreeNode("package");
								}
							}
//							System.out.print(fd.getName());
		
							TypeAccess attType = fd.getType();
							if (attType != null && attType.eCrossReferences() != null) {
								if (attType.eCrossReferences().get(0) instanceof ClassDeclarationImpl) {
//									System.out.println(" : " + ((ClassDeclarationImpl) attType.eCrossReferences().get(0)).getName());
									String typeName = assignName(((ClassDeclarationImpl) attType.eCrossReferences().get(0)).getName(), dictionary);
									javaAttTypeNode = new TreeNode(typeName);
									umlAttTypeNode = new TreeNode(typeName);
								} else if (attType.eCrossReferences().get(0) instanceof PrimitiveType) {
//									System.out.println(" : " + ((PrimitiveType) attType.eCrossReferences().get(0)).getName());
									String ptName = ((PrimitiveType) attType.eCrossReferences().get(0)).getName();
									javaAttTypeNode = javaTypeName(ptName);
									umlAttTypeNode = umlTypeName(ptName);
								}
							}
							
							javaClassNode.addChild(javaAttNode);
							javaAttNode.addChild(javaAttNameNode);
							if (javaAttVisibilityNode!=null) { javaAttNode.addChild(javaAttVisibilityNode); }
							javaAttNode.addChild(javaAttTypeNode);
							
							umlClassNode.addChild(umlAttNode);
							umlAttNode.addChild(umlAttNameNode);
							if (umlAttVisibilityNode!=null) { umlAttNode.addChild(umlAttVisibilityNode); }
							umlAttNode.addChild(umlAttTypeNode);
							
						}
					}
				}
				
//				System.out.println("\t----");
				// Methods
				for (EObject clazzContent : clazzContents) {
					if (clazzContent instanceof MethodDeclaration) {
//						System.out.print("\t");
						MethodDeclaration method = ((MethodDeclaration) clazzContent);
						
						TreeNode javaOpNode = new TreeNode("METHOD");
						TreeNode javaOpVisibilityNode = null;
						TreeNode javaOpNameNode = null;
						if (method.getName().startsWith("get")) { javaOpNameNode = new TreeNode("get" + assignName(method.getName().substring(3), dictionary)); }
						else if (method.getName().startsWith("set")) { javaOpNameNode = new TreeNode("set" + assignName(method.getName().substring(3), dictionary)); }
						else { javaOpNameNode = new TreeNode(assignName(method.getName(), dictionary)); }
						TreeNode javaOpReturnNode = null;
						TreeNode javaOpParamsNode = null;
						
						TreeNode umlOpNode = new TreeNode("OP");
						TreeNode umlOpVisibilityNode = null;
						TreeNode umlOpNameNode = null;
						if (method.getName().startsWith("get")) { umlOpNameNode = new TreeNode("get" + assignName(method.getName().substring(3), dictionary)); }
						else if (method.getName().startsWith("set")) { umlOpNameNode = new TreeNode("set" + assignName(method.getName().substring(3), dictionary)); }
						else { umlOpNameNode = new TreeNode(assignName(method.getName(), dictionary)); }
						
						TreeNode umlOpReturnNode = null;
						TreeNode umlOpParamsNode = null;
						
						if (method.getModifier()!=null) {
							if (method.getModifier().getVisibility()!=null) {
	//							System.out.print(method.getModifier().getVisibility() + " ");
								javaOpVisibilityNode = new TreeNode(method.getModifier().getVisibility().toString());
								umlOpVisibilityNode = new TreeNode(method.getModifier().getVisibility().toString());
							} else {
								javaOpVisibilityNode = null;
								umlOpVisibilityNode = new TreeNode("package");
							}
						}
						if (method.getReturnType() != null && method.getReturnType() instanceof ClassDeclarationImpl) {
							String returnTypeName = assignName(((Type)method.getReturnType().eCrossReferences().get(0)).getName(), dictionary);
							javaOpReturnNode = new TreeNode(returnTypeName);
							umlOpReturnNode = new TreeNode(returnTypeName);
						}
						else if (method.getReturnType() != null && method.getReturnType() instanceof PrimitiveType) {
//							System.out.print( ((Type)method.getReturnType().eCrossReferences().get(0)).getName());
							javaOpReturnNode = javaTypeName(((Type)method.getReturnType().eCrossReferences().get(0)).getName());
							umlOpReturnNode = umlTypeName(((Type)method.getReturnType().eCrossReferences().get(0)).getName());
						} else {
//							System.out.print("void");
							javaOpReturnNode = new TreeNode("void");
							umlOpReturnNode = new TreeNode("void");
						}
//						System.out.print(" " + method.getName() + "(");
						if (method.getParameters()!=null && !method.getParameters().isEmpty()) {
							javaOpParamsNode = new TreeNode("PARAMS");
							umlOpParamsNode = new TreeNode("PARAMS");
							for (SingleVariableDeclaration param : method.getParameters()) {
//								System.out.print(param.getName() + " : " + ((TypeAccess)param.getType()).getType().getName() + ", ");
								String paramName = assignName(param.getName(), dictionary);
								
								TreeNode javaParamNameNode = new TreeNode(paramName);
								TreeNode javaParamTypeNode = null;
								if (((TypeAccess)param.getType()).getType() instanceof ClassDeclarationImpl) {
									javaParamTypeNode = new TreeNode(assignName(((TypeAccess)param.getType()).getType().getName(), dictionary));
								} else  if (((TypeAccess)param.getType()).getType() instanceof PrimitiveType) {
									javaParamTypeNode = javaTypeName(((TypeAccess)param.getType()).getType().getName());
								} else {
									javaParamTypeNode = new TreeNode(assignName(((TypeAccess)param.getType()).getType().getName(), dictionary));
								}
								javaOpParamsNode.addChild(javaParamNameNode);
								javaParamNameNode.addChild(javaParamTypeNode);
								
								TreeNode umlParamNameNode = new TreeNode(paramName);
								TreeNode umlParamTypeNode = null;
								if (((TypeAccess)param.getType()).getType() instanceof ClassDeclarationImpl) {
									umlParamTypeNode = new TreeNode(assignName((((TypeAccess)param.getType()).getType().getName()), dictionary));
								} else if  (((TypeAccess)param.getType()).getType() instanceof PrimitiveType) {
									umlParamTypeNode = umlTypeName(((TypeAccess)param.getType()).getType().getName());
								} else {
									umlParamTypeNode = new TreeNode(assignName((((TypeAccess)param.getType()).getType().getName()), dictionary));
								}
								umlOpParamsNode.addChild(umlParamNameNode);
								umlParamNameNode.addChild(umlParamTypeNode);
							}
//							System.out.println(")");
						}
						
						
						
						boolean isGetterOrSetter = (method.getName().startsWith("get") || method.getName().startsWith("set")) && attributeNames.contains(method.getName().toLowerCase().substring(3)); 
						if (!isGetterOrSetter) {
							
							
							javaClassNode.addChild(javaOpNode);
							javaOpNode.addChild(javaOpNameNode);
							if (javaOpVisibilityNode!=null) { javaOpNode.addChild(javaOpVisibilityNode); }
							javaOpNode.addChild(javaOpReturnNode);
							if (javaOpParamsNode!=null) { javaOpNode.addChild(javaOpParamsNode); }
							
							umlClassNode.addChild(umlOpNode);
							umlOpNode.addChild(umlOpNameNode);
							if (umlOpVisibilityNode!=null) { umlOpNode.addChild(umlOpVisibilityNode); }
							umlOpNode.addChild(umlOpReturnNode);
							if (umlOpParamsNode!=null) { umlOpNode.addChild(umlOpParamsNode); }
						}
					}
				}
				if ((umlRoot.size()>25 || javaRoot.size()>25) && (umlRoot.size()<100 || javaRoot.size()<100)) {
					System.out.println(javaRoot.size());
					String jsontree = printDataset(javaRoot, umlRoot);
					System.out.println(jsontree);
					bw.write(jsontree+",\n");
					System.out.println();
				} else {
//					System.out.println(javaRoot.size());
//					String jsontree = printDataset(javaRoot, umlRoot);
//					System.out.println(jsontree+"\n");
				}
				
			}
		}
		}
		bw.write("]");
		bw.close();
		fw.close();
		
	}

	private static String assignName(String name, Map<String, String> dictionary) {
		String s = "";
		if (dictionary.containsKey(name)) {
			s = dictionary.get(name);
		} else {
			s = generateVariableName(dictionary.keySet().size());
			dictionary.put(name, s);
		}
		return s;
	}
	
	private static TreeNode javaTypeName(String name) {
		if (name.toLowerCase().endsWith("double")) {
			return new TreeNode("double");
		} else if (name.toLowerCase().endsWith("integer") || name.toLowerCase().endsWith("int")) {
			return new TreeNode("int");
		} else if (name.toLowerCase().endsWith("boolean")) {
			return new TreeNode("boolean");
		} else if (name.toLowerCase().endsWith("string")) {
			return new TreeNode("String");
		} else {
			return new TreeNode(name);
		}
	}
	
	private static TreeNode umlTypeName(String name) {
		if (name.toLowerCase().endsWith("double")) {
			return new TreeNode("Real");
		} else if (name.toLowerCase().endsWith("integer")) {
			return new TreeNode("Integer");
		} else if (name.toLowerCase().endsWith("boolean")) {
			return new TreeNode("Boolean");
		} else if (name.toLowerCase().endsWith("string")) {
			return new TreeNode("String");
		} else {
			return new TreeNode(name);
		}
	}

	private static <T> void printTree(TreeNode node, String appender) {
		System.out.println(appender + node.getData());
		node.getChildren().forEach(n -> printTree(n, appender + appender));
	}
	
	private static <T> String printJSON(TreeNode node) {
		String s = "";
		s+= "{\"root\": \""+node.getData()+"\", \"children\": [";
		int i = 0;
		for (TreeNode n : node.getChildren()) {
			s+= printJSON(n);
			if (i<node.getChildren().size()-1) {
				s+= ", ";
			}
			i++;
		}
		s+= "]}";
		return s;
	}
	
	private static String printDataset(TreeNode javaRoot, TreeNode umlRoot) {
		String s = "";
		s+="{\"source_ast\":";
		
		s+= printJSON(umlRoot);
		s+= ", \"target_ast\":";
		s+= printJSON(javaRoot);
		s+="}";
		return s;
	}
}
