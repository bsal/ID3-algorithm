import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

class ID3 {

	/** Each node of the tree contains either the attribute number (for non-leaf
	 *  nodes) or class number (for leaf nodes) in <b>value</b>, and an array of
	 *  tree nodes in <b>children</b> containing each of the children of the
	 *  node (for non-leaf nodes).
	 *  The attribute number corresponds to the column number in the training
	 *  and test files. The children are ordered in the same order as the
	 *  Strings in strings[][]. E.g., if value == 3, then the array of
	 *  children correspond to the branches for attribute 3 (named data[0][3]):
	 *      children[0] is the branch for attribute 3 == strings[3][0]
	 *      children[1] is the branch for attribute 3 == strings[3][1]
	 *      children[2] is the branch for attribute 3 == strings[3][2]
	 *      etc.
	 *  The class number (leaf nodes) also corresponds to the order of classes
	 *  in strings[][]. For example, a leaf with value == 3 corresponds
	 *  to the class label strings[attributes-1][3].
	 **/
	class Tree {

		Tree[] children;
		int value;

		public Tree(Tree[] ch, int val) {
			value = val;
			children = ch;
		} // constructor

		public String toString() {
			return toString("");
		} // toString()
		
		String toString(String indent) {
			if (children != null) {
				String s = "";
				for (int i = 0; i < children.length; i++)
					s += indent + data[0][value] + "=" +
							strings[value][i] + "\n" +
							children[i].toString(indent + '\t');
				return s;
			} else
				return indent + "Class: " + strings[attributes-1][value] + "\n";
		} // toString(String)

	} // inner class Tree

	private int attributes; 	// Number of attributes (including the class)
	private int examples;		// Number of training examples
	private Tree decisionTree;	// Tree learnt in training, used for classifying
	private String[][] data;	// Training data indexed by example, attribute
	private String[][] strings; // Unique strings for each attribute
	private int[] stringCount;  // Number of unique strings for each attribute
	
	public ID3() {
		attributes = 0;
		examples = 0;
		decisionTree = null;
		data = null;
		strings = null;
		stringCount = null;
	} // constructor
	
	public void printTree() {
		if (decisionTree == null)
			error("Attempted to print null Tree");
		else
			System.out.println(decisionTree);
	} // printTree()

	/** Print error message and exit. **/
	static void error(String msg) {
		System.err.println("Error: " + msg);
		System.exit(1);
	} // error()

	static final double LOG2 = Math.log(2.0);
	
	static double xlogx(double x) {
		return x == 0? 0: x * Math.log(x) / LOG2;
	} // xlogx()

	/** Execute the decision tree on the given examples in testData, and print
	 *  the resulting class names, one to a line, for each example in testData.
	 **/
	public void classify(String[][] testData) {
		if (decisionTree == null){
			error("Please run training phase before classification");
			return;
		}
		for (int row=1 ; row<testData.length ; row++) {
			output(decisionTree,testData[row]);
		}
		// PUT  YOUR CODE HERE FOR CLASSIFICATION
	} // classify()
	
	private void output(Tree tree, String[] testData){
		if(tree.children == null){ //reached leaf of the tree and prints class label
			System.out.println(strings[attributes-1][tree.value]);
		}else {
			String value = testData[tree.value]; //attribute value from classifying set
			int index = 0;
			while (!value.equals(strings[tree.value][index])) {
				index++;
			}
			output(tree.children[index],testData);
		}
	}

	public void train(String[][] trainingData) {
		indexStrings(trainingData);
		decisionTree = build(new ArrayList<Integer>(),new ArrayList<String>());
	} // train()
		
	private Tree build(List<Integer> attindex, List<String> attval){
		//Inital entropy calculation for attributes in attindex and attribute values in attval
		int[] set = countClasses(attindex,attval);
		double total = total(set);
		double inital_entropy = entropy(set,total);
		if (inital_entropy == 0.0){ //termination point for particular branch/tree
			for (int val=0 ; val<set.length ; val++) {
				if (set[val] == total) {
					return new Tree(null,val); //leaf node
				}
			}
		}

		//loop to find the best attribute to split based on information gained
		int attributenum = -1; //keep track of attribute at which tree/branch will split
		double largest_gain = -1; //keeps track of largest information gain
		for (int n=0 ; n<attributes-1; n++) {
			if (!attindex.contains(n)) {
				double[] entropy_set = new double[stringCount[n]];
				for (int i=0 ; i<stringCount[n] ; i++) {
					attindex.add(n);
					attval.add(strings[n][i]);
					//calculates entropy for attributes in attindex and attribute values in attval
					int[] temp_set = countClasses(attindex,attval);
					double temp_total = total(temp_set);
					double temp_entrotpy = entropy(temp_set,temp_total);
					entropy_set[i] = temp_entrotpy*temp_total;
					attindex.remove(attindex.size()-1);
					attval.remove(attval.size()-1);
				}
				//calculate information gain for attribute number given by n
				double temp_Gain = inital_entropy;
				for (int i=0 ; i<entropy_set.length ; i++) {
					temp_Gain = temp_Gain - (entropy_set[i]/total);
				}
				if (temp_Gain > largest_gain) {
					largest_gain = temp_Gain;
					attributenum = n;
				}
			}
		}
		//children created for tree/branch at which it is spliting
		Tree[] children = new Tree[stringCount[attributenum]];
		attindex.add(attributenum);
		for (int i=0; i<children.length ; i++) {
			attval.add(strings[attributenum][i]);
			children[i] = build(attindex,attval); //recursive call to build tree
			attval.remove(attval.size()-1);
		}
		attindex.remove(attindex.size()-1);
		return new Tree(children,attributenum);
	}

	private double total(int[] set){
		double total = 0.0;
		for (int i=0 ; i< set.length ; i++) {
			total += set[i];
		}
		return total;
	}

	private double entropy(int[] set, double total){
		double entropy = 0.0;
		for (int i=0 ; i< set.length ; i++) {
			entropy = entropy + (-1) * xlogx(set[i]/total);
		}
		return entropy;
	}
	
	private int[] countClasses(List<Integer> attindex, List<String> attval){
		int[] class_counter = new int[stringCount[attributes-1]];//set to count each class label occurance
		
		for (int i=1 ; i<examples ; i++) {
			//selecting attribute value from training set 
			ArrayList<String> comp = new ArrayList<>();
			for(int n =0; n<attindex.size(); n++){
				comp.add(data[i][attindex.get(n)]);
			}
			if(attval.equals(comp)){ //check if both attribute value are same or not
				int index = 0;
				for (; index<strings[attributes-1].length ; index++) {
					if (data[i][attributes-1].equals(strings[attributes-1][index])) {
						break;
					}
				}
				class_counter[index]++; //increases the occurance of class label given by index
			}
		}
		return class_counter;
	}

	/** Given a 2-dimensional array containing the training data, numbers each
	 *  unique value that each attribute has, and stores these Strings in
	 *  instance variables; for example, for attribute 2, its first value
	 *  would be stored in strings[2][0], its second value in strings[2][1],
	 *  and so on; and the number of different values in stringCount[2].
	 **/
	void indexStrings(String[][] inputData) {
		data = inputData;
		examples = data.length;
		attributes = data[0].length;
		stringCount = new int[attributes];
		strings = new String[attributes][examples];// might not need all columns
		int index = 0;
		for (int attr = 0; attr < attributes; attr++) {
			stringCount[attr] = 0;
			for (int ex = 1; ex < examples; ex++) {
				for (index = 0; index < stringCount[attr]; index++)
					if (data[ex][attr].equals(strings[attr][index]))
						break;	// we've seen this String before
				if (index == stringCount[attr])		// if new String found
					strings[attr][stringCount[attr]++] = data[ex][attr];
			} // for each example
		} // for each attribute
	} // indexStrings()

	/** For debugging: prints the list of attribute values for each attribute
	 *  and their index values.
	 **/
	void printStrings() {
		for (int attr = 0; attr < attributes; attr++)
			for (int index = 0; index < stringCount[attr]; index++)
				System.out.println(data[0][attr] + " value " + index +
									" = " + strings[attr][index]);
	} // printStrings()
		
	/** Reads a text file containing a fixed number of comma-separated values
	 *  on each line, and returns a two dimensional array of these values,
	 *  indexed by line number and position in line.
	 **/
	static String[][] parseCSV(String fileName)
								throws FileNotFoundException, IOException {
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String s = br.readLine();
		int fields = 1;
		int index = 0;
		while ((index = s.indexOf(',', index) + 1) > 0)
			fields++;
		int lines = 1;
		while (br.readLine() != null)
			lines++;
		br.close();
		String[][] data = new String[lines][fields];
		Scanner sc = new Scanner(new File(fileName));
		sc.useDelimiter("[,\n]");
		for (int l = 0; l < lines; l++)
			for (int f = 0; f < fields; f++)
				if (sc.hasNext())
					data[l][f] = sc.next();
				else
					error("Scan error in " + fileName + " at " + l + ":" + f);
		sc.close();
		return data;
	} // parseCSV()

	public static void main(String[] args) throws FileNotFoundException,
												  IOException {
		if (args.length != 2)
			error("Expected 2 arguments: file names of training and test data");
		String[][] trainingData = parseCSV(args[0]);
		String[][] testData = parseCSV(args[1]);
		ID3 classifier = new ID3();
		classifier.train(trainingData);
		classifier.printTree();
		classifier.classify(testData);
	} // main()

} // class ID3
