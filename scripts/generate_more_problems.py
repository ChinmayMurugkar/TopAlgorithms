#!/usr/bin/env python3
"""
Extended Automated Coding Interview Problems Generator
This script generates additional HTML pages for coding interview problems.
"""

import os
import time
from typing import List, Dict

class ExtendedProblemGenerator:
    def __init__(self):
        self.problems_dir = "problems"
        self.template = self.load_template()
        self.problem_data = self.load_extended_problem_data()
        
    def load_template(self) -> str:
        """Load the HTML template with CSS styling."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Coding Interview Problems</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 0;
            margin-bottom: 30px;
        }}
        
        .header-content {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        .breadcrumb {{
            margin-bottom: 10px;
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        
        .breadcrumb a {{
            color: white;
            text-decoration: none;
        }}
        
        .breadcrumb a:hover {{
            text-decoration: underline;
        }}
        
        h1 {{
            font-size: 2.2rem;
            margin-bottom: 10px;
        }}
        
        .meta {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        
        .content {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }}
        
        .problem-description {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 0 6px 6px 0;
        }}
        
        .solution {{
            margin-bottom: 40px;
        }}
        
        .solution h3 {{
            color: #2c3e50;
            font-size: 1.4rem;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #3498db;
        }}
        
        .code-block {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 20px;
            margin: 15px 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
            white-space: pre;
        }}
        
        .complexity {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }}
        
        .complexity strong {{
            color: #856404;
        }}
        
        .example {{
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }}
        
        .example strong {{
            color: #0c5460;
        }}
        
        .visualization {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 20px;
            margin: 15px 0;
            font-family: monospace;
        }}
        
        .navigation {{
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
        }}
        
        .nav-btn {{
            padding: 10px 20px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            transition: background 0.2s ease;
        }}
        
        .nav-btn:hover {{
            background: #2980b9;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 1.8rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="breadcrumb">
                <a href="../index.html">Home</a> > {category} > {problem_name}
            </div>
            <h1>{problem_name}</h1>
            <div class="meta">LeetCode Problem {problem_number}</div>
        </div>
    </div>

    <div class="container">
        <div class="content">
            <div class="problem-description">
                <strong>Problem:</strong> {problem_description}
            </div>

            {solutions}

            <div class="navigation">
                <a href="{prev_link}" class="nav-btn">← Previous: {prev_name}</a>
                <a href="{next_link}" class="nav-btn">Next: {next_name} →</a>
            </div>
        </div>
    </div>
</body>
</html>'''

    def load_extended_problem_data(self) -> List[Dict]:
        """Load extended problem data for additional problems."""
        return [
            # More String/Array Problems
            {
                "filename": "valid-parentheses.html",
                "title": "Valid Parentheses",
                "problem_name": "Valid Parentheses",
                "category": "String/Array/Matrix",
                "problem_number": "20",
                "problem_description": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid. An input string is valid if: Open brackets must be closed by the same type of brackets. Open brackets must be closed in the correct order.",
                "solutions": self.generate_valid_parentheses_solutions(),
                "prev_link": "happy-number.html",
                "prev_name": "Happy Number",
                "next_link": "longest-substring-without-repeating-characters.html",
                "next_name": "Longest Substring Without Repeating Characters"
            },
            {
                "filename": "longest-substring-without-repeating-characters.html",
                "title": "Longest Substring Without Repeating Characters",
                "problem_name": "Longest Substring Without Repeating Characters",
                "category": "String/Array/Matrix",
                "problem_number": "3",
                "problem_description": "Given a string s, find the length of the longest substring without repeating characters.",
                "solutions": self.generate_longest_substring_solutions(),
                "prev_link": "valid-parentheses.html",
                "prev_name": "Valid Parentheses",
                "next_link": "container-with-most-water.html",
                "next_name": "Container With Most Water"
            },
            {
                "filename": "container-with-most-water.html",
                "title": "Container With Most Water",
                "problem_name": "Container With Most Water",
                "category": "String/Array/Matrix",
                "problem_number": "11",
                "problem_description": "Given n non-negative integers height where each represents a point at coordinate (i, height[i]), find two lines that together with the x-axis form a container that would hold the maximum amount of water.",
                "solutions": self.generate_container_water_solutions(),
                "prev_link": "longest-substring-without-repeating-characters.html",
                "prev_name": "Longest Substring Without Repeating Characters",
                "next_link": "trapping-rain-water.html",
                "next_name": "Trapping Rain Water"
            },
            
            # More Dynamic Programming Problems
            {
                "filename": "unique-paths.html",
                "title": "Unique Paths",
                "problem_name": "Unique Paths",
                "category": "Dynamic Programming",
                "problem_number": "62",
                "problem_description": "There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.",
                "solutions": self.generate_unique_paths_solutions(),
                "prev_link": "container-with-most-water.html",
                "prev_name": "Container With Most Water",
                "next_link": "edit-distance.html",
                "next_name": "Edit Distance"
            },
            {
                "filename": "edit-distance.html",
                "title": "Edit Distance",
                "problem_name": "Edit Distance",
                "category": "Dynamic Programming",
                "problem_number": "72",
                "problem_description": "Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2. You have the following three operations permitted on a word: Insert a character, Delete a character, Replace a character.",
                "solutions": self.generate_edit_distance_solutions(),
                "prev_link": "unique-paths.html",
                "prev_name": "Unique Paths",
                "next_link": "longest-increasing-subsequence.html",
                "next_name": "Longest Increasing Subsequence"
            },
            
            # More Graph Problems
            {
                "filename": "clone-graph.html",
                "title": "Clone Graph",
                "problem_name": "Clone Graph",
                "category": "Graph",
                "problem_number": "133",
                "problem_description": "Given a reference of a node in a connected undirected graph. Return a deep copy (clone) of the graph. Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.",
                "solutions": self.generate_clone_graph_solutions(),
                "prev_link": "edit-distance.html",
                "prev_name": "Edit Distance",
                "next_link": "pacific-atlantic-water-flow.html",
                "next_name": "Pacific Atlantic Water Flow"
            },
            
            # More Tree Problems
            {
                "filename": "binary-tree-level-order-traversal.html",
                "title": "Binary Tree Level Order Traversal",
                "problem_name": "Binary Tree Level Order Traversal",
                "category": "Tree, Heap & Trie",
                "problem_number": "102",
                "problem_description": "Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).",
                "solutions": self.generate_level_order_solutions(),
                "prev_link": "clone-graph.html",
                "prev_name": "Clone Graph",
                "next_link": "invert-binary-tree.html",
                "next_name": "Invert Binary Tree"
            },
            {
                "filename": "invert-binary-tree.html",
                "title": "Invert Binary Tree",
                "problem_name": "Invert Binary Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "226",
                "problem_description": "Given the root of a binary tree, invert the tree, and return its root. To invert a binary tree, swap every left node with its corresponding right node.",
                "solutions": self.generate_invert_tree_solutions(),
                "prev_link": "binary-tree-level-order-traversal.html",
                "prev_name": "Binary Tree Level Order Traversal",
                "next_link": "kth-smallest-element-in-bst.html",
                "next_name": "Kth Smallest Element in BST"
            },
            
            # More Sorting Problems
            {
                "filename": "bubble-sort.html",
                "title": "Bubble Sort",
                "problem_name": "Bubble Sort",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Implement bubble sort algorithm to sort an array of integers in ascending order. Bubble sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order.",
                "solutions": self.generate_bubble_sort_solutions(),
                "prev_link": "invert-binary-tree.html",
                "prev_name": "Invert Binary Tree",
                "next_link": "insertion-sort.html",
                "next_name": "Insertion Sort"
            },
            {
                "filename": "insertion-sort.html",
                "title": "Insertion Sort",
                "problem_name": "Insertion Sort",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Implement insertion sort algorithm to sort an array of integers in ascending order. Insertion sort is a simple sorting algorithm that builds the final sorted array one item at a time.",
                "solutions": self.generate_insertion_sort_solutions(),
                "prev_link": "bubble-sort.html",
                "prev_name": "Bubble Sort",
                "next_link": "selection-sort.html",
                "next_name": "Selection Sort"
            }
        ]

    def generate_valid_parentheses_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Stack</h3>
                <p>Use a stack to keep track of opening brackets and match them with closing brackets.</p>
                
                <div class="code-block">
public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    
    for (char c : s.toCharArray()) {
        if (c == '(' || c == '{' || c == '[') {
            stack.push(c);
        } else {
            if (stack.isEmpty()) {
                return false;
            }
            
            char top = stack.pop();
            if ((c == ')' && top != '(') || 
                (c == '}' && top != '{') || 
                (c == ']' && top != '[')) {
                return false;
            }
        }
    }
    
    return stack.isEmpty();
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n)<br>
                    <strong>Space Complexity:</strong> O(n)
                </div>
            </div>
        '''

    def generate_longest_substring_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Sliding Window with HashSet</h3>
                <p>Use sliding window technique with a HashSet to track unique characters.</p>
                
                <div class="code-block">
public int lengthOfLongestSubstring(String s) {
    Set<Character> set = new HashSet<>();
    int maxLength = 0;
    int left = 0;
    
    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        
        while (set.contains(c)) {
            set.remove(s.charAt(left));
            left++;
        }
        
        set.add(c);
        maxLength = Math.max(maxLength, right - left + 1);
    }
    
    return maxLength;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n)<br>
                    <strong>Space Complexity:</strong> O(min(m, n)) where m is charset size
                </div>
            </div>
        '''

    def generate_container_water_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Two Pointers</h3>
                <p>Use two pointers approach to find the maximum area.</p>
                
                <div class="code-block">
public int maxArea(int[] height) {
    int maxArea = 0;
    int left = 0;
    int right = height.length - 1;
    
    while (left < right) {
        int width = right - left;
        int h = Math.min(height[left], height[right]);
        maxArea = Math.max(maxArea, width * h);
        
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }
    
    return maxArea;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n)<br>
                    <strong>Space Complexity:</strong> O(1)
                </div>
            </div>
        '''

    def generate_unique_paths_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Dynamic Programming</h3>
                <p>Use DP to calculate the number of unique paths to each cell.</p>
                
                <div class="code-block">
public int uniquePaths(int m, int n) {
    int[][] dp = new int[m][n];
    
    // Fill first row and column with 1
    for (int i = 0; i < m; i++) {
        dp[i][0] = 1;
    }
    for (int j = 0; j < n; j++) {
        dp[0][j] = 1;
    }
    
    // Fill the rest of the grid
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    
    return dp[m - 1][n - 1];
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(m * n)<br>
                    <strong>Space Complexity:</strong> O(m * n)
                </div>
            </div>
        '''

    def generate_edit_distance_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Dynamic Programming</h3>
                <p>Use DP to find the minimum number of operations to convert one string to another.</p>
                
                <div class="code-block">
public int minDistance(String word1, String word2) {
    int m = word1.length();
    int n = word2.length();
    
    int[][] dp = new int[m + 1][n + 1];
    
    // Fill first row and column
    for (int i = 0; i <= m; i++) {
        dp[i][0] = i;
    }
    for (int j = 0; j <= n; j++) {
        dp[0][j] = j;
    }
    
    // Fill the rest of the table
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + Math.min(dp[i - 1][j - 1], 
                                       Math.min(dp[i - 1][j], dp[i][j - 1]));
            }
        }
    }
    
    return dp[m][n];
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(m * n)<br>
                    <strong>Space Complexity:</strong> O(m * n)
                </div>
            </div>
        '''

    def generate_clone_graph_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – DFS with HashMap</h3>
                <p>Use DFS with a HashMap to clone the graph and avoid cycles.</p>
                
                <div class="code-block">
public Node cloneGraph(Node node) {
    if (node == null) {
        return null;
    }
    
    Map<Node, Node> visited = new HashMap<>();
    return cloneGraphHelper(node, visited);
}

private Node cloneGraphHelper(Node node, Map<Node, Node> visited) {
    if (visited.containsKey(node)) {
        return visited.get(node);
    }
    
    Node clone = new Node(node.val);
    visited.put(node, clone);
    
    for (Node neighbor : node.neighbors) {
        clone.neighbors.add(cloneGraphHelper(neighbor, visited));
    }
    
    return clone;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(V + E) where V is vertices and E is edges<br>
                    <strong>Space Complexity:</strong> O(V)
                </div>
            </div>
        '''

    def generate_level_order_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – BFS with Queue</h3>
                <p>Use BFS with a queue to traverse the tree level by level.</p>
                
                <div class="code-block">
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    
    if (root == null) {
        return result;
    }
    
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    
    while (!queue.isEmpty()) {
        int levelSize = queue.size();
        List<Integer> currentLevel = new ArrayList<>();
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode node = queue.poll();
            currentLevel.add(node.val);
            
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
        
        result.add(currentLevel);
    }
    
    return result;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n)<br>
                    <strong>Space Complexity:</strong> O(n)
                </div>
            </div>
        '''

    def generate_invert_tree_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Recursive DFS</h3>
                <p>Use recursive DFS to swap left and right children of each node.</p>
                
                <div class="code-block">
public TreeNode invertTree(TreeNode root) {
    if (root == null) {
        return null;
    }
    
    // Swap left and right children
    TreeNode temp = root.left;
    root.left = root.right;
    root.right = temp;
    
    // Recursively invert left and right subtrees
    invertTree(root.left);
    invertTree(root.right);
    
    return root;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n)<br>
                    <strong>Space Complexity:</strong> O(h) where h is the height of the tree
                </div>
            </div>
        '''

    def generate_bubble_sort_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Bubble Sort Implementation</h3>
                <p>Implement bubble sort by repeatedly swapping adjacent elements if they are in wrong order.</p>
                
                <div class="code-block">
public void bubbleSort(int[] arr) {
    int n = arr.length;
    boolean swapped;
    
    for (int i = 0; i < n - 1; i++) {
        swapped = false;
        
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap arr[j] and arr[j+1]
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                swapped = true;
            }
        }
        
        // If no swapping occurred, array is sorted
        if (!swapped) {
            break;
        }
    }
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n²) worst and average case, O(n) best case<br>
                    <strong>Space Complexity:</strong> O(1)
                </div>
            </div>
        '''

    def generate_insertion_sort_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Insertion Sort Implementation</h3>
                <p>Implement insertion sort by building the final sorted array one item at a time.</p>
                
                <div class="code-block">
public void insertionSort(int[] arr) {
    int n = arr.length;
    
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        
        // Move elements of arr[0..i-1] that are greater than key
        // to one position ahead of their current position
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n²) worst and average case, O(n) best case<br>
                    <strong>Space Complexity:</strong> O(1)
                </div>
            </div>
        '''

    def generate_problem_page(self, problem: Dict) -> str:
        """Generate the complete HTML content for a problem page."""
        return self.template.format(
            title=problem["title"],
            problem_name=problem["problem_name"],
            category=problem["category"],
            problem_number=problem["problem_number"],
            problem_description=problem["problem_description"],
            solutions=problem["solutions"],
            prev_link=problem["prev_link"],
            prev_name=problem["prev_name"],
            next_link=problem["next_link"],
            next_name=problem["next_name"]
        )

    def create_problems_directory(self):
        """Create the problems directory if it doesn't exist."""
        if not os.path.exists(self.problems_dir):
            os.makedirs(self.problems_dir)
            print(f"Created directory: {self.problems_dir}")

    def generate_all_problems(self, batch_size: int = 5):
        """Generate all problem pages in batches."""
        self.create_problems_directory()
        
        total_problems = len(self.problem_data)
        print(f"Starting to generate {total_problems} additional problem pages...")
        
        for i in range(0, total_problems, batch_size):
            batch = self.problem_data[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, total_problems)} of {total_problems})")
            
            for problem in batch:
                filename = os.path.join(self.problems_dir, problem["filename"])
                content = self.generate_problem_page(problem)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"  ✓ Generated: {problem['filename']}")
            
            # Small delay between batches
            if i + batch_size < total_problems:
                print("  Waiting 2 seconds before next batch...")
                time.sleep(2)
        
        print(f"\n🎉 Successfully generated {total_problems} additional problem pages!")
        print(f"📁 All files are saved in the '{self.problems_dir}' directory")

def main():
    """Main function to run the extended problem generator."""
    print("🚀 Starting Extended Automated Coding Interview Problems Generator")
    print("=" * 70)
    
    generator = ExtendedProblemGenerator()
    
    # Generate all problems in batches of 5
    generator.generate_all_problems(batch_size=5)
    
    print("\n" + "=" * 70)
    print("✅ All additional problem pages have been generated successfully!")
    print("🌐 You can now open index.html to view the complete website.")

if __name__ == "__main__":
    main() 