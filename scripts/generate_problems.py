#!/usr/bin/env python3
"""
Automated Coding Interview Problems Generator
This script generates HTML pages for coding interview problems in batches.
"""

import os
import time
from typing import List, Dict, Tuple

class ProblemGenerator:
    def __init__(self):
        self.problems_dir = "problems"
        self.template = self.load_template()
        self.problem_data = self.load_problem_data()
        
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

    def load_problem_data(self) -> List[Dict]:
        """Load all the problem data that needs to be generated."""
        return [
            # Tree Problems
            {
                "filename": "lowest-common-ancestor.html",
                "title": "Lowest Common Ancestor of a Binary Tree",
                "problem_name": "Lowest Common Ancestor of a Binary Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "236",
                "problem_description": "Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree. The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).",
                "solutions": self.generate_lca_solutions(),
                "prev_link": "flatten-binary-tree.html",
                "prev_name": "Flatten Binary Tree to Linked List",
                "next_link": "serialize-deserialize-binary-tree.html",
                "next_name": "Serialize and Deserialize Binary Tree"
            },
            {
                "filename": "serialize-deserialize-binary-tree.html",
                "title": "Serialize and Deserialize Binary Tree",
                "problem_name": "Serialize and Deserialize Binary Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "297",
                "problem_description": "Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.",
                "solutions": self.generate_serialize_solutions(),
                "prev_link": "lowest-common-ancestor.html",
                "prev_name": "Lowest Common Ancestor of a Binary Tree",
                "next_link": "implement-trie.html",
                "next_name": "Implement Trie (Prefix Tree)"
            },
            {
                "filename": "implement-trie.html",
                "title": "Implement Trie (Prefix Tree)",
                "problem_name": "Implement Trie (Prefix Tree)",
                "category": "Tree, Heap & Trie",
                "problem_number": "208",
                "problem_description": "A trie (pronounced as 'try') or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.",
                "solutions": self.generate_trie_solutions(),
                "prev_link": "serialize-deserialize-binary-tree.html",
                "prev_name": "Serialize and Deserialize Binary Tree",
                "next_link": "word-search.html",
                "next_name": "Word Search"
            },
            
            # Graph Problems
            {
                "filename": "word-search.html",
                "title": "Word Search",
                "problem_name": "Word Search",
                "category": "Graph",
                "problem_number": "79",
                "problem_description": "Given an m x n grid of characters board and a string word, return true if word exists in the grid. The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.",
                "solutions": self.generate_word_search_solutions(),
                "prev_link": "implement-trie.html",
                "prev_name": "Implement Trie (Prefix Tree)",
                "next_link": "number-of-islands.html",
                "next_name": "Number of Islands"
            },
            {
                "filename": "number-of-islands.html",
                "title": "Number of Islands",
                "problem_name": "Number of Islands",
                "category": "Graph",
                "problem_number": "200",
                "problem_description": "Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.",
                "solutions": self.generate_islands_solutions(),
                "prev_link": "word-search.html",
                "prev_name": "Word Search",
                "next_link": "course-schedule.html",
                "next_name": "Course Schedule"
            },
            {
                "filename": "course-schedule.html",
                "title": "Course Schedule",
                "problem_name": "Course Schedule",
                "category": "Graph",
                "problem_number": "207",
                "problem_description": "There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.",
                "solutions": self.generate_course_schedule_solutions(),
                "prev_link": "number-of-islands.html",
                "prev_name": "Number of Islands",
                "next_link": "clone-graph.html",
                "next_name": "Clone Graph"
            },
            
            # Dynamic Programming Problems
            {
                "filename": "climbing-stairs.html",
                "title": "Climbing Stairs",
                "problem_name": "Climbing Stairs",
                "category": "Dynamic Programming",
                "problem_number": "70",
                "problem_description": "You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?",
                "solutions": self.generate_climbing_stairs_solutions(),
                "prev_link": "course-schedule.html",
                "prev_name": "Course Schedule",
                "next_link": "house-robber.html",
                "next_name": "House Robber"
            },
            {
                "filename": "house-robber.html",
                "title": "House Robber",
                "problem_name": "House Robber",
                "category": "Dynamic Programming",
                "problem_number": "198",
                "problem_description": "You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.",
                "solutions": self.generate_house_robber_solutions(),
                "prev_link": "climbing-stairs.html",
                "prev_name": "Climbing Stairs",
                "next_link": "coin-change.html",
                "next_name": "Coin Change"
            },
            {
                "filename": "coin-change.html",
                "title": "Coin Change",
                "problem_name": "Coin Change",
                "category": "Dynamic Programming",
                "problem_number": "322",
                "problem_description": "You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.",
                "solutions": self.generate_coin_change_solutions(),
                "prev_link": "house-robber.html",
                "prev_name": "House Robber",
                "next_link": "unique-paths.html",
                "next_name": "Unique Paths"
            },
            
            # Sorting Problems
            {
                "filename": "merge-sort.html",
                "title": "Merge Sort",
                "problem_name": "Merge Sort",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Implement merge sort algorithm to sort an array of integers in ascending order. Merge sort is a divide-and-conquer algorithm that recursively breaks down a problem into two or more sub-problems of the same or related type.",
                "solutions": self.generate_merge_sort_solutions(),
                "prev_link": "coin-change.html",
                "prev_name": "Coin Change",
                "next_link": "heap-sort.html",
                "next_name": "Heap Sort"
            },
            {
                "filename": "heap-sort.html",
                "title": "Heap Sort",
                "problem_name": "Heap Sort",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Implement heap sort algorithm to sort an array of integers in ascending order. Heap sort is a comparison-based sorting algorithm that uses a binary heap data structure.",
                "solutions": self.generate_heap_sort_solutions(),
                "prev_link": "merge-sort.html",
                "prev_name": "Merge Sort",
                "next_link": "bubble-sort.html",
                "next_name": "Bubble Sort"
            },
            
            # Bit Manipulation Problems
            {
                "filename": "power-of-two.html",
                "title": "Power of Two",
                "problem_name": "Power of Two",
                "category": "Bit Manipulation",
                "problem_number": "231",
                "problem_description": "Given an integer n, return true if it is a power of two. Otherwise, return false. An integer n is a power of two, if there exists an integer x such that n == 2^x.",
                "solutions": self.generate_power_of_two_solutions(),
                "prev_link": "heap-sort.html",
                "prev_name": "Heap Sort",
                "next_link": "counting-bits.html",
                "next_name": "Counting Bits"
            },
            {
                "filename": "counting-bits.html",
                "title": "Counting Bits",
                "problem_name": "Counting Bits",
                "category": "Bit Manipulation",
                "problem_number": "338",
                "problem_description": "Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.",
                "solutions": self.generate_counting_bits_solutions(),
                "prev_link": "power-of-two.html",
                "prev_name": "Power of Two",
                "next_link": "reverse-bits.html",
                "next_name": "Reverse Bits"
            },
            
            # Math Problems
            {
                "filename": "factorial-trailing-zeros.html",
                "title": "Factorial Trailing Zeroes",
                "problem_name": "Factorial Trailing Zeroes",
                "category": "Math",
                "problem_number": "172",
                "problem_description": "Given an integer n, return the number of trailing zeroes in n!. Note that n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1.",
                "solutions": self.generate_factorial_solutions(),
                "prev_link": "counting-bits.html",
                "prev_name": "Counting Bits",
                "next_link": "happy-number.html",
                "next_name": "Happy Number"
            },
            {
                "filename": "happy-number.html",
                "title": "Happy Number",
                "problem_name": "Happy Number",
                "category": "Math",
                "problem_number": "202",
                "problem_description": "Write an algorithm to determine if a number n is happy. A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.",
                "solutions": self.generate_happy_number_solutions(),
                "prev_link": "factorial-trailing-zeros.html",
                "prev_name": "Factorial Trailing Zeroes",
                "next_link": "perfect-squares.html",
                "next_name": "Perfect Squares"
            }
        ]

    def generate_lca_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Recursive DFS</h3>
                <p>Use recursive DFS to find the lowest common ancestor.</p>
                
                <div class="code-block">
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) {
        return root;
    }
    
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    
    if (left != null && right != null) {
        return root;
    }
    
    return left != null ? left : right;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n)<br>
                    <strong>Space Complexity:</strong> O(h) where h is the height of the tree
                </div>
            </div>
        '''

    def generate_serialize_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Preorder with Null Markers</h3>
                <p>Use preorder traversal with null markers to serialize and deserialize.</p>
                
                <div class="code-block">
public String serialize(TreeNode root) {
    if (root == null) {
        return "null";
    }
    
    return root.val + "," + serialize(root.left) + "," + serialize(root.right);
}

public TreeNode deserialize(String data) {
    String[] values = data.split(",");
    return deserializeHelper(values, new int[]{0});
}

private TreeNode deserializeHelper(String[] values, int[] index) {
    if (values[index[0]].equals("null")) {
        index[0]++;
        return null;
    }
    
    TreeNode root = new TreeNode(Integer.parseInt(values[index[0]]));
    index[0]++;
    root.left = deserializeHelper(values, index);
    root.right = deserializeHelper(values, index);
    
    return root;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n)<br>
                    <strong>Space Complexity:</strong> O(n)
                </div>
            </div>
        '''

    def generate_trie_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Trie Implementation</h3>
                <p>Implement a trie data structure with insert, search, and startsWith operations.</p>
                
                <div class="code-block">
class Trie {
    private TrieNode root;
    
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode current = root;
        for (char c : word.toCharArray()) {
            if (!current.children.containsKey(c)) {
                current.children.put(c, new TrieNode());
            }
            current = current.children.get(c);
        }
        current.isEndOfWord = true;
    }
    
    public boolean search(String word) {
        TrieNode current = root;
        for (char c : word.toCharArray()) {
            if (!current.children.containsKey(c)) {
                return false;
            }
            current = current.children.get(c);
        }
        return current.isEndOfWord;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode current = root;
        for (char c : prefix.toCharArray()) {
            if (!current.children.containsKey(c)) {
                return false;
            }
            current = current.children.get(c);
        }
        return true;
    }
}

class TrieNode {
    Map<Character, TrieNode> children;
    boolean isEndOfWord;
    
    TrieNode() {
        children = new HashMap<>();
        isEndOfWord = false;
    }
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(m) for insert/search/startsWith where m is word length<br>
                    <strong>Space Complexity:</strong> O(ALPHABET_SIZE * m * n) where n is number of words
                </div>
            </div>
        '''

    def generate_word_search_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – DFS with Backtracking</h3>
                <p>Use DFS with backtracking to search for the word in the grid.</p>
                
                <div class="code-block">
public boolean exist(char[][] board, String word) {
    int m = board.length;
    int n = board[0].length;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (dfs(board, word, i, j, 0)) {
                return true;
            }
        }
    }
    
    return false;
}

private boolean dfs(char[][] board, String word, int i, int j, int index) {
    if (index == word.length()) {
        return true;
    }
    
    if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || 
        board[i][j] != word.charAt(index)) {
        return false;
    }
    
    char temp = board[i][j];
    board[i][j] = '#';
    
    boolean result = dfs(board, word, i + 1, j, index + 1) ||
                    dfs(board, word, i - 1, j, index + 1) ||
                    dfs(board, word, i, j + 1, index + 1) ||
                    dfs(board, word, i, j - 1, index + 1);
    
    board[i][j] = temp;
    return result;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(m * n * 4^L) where L is word length<br>
                    <strong>Space Complexity:</strong> O(L) for recursion stack
                </div>
            </div>
        '''

    def generate_islands_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – DFS</h3>
                <p>Use DFS to mark visited islands and count the number of islands.</p>
                
                <div class="code-block">
public int numIslands(char[][] grid) {
    if (grid == null || grid.length == 0) {
        return 0;
    }
    
    int m = grid.length;
    int n = grid[0].length;
    int count = 0;
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == '1') {
                count++;
                dfs(grid, i, j);
            }
        }
    }
    
    return count;
}

private void dfs(char[][] grid, int i, int j) {
    if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || 
        grid[i][j] == '0') {
        return;
    }
    
    grid[i][j] = '0';
    dfs(grid, i + 1, j);
    dfs(grid, i - 1, j);
    dfs(grid, i, j + 1);
    dfs(grid, i, j - 1);
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(m * n)<br>
                    <strong>Space Complexity:</strong> O(m * n) in worst case
                </div>
            </div>
        '''

    def generate_course_schedule_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Topological Sort with DFS</h3>
                <p>Use DFS to detect cycles in the course dependency graph.</p>
                
                <div class="code-block">
public boolean canFinish(int numCourses, int[][] prerequisites) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < numCourses; i++) {
        graph.add(new ArrayList<>());
    }
    
    for (int[] prerequisite : prerequisites) {
        graph.get(prerequisite[1]).add(prerequisite[0]);
    }
    
    boolean[] visited = new boolean[numCourses];
    boolean[] recStack = new boolean[numCourses];
    
    for (int i = 0; i < numCourses; i++) {
        if (!visited[i] && hasCycle(graph, i, visited, recStack)) {
            return false;
        }
    }
    
    return true;
}

private boolean hasCycle(List<List<Integer>> graph, int node, 
                        boolean[] visited, boolean[] recStack) {
    visited[node] = true;
    recStack[node] = true;
    
    for (int neighbor : graph.get(node)) {
        if (!visited[neighbor] && hasCycle(graph, neighbor, visited, recStack)) {
            return true;
        } else if (recStack[neighbor]) {
            return true;
        }
    }
    
    recStack[node] = false;
    return false;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(V + E) where V is vertices and E is edges<br>
                    <strong>Space Complexity:</strong> O(V)
                </div>
            </div>
        '''

    def generate_climbing_stairs_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Dynamic Programming</h3>
                <p>Use DP to calculate the number of ways to climb stairs.</p>
                
                <div class="code-block">
public int climbStairs(int n) {
    if (n <= 2) {
        return n;
    }
    
    int[] dp = new int[n + 1];
    dp[1] = 1;
    dp[2] = 2;
    
    for (int i = 3; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    
    return dp[n];
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n)<br>
                    <strong>Space Complexity:</strong> O(n)
                </div>
            </div>
        '''

    def generate_house_robber_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Dynamic Programming</h3>
                <p>Use DP to find the maximum amount that can be robbed.</p>
                
                <div class="code-block">
public int rob(int[] nums) {
    if (nums == null || nums.length == 0) {
        return 0;
    }
    
    if (nums.length == 1) {
        return nums[0];
    }
    
    int[] dp = new int[nums.length];
    dp[0] = nums[0];
    dp[1] = Math.max(nums[0], nums[1]);
    
    for (int i = 2; i < nums.length; i++) {
        dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
    }
    
    return dp[nums.length - 1];
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n)<br>
                    <strong>Space Complexity:</strong> O(n)
                </div>
            </div>
        '''

    def generate_coin_change_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Dynamic Programming</h3>
                <p>Use DP to find the minimum number of coins needed.</p>
                
                <div class="code-block">
public int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1);
    dp[0] = 0;
    
    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    
    return dp[amount] > amount ? -1 : dp[amount];
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(amount * number of coins)<br>
                    <strong>Space Complexity:</strong> O(amount)
                </div>
            </div>
        '''

    def generate_merge_sort_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Merge Sort Implementation</h3>
                <p>Implement merge sort using divide and conquer approach.</p>
                
                <div class="code-block">
public void mergeSort(int[] arr) {
    if (arr == null || arr.length <= 1) {
        return;
    }
    
    int mid = arr.length / 2;
    int[] left = Arrays.copyOfRange(arr, 0, mid);
    int[] right = Arrays.copyOfRange(arr, mid, arr.length);
    
    mergeSort(left);
    mergeSort(right);
    merge(arr, left, right);
}

private void merge(int[] arr, int[] left, int[] right) {
    int i = 0, j = 0, k = 0;
    
    while (i < left.length && j < right.length) {
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }
    
    while (i < left.length) {
        arr[k++] = left[i++];
    }
    
    while (j < right.length) {
        arr[k++] = right[j++];
    }
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n log n)<br>
                    <strong>Space Complexity:</strong> O(n)
                </div>
            </div>
        '''

    def generate_heap_sort_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Heap Sort Implementation</h3>
                <p>Implement heap sort using max heap.</p>
                
                <div class="code-block">
public void heapSort(int[] arr) {
    int n = arr.length;
    
    // Build max heap
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }
    
    // Extract elements from heap one by one
    for (int i = n - 1; i > 0; i--) {
        // Move current root to end
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        
        // Call heapify on the reduced heap
        heapify(arr, i, 0);
    }
}

private void heapify(int[] arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && arr[left] > arr[largest]) {
        largest = left;
    }
    
    if (right < n && arr[right] > arr[largest]) {
        largest = right;
    }
    
    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;
        
        heapify(arr, n, largest);
    }
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n log n)<br>
                    <strong>Space Complexity:</strong> O(1)
                </div>
            </div>
        '''

    def generate_power_of_two_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Bit Manipulation</h3>
                <p>Use bit manipulation to check if a number is power of two.</p>
                
                <div class="code-block">
public boolean isPowerOfTwo(int n) {
    if (n <= 0) {
        return false;
    }
    
    return (n & (n - 1)) == 0;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(1)<br>
                    <strong>Space Complexity:</strong> O(1)
                </div>
            </div>
        '''

    def generate_counting_bits_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Dynamic Programming</h3>
                <p>Use DP to count bits efficiently.</p>
                
                <div class="code-block">
public int[] countBits(int n) {
    int[] result = new int[n + 1];
    
    for (int i = 1; i <= n; i++) {
        result[i] = result[i >> 1] + (i & 1);
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

    def generate_factorial_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Count Factors of 5</h3>
                <p>Count the number of factors of 5 in the factorial.</p>
                
                <div class="code-block">
public int trailingZeroes(int n) {
    int count = 0;
    
    while (n > 0) {
        n /= 5;
        count += n;
    }
    
    return count;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(log n)<br>
                    <strong>Space Complexity:</strong> O(1)
                </div>
            </div>
        '''

    def generate_happy_number_solutions(self) -> str:
        return '''
            <div class="solution">
                <h3>Solution 1 – Floyd's Cycle Detection</h3>
                <p>Use Floyd's cycle detection algorithm to detect cycles.</p>
                
                <div class="code-block">
public boolean isHappy(int n) {
    int slow = n;
    int fast = getNext(n);
    
    while (fast != 1 && slow != fast) {
        slow = getNext(slow);
        fast = getNext(getNext(fast));
    }
    
    return fast == 1;
}

private int getNext(int n) {
    int sum = 0;
    while (n > 0) {
        int digit = n % 10;
        sum += digit * digit;
        n /= 10;
    }
    return sum;
}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(log n)<br>
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
        print(f"Starting to generate {total_problems} problem pages...")
        
        for i in range(0, total_problems, batch_size):
            batch = self.problem_data[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, total_problems)} of {total_problems})")
            
            for problem in batch:
                filename = os.path.join(self.problems_dir, problem["filename"])
                content = self.generate_problem_page(problem)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"  ✓ Generated: {problem['filename']}")
            
            # Small delay between batches to avoid overwhelming the system
            if i + batch_size < total_problems:
                print("  Waiting 2 seconds before next batch...")
                time.sleep(2)
        
        print(f"\n🎉 Successfully generated {total_problems} problem pages!")
        print(f"📁 All files are saved in the '{self.problems_dir}' directory")

def main():
    """Main function to run the problem generator."""
    print("🚀 Starting Automated Coding Interview Problems Generator")
    print("=" * 60)
    
    generator = ProblemGenerator()
    
    # Generate all problems in batches of 5
    generator.generate_all_problems(batch_size=5)
    
    print("\n" + "=" * 60)
    print("✅ All problem pages have been generated successfully!")
    print("🌐 You can now open index.html to view the complete website.")

if __name__ == "__main__":
    main() 