#!/usr/bin/env python3
import os
import re

def get_final_solutions():
    """Return solutions for remaining coding interview problems."""
    return {
        "Binary Tree Level Order Traversal": {
            "solutions": [{
                "title": "Solution 1 – BFS with Queue",
                "description": "Use breadth-first search with a queue to traverse level by level.",
                "code": """public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) return result;
    
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    
    while (!queue.isEmpty()) {
        int levelSize = queue.size();
        List<Integer> currentLevel = new ArrayList<>();
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode current = queue.poll();
            currentLevel.add(current.val);
            
            if (current.left != null) {
                queue.offer(current.left);
            }
            if (current.right != null) {
                queue.offer(current.right);
            }
        }
        
        result.add(currentLevel);
    }
    
    return result;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Binary Tree Traversal": {
            "solutions": [{
                "title": "Solution 1 – Inorder, Preorder, Postorder",
                "description": "Implement all three types of depth-first traversal.",
                "code": """// Inorder Traversal (Left -> Root -> Right)
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    inorderHelper(root, result);
    return result;
}

private void inorderHelper(TreeNode root, List<Integer> result) {
    if (root == null) return;
    
    inorderHelper(root.left, result);
    result.add(root.val);
    inorderHelper(root.right, result);
}

// Preorder Traversal (Root -> Left -> Right)
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    preorderHelper(root, result);
    return result;
}

private void preorderHelper(TreeNode root, List<Integer> result) {
    if (root == null) return;
    
    result.add(root.val);
    preorderHelper(root.left, result);
    preorderHelper(root.right, result);
}

// Postorder Traversal (Left -> Right -> Root)
public List<Integer> postorderTraversal(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    postorderHelper(root, result);
    return result;
}

private void postorderHelper(TreeNode root, List<Integer> result) {
    if (root == null) return;
    
    postorderHelper(root.left, result);
    postorderHelper(root.right, result);
    result.add(root.val);
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Clone Graph": {
            "solutions": [{
                "title": "Solution 1 – DFS with HashMap",
                "description": "Use DFS to clone the graph while keeping track of visited nodes.",
                "code": """public Node cloneGraph(Node node) {
    if (node == null) return null;
    
    Map<Node, Node> visited = new HashMap<>();
    return cloneGraphHelper(node, visited);
}

private Node cloneGraphHelper(Node node, Map<Node, Node> visited) {
    if (visited.containsKey(node)) {
        return visited.get(node);
    }
    
    Node cloneNode = new Node(node.val);
    visited.put(node, cloneNode);
    
    for (Node neighbor : node.neighbors) {
        cloneNode.neighbors.add(cloneGraphHelper(neighbor, visited));
    }
    
    return cloneNode;
}""",
                "time": "O(V + E)",
                "space": "O(V)"
            }]
        },
        "Detect Cycle": {
            "solutions": [{
                "title": "Solution 1 – Floyd's Cycle Finding Algorithm",
                "description": "Use fast and slow pointers to detect cycle in linked list.",
                "code": """public boolean hasCycle(ListNode head) {
    if (head == null || head.next == null) return false;
    
    ListNode slow = head;
    ListNode fast = head;
    
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
        
        if (slow == fast) {
            return true;
        }
    }
    
    return false;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Evaluate Reverse Polish Notation": {
            "solutions": [{
                "title": "Solution 1 – Stack",
                "description": "Use a stack to evaluate reverse polish notation.",
                "code": """public int evalRPN(String[] tokens) {
    Stack<Integer> stack = new Stack<>();
    
    for (String token : tokens) {
        if (token.equals("+")) {
            stack.push(stack.pop() + stack.pop());
        } else if (token.equals("-")) {
            int b = stack.pop();
            int a = stack.pop();
            stack.push(a - b);
        } else if (token.equals("*")) {
            stack.push(stack.pop() * stack.pop());
        } else if (token.equals("/")) {
            int b = stack.pop();
            int a = stack.pop();
            stack.push(a / b);
        } else {
            stack.push(Integer.parseInt(token));
        }
    }
    
    return stack.pop();
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Factorial Trailing Zeros": {
            "solutions": [{
                "title": "Solution 1 – Count Factors of 5",
                "description": "Count the number of factors of 5 in the factorial.",
                "code": """public int trailingZeroes(int n) {
    int count = 0;
    
    while (n > 0) {
        n /= 5;
        count += n;
    }
    
    return count;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Flatten Binary Tree to Linked List": {
            "solutions": [{
                "title": "Solution 1 – Morris Traversal",
                "description": "Use Morris traversal to flatten the tree in-place.",
                "code": """public void flatten(TreeNode root) {
    TreeNode current = root;
    
    while (current != null) {
        if (current.left != null) {
            TreeNode predecessor = current.left;
            
            while (predecessor.right != null) {
                predecessor = predecessor.right;
            }
            
            predecessor.right = current.right;
            current.right = current.left;
            current.left = null;
        }
        
        current = current.right;
    }
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Isomorphic Strings": {
            "solutions": [{
                "title": "Solution 1 – Character Mapping",
                "description": "Use two hash maps to track character mappings in both directions.",
                "code": """public boolean isIsomorphic(String s, String t) {
    if (s.length() != t.length()) return false;
    
    Map<Character, Character> sToT = new HashMap<>();
    Map<Character, Character> tToS = new HashMap<>();
    
    for (int i = 0; i < s.length(); i++) {
        char sChar = s.charAt(i);
        char tChar = t.charAt(i);
        
        if (sToT.containsKey(sChar)) {
            if (sToT.get(sChar) != tChar) return false;
        } else {
            if (tToS.containsKey(tChar)) return false;
            sToT.put(sChar, tChar);
            tToS.put(tChar, sChar);
        }
    }
    
    return true;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Kth Largest Element": {
            "solutions": [{
                "title": "Solution 1 – Quick Select",
                "description": "Use quick select algorithm to find kth largest element.",
                "code": """public int findKthLargest(int[] nums, int k) {
    return quickSelect(nums, 0, nums.length - 1, nums.length - k);
}

private int quickSelect(int[] nums, int left, int right, int k) {
    if (left == right) return nums[left];
    
    int pivotIndex = partition(nums, left, right);
    
    if (k == pivotIndex) {
        return nums[k];
    } else if (k < pivotIndex) {
        return quickSelect(nums, left, pivotIndex - 1, k);
    } else {
        return quickSelect(nums, pivotIndex + 1, right, k);
    }
}

private int partition(int[] nums, int left, int right) {
    int pivot = nums[right];
    int i = left;
    
    for (int j = left; j < right; j++) {
        if (nums[j] <= pivot) {
            swap(nums, i, j);
            i++;
        }
    }
    
    swap(nums, i, right);
    return i;
}

private void swap(int[] nums, int i, int j) {
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}""",
                "time": "O(n) average, O(n²) worst case",
                "space": "O(1)"
            }]
        },
        "Level Order II": {
            "solutions": [{
                "title": "Solution 1 – BFS with Reversal",
                "description": "Use BFS and reverse the result list.",
                "code": """public List<List<Integer>> levelOrderBottom(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) return result;
    
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    
    while (!queue.isEmpty()) {
        int levelSize = queue.size();
        List<Integer> currentLevel = new ArrayList<>();
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode current = queue.poll();
            currentLevel.add(current.val);
            
            if (current.left != null) {
                queue.offer(current.left);
            }
            if (current.right != null) {
                queue.offer(current.right);
            }
        }
        
        result.add(0, currentLevel);
    }
    
    return result;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Median of Two Sorted Arrays": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Use binary search to find the median in O(log(min(m,n))) time.",
                "code": """public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    if (nums1.length > nums2.length) {
        return findMedianSortedArrays(nums2, nums1);
    }
    
    int x = nums1.length;
    int y = nums2.length;
    
    int low = 0;
    int high = x;
    
    while (low <= high) {
        int partitionX = (low + high) / 2;
        int partitionY = (x + y + 1) / 2 - partitionX;
        
        int maxLeftX = (partitionX == 0) ? Integer.MIN_VALUE : nums1[partitionX - 1];
        int minRightX = (partitionX == x) ? Integer.MAX_VALUE : nums1[partitionX];
        
        int maxLeftY = (partitionY == 0) ? Integer.MIN_VALUE : nums2[partitionY - 1];
        int minRightY = (partitionY == y) ? Integer.MAX_VALUE : nums2[partitionY];
        
        if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
            if ((x + y) % 2 == 0) {
                return (Math.max(maxLeftX, maxLeftY) + Math.min(minRightX, minRightY)) / 2.0;
            } else {
                return Math.max(maxLeftX, maxLeftY);
            }
        } else if (maxLeftX > minRightY) {
            high = partitionX - 1;
        } else {
            low = partitionX + 1;
        }
    }
    
    throw new IllegalArgumentException();
}""",
                "time": "O(log(min(m,n)))",
                "space": "O(1)"
            }]
        },
        "Merge Intervals": {
            "solutions": [{
                "title": "Solution 1 – Sort and Merge",
                "description": "Sort intervals by start time and merge overlapping intervals.",
                "code": """public int[][] merge(int[][] intervals) {
    if (intervals.length <= 1) return intervals;
    
    // Sort intervals by start time
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
    
    List<int[]> result = new ArrayList<>();
    int[] current = intervals[0];
    
    for (int i = 1; i < intervals.length; i++) {
        if (current[1] >= intervals[i][0]) {
            // Overlapping intervals, merge them
            current[1] = Math.max(current[1], intervals[i][1]);
        } else {
            // Non-overlapping interval, add current to result
            result.add(current);
            current = intervals[i];
        }
    }
    
    result.add(current);
    
    return result.toArray(new int[result.size()][]);
}""",
                "time": "O(n log n)",
                "space": "O(n)"
            }]
        },
        "Populating Next Right Pointers": {
            "solutions": [{
                "title": "Solution 1 – Level Order Traversal",
                "description": "Use level order traversal to connect nodes at the same level.",
                "code": """public Node connect(Node root) {
    if (root == null) return null;
    
    Node levelStart = root;
    
    while (levelStart.left != null) {
        Node current = levelStart;
        
        while (current != null) {
            current.left.next = current.right;
            
            if (current.next != null) {
                current.right.next = current.next.left;
            }
            
            current = current.next;
        }
        
        levelStart = levelStart.left;
    }
    
    return root;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Power of Two": {
            "solutions": [{
                "title": "Solution 1 – Bit Manipulation",
                "description": "Check if the number has exactly one bit set.",
                "code": """public boolean isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}""",
                "time": "O(1)",
                "space": "O(1)"
            }]
        },
        "Regular Expression Matching": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to match pattern with string.",
                "code": """public boolean isMatch(String s, String p) {
    int m = s.length();
    int n = p.length();
    
    boolean[][] dp = new boolean[m + 1][n + 1];
    dp[0][0] = true;
    
    // Handle patterns like a*, a*b*, etc.
    for (int j = 1; j <= n; j++) {
        if (p.charAt(j - 1) == '*') {
            dp[0][j] = dp[0][j - 2];
        }
    }
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (p.charAt(j - 1) == '.' || p.charAt(j - 1) == s.charAt(i - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            } else if (p.charAt(j - 1) == '*') {
                dp[i][j] = dp[i][j - 2]; // zero occurrence
                if (p.charAt(j - 2) == '.' || p.charAt(j - 2) == s.charAt(i - 1)) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j];
                }
            }
        }
    }
    
    return dp[m][n];
}""",
                "time": "O(m×n)",
                "space": "O(m×n)"
            }]
        },
        "Reorder List": {
            "solutions": [{
                "title": "Solution 1 – Find Middle, Reverse, Merge",
                "description": "Find middle, reverse second half, then merge the two halves.",
                "code": """public void reorderList(ListNode head) {
    if (head == null || head.next == null) return;
    
    // Find the middle
    ListNode slow = head, fast = head;
    while (fast.next != null && fast.next.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    
    // Reverse the second half
    ListNode second = reverseList(slow.next);
    slow.next = null;
    
    // Merge the two halves
    ListNode first = head;
    while (second != null) {
        ListNode temp1 = first.next;
        ListNode temp2 = second.next;
        
        first.next = second;
        second.next = temp1;
        
        first = temp1;
        second = temp2;
    }
}

private ListNode reverseList(ListNode head) {
    ListNode prev = null;
    ListNode current = head;
    
    while (current != null) {
        ListNode next = current.next;
        current.next = prev;
        prev = current;
        current = next;
    }
    
    return prev;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Reverse Words in a String": {
            "solutions": [{
                "title": "Solution 1 – Split and Reverse",
                "description": "Split the string, reverse words, then join.",
                "code": """public String reverseWords(String s) {
    // Remove leading and trailing spaces, split by multiple spaces
    String[] words = s.trim().split("\\s+");
    
    // Reverse the array
    for (int i = 0; i < words.length / 2; i++) {
        String temp = words[i];
        words[i] = words[words.length - 1 - i];
        words[words.length - 1 - i] = temp;
    }
    
    return String.join(" ", words);
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Rotate Array": {
            "solutions": [{
                "title": "Solution 1 – Reverse Method",
                "description": "Reverse the entire array, then reverse first k and last n-k elements.",
                "code": """public void rotate(int[] nums, int k) {
    int n = nums.length;
    k = k % n; // Handle cases where k > n
    
    reverse(nums, 0, n - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, n - 1);
}

private void reverse(int[] nums, int start, int end) {
    while (start < end) {
        int temp = nums[start];
        nums[start] = nums[end];
        nums[end] = temp;
        start++;
        end--;
    }
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Serialize and Deserialize Binary Tree": {
            "solutions": [{
                "title": "Solution 1 – Preorder with Null Markers",
                "description": "Use preorder traversal with null markers for serialization.",
                "code": """public String serialize(TreeNode root) {
    StringBuilder sb = new StringBuilder();
    serializeHelper(root, sb);
    return sb.toString();
}

private void serializeHelper(TreeNode root, StringBuilder sb) {
    if (root == null) {
        sb.append("null,");
        return;
    }
    
    sb.append(root.val).append(",");
    serializeHelper(root.left, sb);
    serializeHelper(root.right, sb);
}

public TreeNode deserialize(String data) {
    String[] values = data.split(",");
    Queue<String> queue = new LinkedList<>(Arrays.asList(values));
    return deserializeHelper(queue);
}

private TreeNode deserializeHelper(Queue<String> queue) {
    String value = queue.poll();
    
    if (value.equals("null")) {
        return null;
    }
    
    TreeNode root = new TreeNode(Integer.parseInt(value));
    root.left = deserializeHelper(queue);
    root.right = deserializeHelper(queue);
    
    return root;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Set Matrix Zeroes": {
            "solutions": [{
                "title": "Solution 1 – Use First Row/Column as Markers",
                "description": "Use the first row and column as markers to avoid extra space.",
                "code": """public void setZeroes(int[][] matrix) {
    int m = matrix.length;
    int n = matrix[0].length;
    boolean firstRowHasZero = false;
    boolean firstColHasZero = false;
    
    // Check if first row has zero
    for (int j = 0; j < n; j++) {
        if (matrix[0][j] == 0) {
            firstRowHasZero = true;
            break;
        }
    }
    
    // Check if first column has zero
    for (int i = 0; i < m; i++) {
        if (matrix[i][0] == 0) {
            firstColHasZero = true;
            break;
        }
    }
    
    // Use first row and column as markers
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (matrix[i][j] == 0) {
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }
    }
    
    // Set zeros based on markers
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                matrix[i][j] = 0;
            }
        }
    }
    
    // Set first row to zero if needed
    if (firstRowHasZero) {
        for (int j = 0; j < n; j++) {
            matrix[0][j] = 0;
        }
    }
    
    // Set first column to zero if needed
    if (firstColHasZero) {
        for (int i = 0; i < m; i++) {
            matrix[i][0] = 0;
        }
    }
}""",
                "time": "O(m×n)",
                "space": "O(1)"
            }]
        },
        "Spiral Matrix": {
            "solutions": [{
                "title": "Solution 1 – Layer by Layer",
                "description": "Traverse the matrix layer by layer in spiral order.",
                "code": """public List<Integer> spiralOrder(int[][] matrix) {
    List<Integer> result = new ArrayList<>();
    if (matrix == null || matrix.length == 0) return result;
    
    int top = 0, bottom = matrix.length - 1;
    int left = 0, right = matrix[0].length - 1;
    
    while (top <= bottom && left <= right) {
        // Traverse right
        for (int j = left; j <= right; j++) {
            result.add(matrix[top][j]);
        }
        top++;
        
        // Traverse down
        for (int i = top; i <= bottom; i++) {
            result.add(matrix[i][right]);
        }
        right--;
        
        // Traverse left
        if (top <= bottom) {
            for (int j = right; j >= left; j--) {
                result.add(matrix[bottom][j]);
            }
            bottom--;
        }
        
        // Traverse up
        if (left <= right) {
            for (int i = bottom; i >= top; i--) {
                result.add(matrix[i][left]);
            }
            left++;
        }
    }
    
    return result;
}""",
                "time": "O(m×n)",
                "space": "O(1)"
            }]
        },
        "Wildcard Matching": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to match pattern with string, handling * and ? wildcards.",
                "code": """public boolean isMatch(String s, String p) {
    int m = s.length();
    int n = p.length();
    
    boolean[][] dp = new boolean[m + 1][n + 1];
    dp[0][0] = true;
    
    // Handle patterns like *, **, etc.
    for (int j = 1; j <= n; j++) {
        if (p.charAt(j - 1) == '*') {
            dp[0][j] = dp[0][j - 1];
        }
    }
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (p.charAt(j - 1) == '?' || p.charAt(j - 1) == s.charAt(i - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            } else if (p.charAt(j - 1) == '*') {
                dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            }
        }
    }
    
    return dp[m][n];
}""",
                "time": "O(m×n)",
                "space": "O(m×n)"
            }]
        },
        "Word Ladder": {
            "solutions": [{
                "title": "Solution 1 – BFS",
                "description": "Use BFS to find the shortest transformation sequence.",
                "code": """public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    Set<String> wordSet = new HashSet<>(wordList);
    if (!wordSet.contains(endWord)) return 0;
    
    Queue<String> queue = new LinkedList<>();
    queue.offer(beginWord);
    Set<String> visited = new HashSet<>();
    visited.add(beginWord);
    
    int level = 1;
    
    while (!queue.isEmpty()) {
        int size = queue.size();
        
        for (int i = 0; i < size; i++) {
            String current = queue.poll();
            
            if (current.equals(endWord)) {
                return level;
            }
            
            // Try all possible transformations
            for (int j = 0; j < current.length(); j++) {
                char[] chars = current.toCharArray();
                
                for (char c = 'a'; c <= 'z'; c++) {
                    chars[j] = c;
                    String newWord = new String(chars);
                    
                    if (wordSet.contains(newWord) && !visited.contains(newWord)) {
                        queue.offer(newWord);
                        visited.add(newWord);
                    }
                }
            }
        }
        
        level++;
    }
    
    return 0;
}""",
                "time": "O(26 × wordLength × wordListSize)",
                "space": "O(wordListSize)"
            }]
        },
        "Word Ladder II": {
            "solutions": [{
                "title": "Solution 1 – BFS + DFS",
                "description": "Use BFS to find shortest paths, then DFS to reconstruct them.",
                "code": """public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
    Set<String> wordSet = new HashSet<>(wordList);
    List<List<String>> result = new ArrayList<>();
    
    if (!wordSet.contains(endWord)) return result;
    
    Map<String, List<String>> graph = new HashMap<>();
    Map<String, Integer> distance = new HashMap<>();
    
    // BFS to build graph and find shortest distance
    Queue<String> queue = new LinkedList<>();
    queue.offer(beginWord);
    distance.put(beginWord, 0);
    
    while (!queue.isEmpty()) {
        String current = queue.poll();
        
        if (current.equals(endWord)) break;
        
        for (int i = 0; i < current.length(); i++) {
            char[] chars = current.toCharArray();
            
            for (char c = 'a'; c <= 'z'; c++) {
                chars[i] = c;
                String newWord = new String(chars);
                
                if (wordSet.contains(newWord)) {
                    if (!distance.containsKey(newWord)) {
                        distance.put(newWord, distance.get(current) + 1);
                        queue.offer(newWord);
                    }
                    
                    if (distance.get(newWord) == distance.get(current) + 1) {
                        graph.computeIfAbsent(current, k -> new ArrayList<>()).add(newWord);
                    }
                }
            }
        }
    }
    
    // DFS to find all shortest paths
    List<String> path = new ArrayList<>();
    path.add(beginWord);
    dfs(beginWord, endWord, graph, distance, path, result);
    
    return result;
}

private void dfs(String current, String endWord, Map<String, List<String>> graph,
                Map<String, Integer> distance, List<String> path, List<List<String>> result) {
    if (current.equals(endWord)) {
        result.add(new ArrayList<>(path));
        return;
    }
    
    List<String> neighbors = graph.get(current);
    if (neighbors == null) return;
    
    for (String neighbor : neighbors) {
        if (distance.get(neighbor) == distance.get(current) + 1) {
            path.add(neighbor);
            dfs(neighbor, endWord, graph, distance, path, result);
            path.remove(path.size() - 1);
        }
    }
}""",
                "time": "O(26 × wordLength × wordListSize)",
                "space": "O(wordListSize)"
            }]
        }
    }

def update_problem_with_solution(filename, problem_name):
    """Update a problem file with actual solution."""
    filepath = f"problems/{filename}"
    
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file has placeholder solution
    if "TODO: Implement solution" in content:
        solutions_data = get_final_solutions()
        solution_data = solutions_data.get(problem_name)
        
        if solution_data:
            # Replace the placeholder solution
            new_solutions_html = ""
            for solution in solution_data["solutions"]:
                new_solutions_html += f'''
            <div class="solution">
                <h3>{solution["title"]}</h3>
                <p>{solution["description"]}</p>
                
                <div class="code-block">
{solution["code"]}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> {solution["time"]}<br>
                    <strong>Space Complexity:</strong> {solution["space"]}
                </div>
            </div>'''
            
            # Replace the placeholder solution section
            pattern = r'<div class="solution">\s*<h3>Solution 1 – Basic Approach</h3>.*?</div>\s*</div>\s*</div>'
            replacement = f'<div class="solution">{new_solutions_html}\n            </div>\n        </div>\n    </div>'
            
            updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✓ Updated: {filename}")
            return True
    
    return False

def main():
    """Update all remaining problem files with final solutions."""
    print("🚀 Starting Final Solutions Update Process")
    print("=" * 70)
    
    # List of problems to update
    problems_to_update = [
        ("binary-tree-level-order-traversal.html", "Binary Tree Level Order Traversal"),
        ("binary-tree-traversal.html", "Binary Tree Traversal"),
        ("clone-graph.html", "Clone Graph"),
        ("detect-cycle.html", "Detect Cycle"),
        ("evaluate-reverse-polish-notation.html", "Evaluate Reverse Polish Notation"),
        ("factorial-trailing-zeros.html", "Factorial Trailing Zeros"),
        ("flatten-binary-tree.html", "Flatten Binary Tree to Linked List"),
        ("isomorphic-strings.html", "Isomorphic Strings"),
        ("kth-largest-element.html", "Kth Largest Element"),
        ("level-order-ii.html", "Level Order II"),
        ("median-two-sorted-arrays.html", "Median of Two Sorted Arrays"),
        ("merge-intervals.html", "Merge Intervals"),
        ("populating-next-right-pointers.html", "Populating Next Right Pointers"),
        ("power-of-two.html", "Power of Two"),
        ("regular-expression-matching.html", "Regular Expression Matching"),
        ("reorder-list.html", "Reorder List"),
        ("reverse-words-string.html", "Reverse Words in a String"),
        ("rotate-array.html", "Rotate Array"),
        ("serialize-deserialize-binary-tree.html", "Serialize and Deserialize Binary Tree"),
        ("set-matrix-zeroes.html", "Set Matrix Zeroes"),
        ("spiral-matrix.html", "Spiral Matrix"),
        ("wildcard-matching.html", "Wildcard Matching"),
        ("word-ladder.html", "Word Ladder"),
        ("word-ladder-ii.html", "Word Ladder II")
    ]
    
    updated_count = 0
    
    for filename, problem_name in problems_to_update:
        if update_problem_with_solution(filename, problem_name):
            updated_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"✅ Updated {updated_count} problem files with final solutions!")
    print("🌐 You can now view the complete solutions in the problem pages.")

if __name__ == "__main__":
    main() 