#!/usr/bin/env python3
import os
import re

def get_remaining_solutions():
    """Return solutions for remaining important coding interview problems."""
    return {
        "Shortest Palindrome": {
            "solutions": [{
                "title": "Solution 1 – KMP Algorithm",
                "description": "Use KMP algorithm to find the longest palindromic prefix.",
                "code": """public String shortestPalindrome(String s) {
    String temp = s + "#" + new StringBuilder(s).reverse().toString();
    int[] table = getTable(temp);
    
    return new StringBuilder(s.substring(table[table.length - 1])).reverse().toString() + s;
}

private int[] getTable(String s) {
    int[] table = new int[s.length()];
    int j = 0;
    
    for (int i = 1; i < s.length(); i++) {
        while (j > 0 && s.charAt(i) != s.charAt(j)) {
            j = table[j - 1];
        }
        if (s.charAt(i) == s.charAt(j)) {
            j++;
        }
        table[i] = j;
    }
    
    return table;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Decode Ways": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to count ways to decode the string.",
                "code": """public int numDecodings(String s) {
    if (s == null || s.length() == 0) return 0;
    
    int n = s.length();
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = s.charAt(0) == '0' ? 0 : 1;
    
    for (int i = 2; i <= n; i++) {
        int oneDigit = Integer.parseInt(s.substring(i - 1, i));
        int twoDigits = Integer.parseInt(s.substring(i - 2, i));
        
        if (oneDigit >= 1 && oneDigit <= 9) {
            dp[i] += dp[i - 1];
        }
        
        if (twoDigits >= 10 && twoDigits <= 26) {
            dp[i] += dp[i - 2];
        }
    }
    
    return dp[n];
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Integer Break": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to find the maximum product.",
                "code": """public int integerBreak(int n) {
    if (n <= 3) return n - 1;
    
    int[] dp = new int[n + 1];
    dp[1] = 1;
    dp[2] = 2;
    dp[3] = 3;
    
    for (int i = 4; i <= n; i++) {
        for (int j = 1; j <= i / 2; j++) {
            dp[i] = Math.max(dp[i], dp[j] * dp[i - j]);
        }
    }
    
    return dp[n];
}""",
                "time": "O(n²)",
                "space": "O(n)"
            }]
        },
        "Reverse Bits": {
            "solutions": [{
                "title": "Solution 1 – Bit Manipulation",
                "description": "Reverse bits using bit manipulation.",
                "code": """public int reverseBits(int n) {
    int result = 0;
    
    for (int i = 0; i < 32; i++) {
        result = (result << 1) | (n & 1);
        n = n >> 1;
    }
    
    return result;
}""",
                "time": "O(1)",
                "space": "O(1)"
            }]
        },
        "Linked List Cycle": {
            "solutions": [{
                "title": "Solution 1 – Floyd's Cycle Finding",
                "description": "Use fast and slow pointers to detect cycle.",
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
        "Largest Number": {
            "solutions": [{
                "title": "Solution 1 – Custom Comparator",
                "description": "Sort numbers using custom comparator.",
                "code": """public String largestNumber(int[] nums) {
    String[] strs = new String[nums.length];
    for (int i = 0; i < nums.length; i++) {
        strs[i] = String.valueOf(nums[i]);
    }
    
    Arrays.sort(strs, (a, b) -> (b + a).compareTo(a + b));
    
    if (strs[0].equals("0")) return "0";
    
    StringBuilder result = new StringBuilder();
    for (String str : strs) {
        result.append(str);
    }
    
    return result.toString();
}""",
                "time": "O(n log n)",
                "space": "O(n)"
            }]
        },
        "Implement strStr()": {
            "solutions": [{
                "title": "Solution 1 – KMP Algorithm",
                "description": "Use KMP algorithm for efficient string matching.",
                "code": """public int strStr(String haystack, String needle) {
    if (needle.isEmpty()) return 0;
    if (haystack.isEmpty()) return -1;
    
    int[] lps = computeLPS(needle);
    int i = 0, j = 0;
    
    while (i < haystack.length()) {
        if (haystack.charAt(i) == needle.charAt(j)) {
            i++;
            j++;
        }
        
        if (j == needle.length()) {
            return i - j;
        } else if (i < haystack.length() && haystack.charAt(i) != needle.charAt(j)) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    
    return -1;
}

private int[] computeLPS(String pattern) {
    int[] lps = new int[pattern.length()];
    int len = 0;
    int i = 1;
    
    while (i < pattern.length()) {
        if (pattern.charAt(i) == pattern.charAt(len)) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    
    return lps;
}""",
                "time": "O(m + n)",
                "space": "O(n)"
            }]
        },
        "Remove Invalid Parentheses": {
            "solutions": [{
                "title": "Solution 1 – BFS",
                "description": "Use BFS to find all valid strings with minimum removals.",
                "code": """public List<String> removeInvalidParentheses(String s) {
    List<String> result = new ArrayList<>();
    if (s == null) return result;
    
    Set<String> visited = new HashSet<>();
    Queue<String> queue = new LinkedList<>();
    queue.offer(s);
    visited.add(s);
    boolean found = false;
    
    while (!queue.isEmpty()) {
        String current = queue.poll();
        
        if (isValid(current)) {
            result.add(current);
            found = true;
        }
        
        if (found) continue;
        
        for (int i = 0; i < current.length(); i++) {
            if (current.charAt(i) != '(' && current.charAt(i) != ')') continue;
            
            String next = current.substring(0, i) + current.substring(i + 1);
            if (!visited.contains(next)) {
                queue.offer(next);
                visited.add(next);
            }
        }
    }
    
    return result;
}

private boolean isValid(String s) {
    int count = 0;
    for (char c : s.toCharArray()) {
        if (c == '(') count++;
        else if (c == ')') count--;
        if (count < 0) return false;
    }
    return count == 0;
}""",
                "time": "O(2^n)",
                "space": "O(n)"
            }]
        },
        "Find Median from Data Stream": {
            "solutions": [{
                "title": "Solution 1 – Two Heaps",
                "description": "Use min heap and max heap to maintain median.",
                "code": """class MedianFinder {
    private PriorityQueue<Integer> minHeap;
    private PriorityQueue<Integer> maxHeap;
    
    public MedianFinder() {
        minHeap = new PriorityQueue<>();
        maxHeap = new PriorityQueue<>(Collections.reverseOrder());
    }
    
    public void addNum(int num) {
        maxHeap.offer(num);
        minHeap.offer(maxHeap.poll());
        
        if (maxHeap.size() < minHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
    }
    
    public double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.peek();
        }
        return (maxHeap.peek() + minHeap.peek()) / 2.0;
    }
}""",
                "time": "O(log n) for add, O(1) for find",
                "space": "O(n)"
            }]
        },
        "Summary Ranges": {
            "solutions": [{
                "title": "Solution 1 – Linear Scan",
                "description": "Scan array and group consecutive numbers.",
                "code": """public List<String> summaryRanges(int[] nums) {
    List<String> result = new ArrayList<>();
    if (nums == null || nums.length == 0) return result;
    
    int start = nums[0];
    int end = nums[0];
    
    for (int i = 1; i < nums.length; i++) {
        if (nums[i] == end + 1) {
            end = nums[i];
        } else {
            result.add(formatRange(start, end));
            start = end = nums[i];
        }
    }
    
    result.add(formatRange(start, end));
    return result;
}

private String formatRange(int start, int end) {
    return start == end ? String.valueOf(start) : start + "->" + end;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Group Anagrams": {
            "solutions": [{
                "title": "Solution 1 – HashMap",
                "description": "Group strings by their sorted character array.",
                "code": """public List<List<String>> groupAnagrams(String[] strs) {
    Map<String, List<String>> map = new HashMap<>();
    
    for (String str : strs) {
        char[] chars = str.toCharArray();
        Arrays.sort(chars);
        String key = new String(chars);
        
        map.computeIfAbsent(key, k -> new ArrayList<>()).add(str);
    }
    
    return new ArrayList<>(map.values());
}""",
                "time": "O(n * k log k)",
                "space": "O(n * k)"
            }]
        },
        "Search a 2D Matrix": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Treat matrix as sorted array and use binary search.",
                "code": """public boolean searchMatrix(int[][] matrix, int target) {
    if (matrix == null || matrix.length == 0) return false;
    
    int m = matrix.length;
    int n = matrix[0].length;
    int left = 0;
    int right = m * n - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int row = mid / n;
        int col = mid % n;
        
        if (matrix[row][col] == target) {
            return true;
        } else if (matrix[row][col] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return false;
}""",
                "time": "O(log(mn))",
                "space": "O(1)"
            }]
        },
        "Gray Code": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Generate gray code recursively.",
                "code": """public List<Integer> grayCode(int n) {
    List<Integer> result = new ArrayList<>();
    result.add(0);
    
    for (int i = 0; i < n; i++) {
        int size = result.size();
        for (int j = size - 1; j >= 0; j--) {
            result.add(result.get(j) | (1 << i));
        }
    }
    
    return result;
}""",
                "time": "O(2^n)",
                "space": "O(2^n)"
            }]
        },
        "Binary Search Tree Iterator": {
            "solutions": [{
                "title": "Solution 1 – Stack",
                "description": "Use stack to simulate inorder traversal.",
                "code": """class BSTIterator {
    private Stack<TreeNode> stack;
    
    public BSTIterator(TreeNode root) {
        stack = new Stack<>();
        pushAll(root);
    }
    
    public int next() {
        TreeNode node = stack.pop();
        pushAll(node.right);
        return node.val;
    }
    
    public boolean hasNext() {
        return !stack.isEmpty();
    }
    
    private void pushAll(TreeNode root) {
        while (root != null) {
            stack.push(root);
            root = root.left;
        }
    }
}""",
                "time": "O(1) amortized",
                "space": "O(h)"
            }]
        },
        "Longest Consecutive Sequence": {
            "solutions": [{
                "title": "Solution 1 – HashSet",
                "description": "Use HashSet to find consecutive sequences.",
                "code": """public int longestConsecutive(int[] nums) {
    Set<Integer> set = new HashSet<>();
    for (int num : nums) {
        set.add(num);
    }
    
    int maxLength = 0;
    
    for (int num : set) {
        if (!set.contains(num - 1)) {
            int currentNum = num;
            int currentLength = 1;
            
            while (set.contains(currentNum + 1)) {
                currentNum++;
                currentLength++;
            }
            
            maxLength = Math.max(maxLength, currentLength);
        }
    }
    
    return maxLength;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Reverse Vowels of a String": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to reverse vowels.",
                "code": """public String reverseVowels(String s) {
    char[] chars = s.toCharArray();
    int left = 0, right = chars.length - 1;
    
    while (left < right) {
        while (left < right && !isVowel(chars[left])) left++;
        while (left < right && !isVowel(chars[right])) right--;
        
        if (left < right) {
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            left++;
            right--;
        }
    }
    
    return new String(chars);
}

private boolean isVowel(char c) {
    return "aeiouAEIOU".indexOf(c) != -1;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Find Minimum in Rotated Sorted Array": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Use binary search to find minimum in rotated array.",
                "code": """public int findMin(int[] nums) {
    int left = 0;
    int right = nums.length - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] > nums[right]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return nums[left];
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "One Edit Distance": {
            "solutions": [{
                "title": "Solution 1 – Character Comparison",
                "description": "Compare characters and allow one edit.",
                "code": """public boolean isOneEditDistance(String s, String t) {
    int m = s.length();
    int n = t.length();
    
    if (Math.abs(m - n) > 1) return false;
    
    int i = 0, j = 0;
    int diff = 0;
    
    while (i < m && j < n) {
        if (s.charAt(i) == t.charAt(j)) {
            i++;
            j++;
        } else {
            diff++;
            if (diff > 1) return false;
            
            if (m > n) {
                i++;
            } else if (m < n) {
                j++;
            } else {
                i++;
                j++;
            }
        }
    }
    
    if (i < m || j < n) diff++;
    
    return diff == 1;
}""",
                "time": "O(min(m,n))",
                "space": "O(1)"
            }]
        },
        "Longest Increasing Subsequence": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to find longest increasing subsequence.",
                "code": """public int lengthOfLIS(int[] nums) {
    if (nums == null || nums.length == 0) return 0;
    
    int[] dp = new int[nums.length];
    Arrays.fill(dp, 1);
    
    int maxLength = 1;
    
    for (int i = 1; i < nums.length; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }
        maxLength = Math.max(maxLength, dp[i]);
    }
    
    return maxLength;
}""",
                "time": "O(n²)",
                "space": "O(n)"
            }]
        },
        "Spiral Matrix II": {
            "solutions": [{
                "title": "Solution 1 – Layer by Layer",
                "description": "Fill matrix in spiral order layer by layer.",
                "code": """public int[][] generateMatrix(int n) {
    int[][] matrix = new int[n][n];
    int num = 1;
    int top = 0, bottom = n - 1;
    int left = 0, right = n - 1;
    
    while (top <= bottom && left <= right) {
        for (int j = left; j <= right; j++) {
            matrix[top][j] = num++;
        }
        top++;
        
        for (int i = top; i <= bottom; i++) {
            matrix[i][right] = num++;
        }
        right--;
        
        if (top <= bottom) {
            for (int j = right; j >= left; j--) {
                matrix[bottom][j] = num++;
            }
            bottom--;
        }
        
        if (left <= right) {
            for (int i = bottom; i >= top; i--) {
                matrix[i][left] = num++;
            }
            left++;
        }
    }
    
    return matrix;
}""",
                "time": "O(n²)",
                "space": "O(1)"
            }]
        },
        "Surrounded Regions": {
            "solutions": [{
                "title": "Solution 1 – DFS",
                "description": "Use DFS to mark regions connected to border.",
                "code": """public void solve(char[][] board) {
    if (board == null || board.length == 0) return;
    
    int m = board.length;
    int n = board[0].length;
    
    // Mark 'O's connected to border
    for (int i = 0; i < m; i++) {
        dfs(board, i, 0);
        dfs(board, i, n - 1);
    }
    
    for (int j = 0; j < n; j++) {
        dfs(board, 0, j);
        dfs(board, m - 1, j);
    }
    
    // Flip remaining 'O's to 'X' and 'E's back to 'O'
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (board[i][j] == 'O') {
                board[i][j] = 'X';
            } else if (board[i][j] == 'E') {
                board[i][j] = 'O';
            }
        }
    }
}

private void dfs(char[][] board, int i, int j) {
    if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != 'O') {
        return;
    }
    
    board[i][j] = 'E';
    
    dfs(board, i + 1, j);
    dfs(board, i - 1, j);
    dfs(board, i, j + 1);
    dfs(board, i, j - 1);
}""",
                "time": "O(m×n)",
                "space": "O(m×n)"
            }]
        },
        "Product of Array Except Self": {
            "solutions": [{
                "title": "Solution 1 – Two Pass",
                "description": "Use two passes to calculate product without division.",
                "code": """public int[] productExceptSelf(int[] nums) {
    int n = nums.length;
    int[] result = new int[n];
    
    // Calculate left products
    result[0] = 1;
    for (int i = 1; i < n; i++) {
        result[i] = result[i - 1] * nums[i - 1];
    }
    
    // Calculate right products and combine
    int right = 1;
    for (int i = n - 1; i >= 0; i--) {
        result[i] = result[i] * right;
        right *= nums[i];
    }
    
    return result;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Copy List with Random Pointer": {
            "solutions": [{
                "title": "Solution 1 – HashMap",
                "description": "Use HashMap to map original nodes to copied nodes.",
                "code": """public Node copyRandomList(Node head) {
    if (head == null) return null;
    
    Map<Node, Node> map = new HashMap<>();
    
    // First pass: create copies
    Node current = head;
    while (current != null) {
        map.put(current, new Node(current.val));
        current = current.next;
    }
    
    // Second pass: set next and random pointers
    current = head;
    while (current != null) {
        Node copy = map.get(current);
        copy.next = map.get(current.next);
        copy.random = map.get(current.random);
        current = current.next;
    }
    
    return map.get(head);
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Valid Sudoku": {
            "solutions": [{
                "title": "Solution 1 – HashSet",
                "description": "Use HashSet to check rows, columns, and boxes.",
                "code": """public boolean isValidSudoku(char[][] board) {
    for (int i = 0; i < 9; i++) {
        Set<Character> row = new HashSet<>();
        Set<Character> col = new HashSet<>();
        Set<Character> box = new HashSet<>();
        
        for (int j = 0; j < 9; j++) {
            // Check row
            if (board[i][j] != '.' && !row.add(board[i][j])) return false;
            
            // Check column
            if (board[j][i] != '.' && !col.add(board[j][i])) return false;
            
            // Check 3x3 box
            int boxRow = 3 * (i / 3) + j / 3;
            int boxCol = 3 * (i % 3) + j % 3;
            if (board[boxRow][boxCol] != '.' && !box.add(board[boxRow][boxCol])) return false;
        }
    }
    
    return true;
}""",
                "time": "O(n²)",
                "space": "O(n)"
            }]
        },
        "Permutation Sequence": {
            "solutions": [{
                "title": "Solution 1 – Mathematical",
                "description": "Use mathematical approach to find kth permutation.",
                "code": """public String getPermutation(int n, int k) {
    List<Integer> numbers = new ArrayList<>();
    int[] factorial = new int[n + 1];
    factorial[0] = 1;
    
    for (int i = 1; i <= n; i++) {
        factorial[i] = factorial[i - 1] * i;
        numbers.add(i);
    }
    
    k--;
    StringBuilder result = new StringBuilder();
    
    for (int i = n; i > 0; i--) {
        int index = k / factorial[i - 1];
        result.append(numbers.get(index));
        numbers.remove(index);
        k %= factorial[i - 1];
    }
    
    return result.toString();
}""",
                "time": "O(n²)",
                "space": "O(n)"
            }]
        },
        "Graph Valid Tree": {
            "solutions": [{
                "title": "Solution 1 – Union Find",
                "description": "Use Union Find to detect cycles and check connectivity.",
                "code": """public boolean validTree(int n, int[][] edges) {
    if (edges.length != n - 1) return false;
    
    UnionFind uf = new UnionFind(n);
    
    for (int[] edge : edges) {
        if (!uf.union(edge[0], edge[1])) {
            return false;
        }
    }
    
    return uf.getCount() == 1;
}

class UnionFind {
    private int[] parent;
    private int[] rank;
    private int count;
    
    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
        count = n;
        
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    public boolean union(int x, int y) {
        int px = find(x);
        int py = find(y);
        
        if (px == py) return false;
        
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        
        count--;
        return true;
    }
    
    public int getCount() {
        return count;
    }
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Scramble String": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to check if strings are scramble.",
                "code": """public boolean isScramble(String s1, String s2) {
    if (s1.equals(s2)) return true;
    if (s1.length() != s2.length()) return false;
    
    int n = s1.length();
    boolean[][][] dp = new boolean[n][n][n + 1];
    
    for (int len = 1; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            for (int j = 0; j <= n - len; j++) {
                if (len == 1) {
                    dp[i][j][len] = s1.charAt(i) == s2.charAt(j);
                } else {
                    for (int k = 1; k < len; k++) {
                        if ((dp[i][j][k] && dp[i + k][j + k][len - k]) ||
                            (dp[i][j + len - k][k] && dp[i + k][j][len - k])) {
                            dp[i][j][len] = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    return dp[0][0][n];
}""",
                "time": "O(n⁴)",
                "space": "O(n³)"
            }]
        },
        "Range Sum Query - Mutable": {
            "solutions": [{
                "title": "Solution 1 – Binary Indexed Tree",
                "description": "Use Binary Indexed Tree for efficient range sum and updates.",
                "code": """class NumArray {
    private int[] tree;
    private int[] nums;
    private int n;
    
    public NumArray(int[] nums) {
        this.nums = nums;
        this.n = nums.length;
        this.tree = new int[n + 1];
        
        for (int i = 0; i < n; i++) {
            updateTree(i + 1, nums[i]);
        }
    }
    
    public void update(int index, int val) {
        int diff = val - nums[index];
        nums[index] = val;
        updateTree(index + 1, diff);
    }
    
    public int sumRange(int left, int right) {
        return query(right + 1) - query(left);
    }
    
    private void updateTree(int index, int val) {
        while (index <= n) {
            tree[index] += val;
            index += index & -index;
        }
    }
    
    private int query(int index) {
        int sum = 0;
        while (index > 0) {
            sum += tree[index];
            index -= index & -index;
        }
        return sum;
    }
}""",
                "time": "O(log n) for update and query",
                "space": "O(n)"
            }]
        },
        "Ugly Number": {
            "solutions": [{
                "title": "Solution 1 – Prime Factorization",
                "description": "Check if number has only 2, 3, 5 as prime factors.",
                "code": """public boolean isUgly(int n) {
    if (n <= 0) return false;
    
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    
    return n == 1;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Insert Interval": {
            "solutions": [{
                "title": "Solution 1 – Linear Scan",
                "description": "Insert new interval and merge overlapping intervals.",
                "code": """public int[][] insert(int[][] intervals, int[] newInterval) {
    List<int[]> result = new ArrayList<>();
    int i = 0;
    
    // Add intervals before newInterval
    while (i < intervals.length && intervals[i][1] < newInterval[0]) {
        result.add(intervals[i]);
        i++;
    }
    
    // Merge overlapping intervals
    while (i < intervals.length && intervals[i][0] <= newInterval[1]) {
        newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
        newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
        i++;
    }
    result.add(newInterval);
    
    // Add remaining intervals
    while (i < intervals.length) {
        result.add(intervals[i]);
        i++;
    }
    
    return result.toArray(new int[result.size()][]);
}""",
                "time": "O(n)",
                "space": "O(n)"
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
        solutions_data = get_remaining_solutions()
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
    """Update remaining problem files with actual solutions."""
    print("🚀 Starting Remaining Solutions Update Process")
    print("=" * 70)
    
    # List of problems to update
    problems_to_update = [
        ("shortest-palindrome.html", "Shortest Palindrome"),
        ("decode-ways.html", "Decode Ways"),
        ("integer-break.html", "Integer Break"),
        ("reverse-bits.html", "Reverse Bits"),
        ("linked-list-cycle.html", "Linked List Cycle"),
        ("largest-number.html", "Largest Number"),
        ("implement-strstr.html", "Implement strStr()"),
        ("remove-invalid-parentheses.html", "Remove Invalid Parentheses"),
        ("find-median-data-stream.html", "Find Median from Data Stream"),
        ("summary-ranges.html", "Summary Ranges"),
        ("group-anagrams.html", "Group Anagrams"),
        ("search-2d-matrix.html", "Search a 2D Matrix"),
        ("gray-code.html", "Gray Code"),
        ("binary-search-tree-iterator.html", "Binary Search Tree Iterator"),
        ("longest-consecutive-sequence.html", "Longest Consecutive Sequence"),
        ("reverse-vowels-string.html", "Reverse Vowels of a String"),
        ("find-minimum-rotated-sorted-array.html", "Find Minimum in Rotated Sorted Array"),
        ("one-edit-distance.html", "One Edit Distance"),
        ("longest-increasing-subsequence.html", "Longest Increasing Subsequence"),
        ("spiral-matrix-ii.html", "Spiral Matrix II"),
        ("surrounded-regions.html", "Surrounded Regions"),
        ("product-array-except-self.html", "Product of Array Except Self"),
        ("copy-list-random-pointer.html", "Copy List with Random Pointer"),
        ("valid-sudoku.html", "Valid Sudoku"),
        ("permutation-sequence.html", "Permutation Sequence"),
        ("graph-valid-tree.html", "Graph Valid Tree"),
        ("scramble-string.html", "Scramble String"),
        ("range-sum-query-mutable.html", "Range Sum Query - Mutable"),
        ("ugly-number.html", "Ugly Number"),
        ("insert-interval.html", "Insert Interval")
    ]
    
    updated_count = 0
    
    for filename, problem_name in problems_to_update:
        if update_problem_with_solution(filename, problem_name):
            updated_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"✅ Updated {updated_count} problem files with remaining solutions!")
    print("🌐 You can now view the complete solutions in the problem pages.")

if __name__ == "__main__":
    main() 