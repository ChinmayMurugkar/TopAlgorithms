#!/usr/bin/env python3
import os
import re

def get_more_solutions():
    """Return solutions for more coding interview problems."""
    return {
        "Longest Substring Without Repeating Characters": {
            "solutions": [{
                "title": "Solution 1 – Sliding Window",
                "description": "Use a sliding window with a hash set to track unique characters.",
                "code": """public int lengthOfLongestSubstring(String s) {
    Set<Character> set = new HashSet<>();
    int left = 0, maxLength = 0;
    
    for (int right = 0; right < s.length(); right++) {
        while (set.contains(s.charAt(right))) {
            set.remove(s.charAt(left));
            left++;
        }
        
        set.add(s.charAt(right));
        maxLength = Math.max(maxLength, right - left + 1);
    }
    
    return maxLength;
}""",
                "time": "O(n)",
                "space": "O(min(m,n))"
            }]
        },
        "Container With Most Water": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to find the maximum area.",
                "code": """public int maxArea(int[] height) {
    int left = 0, right = height.length - 1;
    int maxArea = 0;
    
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
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Remove Nth Node From End of List": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use fast and slow pointers to find the nth node from end.",
                "code": """public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    
    ListNode fast = dummy;
    ListNode slow = dummy;
    
    // Move fast pointer n+1 steps ahead
    for (int i = 0; i <= n; i++) {
        fast = fast.next;
    }
    
    // Move both pointers until fast reaches end
    while (fast != null) {
        fast = fast.next;
        slow = slow.next;
    }
    
    // Remove the nth node
    slow.next = slow.next.next;
    
    return dummy.next;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Palindrome Linked List": {
            "solutions": [{
                "title": "Solution 1 – Reverse Second Half",
                "description": "Find the middle, reverse the second half, then compare.",
                "code": """public boolean isPalindrome(ListNode head) {
    if (head == null || head.next == null) return true;
    
    // Find the middle
    ListNode slow = head, fast = head;
    while (fast.next != null && fast.next.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    
    // Reverse the second half
    ListNode second = reverseList(slow.next);
    ListNode first = head;
    
    // Compare the two halves
    while (second != null) {
        if (first.val != second.val) return false;
        first = first.next;
        second = second.next;
    }
    
    return true;
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
        "Invert Binary Tree": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Recursively swap left and right children.",
                "code": """public TreeNode invertTree(TreeNode root) {
    if (root == null) return null;
    
    // Swap left and right children
    TreeNode temp = root.left;
    root.left = root.right;
    root.right = temp;
    
    // Recursively invert subtrees
    invertTree(root.left);
    invertTree(root.right);
    
    return root;
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Maximum Depth of Binary Tree": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Recursively find the maximum depth of left and right subtrees.",
                "code": """public int maxDepth(TreeNode root) {
    if (root == null) return 0;
    
    int leftDepth = maxDepth(root.left);
    int rightDepth = maxDepth(root.right);
    
    return Math.max(leftDepth, rightDepth) + 1;
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Symmetric Tree": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Check if left and right subtrees are mirror images.",
                "code": """public boolean isSymmetric(TreeNode root) {
    if (root == null) return true;
    return isMirror(root.left, root.right);
}

private boolean isMirror(TreeNode left, TreeNode right) {
    if (left == null && right == null) return true;
    if (left == null || right == null) return false;
    
    return (left.val == right.val) &&
           isMirror(left.left, right.right) &&
           isMirror(left.right, right.left);
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Validate Binary Search Tree": {
            "solutions": [{
                "title": "Solution 1 – Inorder Traversal",
                "description": "Use inorder traversal to check if values are in ascending order.",
                "code": """private TreeNode prev = null;

public boolean isValidBST(TreeNode root) {
    return inorder(root);
}

private boolean inorder(TreeNode root) {
    if (root == null) return true;
    
    if (!inorder(root.left)) return false;
    
    if (prev != null && root.val <= prev.val) return false;
    prev = root;
    
    return inorder(root.right);
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Path Sum": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Recursively check if there's a path from root to leaf with given sum.",
                "code": """public boolean hasPathSum(TreeNode root, int targetSum) {
    if (root == null) return false;
    
    if (root.left == null && root.right == null) {
        return targetSum == root.val;
    }
    
    return hasPathSum(root.left, targetSum - root.val) ||
           hasPathSum(root.right, targetSum - root.val);
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Lowest Common Ancestor of a Binary Search Tree": {
            "solutions": [{
                "title": "Solution 1 – Iterative",
                "description": "Use the BST property to find the LCA.",
                "code": """public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    while (root != null) {
        if (p.val < root.val && q.val < root.val) {
            root = root.left;
        } else if (p.val > root.val && q.val > root.val) {
            root = root.right;
        } else {
            return root;
        }
    }
    return null;
}""",
                "time": "O(h)",
                "space": "O(1)"
            }]
        },
        "Number of Islands": {
            "solutions": [{
                "title": "Solution 1 – DFS",
                "description": "Use DFS to mark connected land cells as visited.",
                "code": """public int numIslands(char[][] grid) {
    if (grid == null || grid.length == 0) return 0;
    
    int numIslands = 0;
    int rows = grid.length;
    int cols = grid[0].length;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (grid[i][j] == '1') {
                numIslands++;
                dfs(grid, i, j);
            }
        }
    }
    
    return numIslands;
}

private void dfs(char[][] grid, int i, int j) {
    int rows = grid.length;
    int cols = grid[0].length;
    
    if (i < 0 || i >= rows || j < 0 || j >= cols || grid[i][j] == '0') {
        return;
    }
    
    grid[i][j] = '0'; // Mark as visited
    
    dfs(grid, i + 1, j);
    dfs(grid, i - 1, j);
    dfs(grid, i, j + 1);
    dfs(grid, i, j - 1);
}""",
                "time": "O(m×n)",
                "space": "O(m×n)"
            }]
        },
        "Word Search": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to search for the word in all directions.",
                "code": """public boolean exist(char[][] board, String word) {
    int rows = board.length;
    int cols = board[0].length;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (dfs(board, word, i, j, 0)) {
                return true;
            }
        }
    }
    
    return false;
}

private boolean dfs(char[][] board, String word, int i, int j, int index) {
    if (index == word.length()) return true;
    
    if (i < 0 || i >= board.length || j < 0 || j >= board[0].length ||
        board[i][j] != word.charAt(index)) {
        return false;
    }
    
    char temp = board[i][j];
    board[i][j] = '#'; // Mark as visited
    
    boolean result = dfs(board, word, i + 1, j, index + 1) ||
                    dfs(board, word, i - 1, j, index + 1) ||
                    dfs(board, word, i, j + 1, index + 1) ||
                    dfs(board, word, i, j - 1, index + 1);
    
    board[i][j] = temp; // Restore
    return result;
}""",
                "time": "O(m×n×4^L)",
                "space": "O(L)"
            }]
        },
        "Course Schedule": {
            "solutions": [{
                "title": "Solution 1 – Topological Sort (DFS)",
                "description": "Use DFS to detect cycles in the course dependency graph.",
                "code": """public boolean canFinish(int numCourses, int[][] prerequisites) {
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

private boolean hasCycle(List<List<Integer>> graph, int node, boolean[] visited, boolean[] recStack) {
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
}""",
                "time": "O(V + E)",
                "space": "O(V + E)"
            }]
        },
        "Implement Trie (Prefix Tree)": {
            "solutions": [{
                "title": "Solution 1 – Trie Implementation",
                "description": "Implement a Trie data structure with insert, search, and startsWith operations.",
                "code": """class TrieNode {
    private TrieNode[] children;
    private boolean isEndOfWord;
    
    public TrieNode() {
        children = new TrieNode[26];
        isEndOfWord = false;
    }
}

class Trie {
    private TrieNode root;
    
    public Trie() {
        root = new TrieNode();
    }
    
    public void insert(String word) {
        TrieNode current = root;
        for (char c : word.toCharArray()) {
            int index = c - 'a';
            if (current.children[index] == null) {
                current.children[index] = new TrieNode();
            }
            current = current.children[index];
        }
        current.isEndOfWord = true;
    }
    
    public boolean search(String word) {
        TrieNode node = searchPrefix(word);
        return node != null && node.isEndOfWord;
    }
    
    public boolean startsWith(String prefix) {
        return searchPrefix(prefix) != null;
    }
    
    private TrieNode searchPrefix(String prefix) {
        TrieNode current = root;
        for (char c : prefix.toCharArray()) {
            int index = c - 'a';
            if (current.children[index] == null) {
                return null;
            }
            current = current.children[index];
        }
        return current;
    }
}""",
                "time": "O(m) for insert/search/startsWith",
                "space": "O(ALPHABET_SIZE × m × n)"
            }]
        },
        "Coin Change": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to find the minimum number of coins needed.",
                "code": """public int coinChange(int[] coins, int amount) {
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
}""",
                "time": "O(amount × number of coins)",
                "space": "O(amount)"
            }]
        },
        "House Robber": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to find the maximum amount that can be robbed.",
                "code": """public int rob(int[] nums) {
    if (nums == null || nums.length == 0) return 0;
    if (nums.length == 1) return nums[0];
    
    int[] dp = new int[nums.length];
    dp[0] = nums[0];
    dp[1] = Math.max(nums[0], nums[1]);
    
    for (int i = 2; i < nums.length; i++) {
        dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i]);
    }
    
    return dp[nums.length - 1];
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Edit Distance": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to find the minimum number of operations to convert one string to another.",
                "code": """public int minDistance(String word1, String word2) {
    int m = word1.length();
    int n = word2.length();
    
    int[][] dp = new int[m + 1][n + 1];
    
    for (int i = 0; i <= m; i++) {
        dp[i][0] = i;
    }
    
    for (int j = 0; j <= n; j++) {
        dp[0][j] = j;
    }
    
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
}""",
                "time": "O(m×n)",
                "space": "O(m×n)"
            }]
        },
        "Longest Common Subsequence": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to find the longest common subsequence between two strings.",
                "code": """public int longestCommonSubsequence(String text1, String text2) {
    int m = text1.length();
    int n = text2.length();
    
    int[][] dp = new int[m + 1][n + 1];
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    
    return dp[m][n];
}""",
                "time": "O(m×n)",
                "space": "O(m×n)"
            }]
        },
        "Bubble Sort": {
            "solutions": [{
                "title": "Solution 1 – Bubble Sort Implementation",
                "description": "Implement bubble sort algorithm to sort an array.",
                "code": """public void bubbleSort(int[] arr) {
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
        if (!swapped) break;
    }
}""",
                "time": "O(n²)",
                "space": "O(1)"
            }]
        },
        "Quick Sort": {
            "solutions": [{
                "title": "Solution 1 – Quick Sort Implementation",
                "description": "Implement quick sort algorithm using divide and conquer.",
                "code": """public void quickSort(int[] arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

private int partition(int[] arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr, i, j);
        }
    }
    
    swap(arr, i + 1, high);
    return i + 1;
}

private void swap(int[] arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}""",
                "time": "O(n log n) average, O(n²) worst case",
                "space": "O(log n)"
            }]
        },
        "Merge Sort": {
            "solutions": [{
                "title": "Solution 1 – Merge Sort Implementation",
                "description": "Implement merge sort algorithm using divide and conquer.",
                "code": """public void mergeSort(int[] arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        
        merge(arr, left, mid, right);
    }
}

private void merge(int[] arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    int[] leftArr = new int[n1];
    int[] rightArr = new int[n2];
    
    for (int i = 0; i < n1; i++) {
        leftArr[i] = arr[left + i];
    }
    for (int j = 0; j < n2; j++) {
        rightArr[j] = arr[mid + 1 + j];
    }
    
    int i = 0, j = 0, k = left;
    
    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }
    
    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
}""",
                "time": "O(n log n)",
                "space": "O(n)"
            }]
        },
        "Insertion Sort": {
            "solutions": [{
                "title": "Solution 1 – Insertion Sort Implementation",
                "description": "Implement insertion sort algorithm to sort an array.",
                "code": """public void insertionSort(int[] arr) {
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
}""",
                "time": "O(n²)",
                "space": "O(1)"
            }]
        },
        "Selection Sort": {
            "solutions": [{
                "title": "Solution 1 – Selection Sort Implementation",
                "description": "Implement selection sort algorithm to sort an array.",
                "code": """public void selectionSort(int[] arr) {
    int n = arr.length;
    
    for (int i = 0; i < n - 1; i++) {
        // Find the minimum element in unsorted array
        int minIndex = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }
        
        // Swap the found minimum element with the first element
        int temp = arr[minIndex];
        arr[minIndex] = arr[i];
        arr[i] = temp;
    }
}""",
                "time": "O(n²)",
                "space": "O(1)"
            }]
        },
        "Heap Sort": {
            "solutions": [{
                "title": "Solution 1 – Heap Sort Implementation",
                "description": "Implement heap sort algorithm using heap data structure.",
                "code": """public void heapSort(int[] arr) {
    int n = arr.length;
    
    // Build heap (rearrange array)
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }
    
    // One by one extract an element from heap
    for (int i = n - 1; i > 0; i--) {
        // Move current root to end
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        
        // call max heapify on the reduced heap
        heapify(arr, i, 0);
    }
}

private void heapify(int[] arr, int n, int i) {
    int largest = i; // Initialize largest as root
    int left = 2 * i + 1; // left = 2*i + 1
    int right = 2 * i + 2; // right = 2*i + 2
    
    // If left child is larger than root
    if (left < n && arr[left] > arr[largest]) {
        largest = left;
    }
    
    // If right child is larger than largest so far
    if (right < n && arr[right] > arr[largest]) {
        largest = right;
    }
    
    // If largest is not root
    if (largest != i) {
        int swap = arr[i];
        arr[i] = arr[largest];
        arr[largest] = swap;
        
        // Recursively heapify the affected sub-tree
        heapify(arr, n, largest);
    }
}""",
                "time": "O(n log n)",
                "space": "O(1)"
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
        solutions_data = get_more_solutions()
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
    """Update all problem files with more solutions."""
    print("🚀 Starting More Solutions Update Process")
    print("=" * 70)
    
    # List of problems to update
    problems_to_update = [
        ("longest-substring-without-repeating-characters.html", "Longest Substring Without Repeating Characters"),
        ("container-with-most-water.html", "Container With Most Water"),
        ("remove-nth-node.html", "Remove Nth Node From End of List"),
        ("palindrome-linked-list.html", "Palindrome Linked List"),
        ("invert-binary-tree.html", "Invert Binary Tree"),
        ("maximum-depth-binary-tree.html", "Maximum Depth of Binary Tree"),
        ("symmetric-tree.html", "Symmetric Tree"),
        ("validate-binary-search-tree.html", "Validate Binary Search Tree"),
        ("path-sum.html", "Path Sum"),
        ("lowest-common-ancestor.html", "Lowest Common Ancestor of a Binary Search Tree"),
        ("number-of-islands.html", "Number of Islands"),
        ("word-search.html", "Word Search"),
        ("course-schedule.html", "Course Schedule"),
        ("implement-trie.html", "Implement Trie (Prefix Tree)"),
        ("coin-change.html", "Coin Change"),
        ("house-robber.html", "House Robber"),
        ("edit-distance.html", "Edit Distance"),
        ("longest-common-subsequence.html", "Longest Common Subsequence"),
        ("bubble-sort.html", "Bubble Sort"),
        ("quick-sort.html", "Quick Sort"),
        ("merge-sort.html", "Merge Sort"),
        ("insertion-sort.html", "Insertion Sort"),
        ("selection-sort.html", "Selection Sort"),
        ("heap-sort.html", "Heap Sort")
    ]
    
    updated_count = 0
    
    for filename, problem_name in problems_to_update:
        if update_problem_with_solution(filename, problem_name):
            updated_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"✅ Updated {updated_count} problem files with more solutions!")
    print("🌐 You can now view the complete solutions in the problem pages.")

if __name__ == "__main__":
    main() 