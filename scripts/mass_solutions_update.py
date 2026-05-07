#!/usr/bin/env python3
import os
import re

def get_mass_solutions():
    """Return solutions for many more coding interview problems."""
    return {
        "Count Primes": {
            "solutions": [{
                "title": "Solution 1 – Sieve of Eratosthenes",
                "description": "Use the Sieve of Eratosthenes to count prime numbers efficiently.",
                "code": """public int countPrimes(int n) {
    if (n <= 2) return 0;
    
    boolean[] isPrime = new boolean[n];
    Arrays.fill(isPrime, true);
    isPrime[0] = isPrime[1] = false;
    
    for (int i = 2; i * i < n; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j < n; j += i) {
                isPrime[j] = false;
            }
        }
    }
    
    int count = 0;
    for (int i = 2; i < n; i++) {
        if (isPrime[i]) count++;
    }
    
    return count;
}""",
                "time": "O(n log log n)",
                "space": "O(n)"
            }]
        },
        "Missing Number": {
            "solutions": [{
                "title": "Solution 1 – XOR",
                "description": "Use XOR to find the missing number.",
                "code": """public int missingNumber(int[] nums) {
    int result = nums.length;
    
    for (int i = 0; i < nums.length; i++) {
        result ^= i ^ nums[i];
    }
    
    return result;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Contains Duplicate": {
            "solutions": [{
                "title": "Solution 1 – HashSet",
                "description": "Use a HashSet to check for duplicates.",
                "code": """public boolean containsDuplicate(int[] nums) {
    Set<Integer> set = new HashSet<>();
    
    for (int num : nums) {
        if (set.contains(num)) {
            return true;
        }
        set.add(num);
    }
    
    return false;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Valid Anagram": {
            "solutions": [{
                "title": "Solution 1 – Character Count",
                "description": "Count characters in both strings and compare.",
                "code": """public boolean isAnagram(String s, String t) {
    if (s.length() != t.length()) return false;
    
    int[] count = new int[26];
    
    for (char c : s.toCharArray()) {
        count[c - 'a']++;
    }
    
    for (char c : t.toCharArray()) {
        count[c - 'a']--;
        if (count[c - 'a'] < 0) return false;
    }
    
    return true;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Palindrome Number": {
            "solutions": [{
                "title": "Solution 1 – Reverse Half",
                "description": "Reverse the second half of the number and compare.",
                "code": """public boolean isPalindrome(int x) {
    if (x < 0 || (x != 0 && x % 10 == 0)) return false;
    
    int reversed = 0;
    while (x > reversed) {
        reversed = reversed * 10 + x % 10;
        x /= 10;
    }
    
    return x == reversed || x == reversed / 10;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Reverse Integer": {
            "solutions": [{
                "title": "Solution 1 – Digit by Digit",
                "description": "Reverse digits one by one, checking for overflow.",
                "code": """public int reverse(int x) {
    int result = 0;
    
    while (x != 0) {
        int digit = x % 10;
        x /= 10;
        
        if (result > Integer.MAX_VALUE / 10 || 
            (result == Integer.MAX_VALUE / 10 && digit > 7)) return 0;
        if (result < Integer.MIN_VALUE / 10 || 
            (result == Integer.MIN_VALUE / 10 && digit < -8)) return 0;
        
        result = result * 10 + digit;
    }
    
    return result;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "String to Integer (atoi)": {
            "solutions": [{
                "title": "Solution 1 – Character by Character",
                "description": "Parse the string character by character, handling edge cases.",
                "code": """public int myAtoi(String s) {
    if (s == null || s.length() == 0) return 0;
    
    int i = 0;
    while (i < s.length() && s.charAt(i) == ' ') {
        i++;
    }
    
    if (i == s.length()) return 0;
    
    boolean negative = false;
    if (s.charAt(i) == '+' || s.charAt(i) == '-') {
        negative = s.charAt(i) == '-';
        i++;
    }
    
    long result = 0;
    while (i < s.length() && Character.isDigit(s.charAt(i))) {
        result = result * 10 + (s.charAt(i) - '0');
        
        if (negative && -result <= Integer.MIN_VALUE) return Integer.MIN_VALUE;
        if (!negative && result >= Integer.MAX_VALUE) return Integer.MAX_VALUE;
        
        i++;
    }
    
    return negative ? (int) -result : (int) result;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Plus One": {
            "solutions": [{
                "title": "Solution 1 – Carry Over",
                "description": "Add one and handle carry over from right to left.",
                "code": """public int[] plusOne(int[] digits) {
    int n = digits.length;
    
    for (int i = n - 1; i >= 0; i--) {
        if (digits[i] < 9) {
            digits[i]++;
            return digits;
        }
        digits[i] = 0;
    }
    
    int[] newDigits = new int[n + 1];
    newDigits[0] = 1;
    return newDigits;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Add Binary": {
            "solutions": [{
                "title": "Solution 1 – Bit by Bit",
                "description": "Add binary strings bit by bit with carry.",
                "code": """public String addBinary(String a, String b) {
    StringBuilder result = new StringBuilder();
    int carry = 0;
    int i = a.length() - 1;
    int j = b.length() - 1;
    
    while (i >= 0 || j >= 0 || carry > 0) {
        int sum = carry;
        
        if (i >= 0) {
            sum += a.charAt(i) - '0';
            i--;
        }
        
        if (j >= 0) {
            sum += b.charAt(j) - '0';
            j--;
        }
        
        result.insert(0, (char) (sum % 2 + '0'));
        carry = sum / 2;
    }
    
    return result.toString();
}""",
                "time": "O(max(m,n))",
                "space": "O(max(m,n))"
            }]
        },
        "Longest Common Prefix": {
            "solutions": [{
                "title": "Solution 1 – Horizontal Scanning",
                "description": "Compare strings horizontally to find common prefix.",
                "code": """public String longestCommonPrefix(String[] strs) {
    if (strs == null || strs.length == 0) return "";
    
    String prefix = strs[0];
    
    for (int i = 1; i < strs.length; i++) {
        while (strs[i].indexOf(prefix) != 0) {
            prefix = prefix.substring(0, prefix.length() - 1);
            if (prefix.isEmpty()) return "";
        }
    }
    
    return prefix;
}""",
                "time": "O(S)",
                "space": "O(1)"
            }]
        },
        "Remove Duplicates from Sorted Array": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to remove duplicates in-place.",
                "code": """public int removeDuplicates(int[] nums) {
    if (nums.length == 0) return 0;
    
    int i = 0;
    for (int j = 1; j < nums.length; j++) {
        if (nums[j] != nums[i]) {
            i++;
            nums[i] = nums[j];
        }
    }
    
    return i + 1;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Remove Element": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to remove elements in-place.",
                "code": """public int removeElement(int[] nums, int val) {
    int i = 0;
    
    for (int j = 0; j < nums.length; j++) {
        if (nums[j] != val) {
            nums[i] = nums[j];
            i++;
        }
    }
    
    return i;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Move Zeroes": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Move non-zero elements to the front, then fill with zeros.",
                "code": """public void moveZeroes(int[] nums) {
    int nonZeroIndex = 0;
    
    for (int i = 0; i < nums.length; i++) {
        if (nums[i] != 0) {
            nums[nonZeroIndex] = nums[i];
            nonZeroIndex++;
        }
    }
    
    for (int i = nonZeroIndex; i < nums.length; i++) {
        nums[i] = 0;
    }
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Majority Element": {
            "solutions": [{
                "title": "Solution 1 – Boyer-Moore Voting",
                "description": "Use Boyer-Moore voting algorithm to find majority element.",
                "code": """public int majorityElement(int[] nums) {
    int count = 0;
    int candidate = 0;
    
    for (int num : nums) {
        if (count == 0) {
            candidate = num;
        }
        count += (num == candidate) ? 1 : -1;
    }
    
    return candidate;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Search Insert Position": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Use binary search to find insertion position.",
                "code": """public int searchInsert(int[] nums, int target) {
    int left = 0;
    int right = nums.length - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) {
            return mid;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return left;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "First Bad Version": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Use binary search to find the first bad version.",
                "code": """public int firstBadVersion(int n) {
    int left = 1;
    int right = n;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (isBadVersion(mid)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Closest Binary Search Tree Value": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Use binary search to find the closest value in BST.",
                "code": """public int closestValue(TreeNode root, double target) {
    int closest = root.val;
    
    while (root != null) {
        if (Math.abs(root.val - target) < Math.abs(closest - target)) {
            closest = root.val;
        }
        
        if (target < root.val) {
            root = root.left;
        } else {
            root = root.right;
        }
    }
    
    return closest;
}""",
                "time": "O(h)",
                "space": "O(1)"
            }]
        },
        "Kth Smallest Element in BST": {
            "solutions": [{
                "title": "Solution 1 – Inorder Traversal",
                "description": "Use inorder traversal to find kth smallest element.",
                "code": """private int count = 0;
private int result = 0;

public int kthSmallest(TreeNode root, int k) {
    count = 0;
    inorder(root, k);
    return result;
}

private void inorder(TreeNode root, int k) {
    if (root == null) return;
    
    inorder(root.left, k);
    
    count++;
    if (count == k) {
        result = root.val;
        return;
    }
    
    inorder(root.right, k);
}""",
                "time": "O(h + k)",
                "space": "O(h)"
            }]
        },
        "Same Tree": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Recursively compare two binary trees.",
                "code": """public boolean isSameTree(TreeNode p, TreeNode q) {
    if (p == null && q == null) return true;
    if (p == null || q == null) return false;
    
    return p.val == q.val && 
           isSameTree(p.left, q.left) && 
           isSameTree(p.right, q.right);
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Balanced Binary Tree": {
            "solutions": [{
                "title": "Solution 1 – Recursive with Height",
                "description": "Check if tree is balanced by checking height difference.",
                "code": """public boolean isBalanced(TreeNode root) {
    return getHeight(root) != -1;
}

private int getHeight(TreeNode root) {
    if (root == null) return 0;
    
    int leftHeight = getHeight(root.left);
    if (leftHeight == -1) return -1;
    
    int rightHeight = getHeight(root.right);
    if (rightHeight == -1) return -1;
    
    if (Math.abs(leftHeight - rightHeight) > 1) return -1;
    
    return Math.max(leftHeight, rightHeight) + 1;
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Minimum Depth of Binary Tree": {
            "solutions": [{
                "title": "Solution 1 – BFS",
                "description": "Use BFS to find the minimum depth.",
                "code": """public int minDepth(TreeNode root) {
    if (root == null) return 0;
    
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    int depth = 1;
    
    while (!queue.isEmpty()) {
        int size = queue.size();
        
        for (int i = 0; i < size; i++) {
            TreeNode current = queue.poll();
            
            if (current.left == null && current.right == null) {
                return depth;
            }
            
            if (current.left != null) {
                queue.offer(current.left);
            }
            if (current.right != null) {
                queue.offer(current.right);
            }
        }
        
        depth++;
    }
    
    return depth;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Convert Sorted Array to Binary Search Tree": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Use the middle element as root and recursively build left and right subtrees.",
                "code": """public TreeNode sortedArrayToBST(int[] nums) {
    return buildBST(nums, 0, nums.length - 1);
}

private TreeNode buildBST(int[] nums, int left, int right) {
    if (left > right) return null;
    
    int mid = left + (right - left) / 2;
    TreeNode root = new TreeNode(nums[mid]);
    
    root.left = buildBST(nums, left, mid - 1);
    root.right = buildBST(nums, mid + 1, right);
    
    return root;
}""",
                "time": "O(n)",
                "space": "O(log n)"
            }]
        },
        "Convert Sorted List to Binary Search Tree": {
            "solutions": [{
                "title": "Solution 1 – Inorder Simulation",
                "description": "Simulate inorder traversal to build the BST.",
                "code": """private ListNode head;

public TreeNode sortedListToBST(ListNode head) {
    this.head = head;
    int size = getSize(head);
    return buildBST(0, size - 1);
}

private int getSize(ListNode head) {
    int size = 0;
    while (head != null) {
        size++;
        head = head.next;
    }
    return size;
}

private TreeNode buildBST(int left, int right) {
    if (left > right) return null;
    
    int mid = left + (right - left) / 2;
    
    TreeNode leftChild = buildBST(left, mid - 1);
    TreeNode root = new TreeNode(head.val);
    head = head.next;
    TreeNode rightChild = buildBST(mid + 1, right);
    
    root.left = leftChild;
    root.right = rightChild;
    
    return root;
}""",
                "time": "O(n)",
                "space": "O(log n)"
            }]
        },
        "Binary Tree Paths": {
            "solutions": [{
                "title": "Solution 1 – DFS",
                "description": "Use DFS to find all root-to-leaf paths.",
                "code": """public List<String> binaryTreePaths(TreeNode root) {
    List<String> result = new ArrayList<>();
    if (root == null) return result;
    
    dfs(root, "", result);
    return result;
}

private void dfs(TreeNode root, String path, List<String> result) {
    if (root.left == null && root.right == null) {
        result.add(path + root.val);
        return;
    }
    
    if (root.left != null) {
        dfs(root.left, path + root.val + "->", result);
    }
    
    if (root.right != null) {
        dfs(root.right, path + root.val + "->", result);
    }
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Sum Root to Leaf Numbers": {
            "solutions": [{
                "title": "Solution 1 – DFS",
                "description": "Use DFS to calculate sum of all root-to-leaf numbers.",
                "code": """public int sumNumbers(TreeNode root) {
    return dfs(root, 0);
}

private int dfs(TreeNode root, int sum) {
    if (root == null) return 0;
    
    sum = sum * 10 + root.val;
    
    if (root.left == null && root.right == null) {
        return sum;
    }
    
    return dfs(root.left, sum) + dfs(root.right, sum);
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Path Sum II": {
            "solutions": [{
                "title": "Solution 1 – DFS with Backtracking",
                "description": "Use DFS with backtracking to find all paths with given sum.",
                "code": """public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
    List<List<Integer>> result = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    
    dfs(root, targetSum, path, result);
    return result;
}

private void dfs(TreeNode root, int targetSum, List<Integer> path, List<List<Integer>> result) {
    if (root == null) return;
    
    path.add(root.val);
    
    if (root.left == null && root.right == null && targetSum == root.val) {
        result.add(new ArrayList<>(path));
    }
    
    dfs(root.left, targetSum - root.val, path, result);
    dfs(root.right, targetSum - root.val, path, result);
    
    path.remove(path.size() - 1);
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Find Leaves of Binary Tree": {
            "solutions": [{
                "title": "Solution 1 – DFS with Height",
                "description": "Use DFS to find leaves based on their height from bottom.",
                "code": """public List<List<Integer>> findLeaves(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    getHeight(root, result);
    return result;
}

private int getHeight(TreeNode root, List<List<Integer>> result) {
    if (root == null) return -1;
    
    int leftHeight = getHeight(root.left, result);
    int rightHeight = getHeight(root.right, result);
    
    int currentHeight = Math.max(leftHeight, rightHeight) + 1;
    
    if (currentHeight >= result.size()) {
        result.add(new ArrayList<>());
    }
    
    result.get(currentHeight).add(root.val);
    
    return currentHeight;
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Binary Tree Right Side View": {
            "solutions": [{
                "title": "Solution 1 – BFS",
                "description": "Use BFS to get the rightmost node at each level.",
                "code": """public List<Integer> rightSideView(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    if (root == null) return result;
    
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    
    while (!queue.isEmpty()) {
        int size = queue.size();
        
        for (int i = 0; i < size; i++) {
            TreeNode current = queue.poll();
            
            if (i == size - 1) {
                result.add(current.val);
            }
            
            if (current.left != null) {
                queue.offer(current.left);
            }
            if (current.right != null) {
                queue.offer(current.right);
            }
        }
    }
    
    return result;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Binary Tree Maximum Path Sum": {
            "solutions": [{
                "title": "Solution 1 – DFS",
                "description": "Use DFS to find the maximum path sum.",
                "code": """private int maxSum = Integer.MIN_VALUE;

public int maxPathSum(TreeNode root) {
    maxSum = Integer.MIN_VALUE;
    dfs(root);
    return maxSum;
}

private int dfs(TreeNode root) {
    if (root == null) return 0;
    
    int leftMax = Math.max(0, dfs(root.left));
    int rightMax = Math.max(0, dfs(root.right));
    
    maxSum = Math.max(maxSum, root.val + leftMax + rightMax);
    
    return root.val + Math.max(leftMax, rightMax);
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Lowest Common Ancestor of a Binary Tree": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Use recursive approach to find LCA.",
                "code": """public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) return root;
    
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    
    if (left != null && right != null) return root;
    
    return left != null ? left : right;
}""",
                "time": "O(n)",
                "space": "O(h)"
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
        "Implement Stack using Arrays": {
            "solutions": [{
                "title": "Solution 1 – Array Implementation",
                "description": "Implement stack using array with dynamic resizing.",
                "code": """class Stack {
    private int[] arr;
    private int top;
    private int capacity;
    
    public Stack(int size) {
        arr = new int[size];
        capacity = size;
        top = -1;
    }
    
    public void push(int x) {
        if (isFull()) {
            int[] newArr = new int[capacity * 2];
            System.arraycopy(arr, 0, newArr, 0, capacity);
            arr = newArr;
            capacity *= 2;
        }
        arr[++top] = x;
    }
    
    public int pop() {
        if (isEmpty()) {
            throw new IllegalStateException("Stack is empty");
        }
        return arr[top--];
    }
    
    public int peek() {
        if (isEmpty()) {
            throw new IllegalStateException("Stack is empty");
        }
        return arr[top];
    }
    
    public boolean isEmpty() {
        return top == -1;
    }
    
    public boolean isFull() {
        return top == capacity - 1;
    }
    
    public int size() {
        return top + 1;
    }
}""",
                "time": "O(1) amortized",
                "space": "O(n)"
            }]
        },
        "Implement Queue using Arrays": {
            "solutions": [{
                "title": "Solution 1 – Circular Array Implementation",
                "description": "Implement queue using circular array.",
                "code": """class Queue {
    private int[] arr;
    private int front;
    private int rear;
    private int size;
    private int capacity;
    
    public Queue(int size) {
        arr = new int[size];
        capacity = size;
        front = 0;
        rear = -1;
        this.size = 0;
    }
    
    public void enqueue(int x) {
        if (isFull()) {
            throw new IllegalStateException("Queue is full");
        }
        rear = (rear + 1) % capacity;
        arr[rear] = x;
        size++;
    }
    
    public int dequeue() {
        if (isEmpty()) {
            throw new IllegalStateException("Queue is empty");
        }
        int x = arr[front];
        front = (front + 1) % capacity;
        size--;
        return x;
    }
    
    public int peek() {
        if (isEmpty()) {
            throw new IllegalStateException("Queue is empty");
        }
        return arr[front];
    }
    
    public boolean isEmpty() {
        return size == 0;
    }
    
    public boolean isFull() {
        return size == capacity;
    }
    
    public int size() {
        return size;
    }
}""",
                "time": "O(1)",
                "space": "O(n)"
            }]
        },
        "Min Stack": {
            "solutions": [{
                "title": "Solution 1 – Two Stacks",
                "description": "Use two stacks to maintain minimum element.",
                "code": """class MinStack {
    private Stack<Integer> stack;
    private Stack<Integer> minStack;
    
    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }
    
    public void push(int val) {
        stack.push(val);
        
        if (minStack.isEmpty() || val <= minStack.peek()) {
            minStack.push(val);
        }
    }
    
    public void pop() {
        if (stack.peek().equals(minStack.peek())) {
            minStack.pop();
        }
        stack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int getMin() {
        return minStack.peek();
    }
}""",
                "time": "O(1)",
                "space": "O(n)"
            }]
        },
        "LRU Cache": {
            "solutions": [{
                "title": "Solution 1 – HashMap + Doubly Linked List",
                "description": "Use HashMap and doubly linked list to implement LRU cache.",
                "code": """class LRUCache {
    private Map<Integer, Node> cache;
    private Node head, tail;
    private int capacity;
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        cache = new HashMap<>();
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }
    
    public int get(int key) {
        if (cache.containsKey(key)) {
            Node node = cache.get(key);
            moveToHead(node);
            return node.value;
        }
        return -1;
    }
    
    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            Node node = cache.get(key);
            node.value = value;
            moveToHead(node);
        } else {
            Node newNode = new Node(key, value);
            cache.put(key, newNode);
            addNode(newNode);
            
            if (cache.size() > capacity) {
                Node lru = removeTail();
                cache.remove(lru.key);
            }
        }
    }
    
    private void addNode(Node node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }
    
    private void removeNode(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    private void moveToHead(Node node) {
        removeNode(node);
        addNode(node);
    }
    
    private Node removeTail() {
        Node res = tail.prev;
        removeNode(res);
        return res;
    }
    
    class Node {
        int key, value;
        Node prev, next;
        
        Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }
}""",
                "time": "O(1)",
                "space": "O(capacity)"
            }]
        },
        "Counting Sort": {
            "solutions": [{
                "title": "Solution 1 – Counting Sort Implementation",
                "description": "Implement counting sort algorithm for integers.",
                "code": """public void countingSort(int[] arr) {
    int n = arr.length;
    if (n <= 1) return;
    
    // Find the maximum element
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    
    // Create count array
    int[] count = new int[max + 1];
    
    // Count occurrences
    for (int i = 0; i < n; i++) {
        count[arr[i]]++;
    }
    
    // Modify count array to store actual positions
    for (int i = 1; i <= max; i++) {
        count[i] += count[i - 1];
    }
    
    // Create output array
    int[] output = new int[n];
    
    // Build output array
    for (int i = n - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }
    
    // Copy output array back to original array
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}""",
                "time": "O(n + k)",
                "space": "O(n + k)"
            }]
        },
        "Bucket Sort": {
            "solutions": [{
                "title": "Solution 1 – Bucket Sort Implementation",
                "description": "Implement bucket sort algorithm.",
                "code": """public void bucketSort(double[] arr) {
    int n = arr.length;
    if (n <= 1) return;
    
    // Create buckets
    List<Double>[] buckets = new List[n];
    for (int i = 0; i < n; i++) {
        buckets[i] = new ArrayList<>();
    }
    
    // Distribute elements into buckets
    for (int i = 0; i < n; i++) {
        int bucketIndex = (int) (n * arr[i]);
        buckets[bucketIndex].add(arr[i]);
    }
    
    // Sort individual buckets
    for (int i = 0; i < n; i++) {
        Collections.sort(buckets[i]);
    }
    
    // Concatenate buckets
    int index = 0;
    for (int i = 0; i < n; i++) {
        for (double value : buckets[i]) {
            arr[index++] = value;
        }
    }
}""",
                "time": "O(n + k)",
                "space": "O(n)"
            }]
        },
        "Radix Sort": {
            "solutions": [{
                "title": "Solution 1 – Radix Sort Implementation",
                "description": "Implement radix sort algorithm for integers.",
                "code": """public void radixSort(int[] arr) {
    int n = arr.length;
    if (n <= 1) return;
    
    // Find the maximum element
    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    
    // Do counting sort for every digit
    for (int exp = 1; max / exp > 0; exp *= 10) {
        countingSortByDigit(arr, exp);
    }
}

private void countingSortByDigit(int[] arr, int exp) {
    int n = arr.length;
    int[] output = new int[n];
    int[] count = new int[10];
    
    // Count occurrences
    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }
    
    // Modify count array to store actual positions
    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }
    
    // Build output array
    for (int i = n - 1; i >= 0; i--) {
        int digit = (arr[i] / exp) % 10;
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }
    
    // Copy output array back to original array
    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}""",
                "time": "O(d(n + k))",
                "space": "O(n + k)"
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
        solutions_data = get_mass_solutions()
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
    """Update many more problem files with actual solutions."""
    print("🚀 Starting Mass Solutions Update Process")
    print("=" * 70)
    
    # List of problems to update
    problems_to_update = [
        ("count-primes.html", "Count Primes"),
        ("missing-number.html", "Missing Number"),
        ("contains-duplicate.html", "Contains Duplicate"),
        ("valid-anagram.html", "Valid Anagram"),
        ("palindrome-number.html", "Palindrome Number"),
        ("reverse-integer.html", "Reverse Integer"),
        ("string-to-integer.html", "String to Integer (atoi)"),
        ("plus-one.html", "Plus One"),
        ("add-binary.html", "Add Binary"),
        ("longest-common-prefix.html", "Longest Common Prefix"),
        ("remove-duplicates-sorted-array.html", "Remove Duplicates from Sorted Array"),
        ("remove-element.html", "Remove Element"),
        ("move-zeroes.html", "Move Zeroes"),
        ("majority-element.html", "Majority Element"),
        ("search-insert-position.html", "Search Insert Position"),
        ("first-bad-version.html", "First Bad Version"),
        ("closest-binary-search-tree-value.html", "Closest Binary Search Tree Value"),
        ("kth-smallest-bst.html", "Kth Smallest Element in BST"),
        ("same-tree.html", "Same Tree"),
        ("balanced-binary-tree.html", "Balanced Binary Tree"),
        ("minimum-depth-binary-tree.html", "Minimum Depth of Binary Tree"),
        ("convert-sorted-array-bst.html", "Convert Sorted Array to Binary Search Tree"),
        ("convert-sorted-list-bst.html", "Convert Sorted List to Binary Search Tree"),
        ("binary-tree-paths.html", "Binary Tree Paths"),
        ("sum-root-leaf-numbers.html", "Sum Root to Leaf Numbers"),
        ("path-sum-ii.html", "Path Sum II"),
        ("find-leaves-binary-tree.html", "Find Leaves of Binary Tree"),
        ("binary-tree-right-side-view.html", "Binary Tree Right Side View"),
        ("binary-tree-maximum-path-sum.html", "Binary Tree Maximum Path Sum"),
        ("lowest-common-ancestor-binary-tree.html", "Lowest Common Ancestor of a Binary Tree"),
        ("serialize-deserialize-binary-tree.html", "Serialize and Deserialize Binary Tree"),
        ("implement-stack-array.html", "Implement Stack using Arrays"),
        ("implement-queue-array.html", "Implement Queue using Arrays"),
        ("min-stack.html", "Min Stack"),
        ("lru-cache.html", "LRU Cache"),
        ("counting-sort.html", "Counting Sort"),
        ("bucket-sort.html", "Bucket Sort"),
        ("radix-sort.html", "Radix Sort")
    ]
    
    updated_count = 0
    
    for filename, problem_name in problems_to_update:
        if update_problem_with_solution(filename, problem_name):
            updated_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"✅ Updated {updated_count} problem files with mass solutions!")
    print("🌐 You can now view the complete solutions in the problem pages.")

if __name__ == "__main__":
    main() 