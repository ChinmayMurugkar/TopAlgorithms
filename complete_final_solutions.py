#!/usr/bin/env python3
import os
import re

def get_final_solutions():
    """Get solutions for another batch of problems."""
    return {
        "Minimum Platforms Required": {
            "solutions": [{
                "title": "Solution 1 – Sorting and Two Pointers",
                "description": "Sort arrival and departure times, then use two pointers to track platforms needed.",
                "code": """public int findPlatform(int arr[], int dep[], int n) {
    Arrays.sort(arr);
    Arrays.sort(dep);
    
    int platforms = 1, result = 1;
    int i = 1, j = 0;
    
    while (i < n && j < n) {
        if (arr[i] <= dep[j]) {
            platforms++;
            i++;
        } else {
            platforms--;
            j++;
        }
        result = Math.max(result, platforms);
    }
    
    return result;
}""",
                "time": "O(n log n)",
                "space": "O(1)"
            }]
        },
        "Find K Pairs with Smallest Sums": {
            "solutions": [{
                "title": "Solution 1 – Min Heap",
                "description": "Use min heap to find k pairs with smallest sums.",
                "code": """public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
    List<List<Integer>> result = new ArrayList<>();
    if (nums1.length == 0 || nums2.length == 0 || k == 0) return result;
    
    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> 
        nums1[a[0]] + nums2[a[1]] - nums1[b[0]] - nums2[b[1]]);
    
    for (int i = 0; i < Math.min(nums1.length, k); i++) {
        pq.offer(new int[]{i, 0});
    }
    
    while (k > 0 && !pq.isEmpty()) {
        int[] pair = pq.poll();
        result.add(Arrays.asList(nums1[pair[0]], nums2[pair[1]]));
        
        if (pair[1] + 1 < nums2.length) {
            pq.offer(new int[]{pair[0], pair[1] + 1});
        }
        k--;
    }
    
    return result;
}""",
                "time": "O(k log k)",
                "space": "O(k)"
            }]
        },
        "Search Range": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Use binary search to find first and last occurrence.",
                "code": """public int[] searchRange(int[] nums, int target) {
    int[] result = {-1, -1};
    if (nums == null || nums.length == 0) return result;
    
    result[0] = findFirst(nums, target);
    result[1] = findLast(nums, target);
    
    return result;
}

private int findFirst(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            result = mid;
            right = mid - 1;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

private int findLast(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) {
            result = mid;
            left = mid + 1;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Binary Tree Longest Consecutive": {
            "solutions": [{
                "title": "Solution 1 – DFS",
                "description": "Use DFS to find longest consecutive sequence.",
                "code": """public int longestConsecutive(TreeNode root) {
    if (root == null) return 0;
    return dfs(root, 1, root.val);
}

private int dfs(TreeNode node, int length, int prev) {
    if (node == null) return length - 1;
    
    int currLength = (node.val == prev + 1) ? length : 1;
    
    int left = dfs(node.left, currLength + 1, node.val);
    int right = dfs(node.right, currLength + 1, node.val);
    
    return Math.max(currLength, Math.max(left, right));
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Sort by Frequency": {
            "solutions": [{
                "title": "Solution 1 – HashMap and PriorityQueue",
                "description": "Use HashMap to count frequency and PriorityQueue to sort.",
                "code": """public String frequencySort(String s) {
    Map<Character, Integer> map = new HashMap<>();
    for (char c : s.toCharArray()) {
        map.put(c, map.getOrDefault(c, 0) + 1);
    }
    
    PriorityQueue<Character> pq = new PriorityQueue<>((a, b) -> 
        map.get(b) - map.get(a));
    
    for (char c : map.keySet()) {
        pq.offer(c);
    }
    
    StringBuilder result = new StringBuilder();
    while (!pq.isEmpty()) {
        char c = pq.poll();
        for (int i = 0; i < map.get(c); i++) {
            result.append(c);
        }
    }
    
    return result.toString();
}""",
                "time": "O(n log n)",
                "space": "O(n)"
            }]
        },
        "Flatten Binary Tree to Linked List": {
            "solutions": [{
                "title": "Solution 1 – Morris Traversal",
                "description": "Use Morris traversal to flatten the tree in-place.",
                "code": """public void flatten(TreeNode root) {
    TreeNode curr = root;
    while (curr != null) {
        if (curr.left != null) {
            TreeNode prev = curr.left;
            while (prev.right != null) {
                prev = prev.right;
            }
            prev.right = curr.right;
            curr.right = curr.left;
            curr.left = null;
        }
        curr = curr.right;
    }
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Missing Number (Bit Manipulation)": {
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
        "Subsets": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to generate all subsets.",
                "code": """public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(nums, 0, new ArrayList<>(), result);
    return result;
}

private void backtrack(int[] nums, int start, List<Integer> current, List<List<Integer>> result) {
    result.add(new ArrayList<>(current));
    
    for (int i = start; i < nums.length; i++) {
        current.add(nums[i]);
        backtrack(nums, i + 1, current, result);
        current.remove(current.size() - 1);
    }
}""",
                "time": "O(2^n)",
                "space": "O(n)"
            }]
        },
        "Rotate Image": {
            "solutions": [{
                "title": "Solution 1 – Transpose and Reverse",
                "description": "Transpose the matrix and reverse each row.",
                "code": """public void rotate(int[][] matrix) {
    int n = matrix.length;
    
    // Transpose
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
    
    // Reverse each row
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n / 2; j++) {
            int temp = matrix[i][j];
            matrix[i][j] = matrix[i][n - 1 - j];
            matrix[i][n - 1 - j] = temp;
        }
    }
}""",
                "time": "O(n²)",
                "space": "O(1)"
            }]
        },
        "Flip Game": {
            "solutions": [{
                "title": "Solution 1 – String Manipulation",
                "description": "Find all possible moves by replacing ++ with --.",
                "code": """public List<String> generatePossibleNextMoves(String s) {
    List<String> result = new ArrayList<>();
    
    for (int i = 0; i < s.length() - 1; i++) {
        if (s.charAt(i) == '+' && s.charAt(i + 1) == '+') {
            result.add(s.substring(0, i) + "--" + s.substring(i + 2));
        }
    }
    
    return result;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Reconstruct Itinerary": {
            "solutions": [{
                "title": "Solution 1 – Hierholzer's Algorithm",
                "description": "Use Hierholzer's algorithm to find Eulerian path.",
                "code": """public List<String> findItinerary(List<List<String>> tickets) {
    Map<String, PriorityQueue<String>> graph = new HashMap<>();
    
    for (List<String> ticket : tickets) {
        graph.computeIfAbsent(ticket.get(0), k -> new PriorityQueue<>()).add(ticket.get(1));
    }
    
    List<String> result = new ArrayList<>();
    dfs("JFK", graph, result);
    Collections.reverse(result);
    return result;
}

private void dfs(String airport, Map<String, PriorityQueue<String>> graph, List<String> result) {
    PriorityQueue<String> destinations = graph.get(airport);
    
    while (destinations != null && !destinations.isEmpty()) {
        String next = destinations.poll();
        dfs(next, graph, result);
    }
    
    result.add(airport);
}""",
                "time": "O(n log n)",
                "space": "O(n)"
            }]
        },
        "Remove Linked List Elements": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to remove elements with given value.",
                "code": """public ListNode removeElements(ListNode head, int val) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode prev = dummy, curr = head;
    
    while (curr != null) {
        if (curr.val == val) {
            prev.next = curr.next;
        } else {
            prev = curr;
        }
        curr = curr.next;
    }
    
    return dummy.next;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Combinations": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to generate all combinations.",
                "code": """public List<List<Integer>> combine(int n, int k) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(1, n, k, new ArrayList<>(), result);
    return result;
}

private void backtrack(int start, int n, int k, List<Integer> current, List<List<Integer>> result) {
    if (current.size() == k) {
        result.add(new ArrayList<>(current));
        return;
    }
    
    for (int i = start; i <= n; i++) {
        current.add(i);
        backtrack(i + 1, n, k, current, result);
        current.remove(current.size() - 1);
    }
}""",
                "time": "O(C(n,k))",
                "space": "O(k)"
            }]
        },
        "Add and Search Word": {
            "solutions": [{
                "title": "Solution 1 – Trie with DFS",
                "description": "Use Trie data structure with DFS for pattern matching.",
                "code": """class WordDictionary {
    private TrieNode root;
    
    public WordDictionary() {
        root = new TrieNode();
    }
    
    public void addWord(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            if (node.children[c - 'a'] == null) {
                node.children[c - 'a'] = new TrieNode();
            }
            node = node.children[c - 'a'];
        }
        node.isEnd = true;
    }
    
    public boolean search(String word) {
        return searchHelper(word, 0, root);
    }
    
    private boolean searchHelper(String word, int index, TrieNode node) {
        if (index == word.length()) return node.isEnd;
        
        char c = word.charAt(index);
        if (c == '.') {
            for (TrieNode child : node.children) {
                if (child != null && searchHelper(word, index + 1, child)) {
                    return true;
                }
            }
            return false;
        } else {
            return node.children[c - 'a'] != null && 
                   searchHelper(word, index + 1, node.children[c - 'a']);
        }
    }
}

class TrieNode {
    TrieNode[] children = new TrieNode[26];
    boolean isEnd = false;
}""",
                "time": "O(n) for add, O(26^m) for search",
                "space": "O(n)"
            }]
        },
        "Odd Even Linked List": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Separate odd and even nodes, then reconnect.",
                "code": """public ListNode oddEvenList(ListNode head) {
    if (head == null || head.next == null) return head;
    
    ListNode odd = head, even = head.next, evenHead = even;
    
    while (even != null && even.next != null) {
        odd.next = even.next;
        odd = odd.next;
        even.next = odd.next;
        even = even.next;
    }
    
    odd.next = evenHead;
    return head;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Merge Overlapping Intervals": {
            "solutions": [{
                "title": "Solution 1 – Sorting",
                "description": "Sort intervals and merge overlapping ones.",
                "code": """public int[][] merge(int[][] intervals) {
    if (intervals.length <= 1) return intervals;
    
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
    
    List<int[]> result = new ArrayList<>();
    int[] current = intervals[0];
    
    for (int i = 1; i < intervals.length; i++) {
        if (current[1] >= intervals[i][0]) {
            current[1] = Math.max(current[1], intervals[i][1]);
        } else {
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
        "Merge K Sorted Arrays": {
            "solutions": [{
                "title": "Solution 1 – Min Heap",
                "description": "Use min heap to merge k sorted arrays.",
                "code": """public int[] mergeKSortedArrays(int[][] arrays) {
    PriorityQueue<ArrayElement> pq = new PriorityQueue<>();
    int totalSize = 0;
    
    for (int i = 0; i < arrays.length; i++) {
        if (arrays[i].length > 0) {
            pq.offer(new ArrayElement(arrays[i][0], i, 0));
            totalSize += arrays[i].length;
        }
    }
    
    int[] result = new int[totalSize];
    int index = 0;
    
    while (!pq.isEmpty()) {
        ArrayElement element = pq.poll();
        result[index++] = element.value;
        
        if (element.elementIndex + 1 < arrays[element.arrayIndex].length) {
            pq.offer(new ArrayElement(
                arrays[element.arrayIndex][element.elementIndex + 1],
                element.arrayIndex,
                element.elementIndex + 1
            ));
        }
    }
    
    return result;
}

class ArrayElement {
    int value, arrayIndex, elementIndex;
    
    ArrayElement(int value, int arrayIndex, int elementIndex) {
        this.value = value;
        this.arrayIndex = arrayIndex;
        this.elementIndex = elementIndex;
    }
}""",
                "time": "O(n log k)",
                "space": "O(k)"
            }]
        },
        "4Sum": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers approach with nested loops.",
                "code": """public List<List<Integer>> fourSum(int[] nums, int target) {
    List<List<Integer>> result = new ArrayList<>();
    if (nums.length < 4) return result;
    
    Arrays.sort(nums);
    
    for (int i = 0; i < nums.length - 3; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;
        
        for (int j = i + 1; j < nums.length - 2; j++) {
            if (j > i + 1 && nums[j] == nums[j - 1]) continue;
            
            int left = j + 1, right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[j] + nums[left] + nums[right];
                
                if (sum == target) {
                    result.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left + 1]) left++;
                    while (left < right && nums[right] == nums[right - 1]) right--;
                    left++;
                    right--;
                } else if (sum < target) {
                    left++;
                } else {
                    right--;
                }
            }
        }
    }
    
    return result;
}""",
                "time": "O(n³)",
                "space": "O(1)"
            }]
        },
        "Triangle": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use bottom-up DP to find minimum path sum.",
                "code": """public int minimumTotal(List<List<Integer>> triangle) {
    if (triangle == null || triangle.isEmpty()) return 0;
    
    int n = triangle.size();
    int[] dp = new int[n];
    
    // Initialize with last row
    for (int i = 0; i < n; i++) {
        dp[i] = triangle.get(n - 1).get(i);
    }
    
    // Bottom-up approach
    for (int i = n - 2; i >= 0; i--) {
        for (int j = 0; j <= i; j++) {
            dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
        }
    }
    
    return dp[0];
}""",
                "time": "O(n²)",
                "space": "O(n)"
            }]
        },
        "Merge Sorted Array": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers from End",
                "description": "Use two pointers starting from the end of both arrays.",
                "code": """public void merge(int[] nums1, int m, int[] nums2, int n) {
    int p1 = m - 1, p2 = n - 1, p = m + n - 1;
    
    while (p1 >= 0 && p2 >= 0) {
        if (nums1[p1] > nums2[p2]) {
            nums1[p--] = nums1[p1--];
        } else {
            nums1[p--] = nums2[p2--];
        }
    }
    
    // Copy remaining elements from nums2
    while (p2 >= 0) {
        nums1[p--] = nums2[p2--];
    }
}""",
                "time": "O(m + n)",
                "space": "O(1)"
            }]
        },
        "Count Complete Tree Nodes": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Use binary search to find the number of nodes.",
                "code": """public int countNodes(TreeNode root) {
    if (root == null) return 0;
    
    int leftHeight = getHeight(root.left);
    int rightHeight = getHeight(root.right);
    
    if (leftHeight == rightHeight) {
        return (1 << leftHeight) + countNodes(root.right);
    } else {
        return (1 << rightHeight) + countNodes(root.left);
    }
}

private int getHeight(TreeNode node) {
    int height = 0;
    while (node != null) {
        height++;
        node = node.left;
    }
    return height;
}""",
                "time": "O(log² n)",
                "space": "O(log n)"
            }]
        },
        "Sort Array of 0s, 1s, and 2s": {
            "solutions": [{
                "title": "Solution 1 – Dutch National Flag",
                "description": "Use three pointers to sort the array in one pass.",
                "code": """public void sortColors(int[] nums) {
    int low = 0, mid = 0, high = nums.length - 1;
    
    while (mid <= high) {
        if (nums[mid] == 0) {
            swap(nums, low, mid);
            low++;
            mid++;
        } else if (nums[mid] == 1) {
            mid++;
        } else {
            swap(nums, mid, high);
            high--;
        }
    }
}

private void swap(int[] nums, int i, int j) {
    int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Permutations": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to generate all permutations.",
                "code": """public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(nums, new ArrayList<>(), new boolean[nums.length], result);
    return result;
}

private void backtrack(int[] nums, List<Integer> current, boolean[] used, List<List<Integer>> result) {
    if (current.size() == nums.length) {
        result.add(new ArrayList<>(current));
        return;
    }
    
    for (int i = 0; i < nums.length; i++) {
        if (!used[i]) {
            used[i] = true;
            current.add(nums[i]);
            backtrack(nums, current, used, result);
            current.remove(current.size() - 1);
            used[i] = false;
        }
    }
}""",
                "time": "O(n!)",
                "space": "O(n)"
            }]
        },
        "Intersection of Two Linked Lists": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to find intersection point.",
                "code": """public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if (headA == null || headB == null) return null;
    
    ListNode a = headA, b = headB;
    
    while (a != b) {
        a = (a == null) ? headB : a.next;
        b = (b == null) ? headA : b.next;
    }
    
    return a;
}""",
                "time": "O(m + n)",
                "space": "O(1)"
            }]
        },
        "Integer to English Words": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Convert number to English words recursively.",
                "code": """public String numberToWords(int num) {
    if (num == 0) return "Zero";
    
    String[] units = {"", "Thousand", "Million", "Billion"};
    String result = "";
    int i = 0;
    
    while (num > 0) {
        if (num % 1000 != 0) {
            result = convertLessThanOneThousand(num % 1000) + units[i] + " " + result;
        }
        num /= 1000;
        i++;
    }
    
    return result.trim();
}

private String convertLessThanOneThousand(int num) {
    String[] ones = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
    String[] teens = {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    String[] tens = {"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    
    if (num == 0) return "";
    if (num < 10) return ones[num] + " ";
    if (num < 20) return teens[num - 10] + " ";
    if (num < 100) return tens[num / 10] + " " + convertLessThanOneThousand(num % 10);
    return ones[num / 100] + " Hundred " + convertLessThanOneThousand(num % 100);
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Minimum Window Substring": {
            "solutions": [{
                "title": "Solution 1 – Sliding Window",
                "description": "Use sliding window with two pointers.",
                "code": """public String minWindow(String s, String t) {
    if (s.length() == 0 || t.length() == 0) return "";
    
    Map<Character, Integer> target = new HashMap<>();
    for (char c : t.toCharArray()) {
        target.put(c, target.getOrDefault(c, 0) + 1);
    }
    
    int left = 0, right = 0;
    int minLen = Integer.MAX_VALUE, minStart = 0;
    int required = target.size(), formed = 0;
    Map<Character, Integer> window = new HashMap<>();
    
    while (right < s.length()) {
        char c = s.charAt(right);
        window.put(c, window.getOrDefault(c, 0) + 1);
        
        if (target.containsKey(c) && window.get(c).equals(target.get(c))) {
            formed++;
        }
        
        while (left <= right && formed == required) {
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                minStart = left;
            }
            
            char leftChar = s.charAt(left);
            window.put(leftChar, window.get(leftChar) - 1);
            
            if (target.containsKey(leftChar) && window.get(leftChar) < target.get(leftChar)) {
                formed--;
            }
            left++;
        }
        right++;
    }
    
    return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
}""",
                "time": "O(n)",
                "space": "O(k)"
            }]
        }
    }

def update_problem_file(filename, problem_data):
    """Update a problem file with real solutions."""
    filepath = os.path.join("problems", filename)
    
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filename}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file already has real solutions
    if "TODO: Implement solution" not in content:
        print(f"✅ Already has real solutions: {filename}")
        return False
    
    # Get the first solution from problem_data
    solution = problem_data["solutions"][0]
    
    # Create the new solution HTML
    new_solution_html = f"""<div class="solution">
    <h3>{solution['title']}</h3>
    <p>{solution['description']}</p>
    <div class="code-block">
{solution['code']}
    </div>
    <div class="complexity">
        <strong>Time Complexity:</strong> {solution['time']}<br>
        <strong>Space Complexity:</strong> {solution['space']}
    </div>
</div>"""
    
    # Replace the placeholder solution
    pattern = r'<div class="solution">.*?<div class="complexity">.*?</div>\s*</div>'
    replacement = new_solution_html
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ Updated: {filename}")
        return True
    else:
        print(f"❌ Failed to update: {filename}")
        return False

def main():
    """Update solutions for problems."""
    solutions = get_final_solutions()
    
    problems_to_update = [
        ("minimum-platforms-required.html", "Minimum Platforms Required"),
        ("find-k-pairs-smallest-sums.html", "Find K Pairs with Smallest Sums"),
        ("search-range.html", "Search Range"),
        ("binary-tree-longest-consecutive.html", "Binary Tree Longest Consecutive"),
        ("sort-by-frequency.html", "Sort by Frequency"),
        ("flatten-binary-tree-linked-list.html", "Flatten Binary Tree to Linked List"),
        ("missing-number-bit.html", "Missing Number (Bit Manipulation)"),
        ("subsets.html", "Subsets"),
        ("rotate-image.html", "Rotate Image"),
        ("flip-game.html", "Flip Game"),
        ("reconstruct-itinerary.html", "Reconstruct Itinerary"),
        ("remove-linked-list-elements.html", "Remove Linked List Elements"),
        ("combinations.html", "Combinations"),
        ("add-search-word.html", "Add and Search Word"),
        ("odd-even-linked-list.html", "Odd Even Linked List"),
        ("merge-overlapping-intervals.html", "Merge Overlapping Intervals"),
        ("merge-k-sorted-arrays.html", "Merge K Sorted Arrays"),
        ("4sum.html", "4Sum"),
        ("triangle.html", "Triangle"),
        ("merge-sorted-array.html", "Merge Sorted Array"),
        ("count-complete-tree-nodes.html", "Count Complete Tree Nodes"),
        ("sort-array-0s-1s-2s.html", "Sort Array of 0s, 1s, and 2s"),
        ("permutations.html", "Permutations"),
        ("intersection-two-linked-lists.html", "Intersection of Two Linked Lists"),
        ("integer-english-words.html", "Integer to English Words"),
        ("minimum-window-substring.html", "Minimum Window Substring")
    ]
    
    updated_count = 0
    for filename, problem_name in problems_to_update:
        if problem_name in solutions:
            if update_problem_file(filename, solutions[problem_name]):
                updated_count += 1
        else:
            print(f"⚠️  No solution found for: {problem_name}")
    
    print(f"\n🎉 Updated {updated_count} problems with real solutions!")

if __name__ == "__main__":
    main() 