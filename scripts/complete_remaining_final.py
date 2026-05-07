#!/usr/bin/env python3
import os
import re

def get_remaining_final_solutions():
    """Get solutions for the remaining problems."""
    return {
        "Permutations II": {
            "solutions": [{
                "title": "Solution 1 – Backtracking with Duplicate Handling",
                "description": "Use backtracking with duplicate detection.",
                "code": """public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    Arrays.sort(nums);
    backtrack(nums, new boolean[nums.length], new ArrayList<>(), result);
    return result;
}

private void backtrack(int[] nums, boolean[] used, List<Integer> current, List<List<Integer>> result) {
    if (current.size() == nums.length) {
        result.add(new ArrayList<>(current));
        return;
    }
    
    for (int i = 0; i < nums.length; i++) {
        if (used[i] || (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])) {
            continue;
        }
        
        used[i] = true;
        current.add(nums[i]);
        backtrack(nums, used, current, result);
        current.remove(current.size() - 1);
        used[i] = false;
    }
}""",
                "time": "O(n!)",
                "space": "O(n)"
            }]
        },
        "H-Index": {
            "solutions": [{
                "title": "Solution 1 – Sorting",
                "description": "Sort citations and find h-index.",
                "code": """public int hIndex(int[] citations) {
    Arrays.sort(citations);
    int n = citations.length;
    
    for (int i = 0; i < n; i++) {
        if (citations[i] >= n - i) {
            return n - i;
        }
    }
    
    return 0;
}""",
                "time": "O(n log n)",
                "space": "O(1)"
            }]
        },
        "Excel Sheet Column Title": {
            "solutions": [{
                "title": "Solution 1 – Base 26 Conversion",
                "description": "Convert number to base 26 column title.",
                "code": """public String convertToTitle(int columnNumber) {
    StringBuilder result = new StringBuilder();
    
    while (columnNumber > 0) {
        columnNumber--;
        result.insert(0, (char) ('A' + columnNumber % 26));
        columnNumber /= 26;
    }
    
    return result.toString();
}""",
                "time": "O(log n)",
                "space": "O(log n)"
            }]
        },
        "Implement Stack using Queues": {
            "solutions": [{
                "title": "Solution 1 – Two Queues",
                "description": "Use two queues to implement stack.",
                "code": """class MyStack {
    private Queue<Integer> q1;
    private Queue<Integer> q2;
    
    public MyStack() {
        q1 = new LinkedList<>();
        q2 = new LinkedList<>();
    }
    
    public void push(int x) {
        q2.offer(x);
        while (!q1.isEmpty()) {
            q2.offer(q1.poll());
        }
        Queue<Integer> temp = q1;
        q1 = q2;
        q2 = temp;
    }
    
    public int pop() {
        return q1.poll();
    }
    
    public int top() {
        return q1.peek();
    }
    
    public boolean empty() {
        return q1.isEmpty();
    }
}""",
                "time": "O(n) for push, O(1) for others",
                "space": "O(n)"
            }]
        },
        "Construct Binary Tree from Inorder and Postorder": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Use recursive approach with HashMap for index lookup.",
                "code": """public TreeNode buildTree(int[] inorder, int[] postorder) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < inorder.length; i++) {
        map.put(inorder[i], i);
    }
    return buildTreeHelper(inorder, 0, inorder.length - 1, postorder, 0, postorder.length - 1, map);
}

private TreeNode buildTreeHelper(int[] inorder, int inStart, int inEnd, 
                               int[] postorder, int postStart, int postEnd, 
                               Map<Integer, Integer> map) {
    if (inStart > inEnd || postStart > postEnd) return null;
    
    TreeNode root = new TreeNode(postorder[postEnd]);
    int rootIndex = map.get(postorder[postEnd]);
    int leftSize = rootIndex - inStart;
    
    root.left = buildTreeHelper(inorder, inStart, rootIndex - 1, 
                               postorder, postStart, postStart + leftSize - 1, map);
    root.right = buildTreeHelper(inorder, rootIndex + 1, inEnd, 
                                postorder, postStart + leftSize, postEnd - 1, map);
    
    return root;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Pascal's Triangle": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Build Pascal's triangle using DP.",
                "code": """public List<List<Integer>> generate(int numRows) {
    List<List<Integer>> result = new ArrayList<>();
    
    for (int i = 0; i < numRows; i++) {
        List<Integer> row = new ArrayList<>();
        for (int j = 0; j <= i; j++) {
            if (j == 0 || j == i) {
                row.add(1);
            } else {
                row.add(result.get(i - 1).get(j - 1) + result.get(i - 1).get(j));
            }
        }
        result.add(row);
    }
    
    return result;
}""",
                "time": "O(n²)",
                "space": "O(n²)"
            }]
        },
        "Simplify Path": {
            "solutions": [{
                "title": "Solution 1 – Stack",
                "description": "Use stack to handle path operations.",
                "code": """public String simplifyPath(String path) {
    String[] parts = path.split("/");
    Stack<String> stack = new Stack<>();
    
    for (String part : parts) {
        if (part.equals("") || part.equals(".")) {
            continue;
        } else if (part.equals("..")) {
            if (!stack.isEmpty()) {
                stack.pop();
            }
        } else {
            stack.push(part);
        }
    }
    
    if (stack.isEmpty()) return "/";
    
    StringBuilder result = new StringBuilder();
    for (String dir : stack) {
        result.append("/").append(dir);
    }
    
    return result.toString();
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Meeting Rooms": {
            "solutions": [{
                "title": "Solution 1 – Sorting",
                "description": "Sort intervals and check for overlap.",
                "code": """public boolean canAttendMeetings(int[][] intervals) {
    Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
    
    for (int i = 1; i < intervals.length; i++) {
        if (intervals[i][0] < intervals[i - 1][1]) {
            return false;
        }
    }
    
    return true;
}""",
                "time": "O(n log n)",
                "space": "O(1)"
            }]
        },
        "Unique Binary Search Trees II": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Generate all possible BSTs recursively.",
                "code": """public List<TreeNode> generateTrees(int n) {
    if (n == 0) return new ArrayList<>();
    return generateTrees(1, n);
}

private List<TreeNode> generateTrees(int start, int end) {
    List<TreeNode> result = new ArrayList<>();
    
    if (start > end) {
        result.add(null);
        return result;
    }
    
    for (int i = start; i <= end; i++) {
        List<TreeNode> leftTrees = generateTrees(start, i - 1);
        List<TreeNode> rightTrees = generateTrees(i + 1, end);
        
        for (TreeNode left : leftTrees) {
            for (TreeNode right : rightTrees) {
                TreeNode root = new TreeNode(i);
                root.left = left;
                root.right = right;
                result.add(root);
            }
        }
    }
    
    return result;
}""",
                "time": "O(C(n))",
                "space": "O(C(n))"
            }]
        },
        "Unique Paths II": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP with obstacle handling.",
                "code": """public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int m = obstacleGrid.length, n = obstacleGrid[0].length;
    
    if (obstacleGrid[0][0] == 1) return 0;
    
    obstacleGrid[0][0] = 1;
    
    for (int i = 1; i < m; i++) {
        obstacleGrid[i][0] = (obstacleGrid[i][0] == 0 && obstacleGrid[i - 1][0] == 1) ? 1 : 0;
    }
    
    for (int j = 1; j < n; j++) {
        obstacleGrid[0][j] = (obstacleGrid[0][j] == 0 && obstacleGrid[0][j - 1] == 1) ? 1 : 0;
    }
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[i][j] == 0) {
                obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
            } else {
                obstacleGrid[i][j] = 0;
            }
        }
    }
    
    return obstacleGrid[m - 1][n - 1];
}""",
                "time": "O(m × n)",
                "space": "O(1)"
            }]
        },
        "Reverse Nodes in k-Group": {
            "solutions": [{
                "title": "Solution 1 – Iterative",
                "description": "Reverse nodes in groups of k iteratively.",
                "code": """public ListNode reverseKGroup(ListNode head, int k) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode prev = dummy;
    
    while (head != null) {
        ListNode tail = head;
        int count = 0;
        
        while (tail != null && count < k) {
            tail = tail.next;
            count++;
        }
        
        if (count == k) {
            ListNode[] reversed = reverse(head, tail);
            prev.next = reversed[0];
            prev = reversed[1];
            head = tail;
        } else {
            break;
        }
    }
    
    return dummy.next;
}

private ListNode[] reverse(ListNode head, ListNode tail) {
    ListNode prev = null;
    ListNode curr = head;
    
    while (curr != tail) {
        ListNode next = curr.next;
        curr.next = prev;
        prev = curr;
        curr = next;
    }
    
    return new ListNode[]{prev, head};
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Excel Sheet Column Number": {
            "solutions": [{
                "title": "Solution 1 – Base 26 Conversion",
                "description": "Convert column title to number.",
                "code": """public int titleToNumber(String columnTitle) {
    int result = 0;
    
    for (char c : columnTitle.toCharArray()) {
        result = result * 26 + (c - 'A' + 1);
    }
    
    return result;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Maximum Binary Gap": {
            "solutions": [{
                "title": "Solution 1 – Bit Manipulation",
                "description": "Find maximum gap between 1s in binary representation.",
                "code": """public int binaryGap(int n) {
    int maxGap = 0;
    int lastOne = -1;
    
    for (int i = 0; i < 32; i++) {
        if ((n & (1 << i)) != 0) {
            if (lastOne != -1) {
                maxGap = Math.max(maxGap, i - lastOne);
            }
            lastOne = i;
        }
    }
    
    return maxGap;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Unique Binary Search Trees": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to count unique BSTs.",
                "code": """public int numTrees(int n) {
    int[] dp = new int[n + 1];
    dp[0] = 1;
    dp[1] = 1;
    
    for (int i = 2; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            dp[i] += dp[j - 1] * dp[i - j];
        }
    }
    
    return dp[n];
}""",
                "time": "O(n²)",
                "space": "O(n)"
            }]
        },
        "Text Justification": {
            "solutions": [{
                "title": "Solution 1 – Greedy",
                "description": "Use greedy approach to justify text.",
                "code": """public List<String> fullJustify(String[] words, int maxWidth) {
    List<String> result = new ArrayList<>();
    int i = 0;
    
    while (i < words.length) {
        int j = i;
        int lineLength = 0;
        
        while (j < words.length && lineLength + words[j].length() + (j - i) <= maxWidth) {
            lineLength += words[j].length();
            j++;
        }
        
        StringBuilder line = new StringBuilder();
        int spaces = maxWidth - lineLength;
        
        if (j == words.length || j - i == 1) {
            // Left justify
            for (int k = i; k < j; k++) {
                line.append(words[k]);
                if (k < j - 1) line.append(" ");
            }
            while (line.length() < maxWidth) {
                line.append(" ");
            }
        } else {
            // Full justify
            int gaps = j - i - 1;
            int spacesPerGap = gaps == 0 ? 0 : spaces / gaps;
            int extraSpaces = gaps == 0 ? 0 : spaces % gaps;
            
            for (int k = i; k < j; k++) {
                line.append(words[k]);
                if (k < j - 1) {
                    int spacesToAdd = spacesPerGap + (extraSpaces-- > 0 ? 1 : 0);
                    for (int s = 0; s < spacesToAdd; s++) {
                        line.append(" ");
                    }
                }
            }
        }
        
        result.add(line.toString());
        i = j;
    }
    
    return result;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Combination Sum": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to find all combinations.",
                "code": """public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(candidates, target, 0, new ArrayList<>(), result);
    return result;
}

private void backtrack(int[] candidates, int target, int start, List<Integer> current, List<List<Integer>> result) {
    if (target == 0) {
        result.add(new ArrayList<>(current));
        return;
    }
    
    if (target < 0) return;
    
    for (int i = start; i < candidates.length; i++) {
        current.add(candidates[i]);
        backtrack(candidates, target - candidates[i], i, current, result);
        current.remove(current.size() - 1);
    }
}""",
                "time": "O(n^(target/min))",
                "space": "O(target/min)"
            }]
        },
        "Partition List": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to partition list.",
                "code": """public ListNode partition(ListNode head, int x) {
    ListNode beforeHead = new ListNode(0);
    ListNode afterHead = new ListNode(0);
    ListNode before = beforeHead;
    ListNode after = afterHead;
    
    while (head != null) {
        if (head.val < x) {
            before.next = head;
            before = before.next;
        } else {
            after.next = head;
            after = after.next;
        }
        head = head.next;
    }
    
    after.next = null;
    before.next = afterHead.next;
    
    return beforeHead.next;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Remove Duplicates from Sorted List II": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to remove all duplicates.",
                "code": """public ListNode deleteDuplicates(ListNode head) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode prev = dummy;
    
    while (head != null && head.next != null) {
        if (head.val == head.next.val) {
            while (head.next != null && head.val == head.next.val) {
                head = head.next;
            }
            prev.next = head.next;
        } else {
            prev = head;
        }
        head = head.next;
    }
    
    return dummy.next;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Longest Substring with At Most Two Distinct Characters": {
            "solutions": [{
                "title": "Solution 1 – Sliding Window",
                "description": "Use sliding window with HashMap.",
                "code": """public int lengthOfLongestSubstringTwoDistinct(String s) {
    Map<Character, Integer> map = new HashMap<>();
    int left = 0, maxLen = 0;
    
    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        map.put(c, map.getOrDefault(c, 0) + 1);
        
        while (map.size() > 2) {
            char leftChar = s.charAt(left);
            map.put(leftChar, map.get(leftChar) - 1);
            if (map.get(leftChar) == 0) {
                map.remove(leftChar);
            }
            left++;
        }
        
        maxLen = Math.max(maxLen, right - left + 1);
    }
    
    return maxLen;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Knapsack Problem": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to solve 0/1 knapsack problem.",
                "code": """public int knapsack(int[] weights, int[] values, int capacity) {
    int n = weights.length;
    int[][] dp = new int[n + 1][capacity + 1];
    
    for (int i = 1; i <= n; i++) {
        for (int w = 0; w <= capacity; w++) {
            if (weights[i - 1] <= w) {
                dp[i][w] = Math.max(dp[i - 1][w], 
                                   dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }
    
    return dp[n][capacity];
}""",
                "time": "O(n × W)",
                "space": "O(n × W)"
            }]
        },
        "Implement Queue using Stacks": {
            "solutions": [{
                "title": "Solution 1 – Two Stacks",
                "description": "Use two stacks to implement queue.",
                "code": """class MyQueue {
    private Stack<Integer> stack1;
    private Stack<Integer> stack2;
    
    public MyQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }
    
    public void push(int x) {
        stack1.push(x);
    }
    
    public int pop() {
        if (stack2.isEmpty()) {
            while (!stack1.isEmpty()) {
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop();
    }
    
    public int peek() {
        if (stack2.isEmpty()) {
            while (!stack1.isEmpty()) {
                stack2.push(stack1.pop());
            }
        }
        return stack2.peek();
    }
    
    public boolean empty() {
        return stack1.isEmpty() && stack2.isEmpty();
    }
}""",
                "time": "O(1) amortized",
                "space": "O(n)"
            }]
        },
        "Minimum Size Subarray Sum": {
            "solutions": [{
                "title": "Solution 1 – Sliding Window",
                "description": "Use sliding window to find minimum subarray.",
                "code": """public int minSubArrayLen(int target, int[] nums) {
    int left = 0, sum = 0, minLen = Integer.MAX_VALUE;
    
    for (int right = 0; right < nums.length; right++) {
        sum += nums[right];
        
        while (sum >= target) {
            minLen = Math.min(minLen, right - left + 1);
            sum -= nums[left];
            left++;
        }
    }
    
    return minLen == Integer.MAX_VALUE ? 0 : minLen;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Two Sum III": {
            "solutions": [{
                "title": "Solution 1 – HashMap",
                "description": "Use HashMap to store numbers and find pairs.",
                "code": """class TwoSum {
    private Map<Integer, Integer> map;
    
    public TwoSum() {
        map = new HashMap<>();
    }
    
    public void add(int number) {
        map.put(number, map.getOrDefault(number, 0) + 1);
    }
    
    public boolean find(int value) {
        for (int num : map.keySet()) {
            int complement = value - num;
            if (complement == num) {
                if (map.get(num) > 1) return true;
            } else {
                if (map.containsKey(complement)) return true;
            }
        }
        return false;
    }
}""",
                "time": "O(n) for find, O(1) for add",
                "space": "O(n)"
            }]
        },
        "Form Largest Number": {
            "solutions": [{
                "title": "Solution 1 – Custom Sorting",
                "description": "Use custom comparator to sort numbers.",
                "code": """public String largestNumber(int[] nums) {
    String[] strings = new String[nums.length];
    for (int i = 0; i < nums.length; i++) {
        strings[i] = String.valueOf(nums[i]);
    }
    
    Arrays.sort(strings, (a, b) -> (b + a).compareTo(a + b));
    
    if (strings[0].equals("0")) return "0";
    
    StringBuilder result = new StringBuilder();
    for (String s : strings) {
        result.append(s);
    }
    
    return result.toString();
}""",
                "time": "O(n log n)",
                "space": "O(n)"
            }]
        },
        "Restore IP Addresses": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to find valid IP addresses.",
                "code": """public List<String> restoreIpAddresses(String s) {
    List<String> result = new ArrayList<>();
    backtrack(s, 0, 0, "", result);
    return result;
}

private void backtrack(String s, int index, int dots, String current, List<String> result) {
    if (dots == 4 && index == s.length()) {
        result.add(current.substring(0, current.length() - 1));
        return;
    }
    
    if (dots > 4) return;
    
    for (int i = 1; i <= 3 && index + i <= s.length(); i++) {
        String segment = s.substring(index, index + i);
        if (segment.length() > 1 && segment.charAt(0) == '0') continue;
        if (Integer.parseInt(segment) > 255) continue;
        
        backtrack(s, index + i, dots + 1, current + segment + ".", result);
    }
}""",
                "time": "O(3^4)",
                "space": "O(n)"
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
    solutions = get_remaining_final_solutions()
    
    problems_to_update = [
        ("permutations-ii.html", "Permutations II"),
        ("h-index.html", "H-Index"),
        ("excel-sheet-column-title.html", "Excel Sheet Column Title"),
        ("implement-stack-queues.html", "Implement Stack using Queues"),
        ("construct-binary-tree-inorder-postorder.html", "Construct Binary Tree from Inorder and Postorder"),
        ("pascals-triangle.html", "Pascal's Triangle"),
        ("simplify-path.html", "Simplify Path"),
        ("meeting-rooms.html", "Meeting Rooms"),
        ("unique-binary-search-trees-ii.html", "Unique Binary Search Trees II"),
        ("unique-paths-ii.html", "Unique Paths II"),
        ("reverse-nodes-k-group.html", "Reverse Nodes in k-Group"),
        ("excel-sheet-column-number.html", "Excel Sheet Column Number"),
        ("maximum-binary-gap.html", "Maximum Binary Gap"),
        ("unique-binary-search-trees.html", "Unique Binary Search Trees"),
        ("text-justification.html", "Text Justification"),
        ("combination-sum.html", "Combination Sum"),
        ("partition-list.html", "Partition List"),
        ("remove-duplicates-sorted-list-ii.html", "Remove Duplicates from Sorted List II"),
        ("longest-substring-2-unique.html", "Longest Substring with At Most Two Distinct Characters"),
        ("knapsack-problem.html", "Knapsack Problem"),
        ("implement-queue-stacks.html", "Implement Queue using Stacks"),
        ("minimum-size-subarray-sum.html", "Minimum Size Subarray Sum"),
        ("two-sum-iii.html", "Two Sum III"),
        ("form-largest-number.html", "Form Largest Number"),
        ("restore-ip-addresses.html", "Restore IP Addresses")
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