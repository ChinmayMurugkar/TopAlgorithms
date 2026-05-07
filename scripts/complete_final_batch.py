#!/usr/bin/env python3
import os
import re

def get_final_batch_solutions():
    """Get solutions for another batch of problems."""
    return {
        "Search in Rotated Sorted Array": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Use binary search with rotation detection.",
                "code": """public int search(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) return mid;
        
        if (nums[left] <= nums[mid]) {
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    return -1;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Sum of Two Integers": {
            "solutions": [{
                "title": "Solution 1 – Bit Manipulation",
                "description": "Use bit manipulation to add without + operator.",
                "code": """public int getSum(int a, int b) {
    while (b != 0) {
        int carry = a & b;
        a = a ^ b;
        b = carry << 1;
    }
    return a;
}""",
                "time": "O(1)",
                "space": "O(1)"
            }]
        },
        "Number of 1 Bits": {
            "solutions": [{
                "title": "Solution 1 – Bit Manipulation",
                "description": "Count 1 bits using bit manipulation.",
                "code": """public int hammingWeight(int n) {
    int count = 0;
    while (n != 0) {
        count += n & 1;
        n = n >>> 1;
    }
    return count;
}""",
                "time": "O(1)",
                "space": "O(1)"
            }]
        },
        "Palindrome Partitioning": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to find all palindrome partitions.",
                "code": """public List<List<String>> partition(String s) {
    List<List<String>> result = new ArrayList<>();
    backtrack(s, 0, new ArrayList<>(), result);
    return result;
}

private void backtrack(String s, int start, List<String> current, List<List<String>> result) {
    if (start == s.length()) {
        result.add(new ArrayList<>(current));
        return;
    }
    
    for (int i = start; i < s.length(); i++) {
        if (isPalindrome(s, start, i)) {
            current.add(s.substring(start, i + 1));
            backtrack(s, i + 1, current, result);
            current.remove(current.size() - 1);
        }
    }
}

private boolean isPalindrome(String s, int start, int end) {
    while (start < end) {
        if (s.charAt(start++) != s.charAt(end--)) {
            return false;
        }
    }
    return true;
}""",
                "time": "O(n × 2^n)",
                "space": "O(n)"
            }]
        },
        "Top K Frequent Elements": {
            "solutions": [{
                "title": "Solution 1 – Heap",
                "description": "Use min heap to find top k frequent elements.",
                "code": """public int[] topKFrequent(int[] nums, int k) {
    Map<Integer, Integer> frequency = new HashMap<>();
    for (int num : nums) {
        frequency.put(num, frequency.getOrDefault(num, 0) + 1);
    }
    
    PriorityQueue<Map.Entry<Integer, Integer>> pq = new PriorityQueue<>(
        (a, b) -> a.getValue() - b.getValue()
    );
    
    for (Map.Entry<Integer, Integer> entry : frequency.entrySet()) {
        pq.offer(entry);
        if (pq.size() > k) {
            pq.poll();
        }
    }
    
    int[] result = new int[k];
    for (int i = k - 1; i >= 0; i--) {
        result[i] = pq.poll().getKey();
    }
    
    return result;
}""",
                "time": "O(n log k)",
                "space": "O(n)"
            }]
        },
        "Delete Node in a Linked List": {
            "solutions": [{
                "title": "Solution 1 – Value Copy",
                "description": "Copy next node's value and delete next node.",
                "code": """public void deleteNode(ListNode node) {
    node.val = node.next.val;
    node.next = node.next.next;
}""",
                "time": "O(1)",
                "space": "O(1)"
            }]
        },
        "Guess Number Higher or Lower": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Use binary search to guess the number.",
                "code": """public int guessNumber(int n) {
    int left = 1, right = n;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int result = guess(mid);
        
        if (result == 0) {
            return mid;
        } else if (result == 1) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Rectangle Area": {
            "solutions": [{
                "title": "Solution 1 – Geometry",
                "description": "Calculate area with overlap consideration.",
                "code": """public int computeArea(int ax1, int ay1, int ax2, int ay2, 
                        int bx1, int by1, int bx2, int by2) {
    int area1 = (ax2 - ax1) * (ay2 - ay1);
    int area2 = (bx2 - bx1) * (by2 - by1);
    
    int overlapWidth = Math.min(ax2, bx2) - Math.max(ax1, bx1);
    int overlapHeight = Math.min(ay2, by2) - Math.max(ay1, by1);
    
    int overlapArea = 0;
    if (overlapWidth > 0 && overlapHeight > 0) {
        overlapArea = overlapWidth * overlapHeight;
    }
    
    return area1 + area2 - overlapArea;
}""",
                "time": "O(1)",
                "space": "O(1)"
            }]
        },
        "Candy": {
            "solutions": [{
                "title": "Solution 1 – Two Pass",
                "description": "Use two passes to distribute candy optimally.",
                "code": """public int candy(int[] ratings) {
    int n = ratings.length;
    int[] candies = new int[n];
    Arrays.fill(candies, 1);
    
    // Left to right pass
    for (int i = 1; i < n; i++) {
        if (ratings[i] > ratings[i - 1]) {
            candies[i] = candies[i - 1] + 1;
        }
    }
    
    // Right to left pass
    for (int i = n - 2; i >= 0; i--) {
        if (ratings[i] > ratings[i + 1]) {
            candies[i] = Math.max(candies[i], candies[i + 1] + 1);
        }
    }
    
    int total = 0;
    for (int candy : candies) {
        total += candy;
    }
    
    return total;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Compare Version Numbers": {
            "solutions": [{
                "title": "Solution 1 – String Parsing",
                "description": "Parse version numbers and compare.",
                "code": """public int compareVersion(String version1, String version2) {
    String[] v1 = version1.split("\\.");
    String[] v2 = version2.split("\\.");
    
    int maxLength = Math.max(v1.length, v2.length);
    
    for (int i = 0; i < maxLength; i++) {
        int num1 = i < v1.length ? Integer.parseInt(v1[i]) : 0;
        int num2 = i < v2.length ? Integer.parseInt(v2[i]) : 0;
        
        if (num1 < num2) return -1;
        if (num1 > num2) return 1;
    }
    
    return 0;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Merge K Sorted Lists": {
            "solutions": [{
                "title": "Solution 1 – Min Heap",
                "description": "Use min heap to merge k sorted lists.",
                "code": """public ListNode mergeKLists(ListNode[] lists) {
    if (lists == null || lists.length == 0) return null;
    
    PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
    
    for (ListNode list : lists) {
        if (list != null) {
            pq.offer(list);
        }
    }
    
    ListNode dummy = new ListNode(0);
    ListNode current = dummy;
    
    while (!pq.isEmpty()) {
        ListNode node = pq.poll();
        current.next = node;
        current = current.next;
        
        if (node.next != null) {
            pq.offer(node.next);
        }
    }
    
    return dummy.next;
}""",
                "time": "O(n log k)",
                "space": "O(k)"
            }]
        },
        "Intersection of Two Arrays": {
            "solutions": [{
                "title": "Solution 1 – HashSet",
                "description": "Use HashSet to find intersection.",
                "code": """public int[] intersection(int[] nums1, int[] nums2) {
    Set<Integer> set1 = new HashSet<>();
    Set<Integer> result = new HashSet<>();
    
    for (int num : nums1) {
        set1.add(num);
    }
    
    for (int num : nums2) {
        if (set1.contains(num)) {
            result.add(num);
        }
    }
    
    int[] intersection = new int[result.size()];
    int i = 0;
    for (int num : result) {
        intersection[i++] = num;
    }
    
    return intersection;
}""",
                "time": "O(n + m)",
                "space": "O(min(n, m))"
            }]
        },
        "Range Addition": {
            "solutions": [{
                "title": "Solution 1 – Prefix Sum",
                "description": "Use prefix sum to handle range updates.",
                "code": """public int[] getModifiedArray(int length, int[][] updates) {
    int[] result = new int[length];
    
    for (int[] update : updates) {
        int start = update[0];
        int end = update[1];
        int inc = update[2];
        
        result[start] += inc;
        if (end + 1 < length) {
            result[end + 1] -= inc;
        }
    }
    
    for (int i = 1; i < length; i++) {
        result[i] += result[i - 1];
    }
    
    return result;
}""",
                "time": "O(n + k)",
                "space": "O(1)"
            }]
        },
        "Divide Two Integers": {
            "solutions": [{
                "title": "Solution 1 – Bit Manipulation",
                "description": "Use bit manipulation to divide without multiplication.",
                "code": """public int divide(int dividend, int divisor) {
    if (dividend == Integer.MIN_VALUE && divisor == -1) {
        return Integer.MAX_VALUE;
    }
    
    boolean negative = (dividend < 0) ^ (divisor < 0);
    long dvd = Math.abs((long) dividend);
    long dvs = Math.abs((long) divisor);
    
    int result = 0;
    while (dvd >= dvs) {
        long temp = dvs;
        int multiple = 1;
        
        while (dvd >= (temp << 1)) {
            temp <<= 1;
            multiple <<= 1;
        }
        
        dvd -= temp;
        result += multiple;
    }
    
    return negative ? -result : result;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Search a 2D Matrix II": {
            "solutions": [{
                "title": "Solution 1 – Search from Corner",
                "description": "Start from top-right corner and move based on comparison.",
                "code": """public boolean searchMatrix(int[][] matrix, int target) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
        return false;
    }
    
    int row = 0, col = matrix[0].length - 1;
    
    while (row < matrix.length && col >= 0) {
        if (matrix[row][col] == target) {
            return true;
        } else if (matrix[row][col] > target) {
            col--;
        } else {
            row++;
        }
    }
    
    return false;
}""",
                "time": "O(m + n)",
                "space": "O(1)"
            }]
        },
        "Pow(x, n)": {
            "solutions": [{
                "title": "Solution 1 – Fast Power",
                "description": "Use fast power algorithm with binary exponentiation.",
                "code": """public double myPow(double x, int n) {
    if (n == 0) return 1;
    if (n == 1) return x;
    if (n == -1) return 1 / x;
    
    double half = myPow(x, n / 2);
    if (n % 2 == 0) {
        return half * half;
    } else {
        return n > 0 ? half * half * x : half * half / x;
    }
}""",
                "time": "O(log n)",
                "space": "O(log n)"
            }]
        },
        "Minimum Height Trees": {
            "solutions": [{
                "title": "Solution 1 – Topological Sort",
                "description": "Use topological sort to find center nodes.",
                "code": """public List<Integer> findMinHeightTrees(int n, int[][] edges) {
    if (n == 1) return Arrays.asList(0);
    
    List<Set<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < n; i++) {
        graph.add(new HashSet<>());
    }
    
    for (int[] edge : edges) {
        graph.get(edge[0]).add(edge[1]);
        graph.get(edge[1]).add(edge[0]);
    }
    
    List<Integer> leaves = new ArrayList<>();
    for (int i = 0; i < n; i++) {
        if (graph.get(i).size() == 1) {
            leaves.add(i);
        }
    }
    
    while (n > 2) {
        n -= leaves.size();
        List<Integer> newLeaves = new ArrayList<>();
        
        for (int leaf : leaves) {
            int neighbor = graph.get(leaf).iterator().next();
            graph.get(neighbor).remove(leaf);
            
            if (graph.get(neighbor).size() == 1) {
                newLeaves.add(neighbor);
            }
        }
        
        leaves = newLeaves;
    }
    
    return leaves;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Print Linked List in Reverse": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Use recursion to print in reverse order.",
                "code": """public void printLinkedListInReverse(ListNode head) {
    if (head == null) return;
    
    printLinkedListInReverse(head.next);
    System.out.println(head.val);
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Jump Game": {
            "solutions": [{
                "title": "Solution 1 – Greedy",
                "description": "Use greedy approach to check if can reach end.",
                "code": """public boolean canJump(int[] nums) {
    int maxReach = 0;
    
    for (int i = 0; i < nums.length; i++) {
        if (i > maxReach) return false;
        maxReach = Math.max(maxReach, i + nums[i]);
    }
    
    return true;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Add Digits": {
            "solutions": [{
                "title": "Solution 1 – Mathematical",
                "description": "Use mathematical formula for digital root.",
                "code": """public int addDigits(int num) {
    if (num == 0) return 0;
    if (num % 9 == 0) return 9;
    return num % 9;
}""",
                "time": "O(1)",
                "space": "O(1)"
            }]
        },
        "Inorder Successor in BST": {
            "solutions": [{
                "title": "Solution 1 – BST Traversal",
                "description": "Use BST property to find successor.",
                "code": """public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
    TreeNode successor = null;
    
    while (root != null) {
        if (p.val >= root.val) {
            root = root.right;
        } else {
            successor = root;
            root = root.left;
        }
    }
    
    return successor;
}""",
                "time": "O(h)",
                "space": "O(1)"
            }]
        },
        "Largest BST Subtree": {
            "solutions": [{
                "title": "Solution 1 – DFS",
                "description": "Use DFS to find largest BST subtree.",
                "code": """public int largestBSTSubtree(TreeNode root) {
    return dfs(root)[2];
}

private int[] dfs(TreeNode node) {
    if (node == null) {
        return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE, 0};
    }
    
    int[] left = dfs(node.left);
    int[] right = dfs(node.right);
    
    if (node.val > left[1] && node.val < right[0]) {
        return new int[]{
            Math.min(left[0], node.val),
            Math.max(right[1], node.val),
            left[2] + right[2] + 1
        };
    }
    
    return new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE, 
                     Math.max(left[2], right[2])};
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Multiply Strings": {
            "solutions": [{
                "title": "Solution 1 – Grade School Multiplication",
                "description": "Implement grade school multiplication algorithm.",
                "code": """public String multiply(String num1, String num2) {
    int m = num1.length(), n = num2.length();
    int[] result = new int[m + n];
    
    for (int i = m - 1; i >= 0; i--) {
        for (int j = n - 1; j >= 0; j--) {
            int product = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
            int sum = result[i + j + 1] + product;
            
            result[i + j + 1] = sum % 10;
            result[i + j] += sum / 10;
        }
    }
    
    StringBuilder sb = new StringBuilder();
    for (int digit : result) {
        if (!(sb.length() == 0 && digit == 0)) {
            sb.append(digit);
        }
    }
    
    return sb.length() == 0 ? "0" : sb.toString();
}""",
                "time": "O(m × n)",
                "space": "O(m + n)"
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
    solutions = get_final_batch_solutions()
    
    problems_to_update = [
        ("search-rotated-array.html", "Search in Rotated Sorted Array"),
        ("sum-two-integers.html", "Sum of Two Integers"),
        ("number-1-bits.html", "Number of 1 Bits"),
        ("palindrome-partitioning.html", "Palindrome Partitioning"),
        ("top-k-frequent-elements.html", "Top K Frequent Elements"),
        ("delete-node-linked-list.html", "Delete Node in a Linked List"),
        ("guess-number-higher-lower.html", "Guess Number Higher or Lower"),
        ("rectangle-area.html", "Rectangle Area"),
        ("candy.html", "Candy"),
        ("compare-version-numbers.html", "Compare Version Numbers"),
        ("merge-k-sorted-lists.html", "Merge K Sorted Lists"),
        ("intersection-two-arrays.html", "Intersection of Two Arrays"),
        ("range-addition.html", "Range Addition"),
        ("divide-two-integers.html", "Divide Two Integers"),
        ("search-2d-matrix-ii.html", "Search a 2D Matrix II"),
        ("pow-x-n.html", "Pow(x, n)"),
        ("minimum-height-trees.html", "Minimum Height Trees"),
        ("print-linked-list-reversed.html", "Print Linked List in Reverse"),
        ("jump-game.html", "Jump Game"),
        ("add-digits.html", "Add Digits"),
        ("inorder-successor-bst.html", "Inorder Successor in BST"),
        ("largest-bst-subtree.html", "Largest BST Subtree"),
        ("multiply-strings.html", "Multiply Strings")
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