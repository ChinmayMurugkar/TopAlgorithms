#!/usr/bin/env python3
import os
import re

def get_last_solutions():
    """Get solutions for the final batch of problems."""
    return {
        "Construct Binary Tree from Preorder and Inorder": {
            "solutions": [{
                "title": "Solution 1 – Recursive",
                "description": "Use recursive approach with HashMap for index lookup.",
                "code": """public TreeNode buildTree(int[] preorder, int[] inorder) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < inorder.length; i++) {
        map.put(inorder[i], i);
    }
    return buildTreeHelper(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1, map);
}

private TreeNode buildTreeHelper(int[] preorder, int preStart, int preEnd, 
                               int[] inorder, int inStart, int inEnd, 
                               Map<Integer, Integer> map) {
    if (preStart > preEnd || inStart > inEnd) return null;
    
    TreeNode root = new TreeNode(preorder[preStart]);
    int rootIndex = map.get(preorder[preStart]);
    int leftSize = rootIndex - inStart;
    
    root.left = buildTreeHelper(preorder, preStart + 1, preStart + leftSize, 
                               inorder, inStart, rootIndex - 1, map);
    root.right = buildTreeHelper(preorder, preStart + leftSize + 1, preEnd, 
                                inorder, rootIndex + 1, inEnd, map);
    
    return root;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Count and Say": {
            "solutions": [{
                "title": "Solution 1 – Iterative",
                "description": "Generate the sequence iteratively.",
                "code": """public String countAndSay(int n) {
    if (n == 1) return "1";
    
    String result = "1";
    for (int i = 2; i <= n; i++) {
        result = generateNext(result);
    }
    return result;
}

private String generateNext(String s) {
    StringBuilder sb = new StringBuilder();
    int count = 1;
    char current = s.charAt(0);
    
    for (int i = 1; i < s.length(); i++) {
        if (s.charAt(i) == current) {
            count++;
        } else {
            sb.append(count).append(current);
            current = s.charAt(i);
            count = 1;
        }
    }
    sb.append(count).append(current);
    
    return sb.toString();
}""",
                "time": "O(n × 2^n)",
                "space": "O(2^n)"
            }]
        },
        "Letter Combinations of Phone Number": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to generate all combinations.",
                "code": """public List<String> letterCombinations(String digits) {
    List<String> result = new ArrayList<>();
    if (digits == null || digits.length() == 0) return result;
    
    String[] mapping = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    backtrack(digits, 0, "", mapping, result);
    return result;
}

private void backtrack(String digits, int index, String current, String[] mapping, List<String> result) {
    if (index == digits.length()) {
        result.add(current);
        return;
    }
    
    String letters = mapping[digits.charAt(index) - '0'];
    for (char letter : letters.toCharArray()) {
        backtrack(digits, index + 1, current + letter, mapping, result);
    }
}""",
                "time": "O(4^n)",
                "space": "O(n)"
            }]
        },
        "Recover Binary Search Tree": {
            "solutions": [{
                "title": "Solution 1 – Inorder Traversal",
                "description": "Use inorder traversal to find swapped nodes.",
                "code": """public void recoverTree(TreeNode root) {
    TreeNode[] swapped = new TreeNode[2];
    TreeNode[] prev = {null};
    inorder(root, prev, swapped);
    
    int temp = swapped[0].val;
    swapped[0].val = swapped[1].val;
    swapped[1].val = temp;
}

private void inorder(TreeNode node, TreeNode[] prev, TreeNode[] swapped) {
    if (node == null) return;
    
    inorder(node.left, prev, swapped);
    
    if (prev[0] != null && prev[0].val > node.val) {
        if (swapped[0] == null) {
            swapped[0] = prev[0];
            swapped[1] = node;
        } else {
            swapped[1] = node;
        }
    }
    prev[0] = node;
    
    inorder(node.right, prev, swapped);
}""",
                "time": "O(n)",
                "space": "O(h)"
            }]
        },
        "Generate Parentheses": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to generate valid parentheses.",
                "code": """public List<String> generateParenthesis(int n) {
    List<String> result = new ArrayList<>();
    backtrack(n, n, "", result);
    return result;
}

private void backtrack(int open, int close, String current, List<String> result) {
    if (open == 0 && close == 0) {
        result.add(current);
        return;
    }
    
    if (open > 0) {
        backtrack(open - 1, close, current + "(", result);
    }
    
    if (close > open) {
        backtrack(open, close - 1, current + ")", result);
    }
}""",
                "time": "O(4^n/√n)",
                "space": "O(n)"
            }]
        },
        "Largest Rectangle in Histogram": {
            "solutions": [{
                "title": "Solution 1 – Monotonic Stack",
                "description": "Use monotonic stack to find largest rectangle.",
                "code": """public int largestRectangleArea(int[] heights) {
    Stack<Integer> stack = new Stack<>();
    int maxArea = 0;
    int i = 0;
    
    while (i < heights.length) {
        if (stack.isEmpty() || heights[stack.peek()] <= heights[i]) {
            stack.push(i++);
        } else {
            int height = heights[stack.pop()];
            int width = stack.isEmpty() ? i : i - stack.peek() - 1;
            maxArea = Math.max(maxArea, height * width);
        }
    }
    
    while (!stack.isEmpty()) {
        int height = heights[stack.pop()];
        int width = stack.isEmpty() ? i : i - stack.peek() - 1;
        maxArea = Math.max(maxArea, height * width);
    }
    
    return maxArea;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Swap Nodes in Pairs": {
            "solutions": [{
                "title": "Solution 1 – Iterative",
                "description": "Swap nodes in pairs iteratively.",
                "code": """public ListNode swapPairs(ListNode head) {
    if (head == null || head.next == null) return head;
    
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode prev = dummy;
    
    while (prev.next != null && prev.next.next != null) {
        ListNode first = prev.next;
        ListNode second = prev.next.next;
        
        first.next = second.next;
        second.next = first;
        prev.next = second;
        
        prev = first;
    }
    
    return dummy.next;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Max Points on a Line": {
            "solutions": [{
                "title": "Solution 1 – HashMap with GCD",
                "description": "Use HashMap to count points on same line.",
                "code": """public int maxPoints(int[][] points) {
    if (points.length <= 2) return points.length;
    
    int maxPoints = 0;
    for (int i = 0; i < points.length; i++) {
        Map<String, Integer> slopes = new HashMap<>();
        int duplicates = 0;
        
        for (int j = 0; j < points.length; j++) {
            if (i == j) continue;
            
            if (points[i][0] == points[j][0] && points[i][1] == points[j][1]) {
                duplicates++;
                continue;
            }
            
            int dx = points[j][0] - points[i][0];
            int dy = points[j][1] - points[i][1];
            int gcd = gcd(dx, dy);
            
            String slope = (dx / gcd) + "/" + (dy / gcd);
            slopes.put(slope, slopes.getOrDefault(slope, 0) + 1);
        }
        
        int maxSlope = 0;
        for (int count : slopes.values()) {
            maxSlope = Math.max(maxSlope, count);
        }
        
        maxPoints = Math.max(maxPoints, maxSlope + duplicates + 1);
    }
    
    return maxPoints;
}

private int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}""",
                "time": "O(n²)",
                "space": "O(n)"
            }]
        },
        "Palindrome Pairs": {
            "solutions": [{
                "title": "Solution 1 – HashMap",
                "description": "Use HashMap to find palindrome pairs.",
                "code": """public List<List<Integer>> palindromePairs(String[] words) {
    List<List<Integer>> result = new ArrayList<>();
    Map<String, Integer> map = new HashMap<>();
    
    for (int i = 0; i < words.length; i++) {
        map.put(words[i], i);
    }
    
    for (int i = 0; i < words.length; i++) {
        String word = words[i];
        
        // Check if reverse exists
        String reverse = new StringBuilder(word).reverse().toString();
        if (map.containsKey(reverse) && map.get(reverse) != i) {
            result.add(Arrays.asList(i, map.get(reverse)));
        }
        
        // Check for palindromes
        for (int j = 0; j < word.length(); j++) {
            if (isPalindrome(word, 0, j)) {
                String suffix = new StringBuilder(word.substring(j + 1)).reverse().toString();
                if (map.containsKey(suffix) && map.get(suffix) != i) {
                    result.add(Arrays.asList(map.get(suffix), i));
                }
            }
            
            if (isPalindrome(word, j, word.length() - 1)) {
                String prefix = new StringBuilder(word.substring(0, j)).reverse().toString();
                if (map.containsKey(prefix) && map.get(prefix) != i) {
                    result.add(Arrays.asList(i, map.get(prefix)));
                }
            }
        }
    }
    
    return result;
}

private boolean isPalindrome(String s, int start, int end) {
    while (start < end) {
        if (s.charAt(start++) != s.charAt(end--)) {
            return false;
        }
    }
    return true;
}""",
                "time": "O(n × k²)",
                "space": "O(n)"
            }]
        },
        "Kth Smallest Element": {
            "solutions": [{
                "title": "Solution 1 – QuickSelect",
                "description": "Use QuickSelect algorithm to find kth element.",
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
                "time": "O(n) average",
                "space": "O(1)"
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
    solutions = get_last_solutions()
    
    problems_to_update = [
        ("construct-binary-tree-preorder-inorder.html", "Construct Binary Tree from Preorder and Inorder"),
        ("count-and-say.html", "Count and Say"),
        ("letter-combinations-phone-number.html", "Letter Combinations of Phone Number"),
        ("recover-binary-search-tree.html", "Recover Binary Search Tree"),
        ("generate-parentheses.html", "Generate Parentheses"),
        ("largest-rectangle-histogram.html", "Largest Rectangle in Histogram"),
        ("swap-nodes-pairs.html", "Swap Nodes in Pairs"),
        ("max-points-line.html", "Max Points on a Line"),
        ("palindrome-pairs.html", "Palindrome Pairs"),
        ("kth-smallest-largest.html", "Kth Smallest Element")
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