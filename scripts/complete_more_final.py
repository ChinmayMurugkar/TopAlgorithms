#!/usr/bin/env python3
import os
import re

def get_more_final_solutions():
    """Get solutions for another batch of problems."""
    return {
        "Minimum Operations to Make Array Distinct": {
            "solutions": [{
                "title": "Solution 1 – Sorting and Two Pointers",
                "description": "Sort array and use two pointers to find minimum operations.",
                "code": """public int minOperations(int[] nums) {
    Arrays.sort(nums);
    int operations = 0;
    
    for (int i = 1; i < nums.length; i++) {
        if (nums[i] <= nums[i - 1]) {
            operations += nums[i - 1] - nums[i] + 1;
            nums[i] = nums[i - 1] + 1;
        }
    }
    
    return operations;
}""",
                "time": "O(n log n)",
                "space": "O(1)"
            }]
        },
        "Substring with Concatenation of All Words": {
            "solutions": [{
                "title": "Solution 1 – Sliding Window",
                "description": "Use sliding window with HashMap to find concatenated substrings.",
                "code": """public List<Integer> findSubstring(String s, String[] words) {
    List<Integer> result = new ArrayList<>();
    if (s == null || words == null || words.length == 0) return result;
    
    int wordLength = words[0].length();
    int totalLength = wordLength * words.length;
    
    Map<String, Integer> wordCount = new HashMap<>();
    for (String word : words) {
        wordCount.put(word, wordCount.getOrDefault(word, 0) + 1);
    }
    
    for (int i = 0; i <= s.length() - totalLength; i++) {
        Map<String, Integer> seen = new HashMap<>();
        int j = 0;
        
        while (j < words.length) {
            String word = s.substring(i + j * wordLength, i + (j + 1) * wordLength);
            if (!wordCount.containsKey(word)) break;
            
            seen.put(word, seen.getOrDefault(word, 0) + 1);
            if (seen.get(word) > wordCount.get(word)) break;
            j++;
        }
        
        if (j == words.length) {
            result.add(i);
        }
    }
    
    return result;
}""",
                "time": "O(n × m × k)",
                "space": "O(m)"
            }]
        },
        "Populating Next Right Pointers II": {
            "solutions": [{
                "title": "Solution 1 – Level Order Traversal",
                "description": "Use level order traversal to connect nodes.",
                "code": """public Node connect(Node root) {
    if (root == null) return null;
    
    Node levelStart = root;
    while (levelStart != null) {
        Node curr = levelStart;
        Node nextLevelStart = null;
        Node prev = null;
        
        while (curr != null) {
            if (curr.left != null) {
                if (prev == null) {
                    nextLevelStart = curr.left;
                } else {
                    prev.next = curr.left;
                }
                prev = curr.left;
            }
            
            if (curr.right != null) {
                if (prev == null) {
                    nextLevelStart = curr.right;
                } else {
                    prev.next = curr.right;
                }
                prev = curr.right;
            }
            
            curr = curr.next;
        }
        
        levelStart = nextLevelStart;
    }
    
    return root;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Lowest Common Ancestor of BST": {
            "solutions": [{
                "title": "Solution 1 – BST Property",
                "description": "Use BST property to find LCA.",
                "code": """public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null) return null;
    
    if (p.val < root.val && q.val < root.val) {
        return lowestCommonAncestor(root.left, p, q);
    }
    
    if (p.val > root.val && q.val > root.val) {
        return lowestCommonAncestor(root.right, p, q);
    }
    
    return root;
}""",
                "time": "O(h)",
                "space": "O(h)"
            }]
        },
        "Find Peak Element": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Use binary search to find peak element.",
                "code": """public int findPeakElement(int[] nums) {
    int left = 0, right = nums.length - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] > nums[mid + 1]) {
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
        "ZigZag Conversion": {
            "solutions": [{
                "title": "Solution 1 – Row by Row",
                "description": "Build result row by row using zigzag pattern.",
                "code": """public String convert(String s, int numRows) {
    if (numRows == 1) return s;
    
    StringBuilder[] rows = new StringBuilder[numRows];
    for (int i = 0; i < numRows; i++) {
        rows[i] = new StringBuilder();
    }
    
    int currentRow = 0;
    boolean goingDown = false;
    
    for (char c : s.toCharArray()) {
        rows[currentRow].append(c);
        
        if (currentRow == 0 || currentRow == numRows - 1) {
            goingDown = !goingDown;
        }
        
        currentRow += goingDown ? 1 : -1;
    }
    
    StringBuilder result = new StringBuilder();
    for (StringBuilder row : rows) {
        result.append(row);
    }
    
    return result.toString();
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Fraction to Recurring Decimal": {
            "solutions": [{
                "title": "Solution 1 – Long Division",
                "description": "Use long division algorithm with HashMap to detect cycles.",
                "code": """public String fractionToDecimal(int numerator, int denominator) {
    if (numerator == 0) return "0";
    
    StringBuilder result = new StringBuilder();
    if ((numerator < 0) ^ (denominator < 0)) {
        result.append("-");
    }
    
    long num = Math.abs((long) numerator);
    long den = Math.abs((long) denominator);
    
    result.append(num / den);
    long remainder = num % den;
    
    if (remainder == 0) {
        return result.toString();
    }
    
    result.append(".");
    Map<Long, Integer> map = new HashMap<>();
    
    while (remainder != 0) {
        if (map.containsKey(remainder)) {
            result.insert(map.get(remainder), "(");
            result.append(")");
            break;
        }
        
        map.put(remainder, result.length());
        remainder *= 10;
        result.append(remainder / den);
        remainder %= den;
    }
    
    return result.toString();
}""",
                "time": "O(log n)",
                "space": "O(log n)"
            }]
        },
        "Minimum Path Sum": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to find minimum path sum.",
                "code": """public int minPathSum(int[][] grid) {
    int m = grid.length, n = grid[0].length;
    
    for (int i = 1; i < m; i++) {
        grid[i][0] += grid[i - 1][0];
    }
    
    for (int j = 1; j < n; j++) {
        grid[0][j] += grid[0][j - 1];
    }
    
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
        }
    }
    
    return grid[m - 1][n - 1];
}""",
                "time": "O(m × n)",
                "space": "O(1)"
            }]
        },
        "Gas Station": {
            "solutions": [{
                "title": "Solution 1 – Greedy",
                "description": "Use greedy approach to find starting gas station.",
                "code": """public int canCompleteCircuit(int[] gas, int[] cost) {
    int totalGas = 0, totalCost = 0;
    int start = 0, currentGas = 0;
    
    for (int i = 0; i < gas.length; i++) {
        totalGas += gas[i];
        totalCost += cost[i];
        currentGas += gas[i] - cost[i];
        
        if (currentGas < 0) {
            start = i + 1;
            currentGas = 0;
        }
    }
    
    return totalGas >= totalCost ? start : -1;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Word Search II": {
            "solutions": [{
                "title": "Solution 1 – Trie + DFS",
                "description": "Use Trie and DFS to find all words.",
                "code": """public List<String> findWords(char[][] board, String[] words) {
    List<String> result = new ArrayList<>();
    TrieNode root = buildTrie(words);
    
    for (int i = 0; i < board.length; i++) {
        for (int j = 0; j < board[0].length; j++) {
            dfs(board, i, j, root, result);
        }
    }
    
    return result;
}

private void dfs(char[][] board, int i, int j, TrieNode node, List<String> result) {
    char c = board[i][j];
    if (c == '#' || node.children[c - 'a'] == null) return;
    
    node = node.children[c - 'a'];
    if (node.word != null) {
        result.add(node.word);
        node.word = null; // Avoid duplicates
    }
    
    board[i][j] = '#';
    if (i > 0) dfs(board, i - 1, j, node, result);
    if (j > 0) dfs(board, i, j - 1, node, result);
    if (i < board.length - 1) dfs(board, i + 1, j, node, result);
    if (j < board[0].length - 1) dfs(board, i, j + 1, node, result);
    board[i][j] = c;
}

private TrieNode buildTrie(String[] words) {
    TrieNode root = new TrieNode();
    for (String word : words) {
        TrieNode node = root;
        for (char c : word.toCharArray()) {
            if (node.children[c - 'a'] == null) {
                node.children[c - 'a'] = new TrieNode();
            }
            node = node.children[c - 'a'];
        }
        node.word = word;
    }
    return root;
}

class TrieNode {
    TrieNode[] children = new TrieNode[26];
    String word;
}""",
                "time": "O(m × n × 4^L)",
                "space": "O(k × L)"
            }]
        },
        "Word Pattern": {
            "solutions": [{
                "title": "Solution 1 – HashMap",
                "description": "Use HashMap to check pattern matching.",
                "code": """public boolean wordPattern(String pattern, String s) {
    String[] words = s.split(" ");
    if (pattern.length() != words.length) return false;
    
    Map<Character, String> charToWord = new HashMap<>();
    Map<String, Character> wordToChar = new HashMap<>();
    
    for (int i = 0; i < pattern.length(); i++) {
        char c = pattern.charAt(i);
        String word = words[i];
        
        if (charToWord.containsKey(c)) {
            if (!charToWord.get(c).equals(word)) return false;
        } else {
            if (wordToChar.containsKey(word)) return false;
            charToWord.put(c, word);
            wordToChar.put(word, c);
        }
    }
    
    return true;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Remove Duplicates from Sorted List": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to remove duplicates.",
                "code": """public ListNode deleteDuplicates(ListNode head) {
    if (head == null || head.next == null) return head;
    
    ListNode current = head;
    while (current.next != null) {
        if (current.val == current.next.val) {
            current.next = current.next.next;
        } else {
            current = current.next;
        }
    }
    
    return head;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Maximum Product of Word Lengths": {
            "solutions": [{
                "title": "Solution 1 – Bit Manipulation",
                "description": "Use bit manipulation to check character overlap.",
                "code": """public int maxProduct(String[] words) {
    int n = words.length;
    int[] masks = new int[n];
    
    for (int i = 0; i < n; i++) {
        for (char c : words[i].toCharArray()) {
            masks[i] |= 1 << (c - 'a');
        }
    }
    
    int maxProduct = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((masks[i] & masks[j]) == 0) {
                maxProduct = Math.max(maxProduct, words[i].length() * words[j].length());
            }
        }
    }
    
    return maxProduct;
}""",
                "time": "O(n² + L)",
                "space": "O(n)"
            }]
        },
        "Bitwise AND of Numbers Range": {
            "solutions": [{
                "title": "Solution 1 – Bit Manipulation",
                "description": "Find common prefix of binary representations.",
                "code": """public int rangeBitwiseAnd(int left, int right) {
    int shift = 0;
    
    while (left < right) {
        left >>= 1;
        right >>= 1;
        shift++;
    }
    
    return left << shift;
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Factor Combinations": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to find all factor combinations.",
                "code": """public List<List<Integer>> getFactors(int n) {
    List<List<Integer>> result = new ArrayList<>();
    backtrack(n, 2, new ArrayList<>(), result);
    return result;
}

private void backtrack(int n, int start, List<Integer> current, List<List<Integer>> result) {
    if (n == 1) {
        if (current.size() > 1) {
            result.add(new ArrayList<>(current));
        }
        return;
    }
    
    for (int i = start; i <= n; i++) {
        if (n % i == 0) {
            current.add(i);
            backtrack(n / i, i, current, result);
            current.remove(current.size() - 1);
        }
    }
}""",
                "time": "O(n^(log n))",
                "space": "O(log n)"
            }]
        },
        "3Sum Closest": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to find closest sum.",
                "code": """public int threeSumClosest(int[] nums, int target) {
    Arrays.sort(nums);
    int closest = nums[0] + nums[1] + nums[2];
    
    for (int i = 0; i < nums.length - 2; i++) {
        int left = i + 1, right = nums.length - 1;
        
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            
            if (Math.abs(sum - target) < Math.abs(closest - target)) {
                closest = sum;
            }
            
            if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
    }
    
    return closest;
}""",
                "time": "O(n²)",
                "space": "O(1)"
            }]
        },
        "Vertical Order Traversal": {
            "solutions": [{
                "title": "Solution 1 – DFS with Sorting",
                "description": "Use DFS to collect nodes and sort by position.",
                "code": """public List<List<Integer>> verticalTraversal(TreeNode root) {
    List<int[]> nodes = new ArrayList<>();
    dfs(root, 0, 0, nodes);
    
    Collections.sort(nodes, (a, b) -> {
        if (a[1] != b[1]) return a[1] - b[1]; // column
        if (a[0] != b[0]) return a[0] - b[0]; // row
        return a[2] - b[2]; // value
    });
    
    List<List<Integer>> result = new ArrayList<>();
    int currentCol = nodes.get(0)[1];
    List<Integer> currentList = new ArrayList<>();
    
    for (int[] node : nodes) {
        if (node[1] != currentCol) {
            result.add(currentList);
            currentList = new ArrayList<>();
            currentCol = node[1];
        }
        currentList.add(node[2]);
    }
    result.add(currentList);
    
    return result;
}

private void dfs(TreeNode node, int row, int col, List<int[]> nodes) {
    if (node == null) return;
    
    nodes.add(new int[]{row, col, node.val});
    dfs(node.left, row + 1, col - 1, nodes);
    dfs(node.right, row + 1, col + 1, nodes);
}""",
                "time": "O(n log n)",
                "space": "O(n)"
            }]
        },
        "Length of Last Word": {
            "solutions": [{
                "title": "Solution 1 – String Traversal",
                "description": "Traverse string from end to find last word length.",
                "code": """public int lengthOfLastWord(String s) {
    int length = 0;
    int i = s.length() - 1;
    
    // Skip trailing spaces
    while (i >= 0 && s.charAt(i) == ' ') {
        i--;
    }
    
    // Count characters of last word
    while (i >= 0 && s.charAt(i) != ' ') {
        length++;
        i--;
    }
    
    return length;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Bulls and Cows": {
            "solutions": [{
                "title": "Solution 1 – HashMap",
                "description": "Use HashMap to count bulls and cows.",
                "code": """public String getHint(String secret, String guess) {
    int bulls = 0, cows = 0;
    int[] secretCount = new int[10];
    int[] guessCount = new int[10];
    
    for (int i = 0; i < secret.length(); i++) {
        if (secret.charAt(i) == guess.charAt(i)) {
            bulls++;
        } else {
            secretCount[secret.charAt(i) - '0']++;
            guessCount[guess.charAt(i) - '0']++;
        }
    }
    
    for (int i = 0; i < 10; i++) {
        cows += Math.min(secretCount[i], guessCount[i]);
    }
    
    return bulls + "A" + cows + "B";
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Increasing Triplet Subsequence": {
            "solutions": [{
                "title": "Solution 1 – Two Variables",
                "description": "Use two variables to track smallest and second smallest.",
                "code": """public boolean increasingTriplet(int[] nums) {
    int first = Integer.MAX_VALUE;
    int second = Integer.MAX_VALUE;
    
    for (int num : nums) {
        if (num <= first) {
            first = num;
        } else if (num <= second) {
            second = num;
        } else {
            return true;
        }
    }
    
    return false;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Basic Calculator": {
            "solutions": [{
                "title": "Solution 1 – Stack",
                "description": "Use stack to handle parentheses and operations.",
                "code": """public int calculate(String s) {
    Stack<Integer> stack = new Stack<>();
    int result = 0;
    int number = 0;
    int sign = 1;
    
    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        
        if (Character.isDigit(c)) {
            number = number * 10 + (c - '0');
        } else if (c == '+') {
            result += sign * number;
            number = 0;
            sign = 1;
        } else if (c == '-') {
            result += sign * number;
            number = 0;
            sign = -1;
        } else if (c == '(') {
            stack.push(result);
            stack.push(sign);
            result = 0;
            sign = 1;
        } else if (c == ')') {
            result += sign * number;
            number = 0;
            result *= stack.pop();
            result += stack.pop();
        }
    }
    
    if (number != 0) {
        result += sign * number;
    }
    
    return result;
}""",
                "time": "O(n)",
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
    solutions = get_more_final_solutions()
    
    problems_to_update = [
        ("minimum-operations-distinct.html", "Minimum Operations to Make Array Distinct"),
        ("substring-concatenation-words.html", "Substring with Concatenation of All Words"),
        ("populating-next-right-pointers-ii.html", "Populating Next Right Pointers II"),
        ("lowest-common-ancestor-bst.html", "Lowest Common Ancestor of BST"),
        ("find-peak-element.html", "Find Peak Element"),
        ("zigzag-conversion.html", "ZigZag Conversion"),
        ("fraction-recurring-decimal.html", "Fraction to Recurring Decimal"),
        ("minimum-path-sum.html", "Minimum Path Sum"),
        ("gas-station.html", "Gas Station"),
        ("word-search-ii.html", "Word Search II"),
        ("word-pattern.html", "Word Pattern"),
        ("remove-duplicates-sorted-list.html", "Remove Duplicates from Sorted List"),
        ("maximum-product-word-lengths.html", "Maximum Product of Word Lengths"),
        ("bitwise-and-numbers-range.html", "Bitwise AND of Numbers Range"),
        ("factor-combinations.html", "Factor Combinations"),
        ("3sum-closest.html", "3Sum Closest"),
        ("vertical-order.html", "Vertical Order Traversal"),
        ("length-last-word.html", "Length of Last Word"),
        ("bulls-and-cows.html", "Bulls and Cows"),
        ("increasing-triplet-subsequence.html", "Increasing Triplet Subsequence"),
        ("basic-calculator.html", "Basic Calculator")
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