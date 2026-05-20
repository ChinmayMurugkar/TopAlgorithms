#!/usr/bin/env python3
"""
Script to:
1. Fix duplicate solutions in majority-element.json and count-primes.json
2. Add new interviewer-accepted solutions to various JSON files
"""

import json
import os

SOLUTIONS_DIR = "/Users/chinmay/Desktop/Chinmay/workspace/TopAlgorithms/solutions"
MARKER = " [✓ Interviewer Accepted]"

updated_files = []


def load_json(filename):
    path = os.path.join(SOLUTIONS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filename, data):
    path = os.path.join(SOLUTIONS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if filename not in updated_files:
        updated_files.append(filename)


def fix_duplicate(filename):
    data = load_json(filename)
    solutions = data["solutions"]
    if len(solutions) >= 2 and solutions[0]["title"] == solutions[1]["title"]:
        data["solutions"] = [solutions[0]]
        save_json(filename, data)
        print(f"  Fixed duplicate in {filename}")


def add_solutions(filename, new_solutions):
    data = load_json(filename)
    data["solutions"].extend(new_solutions)
    save_json(filename, data)
    print(f"  Updated {filename} (+{len(new_solutions)} solution(s))")


# ─────────────────────────────────────────────
# 1. Fix duplicates
# ─────────────────────────────────────────────
print("=== Fixing duplicates ===")
fix_duplicate("majority-element.json")
fix_duplicate("count-primes.json")

# ─────────────────────────────────────────────
# 2. longest-valid-parentheses.json — 2 solutions
# ─────────────────────────────────────────────
print("\n=== Adding solutions ===")

add_solutions("longest-valid-parentheses.json", [
    {
        "title": f"Solution 2 – Two Pointers{MARKER}",
        "description": (
            "Two passes over the string — left-to-right and right-to-left — each using O(1) space. "
            "In the first pass, maintain open and close counters. When close > open, reset both to 0; "
            "when they are equal, update max with 2*close. The second pass mirrors this with reversed "
            "roles to catch cases where the string starts with unmatched '('."
        ),
        "code": (
            "public int longestValidParentheses(String s) {\n"
            "    int open = 0, close = 0, maxLen = 0;\n"
            "    // Left to right\n"
            "    for (int i = 0; i < s.length(); i++) {\n"
            "        if (s.charAt(i) == '(') open++;\n"
            "        else close++;\n"
            "        if (open == close) maxLen = Math.max(maxLen, 2 * close);\n"
            "        else if (close > open) { open = 0; close = 0; }\n"
            "    }\n"
            "    open = 0; close = 0;\n"
            "    // Right to left\n"
            "    for (int i = s.length() - 1; i >= 0; i--) {\n"
            "        if (s.charAt(i) == '(') open++;\n"
            "        else close++;\n"
            "        if (open == close) maxLen = Math.max(maxLen, 2 * open);\n"
            "        else if (open > close) { open = 0; close = 0; }\n"
            "    }\n"
            "    return maxLen;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def longestValidParentheses(self, s: str) -> int:\n"
            "        open_c = close_c = max_len = 0\n"
            "        for c in s:\n"
            "            if c == '(':\n"
            "                open_c += 1\n"
            "            else:\n"
            "                close_c += 1\n"
            "            if open_c == close_c:\n"
            "                max_len = max(max_len, 2 * close_c)\n"
            "            elif close_c > open_c:\n"
            "                open_c = close_c = 0\n"
            "        open_c = close_c = 0\n"
            "        for c in reversed(s):\n"
            "            if c == '(':\n"
            "                open_c += 1\n"
            "            else:\n"
            "                close_c += 1\n"
            "            if open_c == close_c:\n"
            "                max_len = max(max_len, 2 * open_c)\n"
            "            elif open_c > close_c:\n"
            "                open_c = close_c = 0\n"
            "        return max_len"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(1)"}
    },
    {
        "title": f"Solution 3 – Dynamic Programming{MARKER}",
        "description": (
            "Build a dp array where dp[i] = length of longest valid parentheses substring ending at index i. "
            "For each ')' at index i: if s[i-1]=='(' then dp[i] = dp[i-2] + 2 (direct match). "
            "Otherwise if s[i-1]==')' and the matching '(' for the inner valid substring is at j = i - dp[i-1] - 1: "
            "if s[j]=='(' then dp[i] = dp[i-1] + 2 + (dp[j-1] if j>=1 else 0). "
            "Track maximum dp value."
        ),
        "code": (
            "public int longestValidParentheses(String s) {\n"
            "    int n = s.length();\n"
            "    if (n == 0) return 0;\n"
            "    int[] dp = new int[n];\n"
            "    int maxLen = 0;\n"
            "    for (int i = 1; i < n; i++) {\n"
            "        if (s.charAt(i) == ')') {\n"
            "            if (s.charAt(i - 1) == '(') {\n"
            "                dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;\n"
            "            } else {\n"
            "                int j = i - dp[i - 1] - 1;\n"
            "                if (j >= 0 && s.charAt(j) == '(') {\n"
            "                    dp[i] = dp[i - 1] + 2 + (j >= 1 ? dp[j - 1] : 0);\n"
            "                }\n"
            "            }\n"
            "            maxLen = Math.max(maxLen, dp[i]);\n"
            "        }\n"
            "    }\n"
            "    return maxLen;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def longestValidParentheses(self, s: str) -> int:\n"
            "        n = len(s)\n"
            "        if n == 0:\n"
            "            return 0\n"
            "        dp = [0] * n\n"
            "        max_len = 0\n"
            "        for i in range(1, n):\n"
            "            if s[i] == ')':\n"
            "                if s[i - 1] == '(':\n"
            "                    dp[i] = (dp[i - 2] if i >= 2 else 0) + 2\n"
            "                else:\n"
            "                    j = i - dp[i - 1] - 1\n"
            "                    if j >= 0 and s[j] == '(':\n"
            "                        dp[i] = dp[i - 1] + 2 + (dp[j - 1] if j >= 1 else 0)\n"
            "                max_len = max(max_len, dp[i])\n"
            "        return max_len"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# integer-break.json
# ─────────────────────────────────────────────
add_solutions("integer-break.json", [
    {
        "title": f"Solution 2 – Greedy / Math{MARKER}",
        "description": (
            "Any factor >= 5 is better split into 3s and 2s because 3*(n-3) > n for n>=5. "
            "So the optimal strategy is to use as many 3s as possible. "
            "Handle the remainder: if remainder==0, return 3^q; if remainder==1, use one fewer 3 and multiply by 4 (3+1 < 2+2); "
            "if remainder==2, multiply by 2. Special case n==2 returns 1, n==3 returns 2."
        ),
        "code": (
            "public int integerBreak(int n) {\n"
            "    if (n == 2) return 1;\n"
            "    if (n == 3) return 2;\n"
            "    int product = 1;\n"
            "    while (n > 4) {\n"
            "        product *= 3;\n"
            "        n -= 3;\n"
            "    }\n"
            "    // n is now 2, 3, or 4 — all are optimal to keep as-is\n"
            "    return product * n;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def integerBreak(self, n: int) -> int:\n"
            "        if n == 2:\n"
            "            return 1\n"
            "        if n == 3:\n"
            "            return 2\n"
            "        product = 1\n"
            "        while n > 4:\n"
            "            product *= 3\n"
            "            n -= 3\n"
            "        return product * n"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(1)"}
    }
])

# ─────────────────────────────────────────────
# kth-smallest-bst.json
# ─────────────────────────────────────────────
add_solutions("kth-smallest-bst.json", [
    {
        "title": f"Solution 2 – Iterative Inorder with Stack{MARKER}",
        "description": (
            "Use an explicit stack to simulate inorder traversal without recursion. "
            "Push all left children of the current node onto the stack. "
            "Pop a node, decrement k; if k reaches 0, that node's value is the answer. "
            "Then move to the right child and repeat. "
            "O(H+k) time where H is tree height, O(H) space for the stack."
        ),
        "code": (
            "public int kthSmallest(TreeNode root, int k) {\n"
            "    Deque<TreeNode> stack = new ArrayDeque<>();\n"
            "    TreeNode curr = root;\n"
            "    while (curr != null || !stack.isEmpty()) {\n"
            "        while (curr != null) {\n"
            "            stack.push(curr);\n"
            "            curr = curr.left;\n"
            "        }\n"
            "        curr = stack.pop();\n"
            "        if (--k == 0) return curr.val;\n"
            "        curr = curr.right;\n"
            "    }\n"
            "    return -1; // k > number of nodes\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def kthSmallest(self, root, k: int) -> int:\n"
            "        stack = []\n"
            "        curr = root\n"
            "        while curr or stack:\n"
            "            while curr:\n"
            "                stack.append(curr)\n"
            "                curr = curr.left\n"
            "            curr = stack.pop()\n"
            "            k -= 1\n"
            "            if k == 0:\n"
            "                return curr.val\n"
            "            curr = curr.right\n"
            "        return -1"
        ),
        "language": "java",
        "complexity": {"time": "O(H + k)", "space": "O(H)"}
    }
])

# ─────────────────────────────────────────────
# lowest-common-ancestor-bst.json
# ─────────────────────────────────────────────
add_solutions("lowest-common-ancestor-bst.json", [
    {
        "title": f"Solution 2 – Iterative{MARKER}",
        "description": (
            "Apply the same BST property logic as the recursive approach, but with an iterative while loop. "
            "If both p and q are less than root.val, move to the left subtree. "
            "If both are greater, move to the right subtree. "
            "Otherwise root is the LCA. O(H) time O(1) space — no call stack overhead."
        ),
        "code": (
            "public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {\n"
            "    TreeNode curr = root;\n"
            "    while (curr != null) {\n"
            "        if (p.val < curr.val && q.val < curr.val) {\n"
            "            curr = curr.left;\n"
            "        } else if (p.val > curr.val && q.val > curr.val) {\n"
            "            curr = curr.right;\n"
            "        } else {\n"
            "            return curr;\n"
            "        }\n"
            "    }\n"
            "    return null;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def lowestCommonAncestor(self, root, p, q):\n"
            "        curr = root\n"
            "        while curr:\n"
            "            if p.val < curr.val and q.val < curr.val:\n"
            "                curr = curr.left\n"
            "            elif p.val > curr.val and q.val > curr.val:\n"
            "                curr = curr.right\n"
            "            else:\n"
            "                return curr\n"
            "        return None"
        ),
        "language": "java",
        "complexity": {"time": "O(H)", "space": "O(1)"}
    }
])

# ─────────────────────────────────────────────
# lowest-common-ancestor.json
# ─────────────────────────────────────────────
add_solutions("lowest-common-ancestor.json", [
    {
        "title": f"Solution 2 – Parent Pointer + HashSet{MARKER}",
        "description": (
            "Use BFS/DFS to build a parent map for every node in the tree. "
            "Starting from node p, walk up the tree (using the parent map) adding each ancestor to a HashSet. "
            "Then starting from node q, walk up the tree — the first node already present in the ancestor set is the LCA. "
            "O(n) time O(n) space."
        ),
        "code": (
            "public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {\n"
            "    Map<TreeNode, TreeNode> parent = new HashMap<>();\n"
            "    Deque<TreeNode> stack = new ArrayDeque<>();\n"
            "    parent.put(root, null);\n"
            "    stack.push(root);\n"
            "    while (!parent.containsKey(p) || !parent.containsKey(q)) {\n"
            "        TreeNode node = stack.pop();\n"
            "        if (node.left != null) {\n"
            "            parent.put(node.left, node);\n"
            "            stack.push(node.left);\n"
            "        }\n"
            "        if (node.right != null) {\n"
            "            parent.put(node.right, node);\n"
            "            stack.push(node.right);\n"
            "        }\n"
            "    }\n"
            "    Set<TreeNode> ancestors = new HashSet<>();\n"
            "    while (p != null) {\n"
            "        ancestors.add(p);\n"
            "        p = parent.get(p);\n"
            "    }\n"
            "    while (!ancestors.contains(q)) {\n"
            "        q = parent.get(q);\n"
            "    }\n"
            "    return q;\n"
            "}"
        ),
        "code_python": (
            "from collections import deque\n"
            "\n"
            "class Solution:\n"
            "    def lowestCommonAncestor(self, root, p, q):\n"
            "        parent = {root: None}\n"
            "        stack = deque([root])\n"
            "        while p not in parent or q not in parent:\n"
            "            node = stack.pop()\n"
            "            if node.left:\n"
            "                parent[node.left] = node\n"
            "                stack.append(node.left)\n"
            "            if node.right:\n"
            "                parent[node.right] = node\n"
            "                stack.append(node.right)\n"
            "        ancestors = set()\n"
            "        while p:\n"
            "            ancestors.add(p)\n"
            "            p = parent[p]\n"
            "        while q not in ancestors:\n"
            "            q = parent[q]\n"
            "        return q"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# clone-graph.json
# ─────────────────────────────────────────────
add_solutions("clone-graph.json", [
    {
        "title": f"Solution 2 – BFS with HashMap{MARKER}",
        "description": (
            "Use a queue for BFS and a HashMap to map original nodes to their clones. "
            "Start by cloning the source node and adding it to the queue and map. "
            "For each dequeued node, iterate its neighbors: if a neighbor hasn't been cloned yet, "
            "create its clone and enqueue it. Add the clone to the neighbor list of the current cloned node. "
            "O(V+E) time O(V) space."
        ),
        "code": (
            "public Node cloneGraph(Node node) {\n"
            "    if (node == null) return null;\n"
            "    Map<Node, Node> map = new HashMap<>();\n"
            "    Queue<Node> queue = new LinkedList<>();\n"
            "    map.put(node, new Node(node.val));\n"
            "    queue.offer(node);\n"
            "    while (!queue.isEmpty()) {\n"
            "        Node curr = queue.poll();\n"
            "        for (Node neighbor : curr.neighbors) {\n"
            "            if (!map.containsKey(neighbor)) {\n"
            "                map.put(neighbor, new Node(neighbor.val));\n"
            "                queue.offer(neighbor);\n"
            "            }\n"
            "            map.get(curr).neighbors.add(map.get(neighbor));\n"
            "        }\n"
            "    }\n"
            "    return map.get(node);\n"
            "}"
        ),
        "code_python": (
            "from collections import deque\n"
            "\n"
            "class Solution:\n"
            "    def cloneGraph(self, node):\n"
            "        if not node:\n"
            "            return None\n"
            "        clones = {node: Node(node.val)}\n"
            "        queue = deque([node])\n"
            "        while queue:\n"
            "            curr = queue.popleft()\n"
            "            for neighbor in curr.neighbors:\n"
            "                if neighbor not in clones:\n"
            "                    clones[neighbor] = Node(neighbor.val)\n"
            "                    queue.append(neighbor)\n"
            "                clones[curr].neighbors.append(clones[neighbor])\n"
            "        return clones[node]"
        ),
        "language": "java",
        "complexity": {"time": "O(V + E)", "space": "O(V)"}
    }
])

# ─────────────────────────────────────────────
# serialize-deserialize-binary-tree.json
# ─────────────────────────────────────────────
add_solutions("serialize-deserialize-binary-tree.json", [
    {
        "title": f"Solution 2 – BFS (Level Order){MARKER}",
        "description": (
            "Serialize: use a queue for level-order traversal, appending each node's value (or 'null') "
            "to a space-separated string. "
            "Deserialize: split the string by spaces, use an index pointer and a queue of parent nodes. "
            "For each parent node dequeued, assign its left child from data[i] and right child from data[i+1], "
            "enqueuing non-null children. O(n) time and space."
        ),
        "code": (
            "public class Codec {\n"
            "    public String serialize(TreeNode root) {\n"
            "        if (root == null) return \"\";\n"
            "        StringBuilder sb = new StringBuilder();\n"
            "        Queue<TreeNode> queue = new LinkedList<>();\n"
            "        queue.offer(root);\n"
            "        while (!queue.isEmpty()) {\n"
            "            TreeNode node = queue.poll();\n"
            "            if (node == null) {\n"
            "                sb.append(\"null \");\n"
            "            } else {\n"
            "                sb.append(node.val).append(' ');\n"
            "                queue.offer(node.left);\n"
            "                queue.offer(node.right);\n"
            "            }\n"
            "        }\n"
            "        return sb.toString().trim();\n"
            "    }\n"
            "\n"
            "    public TreeNode deserialize(String data) {\n"
            "        if (data == null || data.isEmpty()) return null;\n"
            "        String[] parts = data.split(\" \");\n"
            "        TreeNode root = new TreeNode(Integer.parseInt(parts[0]));\n"
            "        Queue<TreeNode> queue = new LinkedList<>();\n"
            "        queue.offer(root);\n"
            "        int i = 1;\n"
            "        while (!queue.isEmpty() && i < parts.length) {\n"
            "            TreeNode node = queue.poll();\n"
            "            if (!parts[i].equals(\"null\")) {\n"
            "                node.left = new TreeNode(Integer.parseInt(parts[i]));\n"
            "                queue.offer(node.left);\n"
            "            }\n"
            "            i++;\n"
            "            if (i < parts.length && !parts[i].equals(\"null\")) {\n"
            "                node.right = new TreeNode(Integer.parseInt(parts[i]));\n"
            "                queue.offer(node.right);\n"
            "            }\n"
            "            i++;\n"
            "        }\n"
            "        return root;\n"
            "    }\n"
            "}"
        ),
        "code_python": (
            "from collections import deque\n"
            "\n"
            "class Codec:\n"
            "    def serialize(self, root) -> str:\n"
            "        if not root:\n"
            "            return ''\n"
            "        result = []\n"
            "        queue = deque([root])\n"
            "        while queue:\n"
            "            node = queue.popleft()\n"
            "            if node is None:\n"
            "                result.append('null')\n"
            "            else:\n"
            "                result.append(str(node.val))\n"
            "                queue.append(node.left)\n"
            "                queue.append(node.right)\n"
            "        return ' '.join(result)\n"
            "\n"
            "    def deserialize(self, data: str):\n"
            "        if not data:\n"
            "            return None\n"
            "        parts = data.split()\n"
            "        root = TreeNode(int(parts[0]))\n"
            "        queue = deque([root])\n"
            "        i = 1\n"
            "        while queue and i < len(parts):\n"
            "            node = queue.popleft()\n"
            "            if parts[i] != 'null':\n"
            "                node.left = TreeNode(int(parts[i]))\n"
            "                queue.append(node.left)\n"
            "            i += 1\n"
            "            if i < len(parts) and parts[i] != 'null':\n"
            "                node.right = TreeNode(int(parts[i]))\n"
            "                queue.append(node.right)\n"
            "            i += 1\n"
            "        return root"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# same-tree.json
# ─────────────────────────────────────────────
add_solutions("same-tree.json", [
    {
        "title": f"Solution 2 – Iterative with Stack{MARKER}",
        "description": (
            "Use a stack containing pairs of corresponding nodes from both trees. "
            "For each pair popped: if both are null, continue; if only one is null or values differ, return false. "
            "Otherwise push both pairs of children (left-left and right-right). "
            "O(n) time O(n) space."
        ),
        "code": (
            "public boolean isSameTree(TreeNode p, TreeNode q) {\n"
            "    Deque<TreeNode[]> stack = new ArrayDeque<>();\n"
            "    stack.push(new TreeNode[]{p, q});\n"
            "    while (!stack.isEmpty()) {\n"
            "        TreeNode[] pair = stack.pop();\n"
            "        TreeNode a = pair[0], b = pair[1];\n"
            "        if (a == null && b == null) continue;\n"
            "        if (a == null || b == null || a.val != b.val) return false;\n"
            "        stack.push(new TreeNode[]{a.left, b.left});\n"
            "        stack.push(new TreeNode[]{a.right, b.right});\n"
            "    }\n"
            "    return true;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def isSameTree(self, p, q) -> bool:\n"
            "        stack = [(p, q)]\n"
            "        while stack:\n"
            "            a, b = stack.pop()\n"
            "            if a is None and b is None:\n"
            "                continue\n"
            "            if a is None or b is None or a.val != b.val:\n"
            "                return False\n"
            "            stack.append((a.left, b.left))\n"
            "            stack.append((a.right, b.right))\n"
            "        return True"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# edit-distance.json
# ─────────────────────────────────────────────
add_solutions("edit-distance.json", [
    {
        "title": f"Solution 2 – Space-Optimized DP{MARKER}",
        "description": (
            "The standard 2D DP table only needs the previous row to compute the current row, "
            "so we can reduce space to O(min(m,n)) by keeping just two 1D arrays (prev and curr). "
            "Ensure the shorter string is placed along the column axis to minimize the array size. "
            "Same O(m*n) time as the full DP, but O(min(m,n)) space."
        ),
        "code": (
            "public int minDistance(String word1, String word2) {\n"
            "    int m = word1.length(), n = word2.length();\n"
            "    if (m < n) return minDistance(word2, word1); // ensure n <= m\n"
            "    int[] prev = new int[n + 1];\n"
            "    for (int j = 0; j <= n; j++) prev[j] = j;\n"
            "    for (int i = 1; i <= m; i++) {\n"
            "        int[] curr = new int[n + 1];\n"
            "        curr[0] = i;\n"
            "        for (int j = 1; j <= n; j++) {\n"
            "            if (word1.charAt(i - 1) == word2.charAt(j - 1)) {\n"
            "                curr[j] = prev[j - 1];\n"
            "            } else {\n"
            "                curr[j] = 1 + Math.min(prev[j - 1], Math.min(prev[j], curr[j - 1]));\n"
            "            }\n"
            "        }\n"
            "        prev = curr;\n"
            "    }\n"
            "    return prev[n];\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def minDistance(self, word1: str, word2: str) -> int:\n"
            "        m, n = len(word1), len(word2)\n"
            "        if m < n:\n"
            "            word1, word2 = word2, word1\n"
            "            m, n = n, m\n"
            "        prev = list(range(n + 1))\n"
            "        for i in range(1, m + 1):\n"
            "            curr = [i] + [0] * n\n"
            "            for j in range(1, n + 1):\n"
            "                if word1[i - 1] == word2[j - 1]:\n"
            "                    curr[j] = prev[j - 1]\n"
            "                else:\n"
            "                    curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])\n"
            "            prev = curr\n"
            "        return prev[n]"
        ),
        "language": "java",
        "complexity": {"time": "O(m * n)", "space": "O(min(m, n))"}
    }
])

# ─────────────────────────────────────────────
# word-break.json
# ─────────────────────────────────────────────
add_solutions("word-break.json", [
    {
        "title": f"Solution 2 – BFS{MARKER}",
        "description": (
            "Model the problem as a graph where each index in the string is a node and index 0 is the source. "
            "Use BFS: from each reachable index i, try every word in the dictionary — "
            "if s[i:i+len(word)] == word, add i+len(word) to the queue (if not visited). "
            "Return true if index n is ever reached. "
            "O(n * |dict| * max_word_len) time, O(n) space for visited set."
        ),
        "code": (
            "public boolean wordBreak(String s, List<String> wordDict) {\n"
            "    Set<String> wordSet = new HashSet<>(wordDict);\n"
            "    boolean[] visited = new boolean[s.length() + 1];\n"
            "    Queue<Integer> queue = new LinkedList<>();\n"
            "    queue.offer(0);\n"
            "    visited[0] = true;\n"
            "    while (!queue.isEmpty()) {\n"
            "        int start = queue.poll();\n"
            "        for (int end = start + 1; end <= s.length(); end++) {\n"
            "            if (visited[end]) continue;\n"
            "            if (wordSet.contains(s.substring(start, end))) {\n"
            "                if (end == s.length()) return true;\n"
            "                visited[end] = true;\n"
            "                queue.offer(end);\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    return false;\n"
            "}"
        ),
        "code_python": (
            "from collections import deque\n"
            "\n"
            "class Solution:\n"
            "    def wordBreak(self, s: str, wordDict) -> bool:\n"
            "        word_set = set(wordDict)\n"
            "        n = len(s)\n"
            "        visited = [False] * (n + 1)\n"
            "        queue = deque([0])\n"
            "        visited[0] = True\n"
            "        while queue:\n"
            "            start = queue.popleft()\n"
            "            for end in range(start + 1, n + 1):\n"
            "                if visited[end]:\n"
            "                    continue\n"
            "                if s[start:end] in word_set:\n"
            "                    if end == n:\n"
            "                        return True\n"
            "                    visited[end] = True\n"
            "                    queue.append(end)\n"
            "        return False"
        ),
        "language": "java",
        "complexity": {"time": "O(n * |dict| * max_word_len)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# subsets.json
# ─────────────────────────────────────────────
add_solutions("subsets.json", [
    {
        "title": f"Solution 2 – Iterative Bit Manipulation{MARKER}",
        "description": (
            "For an array of n elements there are exactly 2^n subsets. "
            "For each integer i from 0 to 2^n - 1, treat each bit of i as a flag: "
            "if bit j is set, include nums[j] in the current subset. "
            "This generates all subsets without recursion. "
            "O(n * 2^n) time O(n * 2^n) space."
        ),
        "code": (
            "public List<List<Integer>> subsets(int[] nums) {\n"
            "    int n = nums.length;\n"
            "    int total = 1 << n; // 2^n\n"
            "    List<List<Integer>> result = new ArrayList<>();\n"
            "    for (int mask = 0; mask < total; mask++) {\n"
            "        List<Integer> subset = new ArrayList<>();\n"
            "        for (int j = 0; j < n; j++) {\n"
            "            if ((mask & (1 << j)) != 0) {\n"
            "                subset.add(nums[j]);\n"
            "            }\n"
            "        }\n"
            "        result.add(subset);\n"
            "    }\n"
            "    return result;\n"
            "}"
        ),
        "code_python": (
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def subsets(self, nums: List[int]) -> List[List[int]]:\n"
            "        n = len(nums)\n"
            "        result = []\n"
            "        for mask in range(1 << n):\n"
            "            subset = [nums[j] for j in range(n) if mask & (1 << j)]\n"
            "            result.append(subset)\n"
            "        return result"
        ),
        "language": "java",
        "complexity": {"time": "O(n * 2^n)", "space": "O(n * 2^n)"}
    }
])

# ─────────────────────────────────────────────
# knapsack-problem.json
# ─────────────────────────────────────────────
add_solutions("knapsack-problem.json", [
    {
        "title": f"Solution 2 – Space-Optimized 1D DP{MARKER}",
        "description": (
            "Instead of a 2D dp[n+1][W+1] table, use a single 1D array dp[W+1] where dp[w] represents "
            "the maximum value achievable with capacity w. "
            "Iterate items; for each item iterate capacity from W down to item's weight — "
            "iterating downward prevents using the same item twice. "
            "O(n*W) time O(W) space."
        ),
        "code": (
            "public int knapsack(int W, int[] weights, int[] values) {\n"
            "    int n = weights.length;\n"
            "    int[] dp = new int[W + 1];\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        for (int w = W; w >= weights[i]; w--) {\n"
            "            dp[w] = Math.max(dp[w], dp[w - weights[i]] + values[i]);\n"
            "        }\n"
            "    }\n"
            "    return dp[W];\n"
            "}"
        ),
        "code_python": (
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def knapsack(self, W: int, weights: List[int], values: List[int]) -> int:\n"
            "        dp = [0] * (W + 1)\n"
            "        for w, v in zip(weights, values):\n"
            "            for cap in range(W, w - 1, -1):\n"
            "                dp[cap] = max(dp[cap], dp[cap - w] + v)\n"
            "        return dp[W]"
        ),
        "language": "java",
        "complexity": {"time": "O(n * W)", "space": "O(W)"}
    }
])

# ─────────────────────────────────────────────
# unique-paths-ii.json
# ─────────────────────────────────────────────
add_solutions("unique-paths-ii.json", [
    {
        "title": f"Solution 2 – Space-Optimized 1D DP{MARKER}",
        "description": (
            "Use a single 1D array dp of length n (number of columns). "
            "Initialize dp[0] = 1 if obstacleGrid[0][0] == 0, else 0. "
            "For each row, update left to right: if obstacle, set dp[j] = 0; "
            "otherwise dp[j] += dp[j-1]. This accumulates paths from above (dp[j] before update) "
            "and from the left (dp[j-1] after update). O(m*n) time O(n) space."
        ),
        "code": (
            "public int uniquePathsWithObstacles(int[][] obstacleGrid) {\n"
            "    int m = obstacleGrid.length, n = obstacleGrid[0].length;\n"
            "    int[] dp = new int[n];\n"
            "    dp[0] = obstacleGrid[0][0] == 0 ? 1 : 0;\n"
            "    for (int i = 0; i < m; i++) {\n"
            "        for (int j = 0; j < n; j++) {\n"
            "            if (obstacleGrid[i][j] == 1) {\n"
            "                dp[j] = 0;\n"
            "            } else if (j > 0) {\n"
            "                dp[j] += dp[j - 1];\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    return dp[n - 1];\n"
            "}"
        ),
        "code_python": (
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:\n"
            "        m, n = len(obstacleGrid), len(obstacleGrid[0])\n"
            "        dp = [0] * n\n"
            "        dp[0] = 1 if obstacleGrid[0][0] == 0 else 0\n"
            "        for i in range(m):\n"
            "            for j in range(n):\n"
            "                if obstacleGrid[i][j] == 1:\n"
            "                    dp[j] = 0\n"
            "                elif j > 0:\n"
            "                    dp[j] += dp[j - 1]\n"
            "        return dp[n - 1]"
        ),
        "language": "java",
        "complexity": {"time": "O(m * n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# minimum-path-sum.json
# ─────────────────────────────────────────────
add_solutions("minimum-path-sum.json", [
    {
        "title": f"Solution 2 – Space-Optimized 1D DP{MARKER}",
        "description": (
            "Use a 1D array dp of size n (columns). Initialize the first row in dp directly from grid[0]. "
            "For each subsequent row i and column j: "
            "dp[j] = grid[i][j] + min(dp[j] /*from above*/, dp[j-1] /*from left, already updated*/). "
            "For j==0 just add grid[i][0] to dp[0]. "
            "O(m*n) time O(n) space."
        ),
        "code": (
            "public int minPathSum(int[][] grid) {\n"
            "    int m = grid.length, n = grid[0].length;\n"
            "    int[] dp = new int[n];\n"
            "    dp[0] = grid[0][0];\n"
            "    for (int j = 1; j < n; j++) dp[j] = dp[j - 1] + grid[0][j];\n"
            "    for (int i = 1; i < m; i++) {\n"
            "        dp[0] += grid[i][0];\n"
            "        for (int j = 1; j < n; j++) {\n"
            "            dp[j] = grid[i][j] + Math.min(dp[j], dp[j - 1]);\n"
            "        }\n"
            "    }\n"
            "    return dp[n - 1];\n"
            "}"
        ),
        "code_python": (
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def minPathSum(self, grid: List[List[int]]) -> int:\n"
            "        m, n = len(grid), len(grid[0])\n"
            "        dp = [0] * n\n"
            "        dp[0] = grid[0][0]\n"
            "        for j in range(1, n):\n"
            "            dp[j] = dp[j - 1] + grid[0][j]\n"
            "        for i in range(1, m):\n"
            "            dp[0] += grid[i][0]\n"
            "            for j in range(1, n):\n"
            "                dp[j] = grid[i][j] + min(dp[j], dp[j - 1])\n"
            "        return dp[n - 1]"
        ),
        "language": "java",
        "complexity": {"time": "O(m * n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# merge-k-sorted-lists.json
# ─────────────────────────────────────────────
add_solutions("merge-k-sorted-lists.json", [
    {
        "title": f"Solution 2 – Divide and Conquer{MARKER}",
        "description": (
            "Repeatedly pair up lists and merge each pair until only one list remains. "
            "In each round, the number of lists halves. "
            "Merging two sorted lists of combined length L takes O(L). "
            "Across log k rounds, total work is O(n log k) where n is the total number of nodes. "
            "O(log k) recursion stack space."
        ),
        "code": (
            "public ListNode mergeKLists(ListNode[] lists) {\n"
            "    if (lists == null || lists.length == 0) return null;\n"
            "    int interval = 1;\n"
            "    int len = lists.length;\n"
            "    while (interval < len) {\n"
            "        for (int i = 0; i + interval < len; i += interval * 2) {\n"
            "            lists[i] = mergeTwoLists(lists[i], lists[i + interval]);\n"
            "        }\n"
            "        interval *= 2;\n"
            "    }\n"
            "    return lists[0];\n"
            "}\n"
            "\n"
            "private ListNode mergeTwoLists(ListNode l1, ListNode l2) {\n"
            "    ListNode dummy = new ListNode(0);\n"
            "    ListNode curr = dummy;\n"
            "    while (l1 != null && l2 != null) {\n"
            "        if (l1.val <= l2.val) { curr.next = l1; l1 = l1.next; }\n"
            "        else { curr.next = l2; l2 = l2.next; }\n"
            "        curr = curr.next;\n"
            "    }\n"
            "    curr.next = (l1 != null) ? l1 : l2;\n"
            "    return dummy.next;\n"
            "}"
        ),
        "code_python": (
            "from typing import List, Optional\n"
            "\n"
            "class Solution:\n"
            "    def mergeKLists(self, lists: List[Optional[object]]) -> Optional[object]:\n"
            "        if not lists:\n"
            "            return None\n"
            "        interval = 1\n"
            "        n = len(lists)\n"
            "        while interval < n:\n"
            "            for i in range(0, n - interval, interval * 2):\n"
            "                lists[i] = self._merge_two(lists[i], lists[i + interval])\n"
            "            interval *= 2\n"
            "        return lists[0]\n"
            "\n"
            "    def _merge_two(self, l1, l2):\n"
            "        dummy = type('Node', (), {'val': 0, 'next': None})()\n"
            "        curr = dummy\n"
            "        while l1 and l2:\n"
            "            if l1.val <= l2.val:\n"
            "                curr.next, l1 = l1, l1.next\n"
            "            else:\n"
            "                curr.next, l2 = l2, l2.next\n"
            "            curr = curr.next\n"
            "        curr.next = l1 if l1 else l2\n"
            "        return dummy.next"
        ),
        "language": "java",
        "complexity": {"time": "O(n log k)", "space": "O(log k)"}
    }
])

# ─────────────────────────────────────────────
# pow-x-n.json
# ─────────────────────────────────────────────
add_solutions("pow-x-n.json", [
    {
        "title": f"Solution 2 – Iterative Fast Power{MARKER}",
        "description": (
            "Iterative version of binary exponentiation (exponentiation by squaring). "
            "Handle negative n by using x = 1/x and n = -n. "
            "Keep a result accumulator: if the current bit of n is 1, multiply result by x; "
            "then square x and halve n. This avoids the call stack entirely. "
            "O(log n) time O(1) space."
        ),
        "code": (
            "public double myPow(double x, int n) {\n"
            "    long N = n; // use long to handle Integer.MIN_VALUE\n"
            "    if (N < 0) { x = 1.0 / x; N = -N; }\n"
            "    double result = 1.0;\n"
            "    while (N > 0) {\n"
            "        if ((N & 1) == 1) result *= x;\n"
            "        x *= x;\n"
            "        N >>= 1;\n"
            "    }\n"
            "    return result;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def myPow(self, x: float, n: int) -> float:\n"
            "        if n < 0:\n"
            "            x, n = 1.0 / x, -n\n"
            "        result = 1.0\n"
            "        while n > 0:\n"
            "            if n & 1:\n"
            "                result *= x\n"
            "            x *= x\n"
            "            n >>= 1\n"
            "        return result"
        ),
        "language": "java",
        "complexity": {"time": "O(log n)", "space": "O(1)"}
    }
])

# ─────────────────────────────────────────────
# swap-nodes-pairs.json
# ─────────────────────────────────────────────
add_solutions("swap-nodes-pairs.json", [
    {
        "title": f"Solution 2 – Recursive{MARKER}",
        "description": (
            "Base case: if the list has 0 or 1 nodes, return head unchanged. "
            "Otherwise, take the first two nodes, recursively solve the rest of the list, "
            "attach the result to the second node's next pointer, then point the second node back to the first. "
            "Return the second node as the new head. O(n) time O(n) stack space."
        ),
        "code": (
            "public ListNode swapPairs(ListNode head) {\n"
            "    if (head == null || head.next == null) return head;\n"
            "    ListNode first = head;\n"
            "    ListNode second = head.next;\n"
            "    first.next = swapPairs(second.next);\n"
            "    second.next = first;\n"
            "    return second;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def swapPairs(self, head):\n"
            "        if not head or not head.next:\n"
            "            return head\n"
            "        first, second = head, head.next\n"
            "        first.next = self.swapPairs(second.next)\n"
            "        second.next = first\n"
            "        return second"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# print-linked-list-reversed.json
# ─────────────────────────────────────────────
add_solutions("print-linked-list-reversed.json", [
    {
        "title": f"Solution 2 – Stack-Based Iterative{MARKER}",
        "description": (
            "Traverse the linked list from head to tail, pushing each node's value onto a stack. "
            "Then pop all values from the stack and collect/print them — "
            "since a stack is LIFO, this reverses the order. "
            "O(n) time O(n) space for the stack."
        ),
        "code": (
            "public List<Integer> printReversed(ListNode head) {\n"
            "    Deque<Integer> stack = new ArrayDeque<>();\n"
            "    ListNode curr = head;\n"
            "    while (curr != null) {\n"
            "        stack.push(curr.val);\n"
            "        curr = curr.next;\n"
            "    }\n"
            "    List<Integer> result = new ArrayList<>();\n"
            "    while (!stack.isEmpty()) {\n"
            "        result.add(stack.pop());\n"
            "    }\n"
            "    return result;\n"
            "}"
        ),
        "code_python": (
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def printReversed(self, head) -> List[int]:\n"
            "        stack = []\n"
            "        curr = head\n"
            "        while curr:\n"
            "            stack.append(curr.val)\n"
            "            curr = curr.next\n"
            "        return stack[::-1]"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# intersection-two-linked-lists.json
# ─────────────────────────────────────────────
add_solutions("intersection-two-linked-lists.json", [
    {
        "title": f"Solution 2 – HashSet{MARKER}",
        "description": (
            "First pass: traverse list A and add every node (by reference/identity) to a HashSet. "
            "Second pass: traverse list B — the first node found in the set is the intersection node. "
            "If no intersection exists, return null. "
            "O(m+n) time O(m) space."
        ),
        "code": (
            "public ListNode getIntersectionNode(ListNode headA, ListNode headB) {\n"
            "    Set<ListNode> visited = new HashSet<>();\n"
            "    ListNode curr = headA;\n"
            "    while (curr != null) {\n"
            "        visited.add(curr);\n"
            "        curr = curr.next;\n"
            "    }\n"
            "    curr = headB;\n"
            "    while (curr != null) {\n"
            "        if (visited.contains(curr)) return curr;\n"
            "        curr = curr.next;\n"
            "    }\n"
            "    return null;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def getIntersectionNode(self, headA, headB):\n"
            "        visited = set()\n"
            "        curr = headA\n"
            "        while curr:\n"
            "            visited.add(id(curr))\n"
            "            curr = curr.next\n"
            "        curr = headB\n"
            "        while curr:\n"
            "            if id(curr) in visited:\n"
            "                return curr\n"
            "            curr = curr.next\n"
            "        return None"
        ),
        "language": "java",
        "complexity": {"time": "O(m + n)", "space": "O(m)"}
    }
])

# ─────────────────────────────────────────────
# binary-tree-level-order-traversal.json
# ─────────────────────────────────────────────
add_solutions("binary-tree-level-order-traversal.json", [
    {
        "title": f"Solution 2 – DFS Recursive{MARKER}",
        "description": (
            "Perform a DFS (preorder) with a depth/level counter. "
            "If the result list doesn't have an entry for the current level yet, add a new list. "
            "Append the current node's value to result[level], then recurse left with level+1 and right with level+1. "
            "O(n) time O(n) space (output + recursion stack)."
        ),
        "code": (
            "public List<List<Integer>> levelOrder(TreeNode root) {\n"
            "    List<List<Integer>> result = new ArrayList<>();\n"
            "    dfs(root, 0, result);\n"
            "    return result;\n"
            "}\n"
            "\n"
            "private void dfs(TreeNode node, int level, List<List<Integer>> result) {\n"
            "    if (node == null) return;\n"
            "    if (result.size() == level) {\n"
            "        result.add(new ArrayList<>());\n"
            "    }\n"
            "    result.get(level).add(node.val);\n"
            "    dfs(node.left, level + 1, result);\n"
            "    dfs(node.right, level + 1, result);\n"
            "}"
        ),
        "code_python": (
            "from typing import List, Optional\n"
            "\n"
            "class Solution:\n"
            "    def levelOrder(self, root) -> List[List[int]]:\n"
            "        result = []\n"
            "\n"
            "        def dfs(node, level):\n"
            "            if not node:\n"
            "                return\n"
            "            if len(result) == level:\n"
            "                result.append([])\n"
            "            result[level].append(node.val)\n"
            "            dfs(node.left, level + 1)\n"
            "            dfs(node.right, level + 1)\n"
            "\n"
            "        dfs(root, 0)\n"
            "        return result"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# min-stack.json
# ─────────────────────────────────────────────
add_solutions("min-stack.json", [
    {
        "title": f"Solution 2 – Single Stack with Value-Min Pairs{MARKER}",
        "description": (
            "Instead of maintaining two separate stacks, store each element as a pair (value, current_min). "
            "On push, the current_min in the pair is min(val, stack.peek().min). "
            "getMin() simply returns the min from the top pair. "
            "All operations remain O(1) with no second stack needed."
        ),
        "code": (
            "class MinStack {\n"
            "    private Deque<int[]> stack; // each entry: [value, currentMin]\n"
            "\n"
            "    public MinStack() {\n"
            "        stack = new ArrayDeque<>();\n"
            "    }\n"
            "\n"
            "    public void push(int val) {\n"
            "        int currentMin = stack.isEmpty() ? val : Math.min(val, stack.peek()[1]);\n"
            "        stack.push(new int[]{val, currentMin});\n"
            "    }\n"
            "\n"
            "    public void pop() {\n"
            "        stack.pop();\n"
            "    }\n"
            "\n"
            "    public int top() {\n"
            "        return stack.peek()[0];\n"
            "    }\n"
            "\n"
            "    public int getMin() {\n"
            "        return stack.peek()[1];\n"
            "    }\n"
            "}"
        ),
        "code_python": (
            "class MinStack:\n"
            "    def __init__(self):\n"
            "        self._stack = []  # stores (value, current_min)\n"
            "\n"
            "    def push(self, val: int) -> None:\n"
            "        current_min = val if not self._stack else min(val, self._stack[-1][1])\n"
            "        self._stack.append((val, current_min))\n"
            "\n"
            "    def pop(self) -> None:\n"
            "        self._stack.pop()\n"
            "\n"
            "    def top(self) -> int:\n"
            "        return self._stack[-1][0]\n"
            "\n"
            "    def getMin(self) -> int:\n"
            "        return self._stack[-1][1]"
        ),
        "language": "java",
        "complexity": {"time": "O(1) all ops", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# number-1-bits.json
# ─────────────────────────────────────────────
add_solutions("number-1-bits.json", [
    {
        "title": f"Solution 2 – Brian Kernighan's Algorithm{MARKER}",
        "description": (
            "Repeatedly clear the lowest set bit using the operation n = n & (n - 1). "
            "Each iteration removes exactly one set bit (the least significant one). "
            "Count how many iterations are needed until n becomes 0. "
            "This is faster than checking every bit when the number of set bits is small. "
            "O(number of 1-bits) time O(1) space."
        ),
        "code": (
            "public int hammingWeight(int n) {\n"
            "    int count = 0;\n"
            "    while (n != 0) {\n"
            "        n &= (n - 1); // clear lowest set bit\n"
            "        count++;\n"
            "    }\n"
            "    return count;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def hammingWeight(self, n: int) -> int:\n"
            "        count = 0\n"
            "        while n:\n"
            "            n &= n - 1  # clear lowest set bit\n"
            "            count += 1\n"
            "        return count"
        ),
        "language": "java",
        "complexity": {"time": "O(k) where k = number of set bits", "space": "O(1)"}
    }
])

# ─────────────────────────────────────────────
# palindrome-number.json
# ─────────────────────────────────────────────
add_solutions("palindrome-number.json", [
    {
        "title": f"Solution 2 – String Conversion{MARKER}",
        "description": (
            "Convert the integer to its string representation and compare it with the reversed string. "
            "Negative numbers are never palindromes (due to the leading '-'). "
            "O(log n) time to convert and compare (proportional to digit count) "
            "O(log n) space for the string."
        ),
        "code": (
            "public boolean isPalindrome(int x) {\n"
            "    if (x < 0) return false;\n"
            "    String s = Integer.toString(x);\n"
            "    int left = 0, right = s.length() - 1;\n"
            "    while (left < right) {\n"
            "        if (s.charAt(left) != s.charAt(right)) return false;\n"
            "        left++;\n"
            "        right--;\n"
            "    }\n"
            "    return true;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def isPalindrome(self, x: int) -> bool:\n"
            "        if x < 0:\n"
            "            return False\n"
            "        s = str(x)\n"
            "        return s == s[::-1]"
        ),
        "language": "java",
        "complexity": {"time": "O(log n)", "space": "O(log n)"}
    }
])

# ─────────────────────────────────────────────
# gray-code.json
# ─────────────────────────────────────────────
add_solutions("gray-code.json", [
    {
        "title": f"Solution 2 – Bit Manipulation Formula{MARKER}",
        "description": (
            "The i-th Gray code value is given directly by the formula: i XOR (i >> 1). "
            "For n bits, there are 2^n Gray codes. Generate each by applying the formula to integers 0 through 2^n - 1. "
            "This is O(2^n) time and requires no recursion or explicit reflection. "
            "Adjacent values differ by exactly one bit."
        ),
        "code": (
            "public List<Integer> grayCode(int n) {\n"
            "    List<Integer> result = new ArrayList<>();\n"
            "    int total = 1 << n; // 2^n\n"
            "    for (int i = 0; i < total; i++) {\n"
            "        result.add(i ^ (i >> 1));\n"
            "    }\n"
            "    return result;\n"
            "}"
        ),
        "code_python": (
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def grayCode(self, n: int) -> List[int]:\n"
            "        return [i ^ (i >> 1) for i in range(1 << n)]"
        ),
        "language": "java",
        "complexity": {"time": "O(2^n)", "space": "O(2^n)"}
    }
])

# ─────────────────────────────────────────────
# graph-valid-tree.json
# ─────────────────────────────────────────────
add_solutions("graph-valid-tree.json", [
    {
        "title": f"Solution 2 – DFS Cycle Detection{MARKER}",
        "description": (
            "Build an undirected adjacency list from the given edges. "
            "DFS from node 0, tracking the parent of each node to avoid false cycle detection on the edge we came from. "
            "If we encounter a visited node that is not the parent, a cycle exists — return false. "
            "After DFS, check that all n nodes were visited (ensures connectivity). "
            "O(V+E) time O(V+E) space."
        ),
        "code": (
            "public boolean validTree(int n, int[][] edges) {\n"
            "    if (edges.length != n - 1) return false; // quick check\n"
            "    List<List<Integer>> adj = new ArrayList<>();\n"
            "    for (int i = 0; i < n; i++) adj.add(new ArrayList<>());\n"
            "    for (int[] e : edges) {\n"
            "        adj.get(e[0]).add(e[1]);\n"
            "        adj.get(e[1]).add(e[0]);\n"
            "    }\n"
            "    boolean[] visited = new boolean[n];\n"
            "    if (!dfs(adj, visited, 0, -1)) return false;\n"
            "    for (boolean v : visited) if (!v) return false;\n"
            "    return true;\n"
            "}\n"
            "\n"
            "private boolean dfs(List<List<Integer>> adj, boolean[] visited, int node, int parent) {\n"
            "    visited[node] = true;\n"
            "    for (int neighbor : adj.get(node)) {\n"
            "        if (neighbor == parent) continue;\n"
            "        if (visited[neighbor]) return false;\n"
            "        if (!dfs(adj, visited, neighbor, node)) return false;\n"
            "    }\n"
            "    return true;\n"
            "}"
        ),
        "code_python": (
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def validTree(self, n: int, edges: List[List[int]]) -> bool:\n"
            "        if len(edges) != n - 1:\n"
            "            return False\n"
            "        adj = [[] for _ in range(n)]\n"
            "        for u, v in edges:\n"
            "            adj[u].append(v)\n"
            "            adj[v].append(u)\n"
            "        visited = [False] * n\n"
            "\n"
            "        def dfs(node, parent):\n"
            "            visited[node] = True\n"
            "            for neighbor in adj[node]:\n"
            "                if neighbor == parent:\n"
            "                    continue\n"
            "                if visited[neighbor]:\n"
            "                    return False\n"
            "                if not dfs(neighbor, node):\n"
            "                    return False\n"
            "            return True\n"
            "\n"
            "        if not dfs(0, -1):\n"
            "            return False\n"
            "        return all(visited)"
        ),
        "language": "java",
        "complexity": {"time": "O(V + E)", "space": "O(V + E)"}
    }
])

# ─────────────────────────────────────────────
# happy-number.json
# ─────────────────────────────────────────────
add_solutions("happy-number.json", [
    {
        "title": f"Solution 2 – HashSet{MARKER}",
        "description": (
            "Compute the sum of squares of digits in a loop. "
            "If the result equals 1, return true — it's a happy number. "
            "If the result has been seen before (detected via a HashSet), return false — we're in a cycle. "
            "Otherwise add the result to the set and repeat. "
            "O(log n) per step, terminates because the sequence either reaches 1 or enters a known cycle."
        ),
        "code": (
            "public boolean isHappy(int n) {\n"
            "    Set<Integer> seen = new HashSet<>();\n"
            "    while (n != 1) {\n"
            "        if (seen.contains(n)) return false;\n"
            "        seen.add(n);\n"
            "        int sum = 0;\n"
            "        while (n > 0) {\n"
            "            int d = n % 10;\n"
            "            sum += d * d;\n"
            "            n /= 10;\n"
            "        }\n"
            "        n = sum;\n"
            "    }\n"
            "    return true;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def isHappy(self, n: int) -> bool:\n"
            "        seen = set()\n"
            "        while n != 1:\n"
            "            if n in seen:\n"
            "                return False\n"
            "            seen.add(n)\n"
            "            n = sum(int(d) ** 2 for d in str(n))\n"
            "        return True"
        ),
        "language": "java",
        "complexity": {"time": "O(log n) per step", "space": "O(log n)"}
    }
])

# ─────────────────────────────────────────────
# triangle.json
# ─────────────────────────────────────────────
add_solutions("triangle.json", [
    {
        "title": f"Solution 2 – Bottom-Up In-Place DP{MARKER}",
        "description": (
            "Start from the second-to-last row of the triangle and work upward. "
            "For each cell triangle[i][j], add the minimum of its two children: "
            "triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1]). "
            "After processing all rows, triangle[0][0] holds the minimum path sum. "
            "O(n^2) time O(1) extra space (modifies the input in place)."
        ),
        "code": (
            "public int minimumTotal(List<List<Integer>> triangle) {\n"
            "    int n = triangle.size();\n"
            "    for (int i = n - 2; i >= 0; i--) {\n"
            "        for (int j = 0; j <= i; j++) {\n"
            "            int best = Math.min(triangle.get(i + 1).get(j),\n"
            "                               triangle.get(i + 1).get(j + 1));\n"
            "            triangle.get(i).set(j, triangle.get(i).get(j) + best);\n"
            "        }\n"
            "    }\n"
            "    return triangle.get(0).get(0);\n"
            "}"
        ),
        "code_python": (
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def minimumTotal(self, triangle: List[List[int]]) -> int:\n"
            "        n = len(triangle)\n"
            "        for i in range(n - 2, -1, -1):\n"
            "            for j in range(i + 1):\n"
            "                triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1])\n"
            "        return triangle[0][0]"
        ),
        "language": "java",
        "complexity": {"time": "O(n^2)", "space": "O(1)"}
    }
])

# ─────────────────────────────────────────────
# counting-bits.json
# ─────────────────────────────────────────────
add_solutions("counting-bits.json", [
    {
        "title": f"Solution 2 – Brian Kernighan Bit Count{MARKER}",
        "description": (
            "For each integer i from 0 to n, count its set bits using Brian Kernighan's trick: "
            "repeatedly clear the lowest set bit with n = n & (n-1) and count iterations. "
            "O(n log n) total time (O(log i) per number) O(n) space for the output array. "
            "Less optimal than the DP approach but demonstrates a classical bit manipulation technique."
        ),
        "code": (
            "public int[] countBits(int n) {\n"
            "    int[] result = new int[n + 1];\n"
            "    for (int i = 0; i <= n; i++) {\n"
            "        int num = i, count = 0;\n"
            "        while (num != 0) {\n"
            "            num &= (num - 1);\n"
            "            count++;\n"
            "        }\n"
            "        result[i] = count;\n"
            "    }\n"
            "    return result;\n"
            "}"
        ),
        "code_python": (
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def countBits(self, n: int) -> List[int]:\n"
            "        def bit_count(x):\n"
            "            count = 0\n"
            "            while x:\n"
            "                x &= x - 1\n"
            "                count += 1\n"
            "            return count\n"
            "        return [bit_count(i) for i in range(n + 1)]"
        ),
        "language": "java",
        "complexity": {"time": "O(n log n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# unique-binary-search-trees.json
# ─────────────────────────────────────────────
add_solutions("unique-binary-search-trees.json", [
    {
        "title": f"Solution 2 – Catalan Number Formula{MARKER}",
        "description": (
            "The answer is the nth Catalan number: C(n) = C(2n, n) / (n+1). "
            "Compute it using the multiplicative formula: "
            "C(n) = product of (n+k+1)/k for k in 1..n, all divided by (n+1). "
            "Equivalently use: C_n = (2n)! / ((n+1)! * n!). "
            "In practice compute iteratively: result = result * (n+k) / k for k=2..n, then divide by (n+1). "
            "O(n) time O(1) space."
        ),
        "code": (
            "public int numTrees(int n) {\n"
            "    // Catalan number: C(2n, n) / (n+1)\n"
            "    long result = 1;\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        result = result * (2 * n - i) / (i + 1);\n"
            "    }\n"
            "    return (int)(result / (n + 1));\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def numTrees(self, n: int) -> int:\n"
            "        # Catalan number C(2n, n) / (n+1)\n"
            "        from math import comb\n"
            "        return comb(2 * n, n) // (n + 1)"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(1)"}
    }
])

# ─────────────────────────────────────────────
# flatten-binary-tree-linked-list.json
# ─────────────────────────────────────────────
add_solutions("flatten-binary-tree-linked-list.json", [
    {
        "title": f"Solution 2 – Recursive Right-First Postorder{MARKER}",
        "description": (
            "Use a reverse-postorder traversal (right subtree first, then left). "
            "Keep a `prev` pointer (or instance variable) that tracks the previously visited node. "
            "For each node: recursively flatten the right subtree, then the left subtree. "
            "Set node.right = prev, node.left = null, and update prev = node. "
            "This threads the tree into a linked list in correct preorder. "
            "O(n) time O(n) recursion stack space."
        ),
        "code": (
            "class Solution {\n"
            "    private TreeNode prev = null;\n"
            "\n"
            "    public void flatten(TreeNode root) {\n"
            "        if (root == null) return;\n"
            "        flatten(root.right);\n"
            "        flatten(root.left);\n"
            "        root.right = prev;\n"
            "        root.left = null;\n"
            "        prev = root;\n"
            "    }\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def flatten(self, root) -> None:\n"
            "        self.prev = None\n"
            "\n"
            "        def dfs(node):\n"
            "            if not node:\n"
            "                return\n"
            "            dfs(node.right)\n"
            "            dfs(node.left)\n"
            "            node.right = self.prev\n"
            "            node.left = None\n"
            "            self.prev = node\n"
            "\n"
            "        dfs(root)"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# balanced-binary-tree.json
# ─────────────────────────────────────────────
add_solutions("balanced-binary-tree.json", [
    {
        "title": f"Solution 2 – Iterative Post-Order with Stack{MARKER}",
        "description": (
            "Perform an iterative post-order traversal using a stack and a HashMap to store heights. "
            "For each node after both children have been processed: "
            "retrieve left and right heights from the map, check |left_h - right_h| <= 1 "
            "(set a flag if violated), and store this node's height as 1 + max(left_h, right_h). "
            "O(n) time O(n) space."
        ),
        "code": (
            "public boolean isBalanced(TreeNode root) {\n"
            "    if (root == null) return true;\n"
            "    Map<TreeNode, Integer> heights = new HashMap<>();\n"
            "    heights.put(null, 0);\n"
            "    Deque<TreeNode> stack = new ArrayDeque<>();\n"
            "    TreeNode curr = root, lastVisited = null;\n"
            "    boolean balanced = true;\n"
            "    while (curr != null || !stack.isEmpty()) {\n"
            "        if (curr != null) {\n"
            "            stack.push(curr);\n"
            "            curr = curr.left;\n"
            "        } else {\n"
            "            TreeNode peekNode = stack.peek();\n"
            "            if (peekNode.right != null && lastVisited != peekNode.right) {\n"
            "                curr = peekNode.right;\n"
            "            } else {\n"
            "                stack.pop();\n"
            "                int leftH = heights.getOrDefault(peekNode.left, 0);\n"
            "                int rightH = heights.getOrDefault(peekNode.right, 0);\n"
            "                if (Math.abs(leftH - rightH) > 1) balanced = false;\n"
            "                heights.put(peekNode, 1 + Math.max(leftH, rightH));\n"
            "                lastVisited = peekNode;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    return balanced;\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def isBalanced(self, root) -> bool:\n"
            "        heights = {None: 0}\n"
            "        stack = []\n"
            "        curr, last_visited = root, None\n"
            "        balanced = True\n"
            "        while curr or stack:\n"
            "            if curr:\n"
            "                stack.append(curr)\n"
            "                curr = curr.left\n"
            "            else:\n"
            "                peek = stack[-1]\n"
            "                if peek.right and last_visited is not peek.right:\n"
            "                    curr = peek.right\n"
            "                else:\n"
            "                    stack.pop()\n"
            "                    lh = heights.get(peek.left, 0)\n"
            "                    rh = heights.get(peek.right, 0)\n"
            "                    if abs(lh - rh) > 1:\n"
            "                        balanced = False\n"
            "                    heights[peek] = 1 + max(lh, rh)\n"
            "                    last_visited = peek\n"
            "        return balanced"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# minimum-depth-binary-tree.json
# ─────────────────────────────────────────────
add_solutions("minimum-depth-binary-tree.json", [
    {
        "title": f"Solution 2 – DFS Recursive{MARKER}",
        "description": (
            "Recursively compute the minimum depth. "
            "For a null node, return 0. For a leaf node (no children), return 1. "
            "If only the left child exists, return 1 + minDepth(left) — "
            "we must not count the missing right child as a depth-1 leaf. "
            "Similarly if only the right child exists. "
            "If both children exist, return 1 + min(minDepth(left), minDepth(right)). "
            "O(n) time O(H) space."
        ),
        "code": (
            "public int minDepth(TreeNode root) {\n"
            "    if (root == null) return 0;\n"
            "    if (root.left == null && root.right == null) return 1;\n"
            "    if (root.left == null) return 1 + minDepth(root.right);\n"
            "    if (root.right == null) return 1 + minDepth(root.left);\n"
            "    return 1 + Math.min(minDepth(root.left), minDepth(root.right));\n"
            "}"
        ),
        "code_python": (
            "class Solution:\n"
            "    def minDepth(self, root) -> int:\n"
            "        if not root:\n"
            "            return 0\n"
            "        if not root.left and not root.right:\n"
            "            return 1\n"
            "        if not root.left:\n"
            "            return 1 + self.minDepth(root.right)\n"
            "        if not root.right:\n"
            "            return 1 + self.minDepth(root.left)\n"
            "        return 1 + min(self.minDepth(root.left), self.minDepth(root.right))"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(H)"}
    }
])

# ─────────────────────────────────────────────
# construct-binary-tree-preorder-inorder.json
# ─────────────────────────────────────────────
add_solutions("construct-binary-tree-preorder-inorder.json", [
    {
        "title": f"Solution 2 – Iterative with Stack + HashMap{MARKER}",
        "description": (
            "Use a HashMap for O(1) inorder index lookups. "
            "Iterate through preorder values; maintain a stack of nodes. "
            "For each value: create a new node. "
            "If inorder[inorderIdx] != stack.peek().val, the new node is a left child of the top. "
            "Otherwise pop the stack while the top matches inorder[inorderIdx], incrementing inorderIdx; "
            "the new node becomes the right child of the last popped node. "
            "Push the new node. O(n) time O(n) space."
        ),
        "code": (
            "public TreeNode buildTree(int[] preorder, int[] inorder) {\n"
            "    if (preorder.length == 0) return null;\n"
            "    Map<Integer, Integer> inMap = new HashMap<>();\n"
            "    for (int i = 0; i < inorder.length; i++) inMap.put(inorder[i], i);\n"
            "    Deque<TreeNode> stack = new ArrayDeque<>();\n"
            "    TreeNode root = new TreeNode(preorder[0]);\n"
            "    stack.push(root);\n"
            "    int inIdx = 0;\n"
            "    for (int i = 1; i < preorder.length; i++) {\n"
            "        TreeNode node = new TreeNode(preorder[i]);\n"
            "        if (inMap.get(preorder[i]) < inMap.get(stack.peek().val)) {\n"
            "            stack.peek().left = node;\n"
            "        } else {\n"
            "            TreeNode parent = null;\n"
            "            while (!stack.isEmpty() && inMap.get(stack.peek().val) < inMap.get(preorder[i])) {\n"
            "                parent = stack.pop();\n"
            "            }\n"
            "            parent.right = node;\n"
            "        }\n"
            "        stack.push(node);\n"
            "    }\n"
            "    return root;\n"
            "}"
        ),
        "code_python": (
            "from typing import List, Optional\n"
            "\n"
            "class Solution:\n"
            "    def buildTree(self, preorder: List[int], inorder: List[int]):\n"
            "        if not preorder:\n"
            "            return None\n"
            "        in_map = {val: i for i, val in enumerate(inorder)}\n"
            "        root = TreeNode(preorder[0])\n"
            "        stack = [root]\n"
            "        for i in range(1, len(preorder)):\n"
            "            node = TreeNode(preorder[i])\n"
            "            if in_map[preorder[i]] < in_map[stack[-1].val]:\n"
            "                stack[-1].left = node\n"
            "            else:\n"
            "                parent = None\n"
            "                while stack and in_map[stack[-1].val] < in_map[preorder[i]]:\n"
            "                    parent = stack.pop()\n"
            "                parent.right = node\n"
            "            stack.append(node)\n"
            "        return root"
        ),
        "language": "java",
        "complexity": {"time": "O(n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# vertical-order.json
# ─────────────────────────────────────────────
add_solutions("vertical-order.json", [
    {
        "title": f"Solution 2 – BFS with Column Tracking{MARKER}",
        "description": (
            "BFS using a queue of (node, column) pairs. "
            "Use a TreeMap (sorted by column key) mapping column index to a list of node values. "
            "When dequeuing a node at column c: add its value to the map at key c; "
            "enqueue left child at c-1 and right child at c+1. "
            "Because BFS processes nodes level by level, values within the same column "
            "are automatically ordered top-to-bottom. "
            "O(n log n) time due to TreeMap operations, O(n) space."
        ),
        "code": (
            "public List<List<Integer>> verticalOrder(TreeNode root) {\n"
            "    List<List<Integer>> result = new ArrayList<>();\n"
            "    if (root == null) return result;\n"
            "    TreeMap<Integer, List<Integer>> colMap = new TreeMap<>();\n"
            "    Queue<int[]> queue = new LinkedList<>(); // [node_id, col]\n"
            "    Map<Integer, TreeNode> idToNode = new HashMap<>();\n"
            "    // Use a queue of Object arrays for (node, col)\n"
            "    Queue<Object[]> bfsQueue = new LinkedList<>();\n"
            "    bfsQueue.offer(new Object[]{root, 0});\n"
            "    while (!bfsQueue.isEmpty()) {\n"
            "        Object[] curr = bfsQueue.poll();\n"
            "        TreeNode node = (TreeNode) curr[0];\n"
            "        int col = (int) curr[1];\n"
            "        colMap.computeIfAbsent(col, k -> new ArrayList<>()).add(node.val);\n"
            "        if (node.left != null) bfsQueue.offer(new Object[]{node.left, col - 1});\n"
            "        if (node.right != null) bfsQueue.offer(new Object[]{node.right, col + 1});\n"
            "    }\n"
            "    result.addAll(colMap.values());\n"
            "    return result;\n"
            "}"
        ),
        "code_python": (
            "from collections import deque, defaultdict\n"
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def verticalOrder(self, root) -> List[List[int]]:\n"
            "        if not root:\n"
            "            return []\n"
            "        col_map = defaultdict(list)\n"
            "        queue = deque([(root, 0)])\n"
            "        while queue:\n"
            "            node, col = queue.popleft()\n"
            "            col_map[col].append(node.val)\n"
            "            if node.left:\n"
            "                queue.append((node.left, col - 1))\n"
            "            if node.right:\n"
            "                queue.append((node.right, col + 1))\n"
            "        return [col_map[c] for c in sorted(col_map)]"
        ),
        "language": "java",
        "complexity": {"time": "O(n log n)", "space": "O(n)"}
    }
])

# ─────────────────────────────────────────────
# course-schedule.json
# ─────────────────────────────────────────────
add_solutions("course-schedule.json", [
    {
        "title": f"Solution 2 – BFS Topological Sort (Kahn's Algorithm){MARKER}",
        "description": (
            "Build an adjacency list and an in-degree array. "
            "Initialize a queue with all nodes whose in-degree is 0. "
            "Process: for each dequeued node, decrement the in-degree of all its neighbors; "
            "if a neighbor's in-degree drops to 0, enqueue it. "
            "Count nodes processed. If count == numCourses, no cycle exists and all courses can be finished. "
            "O(V+E) time O(V+E) space."
        ),
        "code": (
            "public boolean canFinish(int numCourses, int[][] prerequisites) {\n"
            "    int[] inDegree = new int[numCourses];\n"
            "    List<List<Integer>> adj = new ArrayList<>();\n"
            "    for (int i = 0; i < numCourses; i++) adj.add(new ArrayList<>());\n"
            "    for (int[] pre : prerequisites) {\n"
            "        adj.get(pre[1]).add(pre[0]);\n"
            "        inDegree[pre[0]]++;\n"
            "    }\n"
            "    Queue<Integer> queue = new LinkedList<>();\n"
            "    for (int i = 0; i < numCourses; i++) {\n"
            "        if (inDegree[i] == 0) queue.offer(i);\n"
            "    }\n"
            "    int processed = 0;\n"
            "    while (!queue.isEmpty()) {\n"
            "        int course = queue.poll();\n"
            "        processed++;\n"
            "        for (int next : adj.get(course)) {\n"
            "            if (--inDegree[next] == 0) queue.offer(next);\n"
            "        }\n"
            "    }\n"
            "    return processed == numCourses;\n"
            "}"
        ),
        "code_python": (
            "from collections import deque\n"
            "from typing import List\n"
            "\n"
            "class Solution:\n"
            "    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:\n"
            "        in_degree = [0] * numCourses\n"
            "        adj = [[] for _ in range(numCourses)]\n"
            "        for a, b in prerequisites:\n"
            "            adj[b].append(a)\n"
            "            in_degree[a] += 1\n"
            "        queue = deque(i for i in range(numCourses) if in_degree[i] == 0)\n"
            "        processed = 0\n"
            "        while queue:\n"
            "            course = queue.popleft()\n"
            "            processed += 1\n"
            "            for nxt in adj[course]:\n"
            "                in_degree[nxt] -= 1\n"
            "                if in_degree[nxt] == 0:\n"
            "                    queue.append(nxt)\n"
            "        return processed == numCourses"
        ),
        "language": "java",
        "complexity": {"time": "O(V + E)", "space": "O(V + E)"}
    }
])

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print(f"\n=== Done. {len(updated_files)} files updated ===")
for f in updated_files:
    print(f"  - {f}")
