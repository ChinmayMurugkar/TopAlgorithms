"""
Fill in missing examples (and improve assumptions) for all solution JSON files.
Run from repo root: python3 scripts/fill_examples.py
"""
import json, os, glob

# Map: slug -> { input, output, explanation }
# Also optionally: assumptions list
EXAMPLES = {
    "3sum": {
        "example": {"input": "nums = [-1, 0, 1, 2, -1, -4]", "output": "[[-1,-1,2],[-1,0,1]]",
                    "explanation": "The two triplets that sum to zero are [-1,-1,2] and [-1,0,1]."},
        "assumptions": ["The solution set must not contain duplicate triplets.", "Output order does not matter."]
    },
    "3sum-closest": {
        "example": {"input": "nums = [-1,2,1,-4], target = 1", "output": "2",
                    "explanation": "The sum that is closest to the target is -1+2+1 = 2."},
        "assumptions": ["Exactly one answer exists.", "Array has at least 3 elements."]
    },
    "4sum": {
        "example": {"input": "nums = [1,0,-1,0,-2,2], target = 0", "output": "[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]",
                    "explanation": "All unique quadruplets that sum to 0."},
        "assumptions": ["The solution set must not contain duplicate quadruplets."]
    },
    "add-binary": {
        "example": {"input": 'a = "11", b = "1"', "output": '"100"',
                    "explanation": "Binary 11 (3) + 1 (1) = 100 (4)."},
        "assumptions": ["Both strings are non-empty and contain only '0' or '1'."]
    },
    "add-digits": {
        "example": {"input": "num = 38", "output": "2",
                    "explanation": "3+8=11 → 1+1=2. The process repeats until a single digit remains."},
        "assumptions": ["0 ≤ num ≤ 2³¹ − 1"]
    },
    "add-search-word": {
        "example": {"input": 'addWord("bad"), addWord("dad"), addWord("mad"), search("pad"), search("bad"), search(".ad"), search("b..")',
                    "output": "false, true, true, true",
                    "explanation": "'.ad' matches any char in position 0, so it matches 'bad','dad','mad'. 'b..' matches 'bad'."},
        "assumptions": ["'.' in search pattern matches any single letter.", "Only lowercase letters a-z used."]
    },
    "add-two-numbers": {
        "example": {"input": "l1 = [2,4,3], l2 = [5,6,4]", "output": "[7,0,8]",
                    "explanation": "342 + 465 = 807. Digits stored in reverse order, so answer list is [7,0,8]."},
        "assumptions": ["Each linked list node contains a single digit (0-9).", "Digits are stored in reverse order.", "Neither list has leading zeros (except the number 0 itself)."]
    },
    "balanced-binary-tree": {
        "example": {"input": "root = [3,9,20,null,null,15,7]", "output": "true",
                    "explanation": "Every node's left and right subtrees differ in height by at most 1."},
        "assumptions": ["A height-balanced binary tree is one where every node's left and right subtree heights differ by at most 1."]
    },
    "basic-calculator": {
        "example": {"input": 's = "1 + 1"', "output": "2",
                    "explanation": 'Also handles: "(1+(4+5+2)-3)+(6+8)" → 23.'},
        "assumptions": ["The expression is always valid.", "No division or multiplication — only +, -, (, ).", "Input may contain spaces."]
    },
    "best-meeting-point": {
        "example": {"input": "grid = [[1,0,0],[0,0,0],[0,0,1]]", "output": "2",
                    "explanation": "Optimal meeting point is (0,2) or (1,1) with total distance 2. The median row and median column minimise Manhattan distance."},
        "assumptions": ["At least two people exist in the grid.", "Grid contains only 0s and 1s."]
    },
    "best-time-buy-sell-stock": {
        "example": {"input": "prices = [7,1,5,3,6,4]", "output": "5",
                    "explanation": "Buy on day 2 (price=1), sell on day 5 (price=6). Profit = 6-1 = 5."},
        "assumptions": ["You may only hold at most one share at a time.", "You must buy before you sell.", "Return 0 if no profit is possible."]
    },
    "binary-search-tree-iterator": {
        "example": {"input": "root = [7,3,15,null,null,9,20]\nnext() → 3, next() → 7, next() → 9, hasNext() → true, next() → 15",
                    "output": "3, 7, 9, true, 15",
                    "explanation": "In-order traversal of BST yields sorted order: 3,7,9,15,20."},
        "assumptions": ["next() returns the next smallest number.", "hasNext() returns true if more elements exist.", "next() and hasNext() run in O(1) average time with O(h) space where h = tree height."]
    },
    "binary-tree-level-order-traversal": {
        "example": {"input": "root = [3,9,20,null,null,15,7]", "output": "[[3],[9,20],[15,7]]",
                    "explanation": "Level 0: [3], Level 1: [9,20], Level 2: [15,7]."},
        "assumptions": ["Return a list of lists, one per level.", "Left child appears before right child in each level."]
    },
    "binary-tree-longest-consecutive": {
        "example": {"input": "root = [1,null,3,2,4,null,null,null,5]", "output": "3",
                    "explanation": "Longest consecutive path is 3→4→5, length = 3."},
        "assumptions": ["Consecutive means each parent-child pair differs by exactly 1 (increasing).", "The path goes parent to child only (top-down)."]
    },
    "binary-tree-maximum-path-sum": {
        "example": {"input": "root = [-10,9,20,null,null,15,7]", "output": "42",
                    "explanation": "The optimal path is 15 → 20 → 7, giving sum 15+20+7 = 42."},
        "assumptions": ["A path must contain at least one node.", "The path does not need to pass through the root.", "A node can appear at most once in the path."]
    },
    "binary-tree-paths": {
        "example": {"input": "root = [1,2,3,null,5]", "output": '["1->2->5","1->3"]',
                    "explanation": "Two root-to-leaf paths: 1→2→5 and 1→3."},
        "assumptions": ["A leaf is a node with no children.", "Return all root-to-leaf paths in any order."]
    },
    "binary-tree-right-side-view": {
        "example": {"input": "root = [1,2,3,null,5,null,4]", "output": "[1,3,4]",
                    "explanation": "Looking from the right side: level 0 → 1, level 1 → 3, level 2 → 4."},
        "assumptions": ["Return the rightmost visible node at each level.", "Nodes are returned top to bottom."]
    },
    "binary-tree-traversal": {
        "example": {"input": "root = [1,null,2,3]\nInorder / Preorder / Postorder", "output": "Inorder: [1,3,2]\nPreorder: [1,2,3]\nPostorder: [3,2,1]",
                    "explanation": "Inorder = Left,Root,Right. Preorder = Root,Left,Right. Postorder = Left,Right,Root."},
        "assumptions": []
    },
    "bitwise-and-numbers-range": {
        "example": {"input": "left = 5, right = 7", "output": "4",
                    "explanation": "5&6&7 = 101&110&111 = 100 = 4. Equivalently, find the common left prefix of left and right in binary."},
        "assumptions": ["0 ≤ left ≤ right ≤ 2³¹ − 1"]
    },
    "bubble-sort": {
        "example": {"input": "arr = [64, 34, 25, 12, 22, 11, 90]", "output": "[11, 12, 22, 25, 34, 64, 90]",
                    "explanation": "Repeatedly swap adjacent elements that are in the wrong order until the array is sorted."},
        "assumptions": ["Sort in ascending order.", "In-place sort, O(n²) time."]
    },
    "bucket-sort": {
        "example": {"input": "arr = [0.897, 0.565, 0.656, 0.123, 0.665, 0.343]", "output": "[0.123, 0.343, 0.565, 0.656, 0.665, 0.897]",
                    "explanation": "Distribute elements into buckets, sort each bucket, then concatenate."},
        "assumptions": ["Input values are uniformly distributed over [0, 1).", "O(n) average time."]
    },
    "bulls-and-cows": {
        "example": {"input": 'secret = "1807", guess = "7810"', "output": '"1A3B"',
                    "explanation": "1 bull (8 in position 1 both). 3 cows (1,7,0 are correct digits in wrong position)."},
        "assumptions": ["A bull = right digit, right position.", "A cow = right digit, wrong position (not already counted as bull)."]
    },
    "candy": {
        "example": {"input": "ratings = [1,0,2]", "output": "5",
                    "explanation": "Minimum candies: [2,1,2]. Each child must have ≥1 and children with higher ratings than neighbors get more."},
        "assumptions": ["Each child must have at least 1 candy.", "Children with a higher rating than an adjacent neighbor get more candies than that neighbor."]
    },
    "case-specific-sorting-strings": {
        "example": {"input": 's = "dcBA"', "output": '"dcBA" sorted as "ABcd" but preserving case positions → "ABcd"',
                    "explanation": "Sort uppercase letters among themselves and lowercase among themselves, keeping case positions fixed."},
        "assumptions": ["Uppercase letters maintain relative uppercase positions; lowercase letters maintain relative lowercase positions."]
    },
    "climbing-stairs": {
        "example": {"input": "n = 4", "output": "5",
                    "explanation": "Ways: (1+1+1+1), (1+1+2), (1+2+1), (2+1+1), (2+2) = 5 ways."},
        "assumptions": ["You can take 1 or 2 steps at a time.", "n ≥ 1"]
    },
    "clone-graph": {
        "example": {"input": "adjList = [[2,4],[1,3],[2,4],[1,3]]", "output": "Deep copy of the same graph",
                    "explanation": "Node 1 connects to 2 and 4. Clone every node and edge such that no original node is reused."},
        "assumptions": ["The graph is connected and undirected.", "Each node has a unique value 1..n.", "Return the clone of the given node."]
    },
    "closest-binary-search-tree-value": {
        "example": {"input": "root = [4,2,5,1,3], target = 3.714286", "output": "4",
                    "explanation": "|4-3.714|=0.286 vs |3-3.714|=0.714. Node 4 is closest."},
        "assumptions": ["Tree is a valid BST.", "If two values are equally close, return the smaller one."]
    },
    "coin-change": {
        "example": {"input": "coins = [1,5,10,25], amount = 36", "output": "3",
                    "explanation": "25 + 10 + 1 = 36 uses 3 coins (minimum)."},
        "assumptions": ["You have an infinite supply of each coin.", "Return -1 if the amount cannot be made."]
    },
    "combination-sum": {
        "example": {"input": "candidates = [2,3,6,7], target = 7", "output": "[[2,2,3],[7]]",
                    "explanation": "2+2+3=7 and 7=7. Each candidate can be reused any number of times."},
        "assumptions": ["All candidate numbers are distinct.", "Same number may be chosen multiple times.", "Return all unique combinations (no duplicate sets)."]
    },
    "combinations": {
        "example": {"input": "n = 4, k = 2", "output": "[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]",
                    "explanation": "All 2-element subsets of {1,2,3,4}, listed in sorted order."},
        "assumptions": ["1 ≤ k ≤ n ≤ 20", "Combinations are unique (order doesn't matter)."]
    },
    "compare-version-numbers": {
        "example": {"input": 'version1 = "1.01", version2 = "1.001"', "output": "0",
                    "explanation": "Ignoring leading zeros: 1.1 == 1.1. Return -1 if v1 < v2, 1 if v1 > v2, 0 if equal."},
        "assumptions": ["Revisions with leading zeros are equal to the same revision without (e.g. '01' == '1').", "Missing revisions default to 0."]
    },
    "construct-binary-tree": {
        "example": {"input": "preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]", "output": "[3,9,20,null,null,15,7]",
                    "explanation": "Preorder's first element (3) is the root. Its position in inorder splits left [9] and right [15,20,7] subtrees."},
        "assumptions": ["No duplicate values in the tree.", "Both arrays represent the same tree."]
    },
    "construct-binary-tree-inorder-postorder": {
        "example": {"input": "inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]", "output": "[3,9,20,null,null,15,7]",
                    "explanation": "Postorder's last element (3) is the root. Inorder splits left [9] and right [15,20,7] subtrees."},
        "assumptions": ["No duplicate values.", "Both arrays represent the same tree."]
    },
    "construct-binary-tree-preorder-inorder": {
        "example": {"input": "preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]", "output": "[3,9,20,null,null,15,7]",
                    "explanation": "Preorder's first element is root. Find it in inorder to split left/right subtrees."},
        "assumptions": ["No duplicate values."]
    },
    "container-most-water": {
        "example": {"input": "height = [1,8,6,2,5,4,8,3,7]", "output": "49",
                    "explanation": "Lines at index 1 (height 8) and index 8 (height 7): min(8,7)*(8-1) = 7*7 = 49."},
        "assumptions": ["You cannot slant the container.", "Return the maximum area of water that can be trapped."]
    },
    "container-with-most-water": {
        "example": {"input": "height = [1,8,6,2,5,4,8,3,7]", "output": "49",
                    "explanation": "Lines at index 1 and 8 form container with area min(8,7)*(8-1) = 49."},
        "assumptions": ["n ≥ 2", "Cannot slant the container."]
    },
    "contains-duplicate": {
        "example": {"input": "nums = [1,2,3,1]", "output": "true",
                    "explanation": "1 appears twice. Return true if any value appears at least twice."},
        "assumptions": []
    },
    "convert-sorted-array-bst": {
        "example": {"input": "nums = [-10,-3,0,5,9]", "output": "[0,-3,9,-10,null,5]",
                    "explanation": "Middle element (0) is root. Recursively build left from [-10,-3] and right from [5,9]."},
        "assumptions": ["The array is sorted in ascending order.", "Resulting BST must be height-balanced."]
    },
    "convert-sorted-list-bst": {
        "example": {"input": "head = [-10,-3,0,5,9]", "output": "[0,-3,9,-10,null,5]",
                    "explanation": "Find mid of linked list as root, recursively build left and right halves."},
        "assumptions": ["Input is a sorted singly linked list.", "Result must be height-balanced BST."]
    },
    "copy-list-random-pointer": {
        "example": {"input": "head = [[7,null],[13,0],[11,4],[10,2],[1,0]]", "output": "Deep copy of list",
                    "explanation": "Each node has val, next, and random pointer. Create new nodes with same structure; no original nodes in result."},
        "assumptions": ["random pointer can point to any node or null.", "Return the head of the deep-copied list."]
    },
    "count-and-say": {
        "example": {"input": "n = 4", "output": '"1211"',
                    "explanation": 'n=1:"1", n=2:"11"(one 1), n=3:"21"(two 1s), n=4:"1211"(one 2, one 1).'},
        "assumptions": ["n ≥ 1", "Read digits aloud: count consecutive identical digits."]
    },
    "count-complete-tree-nodes": {
        "example": {"input": "root = [1,2,3,4,5,6]", "output": "6",
                    "explanation": "Count all nodes. Use binary search on last level for O(log²n) solution."},
        "assumptions": ["Tree is a complete binary tree (all levels fully filled except possibly the last, filled left to right)."]
    },
    "count-primes": {
        "example": {"input": "n = 10", "output": "4",
                    "explanation": "Primes less than 10: 2, 3, 5, 7. Count = 4."},
        "assumptions": ["Count primes strictly less than n.", "Use Sieve of Eratosthenes for efficiency."]
    },
    "counting-bits": {
        "example": {"input": "n = 5", "output": "[0,1,1,2,1,2]",
                    "explanation": "0→0 bits, 1→1 bit, 2→1 bit (10), 3→2 bits (11), 4→1 bit (100), 5→2 bits (101)."},
        "assumptions": ["Return array ans of length n+1 where ans[i] = number of 1s in binary representation of i."]
    },
    "counting-sort": {
        "example": {"input": "arr = [4,2,2,8,3,3,1]", "output": "[1,2,2,3,3,4,8]",
                    "explanation": "Count occurrences of each value, then reconstruct sorted array. O(n+k) time."},
        "assumptions": ["Works best when values are in a known, bounded range.", "O(n+k) time, O(k) space."]
    },
    "course-schedule": {
        "example": {"input": "numCourses = 2, prerequisites = [[1,0]]", "output": "true",
                    "explanation": "Take course 0 first, then course 1. No cycle → possible."},
        "assumptions": ["prerequisites[i] = [a, b] means b must be taken before a.", "Return true if you can finish all courses (no cycle exists in the prerequisite graph)."]
    },
    "decode-ways": {
        "example": {"input": 's = "226"', "output": "3",
                    "explanation": '"226" → "BZ"(2,26), "VF"(22,6), "BBF"(2,2,6). 3 ways.'},
        "assumptions": ["'A'=1, 'B'=2, ..., 'Z'=26.", "Leading zeros like '06' are invalid.", "Return number of ways to decode the string."]
    },
    "delete-node-linked-list": {
        "example": {"input": "head = [4,5,1,9], node = 5", "output": "[4,1,9]",
                    "explanation": "Copy next node's value into current node and skip next node: 5→copy 1→node.next = node.next.next."},
        "assumptions": ["You are given direct access to the node to delete (not the head).", "The node is not the tail.", "All values are unique."]
    },
    "detect-cycle": {
        "example": {"input": "head = [3,2,0,-4], pos = 1 (tail connects to index 1)", "output": "true (node at index 1)",
                    "explanation": "Floyd's cycle detection: slow moves 1 step, fast moves 2. If they meet, a cycle exists."},
        "assumptions": ["pos = -1 means no cycle.", "Return the node where the cycle begins, or null."]
    },
    "divide-two-integers": {
        "example": {"input": "dividend = 10, divisor = 3", "output": "3",
                    "explanation": "10 / 3 = 3.33, truncated toward zero = 3."},
        "assumptions": ["Cannot use multiplication, division, or mod operators.", "Truncate toward zero.", "Clamp result to 32-bit signed integer range."]
    },
    "edit-distance": {
        "example": {"input": 'word1 = "horse", word2 = "ros"', "output": "3",
                    "explanation": "horse→rorse(replace h→r)→rose(remove r)→ros(remove e). 3 operations."},
        "assumptions": ["Operations: insert a character, delete a character, replace a character.", "Return minimum number of operations to convert word1 to word2."]
    },
    "evaluate-reverse-polish-notation": {
        "example": {"input": 'tokens = ["2","1","+","3","*"]', "output": "9",
                    "explanation": "((2+1)*3) = 9. Operators act on the two most recent operands."},
        "assumptions": ["Valid operators: +, -, *, /", "Division truncates toward zero.", "Input is always valid RPN."]
    },
    "excel-sheet-column-number": {
        "example": {"input": 'columnTitle = "AB"', "output": "28",
                    "explanation": "A=1, B=2, ..., Z=26. AB = 1*26 + 2 = 28."},
        "assumptions": ["Input is a valid Excel column title (A-Z, AA-AZ, ...)."]
    },
    "excel-sheet-column-title": {
        "example": {"input": "columnNumber = 28", "output": '"AB"',
                    "explanation": "28 = 1*26 + 2 → 'A'+'B' = 'AB'. It's base-26 but with no zero (A=1, ..., Z=26)."},
        "assumptions": ["1-indexed (no column 0).", "Like base-26 but with A=1 instead of 0."]
    },
    "factor-combinations": {
        "example": {"input": "n = 12", "output": "[[2,6],[2,2,3],[3,4]]",
                    "explanation": "12 = 2×6 = 2×2×3 = 3×4. The number itself (12) is not included."},
        "assumptions": ["Factors must be ≥ 2.", "Each combination is non-decreasing.", "n is guaranteed to be > 1."]
    },
    "factorial-trailing-zeroes": {
        "example": {"input": "n = 5", "output": "1",
                    "explanation": "5! = 120, which has 1 trailing zero. Count factors of 5 in n!."},
        "assumptions": ["Each trailing zero comes from a factor of 10 = 2×5. Since 2s are plentiful, count multiples of 5."]
    },
    "factorial-trailing-zeros": {
        "example": {"input": "n = 25", "output": "6",
                    "explanation": "25! has ⌊25/5⌋+⌊25/25⌋ = 5+1 = 6 factors of 5, hence 6 trailing zeros."},
        "assumptions": []
    },
    "find-k-pairs-smallest-sums": {
        "example": {"input": "nums1 = [1,7,11], nums2 = [2,4,6], k = 3", "output": "[[1,2],[1,4],[1,6]]",
                    "explanation": "Smallest 3 sums: 1+2=3, 1+4=5, 1+6=7. Use a min-heap."},
        "assumptions": ["Both arrays sorted in non-decreasing order.", "Return k pairs (u,v) with smallest u+v."]
    },
    "find-leaves-binary-tree": {
        "example": {"input": "root = [1,2,3,4,5]", "output": "[[4,5,3],[2],[1]]",
                    "explanation": "Remove leaves [4,5,3], then [2], then [1]. Group nodes by their height from leaves."},
        "assumptions": ["A leaf is a node with no children.", "Return groups of nodes removed at each step."]
    },
    "find-median-data-stream": {
        "example": {"input": "addNum(1), addNum(2), findMedian() → 1.5, addNum(3), findMedian() → 2.0", "output": "1.5, 2.0",
                    "explanation": "Use two heaps: max-heap for left half, min-heap for right half. Median is the top(s)."},
        "assumptions": ["findMedian returns exact median (average of two middle values if even count).", "O(log n) add, O(1) findMedian."]
    },
    "find-minimum-rotated-sorted-array": {
        "example": {"input": "nums = [3,4,5,1,2]", "output": "1",
                    "explanation": "Array was rotated 3 times. Minimum is 1 at index 3. Use binary search: O(log n)."},
        "assumptions": ["Array has no duplicates.", "Originally sorted in ascending order, then rotated."]
    },
    "find-peak-element": {
        "example": {"input": "nums = [1,2,3,1]", "output": "2",
                    "explanation": "nums[2]=3 is a peak since nums[1]=2 < 3 > nums[3]=1. Return any peak index."},
        "assumptions": ["nums[-1] = nums[n] = -∞ (boundaries are -infinity).", "Any peak index is acceptable.", "O(log n) solution exists using binary search."]
    },
    "first-bad-version": {
        "example": {"input": "n = 5, bad = 4", "output": "4",
                    "explanation": "isBadVersion(3)=false, isBadVersion(4)=true. Binary search to find first bad version."},
        "assumptions": ["All versions after the first bad one are also bad.", "Minimize calls to the API."]
    },
    "flatten-binary-tree": {
        "example": {"input": "root = [1,2,5,3,4,null,6]", "output": "[1,null,2,null,3,null,4,null,5,null,6]",
                    "explanation": "Flatten in-place to a linked list in pre-order: 1→2→3→4→5→6, all right pointers, no left."},
        "assumptions": ["Flatten in-place — don't create new nodes.", "All left pointers become null.", "Right pointers follow pre-order sequence."]
    },
    "flatten-binary-tree-linked-list": {
        "example": {"input": "root = [1,2,5,3,4,null,6]", "output": "[1,null,2,null,3,null,4,null,5,null,6]",
                    "explanation": "Pre-order traversal: 1,2,3,4,5,6. Chain via right pointers, set all left to null."},
        "assumptions": []
    },
    "flip-game": {
        "example": {"input": 's = "++++"', "output": '["--++","+--+","++--"]',
                    "explanation": "Replace each '++' pair with '--'. Return all states after one move."},
        "assumptions": ["Only one move per call.", "A move flips exactly one '++' into '--'."]
    },
    "form-largest-number": {
        "example": {"input": "nums = [3,30,34,5,9]", "output": '"9534330"',
                    "explanation": "Sort by comparing concatenation: 9>5>34>3>30 when determining which comes first."},
        "assumptions": ["Output is a string (may have leading zeros if all are 0).", "Custom comparator: a+b vs b+a as strings."]
    },
    "fraction-recurring-decimal": {
        "example": {"input": "numerator = 1, denominator = 3", "output": '"0.(3)"',
                    "explanation": "1/3 = 0.333... Repeating part goes in parentheses. Detect cycle via remainder tracking."},
        "assumptions": ["Non-repeating part has no parentheses.", "Integer result has no decimal point.", "Handle negative fractions."]
    },
    "game-of-life": {
        "example": {"input": "board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]", "output": "[[0,0,0],[1,0,1],[0,1,1],[0,1,0]]",
                    "explanation": "Apply Conway's rules simultaneously: live cell with <2 or >3 neighbors dies; dead cell with exactly 3 neighbors lives."},
        "assumptions": ["Update in-place.", "All cells update simultaneously (use encoded states to avoid overwriting needed values)."]
    },
    "gas-station": {
        "example": {"input": "gas = [1,2,3,4,5], cost = [3,4,5,1,2]", "output": "3",
                    "explanation": "Start at station 3 (0-indexed): 4-1=3→3+5-2=6→6+1-3=4→4+2-4=2→2+3-5=0. Completes circuit."},
        "assumptions": ["If solution exists, it is guaranteed unique.", "Return -1 if no solution."]
    },
    "generate-parentheses": {
        "example": {"input": "n = 3", "output": '["((()))","(()())","(())()","()(())","()()()"]',
                    "explanation": "All combinations of 3 pairs of well-formed parentheses."},
        "assumptions": ["Only generate valid combinations (never more close than open at any prefix)."]
    },
    "get-target-arithmetic": {
        "example": {"input": "nums = [1,2,3], target = 6", "output": '"1+2+3=6"',
                    "explanation": "Insert +, -, or * between digits to reach target. Return expression if found."},
        "assumptions": ["Numbers cannot have leading zeros.", "Return any valid expression, or empty string if impossible."]
    },
    "graph-valid-tree": {
        "example": {"input": "n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]", "output": "true",
                    "explanation": "A graph is a valid tree if it has n-1 edges and is fully connected with no cycles."},
        "assumptions": ["n nodes labeled 0..n-1.", "No duplicate edges.", "Undirected graph."]
    },
    "gray-code": {
        "example": {"input": "n = 2", "output": "[0,1,3,2]",
                    "explanation": "00→01→11→10. Each consecutive pair differs by exactly 1 bit. This is the standard Gray code."},
        "assumptions": ["Return any valid n-bit Gray code sequence starting at 0.", "Sequence must have 2ⁿ elements."]
    },
    "group-anagrams": {
        "example": {"input": 'strs = ["eat","tea","tan","ate","nat","bat"]', "output": '[["bat"],["nat","tan"],["ate","eat","tea"]]',
                    "explanation": "Group strings that are anagrams of each other. Sort each word as key."},
        "assumptions": ["Input strings contain only lowercase letters.", "Output order doesn't matter."]
    },
    "guess-number-higher-lower": {
        "example": {"input": "n = 10, pick = 6", "output": "6",
                    "explanation": "Binary search: guess(5)→1(higher), guess(8)→-1(lower), guess(6)→0(correct)."},
        "assumptions": ["guess(num) returns -1 if too high, 1 if too low, 0 if correct.", "1 ≤ pick ≤ n."]
    },
    "h-index": {
        "example": {"input": "citations = [3,0,6,1,5]", "output": "3",
                    "explanation": "h=3: researcher has 3 papers with ≥3 citations. Papers: 6,5,3 ≥ 3; but 4 papers don't all have ≥4 cites."},
        "assumptions": ["h-index: largest h such that h papers have ≥h citations each."]
    },
    "happy-number": {
        "example": {"input": "n = 19", "output": "true",
                    "explanation": "1²+9²=82 → 8²+2²=68 → 6²+8²=100 → 1²+0+0=1. Reaches 1, so happy!"},
        "assumptions": ["Repeatedly sum squares of digits. If reaches 1 → happy. If enters cycle not including 1 → not happy."]
    },
    "heap-sort": {
        "example": {"input": "arr = [12,11,13,5,6,7]", "output": "[5,6,7,11,12,13]",
                    "explanation": "Build max-heap, then repeatedly extract maximum to end of array. O(n log n), in-place."},
        "assumptions": ["Sort in ascending order.", "In-place, O(n log n) time."]
    },
    "house-robber": {
        "example": {"input": "nums = [2,7,9,3,1]", "output": "12",
                    "explanation": "Rob houses 1,3,5 (0-indexed: 0,2,4): 2+9+1=12. Cannot rob adjacent houses."},
        "assumptions": ["Adjacent houses cannot both be robbed.", "Return maximum amount you can rob."]
    },
    "implement-queue-array": {
        "example": {"input": "enqueue(1), enqueue(2), dequeue() → 1, peek() → 2", "output": "1, 2",
                    "explanation": "FIFO: First In First Out. Enqueue adds to back, dequeue removes from front."},
        "assumptions": ["Implement using array.", "enqueue: O(1) amortized, dequeue: O(1)."]
    },
    "implement-queue-stacks": {
        "example": {"input": "push(1), push(2), peek() → 1, pop() → 1, empty() → false", "output": "1, 1, false",
                    "explanation": "Use two stacks: push to stack1; on pop/peek, if stack2 empty, move all from stack1."},
        "assumptions": ["Simulate FIFO queue using only two LIFO stacks.", "pop/peek amortized O(1)."]
    },
    "implement-stack-array": {
        "example": {"input": "push(1), push(2), top() → 2, pop() → 2, empty() → false", "output": "2, 2, false",
                    "explanation": "LIFO: Last In First Out. All operations O(1) with array."},
        "assumptions": []
    },
    "implement-stack-queues": {
        "example": {"input": "push(1), push(2), top() → 2, pop() → 2", "output": "2, 2",
                    "explanation": "Use two queues. On push, enqueue to q2 then move all q1 elements to q2, swap q1 and q2."},
        "assumptions": ["Simulate LIFO stack using only queues.", "push O(n), pop/top O(1) — or vice versa."]
    },
    "implement-strstr": {
        "example": {"input": 'haystack = "hello", needle = "ll"', "output": "2",
                    "explanation": "'ll' first occurs at index 2 in 'hello'. Return -1 if not found. Return 0 if needle is empty."},
        "assumptions": ["Return the index of the first occurrence of needle in haystack.", "Return -1 if not found."]
    },
    "implement-trie": {
        "example": {"input": 'insert("apple"), search("apple") → true, search("app") → false, startsWith("app") → true, insert("app"), search("app") → true',
                    "output": "true, false, true, true",
                    "explanation": "Trie stores strings by prefix. search checks complete word; startsWith checks any prefix."},
        "assumptions": ["Only lowercase English letters.", "Implement insert, search, and startsWith."]
    },
    "increasing-triplet-subsequence": {
        "example": {"input": "nums = [2,1,5,0,4,6]", "output": "true",
                    "explanation": "Triplet (1,4,6) or (1,5,6) satisfies i<j<k, nums[i]<nums[j]<nums[k]."},
        "assumptions": ["Return true if such a triplet exists.", "O(n) time, O(1) space solution exists."]
    },
    "inorder-successor-bst": {
        "example": {"input": "root = [2,1,3], p = 1", "output": "2",
                    "explanation": "In-order of BST: 1,2,3. Successor of 1 is 2 (next larger value)."},
        "assumptions": ["In-order successor = next node in in-order traversal (next larger value).", "Return null if no successor."]
    },
    "insert-interval": {
        "example": {"input": "intervals = [[1,3],[6,9]], newInterval = [2,5]", "output": "[[1,5],[6,9]]",
                    "explanation": "[2,5] overlaps [1,3], merge to [1,5]. Then [6,9] doesn't overlap."},
        "assumptions": ["Existing intervals are non-overlapping and sorted.", "Insert and merge as needed.", "Return merged result."]
    },
    "insertion-sort": {
        "example": {"input": "arr = [12,11,13,5,6]", "output": "[5,6,11,12,13]",
                    "explanation": "Build sorted portion left-to-right, inserting each new element into correct position. O(n²)."},
        "assumptions": ["Sort in ascending order.", "Stable sort, O(n²) worst case, O(n) best (already sorted)."]
    },
    "integer-break": {
        "example": {"input": "n = 10", "output": "36",
                    "explanation": "10 = 3+3+4, product = 3×3×4 = 36. Optimal to break into 3s (and possibly a 2 or 4)."},
        "assumptions": ["n ≥ 2", "Split n into at least 2 positive integers that sum to n, maximizing product."]
    },
    "integer-english-words": {
        "example": {"input": "num = 1234567", "output": '"One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"',
                    "explanation": "Process in groups of three digits from right: ones, thousands, millions, billions."},
        "assumptions": ["0 ≤ num ≤ 2³¹ − 1", "Special case: num = 0 → 'Zero'."]
    },
    "intersection-two-arrays": {
        "example": {"input": "nums1 = [1,2,2,1], nums2 = [2,2]", "output": "[2]",
                    "explanation": "Intersection contains each unique element that appears in both arrays. 2 appears in both, but result has no duplicates."},
        "assumptions": ["Return unique intersection elements.", "Order does not matter."]
    },
    "intersection-two-linked-lists": {
        "example": {"input": "listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], intersection starts at node with value 8",
                    "output": "Reference to node with value 8",
                    "explanation": "Both lists share a common suffix starting at node 8. Use two-pointer trick: advance each pointer; when one reaches end, redirect to other list's head."},
        "assumptions": ["Return the node where the two lists intersect, or null.", "No cycles.", "The linked lists retain their original structure after the function."]
    },
    "invert-binary-tree": {
        "example": {"input": "root = [4,2,7,1,3,6,9]", "output": "[4,7,2,9,6,3,1]",
                    "explanation": "Swap left and right children at every node recursively."},
        "assumptions": []
    },
    "isomorphic-strings": {
        "example": {"input": 's = "egg", t = "add"', "output": "true",
                    "explanation": "e→a, g→d. Consistent mapping. No two characters in s may map to the same character in t."},
        "assumptions": ["All characters in s map consistently to characters in t and vice versa.", "A character may map to itself."]
    },
    "jump-game": {
        "example": {"input": "nums = [2,3,1,1,4]", "output": "true",
                    "explanation": "Jump 1 step from index 0 to 1, then 3 steps to last index. Always track max reachable index."},
        "assumptions": ["nums[i] = max jump length from index i.", "Return true if you can reach the last index."]
    },
    "knapsack-problem": {
        "example": {"input": "weights = [2,3,4,5], values = [3,4,5,6], capacity = 8", "output": "10",
                    "explanation": "Select items with weight 3 (value 4) + weight 5 (value 6) = 8 ≤ 8, total value 10 (max)."},
        "assumptions": ["0/1 knapsack: each item can be taken at most once.", "Maximize total value without exceeding capacity."]
    },
    "kth-largest-element": {
        "example": {"input": "nums = [3,2,1,5,6,4], k = 2", "output": "5",
                    "explanation": "2nd largest is 5. Use min-heap of size k or QuickSelect for O(n) average."},
        "assumptions": ["Find kth largest in the array (not distinct kth largest).", "1 ≤ k ≤ nums.length."]
    },
    "kth-smallest-bst": {
        "example": {"input": "root = [3,1,4,null,2], k = 1", "output": "1",
                    "explanation": "In-order traversal of BST gives sorted order: 1,2,3,4. 1st smallest is 1."},
        "assumptions": ["1 ≤ k ≤ number of nodes.", "All node values are unique."]
    },
    "kth-smallest-largest": {
        "example": {"input": "nums = [7,10,4,3,20,15], k = 3", "output": "7 (3rd smallest)",
                    "explanation": "Sorted: [3,4,7,10,15,20]. 3rd smallest = 7. Use QuickSelect or heap."},
        "assumptions": ["Find kth smallest (or largest with similar approach).", "Elements may not be sorted."]
    },
    "largest-bst-subtree": {
        "example": {"input": "root = [10,5,15,1,8,null,7]", "output": "3",
                    "explanation": "The subtree rooted at 5 (nodes 1,5,8) is a valid BST of size 3. Subtree at 15 is invalid (7 < 15 but in right subtree)."},
        "assumptions": ["Return the size (number of nodes) of the largest subtree that is a BST.", "A single node is always a BST."]
    },
    "largest-number": {
        "example": {"input": "nums = [10,2]", "output": '"210"',
                    "explanation": "Sort by comparing a+b vs b+a as strings: '210' > '102', so 2 before 10."},
        "assumptions": ["Return result as a string.", "Handle edge case where all are 0s."]
    },
    "largest-rectangle-histogram": {
        "example": {"input": "heights = [2,1,5,6,2,3]", "output": "10",
                    "explanation": "Largest rectangle spans heights 5,6 with height 5: area = 5×2 = 10. Use monotonic stack."},
        "assumptions": ["Each bar has width 1.", "Return maximum area of a rectangle formed in the histogram."]
    },
    "length-last-word": {
        "example": {"input": 's = "Hello World"', "output": "5",
                    "explanation": "Last word is 'World' with length 5. Trim trailing spaces first."},
        "assumptions": ["A word is a maximal substring with no spaces.", "At least one word in the string."]
    },
    "letter-combinations-phone-number": {
        "example": {"input": 'digits = "23"', "output": '["ad","ae","af","bd","be","bf","cd","ce","cf"]',
                    "explanation": "2→abc, 3→def. All combinations of one letter from each digit group."},
        "assumptions": ["Return empty list if input is empty.", "Phone mapping: 2=abc, 3=def, 4=ghi, 5=jkl, 6=mno, 7=pqrs, 8=tuv, 9=wxyz."]
    },
    "level-order-ii": {
        "example": {"input": "root = [3,9,20,null,null,15,7]", "output": "[[15,7],[9,20],[3]]",
                    "explanation": "Bottom-up level order: deepest level first. Reverse of standard level order."},
        "assumptions": []
    },
    "linked-list-cycle": {
        "example": {"input": "head = [3,2,0,-4], pos = 1", "output": "true",
                    "explanation": "Tail connects back to node at index 1. Floyd's algorithm: fast and slow pointers meet iff cycle exists."},
        "assumptions": ["pos = -1 means no cycle.", "Return true/false (or the cycle node for follow-up)."]
    },
    "longest-common-prefix": {
        "example": {"input": 'strs = ["flower","flow","flight"]', "output": '"fl"',
                    "explanation": "All strings start with 'fl'. 'flo' is not common to 'flight'."},
        "assumptions": ["Return empty string if no common prefix.", "All inputs are lowercase letters."]
    },
    "longest-common-subsequence": {
        "example": {"input": 'text1 = "abcde", text2 = "ace"', "output": "3",
                    "explanation": "LCS is 'ace' with length 3. Classic DP: dp[i][j] = LCS of text1[:i] and text2[:j]."},
        "assumptions": ["Subsequence: characters in order but not necessarily contiguous.", "Return length of LCS."]
    },
    "longest-consecutive-sequence": {
        "example": {"input": "nums = [100,4,200,1,3,2]", "output": "4",
                    "explanation": "Consecutive sequence: 1,2,3,4 has length 4. Use hash set for O(n)."},
        "assumptions": ["O(n) time required.", "Sequence values must be consecutive integers."]
    },
    "longest-increasing-path-matrix": {
        "example": {"input": "matrix = [[9,9,4],[6,6,8],[2,1,1]]", "output": "4",
                    "explanation": "Longest increasing path: 1→2→6→9, length 4. Use DFS with memoization."},
        "assumptions": ["Can move up/down/left/right.", "Cannot move diagonally or revisit cells.", "Each step must be strictly increasing."]
    },
    "longest-increasing-subsequence": {
        "example": {"input": "nums = [10,9,2,5,3,7,101,18]", "output": "4",
                    "explanation": "LIS: [2,3,7,101] or [2,5,7,101], length 4. O(n log n) with patience sorting."},
        "assumptions": ["Subsequence must be strictly increasing.", "Return length (not the subsequence itself)."]
    },
    "longest-substring-2-unique": {
        "example": {"input": 's = "eceba"', "output": "3",
                    "explanation": '"ece" contains 2 unique chars and has length 3. Sliding window with char count map.'},
        "assumptions": ["At most 2 distinct characters.", "Return length of the longest such substring."]
    },
    "longest-substring-without-repeating": {
        "example": {"input": 's = "abcabcbb"', "output": "3",
                    "explanation": '"abc" has length 3 with no repeating characters. Sliding window with seen-char index map.'},
        "assumptions": ["Return length of the longest substring without repeating characters."]
    },
    "longest-substring-without-repeating-characters": {
        "example": {"input": 's = "pwwkew"', "output": "3",
                    "explanation": '"wke" is the longest without repeating characters, length 3.'},
        "assumptions": []
    },
    "longest-valid-parentheses": {
        "example": {"input": 's = ")()())"', "output": "4",
                    "explanation": '"()()" is the longest valid substring, length 4. Use stack or DP.'},
        "assumptions": ["Return the length of the longest valid parentheses substring.", "Valid = properly nested '(' and ')'."]
    },
    "lowest-common-ancestor": {
        "example": {"input": "root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1", "output": "3",
                    "explanation": "LCA of 5 and 1 is 3 (root). LCA = deepest node that has both p and q as descendants."},
        "assumptions": ["All node values are unique.", "Both p and q exist in the tree.", "A node is a descendant of itself."]
    },
    "lowest-common-ancestor-binary-tree": {
        "example": {"input": "root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4", "output": "5",
                    "explanation": "5 is an ancestor of 4, so LCA(5,4) = 5."},
        "assumptions": []
    },
    "lowest-common-ancestor-bst": {
        "example": {"input": "root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8", "output": "6",
                    "explanation": "In BST: if both p,q < root → go left; if both > root → go right; else root is LCA."},
        "assumptions": ["Tree is a valid BST.", "p and q are distinct and both exist in the tree."]
    },
    "lru-cache": {
        "example": {"input": "LRUCache(2): put(1,1), put(2,2), get(1)→1, put(3,3), get(2)→-1, put(4,4), get(1)→-1, get(3)→3, get(4)→4",
                    "output": "1, -1, -1, 3, 4",
                    "explanation": "Capacity 2. After put(3,3), key 2 is evicted (least recently used). Use HashMap + doubly linked list for O(1) ops."},
        "assumptions": ["get: return value or -1 if key not found.", "put: insert or update; evict LRU key if capacity exceeded.", "O(1) for both operations."]
    },
    "majority-element": {
        "example": {"input": "nums = [2,2,1,1,1,2,2]", "output": "2",
                    "explanation": "2 appears 4 times (> 7/2). Boyer-Moore voting: cancel out different elements."},
        "assumptions": ["Majority element appears more than n/2 times.", "Guaranteed to always exist."]
    },
    "matrix-chain-multiplication": {
        "example": {"input": "dims = [40, 20, 30, 10, 30]", "output": "26000",
                    "explanation": "4 matrices: A(40×20), B(20×30), C(30×10), D(10×30). Optimal: ((AB)C)D = 26000 multiplications."},
        "assumptions": ["Find parenthesisation that minimizes scalar multiplications.", "Classic interval DP."]
    },
    "max-points-line": {
        "example": {"input": "points = [[1,1],[2,2],[3,3]]", "output": "3",
                    "explanation": "All 3 points are on the line y=x. For each point, count others with same slope."},
        "assumptions": ["At least 1 point.", "Use fractions (gcd) to represent slope exactly.", "Handle vertical lines (undefined slope) separately."]
    },
    "maximal-rectangle": {
        "example": {"input": 'matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]', "output": "6",
                    "explanation": "Largest rectangle of 1s has area 6. Treat each row as histogram base and apply largest-rectangle-in-histogram."},
        "assumptions": ["Matrix contains only '0' and '1'.", "O(m*n) solution using histogram approach."]
    },
    "maximal-square": {
        "example": {"input": 'matrix = [["1","0","1","1","1"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]', "output": "9",
                    "explanation": "Largest square of 1s has side 3, area 9. DP: dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1 if cell is '1'."},
        "assumptions": ["Matrix contains only '0' and '1'.", "Return area of largest square."]
    },
    "maximum-binary-gap": {
        "example": {"input": "n = 22", "output": "2",
                    "explanation": "22 = 10110 in binary. Gaps between consecutive 1s: positions 1,2,4 → gaps 1,2. Maximum = 2."},
        "assumptions": ["Binary gap = distance between two consecutive 1s.", "Return 0 if fewer than two 1s."]
    },
    "maximum-depth-binary-tree": {
        "example": {"input": "root = [3,9,20,null,null,15,7]", "output": "3",
                    "explanation": "Deepest path: 3→20→15 or 3→20→7, depth = 3."},
        "assumptions": ["Depth = number of nodes along the longest root-to-leaf path.", "Empty tree has depth 0."]
    },
    "maximum-product-word-lengths": {
        "example": {"input": 'words = ["abcw","baz","foo","bar","xtfn","abcdef"]', "output": "16",
                    "explanation": '"abcw" (length 4) and "xtfn" (length 4) share no letters: 4×4=16. Use bitmask to check disjointness.'},
        "assumptions": ["Return max product len(a)*len(b) where a and b share no common letters.", "0 if no such pair exists."]
    },
    "maximum-subarray": {
        "example": {"input": "nums = [-2,1,-3,4,-1,2,1,-5,4]", "output": "6",
                    "explanation": "Subarray [4,-1,2,1] has sum 6. Kadane's algorithm: track current sum; reset if negative."},
        "assumptions": ["Return the largest sum contiguous subarray.", "Array has at least one element."]
    },
    "median-two-sorted-arrays": {
        "example": {"input": "nums1 = [1,3], nums2 = [2]", "output": "2.0",
                    "explanation": "Merged: [1,2,3], median = 2. For even length: average of two middle elements."},
        "assumptions": ["O(log(m+n)) time required.", "Use binary search on the smaller array."]
    },
    "meeting-rooms": {
        "example": {"input": "intervals = [[0,30],[5,10],[15,20]]", "output": "false",
                    "explanation": "[0,30] overlaps with [5,10]. Sort by start, check consecutive overlaps."},
        "assumptions": ["Return true if a person can attend all meetings (no overlaps)."]
    },
    "merge-intervals": {
        "example": {"input": "intervals = [[1,3],[2,6],[8,10],[15,18]]", "output": "[[1,6],[8,10],[15,18]]",
                    "explanation": "[1,3] and [2,6] overlap → merge to [1,6]. Sort by start time, then merge overlapping."},
        "assumptions": ["Intervals may be given in any order.", "Return merged list of non-overlapping intervals."]
    },
    "merge-k-sorted-arrays": {
        "example": {"input": "arrays = [[1,4,7],[2,5,8],[3,6,9]]", "output": "[1,2,3,4,5,6,7,8,9]",
                    "explanation": "Use min-heap of size k: push first element of each array, then repeatedly extract min and push next."},
        "assumptions": ["k sorted arrays.", "Use a min-heap for O(N log k) where N = total elements."]
    },
    "merge-k-sorted-lists": {
        "example": {"input": "lists = [[1,4,5],[1,3,4],[2,6]]", "output": "[1,1,2,3,4,4,5,6]",
                    "explanation": "Merge k sorted linked lists using a min-heap of size k tracking (value, list index)."},
        "assumptions": ["k sorted linked lists.", "O(N log k) using priority queue."]
    },
    "merge-overlapping-intervals": {
        "example": {"input": "intervals = [[1,3],[2,4],[6,8],[9,10]]", "output": "[[1,4],[6,8],[9,10]]",
                    "explanation": "[1,3] and [2,4] overlap → [1,4]. Sort by start, merge if next.start ≤ current.end."},
        "assumptions": []
    },
    "merge-sort": {
        "example": {"input": "arr = [38,27,43,3,9,82,10]", "output": "[3,9,10,27,38,43,82]",
                    "explanation": "Divide in half recursively, sort each half, then merge. O(n log n) time, O(n) space."},
        "assumptions": ["Stable sort.", "O(n log n) time, O(n) extra space."]
    },
    "merge-sorted-array": {
        "example": {"input": "nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3", "output": "[1,2,2,3,5,6]",
                    "explanation": "Merge nums2 into nums1 in-place. Start from the back to avoid overwriting."},
        "assumptions": ["nums1 has enough space (m+n) to hold all elements.", "Modify nums1 in-place."]
    },
    "merge-two-sorted-lists": {
        "example": {"input": "list1 = [1,2,4], list2 = [1,3,4]", "output": "[1,1,2,3,4,4]",
                    "explanation": "Compare heads, attach the smaller, advance that pointer. Append remaining."},
        "assumptions": ["Both lists are already sorted.", "Return the head of the merged list."]
    },
    "min-stack": {
        "example": {"input": "push(-2), push(0), push(-3), getMin()→-3, pop(), top()→0, getMin()→-2",
                    "output": "-3, 0, -2",
                    "explanation": "Maintain a second stack that tracks the minimum at each level. O(1) for all operations."},
        "assumptions": ["getMin() must return the minimum element in O(1).", "pop, push, top also O(1)."]
    },
    "minimum-depth-binary-tree": {
        "example": {"input": "root = [2,null,3,null,4,null,5,null,6]", "output": "5",
                    "explanation": "The only leaf is node 6 at depth 5. BFS finds the shallowest leaf."},
        "assumptions": ["Minimum depth = shortest path from root to any leaf.", "A leaf has no children."]
    },
    "minimum-height-trees": {
        "example": {"input": "n = 4, edges = [[1,0],[1,2],[1,3]]", "output": "[1]",
                    "explanation": "Node 1 is the center. Repeatedly remove leaf nodes; remaining node(s) are roots of minimum height trees."},
        "assumptions": ["Tree is undirected.", "At most 2 nodes can be MHT roots.", "Topological sort from leaves inward."]
    },
    "minimum-operations-distinct": {
        "example": {"input": "nums = [1,1,2,2,3,3]", "output": "3",
                    "explanation": "Remove from the end until all remaining elements are distinct. Count removals."},
        "assumptions": ["Can only remove from the end of the array.", "Return minimum removals for all distinct elements."]
    },
    "minimum-path-sum": {
        "example": {"input": "grid = [[1,3,1],[1,5,1],[4,2,1]]", "output": "7",
                    "explanation": "Path: 1→3→1→1→1 = 7. Only move right or down. DP: dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])."},
        "assumptions": ["Only move right or down.", "Return minimum path sum from top-left to bottom-right."]
    },
    "minimum-platforms-required": {
        "example": {"input": "arrivals = [900,940,950,1100,1500,1800], departures = [910,1200,1120,1130,1900,2000]", "output": "3",
                    "explanation": "At 950: trains at 940-1200, 950-1120, and 900-910 (departed) → 2 trains. Max overlap = 3 at some point."},
        "assumptions": ["Count maximum number of trains present at the station simultaneously.", "Sort both arrays; use two pointers."]
    },
    "minimum-size-subarray-sum": {
        "example": {"input": "target = 7, nums = [2,3,1,2,4,3]", "output": "2",
                    "explanation": "Subarray [4,3] sums to 7 with length 2. Sliding window: expand right, shrink left when sum ≥ target."},
        "assumptions": ["Return 0 if no valid subarray.", "Subarray must be contiguous."]
    },
    "minimum-window-substring": {
        "example": {"input": 's = "ADOBECODEBANC", t = "ABC"', "output": '"BANC"',
                    "explanation": "Smallest window containing A, B, C is 'BANC'. Sliding window with character frequency map."},
        "assumptions": ["Return empty string if no valid window.", "All characters in t must appear in window (with correct frequency)."]
    },
    "missing-number": {
        "example": {"input": "nums = [3,0,1]", "output": "2",
                    "explanation": "Array contains 0..3 except 2. Use sum formula: n*(n+1)/2 - sum(nums), or XOR."},
        "assumptions": ["nums contains n distinct numbers in range [0, n].", "Exactly one number is missing."]
    },
    "missing-number-bit": {
        "example": {"input": "nums = [9,6,4,2,3,5,7,0,1]", "output": "8",
                    "explanation": "XOR all indices 0..n with all values. Paired numbers cancel; missing number remains."},
        "assumptions": []
    },
    "move-zeroes": {
        "example": {"input": "nums = [0,1,0,3,12]", "output": "[1,3,12,0,0]",
                    "explanation": "Move all zeros to end while preserving order of non-zeros. Two-pointer in-place."},
        "assumptions": ["In-place, no extra array.", "Maintain relative order of non-zero elements."]
    },
    "multiply-strings": {
        "example": {"input": 'num1 = "123", num2 = "456"', "output": '"56088"',
                    "explanation": "Simulate grade-school multiplication digit by digit. result[i+j] and result[i+j+1] store partial products."},
        "assumptions": ["No leading zeros (except '0' itself).", "Inputs are non-negative integers as strings."]
    },
    "number-1-bits": {
        "example": {"input": "n = 00000000000000000000000000001011", "output": "3",
                    "explanation": "1011 has three '1' bits. Use n & (n-1) to clear the lowest set bit repeatedly."},
        "assumptions": ["Input is a 32-bit unsigned integer.", "Return number of '1' bits (Hamming weight)."]
    },
    "number-connected-components": {
        "example": {"input": "n = 5, edges = [[0,1],[1,2],[3,4]]", "output": "2",
                    "explanation": "Component 1: {0,1,2}. Component 2: {3,4}. Use Union-Find or BFS/DFS."},
        "assumptions": ["Undirected graph.", "n nodes labeled 0..n-1."]
    },
    "number-of-islands": {
        "example": {"input": 'grid = [["1","1","0","0"],["1","1","0","0"],["0","0","1","0"],["0","0","0","1"]]', "output": "3",
                    "explanation": "3 groups of connected '1's (islands). BFS/DFS to mark visited cells."},
        "assumptions": ["'1' = land, '0' = water.", "Islands are 4-directionally connected."]
    },
    "number-of-islands-ii": {
        "example": {"input": "m=3, n=3, positions=[[0,0],[0,1],[1,2],[2,1]]", "output": "[1,1,2,3]",
                    "explanation": "After each land addition, count islands using Union-Find. Merge with adjacent land cells."},
        "assumptions": ["Dynamic island addition.", "Use Union-Find for O(α) per operation."]
    },
    "odd-even-linked-list": {
        "example": {"input": "head = [1,2,3,4,5]", "output": "[1,3,5,2,4]",
                    "explanation": "Group odd-indexed nodes (1,3,5) followed by even-indexed (2,4). In-place, O(1) space."},
        "assumptions": ["1-indexed positions.", "Preserve relative order within odd and even groups."]
    },
    "one-edit-distance": {
        "example": {"input": 's = "ab", t = "acb"', "output": "true",
                    "explanation": "Insert 'c' into 'ab' → 'acb'. Exactly one edit (insert/delete/replace)."},
        "assumptions": ["Return true if s and t are exactly 1 edit distance apart.", "Not 0 edits (identical strings return false)."]
    },
    "paint-house": {
        "example": {"input": "costs = [[17,2,17],[16,16,5],[14,3,19]]", "output": "10",
                    "explanation": "Paint house 0 blue (2), house 1 green (5), house 2 blue (3) = 10. Adjacent houses must differ in color."},
        "assumptions": ["3 colors (red, blue, green).", "Adjacent houses cannot have the same color.", "Return minimum total cost."]
    },
    "palindrome-linked-list": {
        "example": {"input": "head = [1,2,2,1]", "output": "true",
                    "explanation": "Find mid, reverse second half, compare. O(n) time, O(1) space."},
        "assumptions": ["Return true if linked list reads the same forward and backward."]
    },
    "palindrome-number": {
        "example": {"input": "x = 121", "output": "true",
                    "explanation": "121 reversed is 121. Negative numbers and numbers ending in 0 (except 0) are not palindromes."},
        "assumptions": ["Negative numbers are not palindromes.", "Try without converting to string."]
    },
    "palindrome-pairs": {
        "example": {"input": 'words = ["abcd","dcba","lls","s","sssll"]', "output": "[[0,1],[1,0],[3,2],[2,4]]",
                    "explanation": 'words[0]+words[1]="abcddcba" (palindrome). Use hash map for O(n*k²).'},
        "assumptions": ["Return all unique index pairs [i,j] where words[i]+words[j] is a palindrome.", "i ≠ j."]
    },
    "palindrome-partitioning": {
        "example": {"input": 's = "aab"', "output": '[["a","a","b"],["aa","b"]]',
                    "explanation": "All ways to partition 'aab' such that every substring is a palindrome. Backtracking + DP palindrome check."},
        "assumptions": ["Every substring in the partition must be a palindrome.", "Return all possible partitions."]
    },
    "partition-list": {
        "example": {"input": "head = [1,4,3,2,5,2], x = 3", "output": "[1,2,2,4,3,5]",
                    "explanation": "All nodes < 3 come before nodes ≥ 3. Maintain two lists, join at end."},
        "assumptions": ["Preserve original relative order within each partition."]
    },
    "pascals-triangle": {
        "example": {"input": "numRows = 5", "output": "[[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]",
                    "explanation": "Each element = sum of two elements directly above it. Edges are always 1."},
        "assumptions": ["numRows ≥ 1."]
    },
    "path-sum": {
        "example": {"input": "root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22", "output": "true",
                    "explanation": "Path 5→4→11→2 sums to 22. DFS: subtract node value from target, check leaf when target=0."},
        "assumptions": ["Root-to-leaf path.", "A leaf is a node with no children."]
    },
    "path-sum-ii": {
        "example": {"input": "root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22", "output": "[[5,4,11,2],[5,8,4,5]]",
                    "explanation": "Return all root-to-leaf paths that sum to target. Backtracking DFS."},
        "assumptions": []
    },
    "permutation-sequence": {
        "example": {"input": "n = 3, k = 3", "output": '"213"',
                    "explanation": "Permutations of [1,2,3] in order: 123,132,213,231,312,321. 3rd is 213."},
        "assumptions": ["1 ≤ k ≤ n!", "Use factorial number system to find kth permutation directly without generating all."]
    },
    "permutations": {
        "example": {"input": "nums = [1,2,3]", "output": "[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]",
                    "explanation": "All 3! = 6 permutations. Backtracking: swap elements into position recursively."},
        "assumptions": ["All nums are distinct.", "Return all permutations in any order."]
    },
    "permutations-ii": {
        "example": {"input": "nums = [1,1,2]", "output": "[[1,1,2],[1,2,1],[2,1,1]]",
                    "explanation": "3 unique permutations (not 3!=6 because of duplicate 1s). Skip duplicate branches in backtracking."},
        "assumptions": ["Input may contain duplicates.", "Return only unique permutations."]
    },
    "plus-one": {
        "example": {"input": "digits = [1,2,9]", "output": "[1,3,0]",
                    "explanation": "129 + 1 = 130. Handle carry propagation and edge case [9,9,9] → [1,0,0,0]."},
        "assumptions": ["Digits represent a non-negative integer with no leading zeros.", "Most significant digit is first."]
    },
    "populating-next-right-pointers": {
        "example": {"input": "root = [1,2,3,4,5,6,7]", "output": "Each node's next points to its right neighbor",
                    "explanation": "Level 1: 1→null. Level 2: 2→3→null. Level 3: 4→5→6→7→null. Perfect binary tree: use level-order traversal."},
        "assumptions": ["Perfect binary tree (all leaves at same level, all non-leaves have two children).", "O(1) extra space using existing next pointers."]
    },
    "populating-next-right-pointers-ii": {
        "example": {"input": "root = [1,2,3,4,5,null,7]", "output": "Same next-pointer population but for arbitrary binary tree",
                    "explanation": "Not a perfect binary tree. Use level-order or dummy-head trick to traverse current level and wire up next level."},
        "assumptions": ["Works for any binary tree.", "O(1) extra space solution exists."]
    },
    "pow-x-n": {
        "example": {"input": "x = 2.00000, n = 10", "output": "1024.00000",
                    "explanation": "Fast power: x^n = (x^(n/2))^2. Handles negative n: x^(-n) = 1/x^n. O(log n)."},
        "assumptions": ["n can be negative.", "-100 ≤ n ≤ 100."]
    },
    "power-of-two": {
        "example": {"input": "n = 16", "output": "true",
                    "explanation": "16 = 2⁴. Powers of 2 in binary have exactly one '1' bit. Check: n > 0 && (n & (n-1)) == 0."},
        "assumptions": []
    },
    "print-linked-list-reversed": {
        "example": {"input": "head = [1,2,3,4,5]", "output": "[5,4,3,2,1]",
                    "explanation": "Traverse to end using recursion (call stack), print on way back. Or reverse then print."},
        "assumptions": []
    },
    "product-array-except-self": {
        "example": {"input": "nums = [1,2,3,4]", "output": "[24,12,8,6]",
                    "explanation": "output[i] = product of all elements except nums[i]. Two passes: left products then right products. O(n), no division."},
        "assumptions": ["O(n) time, O(1) extra space (output array doesn't count).", "No division allowed."]
    },
    "quick-sort": {
        "example": {"input": "arr = [10,7,8,9,1,5]", "output": "[1,5,7,8,9,10]",
                    "explanation": "Choose pivot, partition array (smaller left, larger right), recurse. O(n log n) avg, O(n²) worst."},
        "assumptions": ["Average O(n log n), worst O(n²) with bad pivot choice.", "In-place, O(log n) stack space."]
    },
    "radix-sort": {
        "example": {"input": "arr = [170,45,75,90,802,24,2,66]", "output": "[2,24,45,66,75,90,170,802]",
                    "explanation": "Sort by least significant digit first, then next, etc. Uses counting sort at each digit. O(d*(n+k))."},
        "assumptions": ["Non-negative integers.", "d = number of digits, k = base (10)."]
    },
    "range-addition": {
        "example": {"input": "length = 5, updates = [[1,3,2],[2,4,3],[0,2,-2]]", "output": "[-2,0,3,5,3]",
                    "explanation": "Use difference array: add val at start, subtract at end+1. O(n+k) with prefix sum."},
        "assumptions": ["Apply all range updates efficiently.", "Return final array after all updates."]
    },
    "range-sum-query-2d": {
        "example": {"input": "matrix = [[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]\nsumRegion(2,1,4,3) → 8",
                    "output": "8",
                    "explanation": "Precompute 2D prefix sums. Query in O(1): sum(r1,c1,r2,c2) = prefix[r2][c2] - prefix[r1-1][c2] - prefix[r2][c1-1] + prefix[r1-1][c1-1]."},
        "assumptions": ["Multiple sum queries after single matrix initialization.", "O(1) per query after O(m*n) preprocessing."]
    },
    "range-sum-query-mutable": {
        "example": {"input": "nums = [1,3,5], update(1,2), sumRange(0,2) → 8", "output": "8",
                    "explanation": "After update: [1,2,5]. Sum(0,2)=8. Use Binary Indexed Tree (Fenwick) or Segment Tree for O(log n) update and query."},
        "assumptions": ["Both update and sumRange in O(log n).", "Segment tree or Fenwick tree needed."]
    },
    "reconstruct-itinerary": {
        "example": {"input": 'tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]', "output": '["JFK","MUC","LHR","SFO","SJC"]',
                    "explanation": "Find Eulerian path starting from JFK. Use Hierholzer's algorithm with DFS."},
        "assumptions": ["Always start from JFK.", "All tickets must be used.", "If multiple valid itineraries, return lexicographically smallest."]
    },
    "recover-binary-search-tree": {
        "example": {"input": "root = [1,3,null,null,2]", "output": "[3,1,null,null,2]",
                    "explanation": "Nodes 1 and 3 are swapped. In-order traversal detects two inversions; swap those two nodes."},
        "assumptions": ["Exactly two nodes are swapped.", "Recover without changing tree structure.", "O(1) space with Morris traversal."]
    },
    "rectangle-area": {
        "example": {"input": "ax1=-3,ay1=0,ax2=3,ay2=4, bx1=0,by1=-1,bx2=9,by2=2", "output": "45",
                    "explanation": "Area A + Area B - overlap. Overlap = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0, min(ay2,by2)-max(ay1,by1))."},
        "assumptions": ["Return total area covered by two rectangles (union)."]
    },
    "regular-expression-matching": {
        "example": {"input": 's = "aab", p = "c*a*b"', "output": "true",
                    "explanation": "c* matches empty, a* matches 'aa', b matches 'b'. DP: dp[i][j] = does s[:i] match p[:j]."},
        "assumptions": ["'.' matches any single character.", "'*' matches zero or more of the preceding element.", "Full string must match (not partial)."]
    },
    "remove-duplicates-sorted-array": {
        "example": {"input": "nums = [0,0,1,1,1,2,2,3,3,4]", "output": "5, nums = [0,1,2,3,4,...]",
                    "explanation": "Two pointers: slow tracks position of next unique, fast scans ahead. In-place O(1) space."},
        "assumptions": ["In-place modification.", "Return new length k; first k elements must be unique and sorted."]
    },
    "remove-duplicates-sorted-list": {
        "example": {"input": "head = [1,1,2,3,3]", "output": "[1,2,3]",
                    "explanation": "Skip nodes whose value equals next node's value. O(n) time, O(1) space."},
        "assumptions": ["Keep one copy of each duplicate."]
    },
    "remove-duplicates-sorted-list-ii": {
        "example": {"input": "head = [1,2,3,3,4,4,5]", "output": "[1,2,5]",
                    "explanation": "Remove all nodes that had duplicates (3 and 4 are gone entirely). Use dummy head."},
        "assumptions": ["Remove ALL occurrences of duplicated values (unlike version I which keeps one)."]
    },
    "remove-element": {
        "example": {"input": "nums = [3,2,2,3], val = 3", "output": "2, nums = [2,2,...]",
                    "explanation": "Remove all occurrences of val in-place. Return count of remaining elements."},
        "assumptions": ["In-place, O(1) space.", "Order of remaining elements doesn't matter."]
    },
    "remove-invalid-parentheses": {
        "example": {"input": 's = "()())()"', "output": '["()()()", "(())()"]',
                    "explanation": "Remove minimum parentheses to make valid. BFS level by level, stop at first level with valid strings."},
        "assumptions": ["Return all unique valid strings with minimum removals.", "May contain letters which stay unchanged."]
    },
    "remove-linked-list-elements": {
        "example": {"input": "head = [1,2,6,3,4,5,6], val = 6", "output": "[1,2,3,4,5]",
                    "explanation": "Remove all nodes with value = val. Use dummy head to handle edge case of removing head."},
        "assumptions": []
    },
    "remove-nth-node": {
        "example": {"input": "head = [1,2,3,4,5], n = 2", "output": "[1,2,3,5]",
                    "explanation": "Remove 2nd node from end (value 4). Two-pointer: advance fast by n+1, then move both until fast=null."},
        "assumptions": ["n is valid (1 ≤ n ≤ length).", "One-pass solution with two pointers."]
    },
    "remove-nth-node-end": {
        "example": {"input": "head = [1,2,3,4,5], n = 2", "output": "[1,2,3,5]",
                    "explanation": "Same as remove-nth-node: remove nth from end in one pass."},
        "assumptions": []
    },
    "reorder-list": {
        "example": {"input": "head = [1,2,3,4,5]", "output": "[1,5,2,4,3]",
                    "explanation": "Interleave: L0→Ln→L1→Ln-1→... Find mid, reverse second half, merge."},
        "assumptions": ["In-place modification.", "Do not return anything; modify head in-place."]
    },
    "restore-ip-addresses": {
        "example": {"input": 's = "25525511135"', "output": '["255.255.11.135","255.255.111.35"]',
                    "explanation": "Each segment must be 0-255 with no leading zeros. Backtracking with 4 segments."},
        "assumptions": ["Exactly 4 parts.", "Each part is a number 0-255 with no leading zeros (except '0' itself)."]
    },
    "reverse-bits": {
        "example": {"input": "n = 00000010100101000001111010011100", "output": "964176192 (00111001011110000010100101000000)",
                    "explanation": "Reverse all 32 bits. Shift result left and n right, OR the last bit of n each iteration."},
        "assumptions": ["Input is a 32-bit unsigned integer.", "Reverse all 32 bits (including leading zeros)."]
    },
    "reverse-integer": {
        "example": {"input": "x = 123", "output": "321",
                    "explanation": "Reverse digits of integer. Handle overflow: return 0 if result outside 32-bit signed range [-2³¹, 2³¹-1]."},
        "assumptions": ["Return 0 if reversed integer overflows 32-bit signed integer.", "Handle negatives: -123 → -321."]
    },
    "reverse-linked-list": {
        "example": {"input": "head = [1,2,3,4,5]", "output": "[5,4,3,2,1]",
                    "explanation": "Iterative: use prev/curr/next pointers. Or recursive: reverse rest, point head.next.next to head."},
        "assumptions": ["In-place reversal.", "Return new head."]
    },
    "reverse-nodes-k-group": {
        "example": {"input": "head = [1,2,3,4,5], k = 2", "output": "[2,1,4,3,5]",
                    "explanation": "Reverse every k nodes. [1,2]→[2,1], [3,4]→[4,3], [5] stays (fewer than k nodes)."},
        "assumptions": ["If nodes remaining < k, leave them as-is.", "In-place reversal."]
    },
    "reverse-vowels-string": {
        "example": {"input": 's = "hello"', "output": '"holle"',
                    "explanation": "Vowels in 'hello' are e(1) and o(4). Swap them: 'holle'. Two pointers from each end."},
        "assumptions": ["Vowels: a,e,i,o,u (both cases)."]
    },
    "reverse-words-string": {
        "example": {"input": 's = "the sky is blue"', "output": '"blue is sky the"',
                    "explanation": "Reverse word order. Trim leading/trailing spaces, reduce multiple spaces to one."},
        "assumptions": ["Return single spaces between words.", "No leading or trailing spaces in output."]
    },
    "rotate-array": {
        "example": {"input": "nums = [1,2,3,4,5,6,7], k = 3", "output": "[5,6,7,1,2,3,4]",
                    "explanation": "Rotate right by k: last k elements move to front. Trick: reverse all, reverse first k, reverse rest."},
        "assumptions": ["k can be larger than array length (use k % n).", "In-place, O(1) space."]
    },
    "rotate-image": {
        "example": {"input": "matrix = [[1,2,3],[4,5,6],[7,8,9]]", "output": "[[7,4,1],[8,5,2],[9,6,3]]",
                    "explanation": "Rotate 90° clockwise in-place: transpose then reverse each row."},
        "assumptions": ["n×n matrix.", "Rotate in-place."]
    },
    "same-tree": {
        "example": {"input": "p = [1,2,3], q = [1,2,3]", "output": "true",
                    "explanation": "Same structure and node values. Recursive: check root values equal and both subtrees are same."},
        "assumptions": []
    },
    "scramble-string": {
        "example": {"input": 's1 = "great", s2 = "rgeat"', "output": "true",
                    "explanation": "'great' can be split to 'gr|eat', swap to 'eat|gr', then 'rg|eat' = 'rgeat'. Recursive DP with memoization."},
        "assumptions": ["A scramble swaps any two non-overlapping substrings recursively.", "Use memoization on (s1,s2) pairs."]
    },
    "search-2d-matrix": {
        "example": {"input": "matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3", "output": "true",
                    "explanation": "Rows are sorted, first element of each row > last of previous. Treat as flat sorted array: binary search on row then column."},
        "assumptions": ["Rows sorted, first element of each row > last element of previous row.", "O(log(m*n)) binary search."]
    },
    "search-2d-matrix-ii": {
        "example": {"input": "matrix = [[1,4,7,11],[2,5,8,12],[3,6,9,16],[10,13,14,17]], target = 5", "output": "true",
                    "explanation": "Start top-right: if > target go left, if < target go down. O(m+n)."},
        "assumptions": ["Rows sorted ascending, columns sorted ascending.", "First element NOT necessarily > last of previous row."]
    },
    "search-insert-position": {
        "example": {"input": "nums = [1,3,5,6], target = 5", "output": "2",
                    "explanation": "5 is at index 2. If target=2, return 1 (insert position). Binary search."},
        "assumptions": ["Array has no duplicates.", "Return index if found, else return index where it would be inserted."]
    },
    "search-range": {
        "example": {"input": "nums = [5,7,7,8,8,10], target = 8", "output": "[3,4]",
                    "explanation": "8 appears at indices 3 and 4. Two binary searches: find leftmost and rightmost occurrence."},
        "assumptions": ["O(log n) time.", "Return [-1,-1] if target not found."]
    },
    "search-rotated-array": {
        "example": {"input": "nums = [4,5,6,7,0,1,2], target = 0", "output": "4",
                    "explanation": "Array rotated at index 4. Binary search: determine which half is sorted, check if target in that half."},
        "assumptions": ["All values are unique.", "Originally sorted, then rotated at some pivot.", "O(log n)."]
    },
    "selection-sort": {
        "example": {"input": "arr = [64,25,12,22,11]", "output": "[11,12,22,25,64]",
                    "explanation": "Find minimum in unsorted portion, swap to front. O(n²) always, O(1) space."},
        "assumptions": ["Sort in ascending order.", "O(n²) time regardless of input."]
    },
    "serialize-deserialize-binary-tree": {
        "example": {"input": "root = [1,2,3,null,null,4,5]", "output": "Serialize to string, deserialize back to same tree",
                    "explanation": "BFS or pre-order serialization with null markers. '1,2,3,null,null,4,5'."},
        "assumptions": ["Your serialization format can be anything, as long as it deserializes correctly.", "No constraints on algorithm used."]
    },
    "set-matrix-zeroes": {
        "example": {"input": "matrix = [[1,1,1],[1,0,1],[1,1,1]]", "output": "[[1,0,1],[0,0,0],[1,0,1]]",
                    "explanation": "If cell is 0, set entire row and column to 0. Use first row/col as markers for O(1) space."},
        "assumptions": ["In-place.", "O(1) space solution exists using first row/column as markers."]
    },
    "shortest-distance-buildings": {
        "example": {"input": "grid = [[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]]", "output": "7",
                    "explanation": "BFS from each building to find distances to empty lands. Find empty land minimizing sum of distances to all buildings."},
        "assumptions": ["0=empty, 1=building, 2=obstacle.", "Return -1 if no valid location."]
    },
    "shortest-palindrome": {
        "example": {"input": 's = "aacecaaa"', "output": '"aaacecaaa"',
                    "explanation": "Find longest palindromic prefix, prepend reverse of remaining suffix. Use KMP to find overlap."},
        "assumptions": ["Add characters only to the front.", "Minimize additions."]
    },
    "simplify-path": {
        "example": {"input": 'path = "/home//foo/"', "output": '"/home/foo"',
                    "explanation": "Split by '/', handle '.' (current dir), '..' (parent dir), empty parts. Use stack."},
        "assumptions": ["Unix-style path.", "'..' goes up one directory.", "'.' is current directory."]
    },
    "single-number": {
        "example": {"input": "nums = [4,1,2,1,2]", "output": "4",
                    "explanation": "XOR all numbers: pairs cancel out (a^a=0), leaving the single number (a^0=a)."},
        "assumptions": ["Every element appears exactly twice except for one.", "O(n) time, O(1) space."]
    },
    "skyline-problem": {
        "example": {"input": "buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]",
                    "output": "[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]",
                    "explanation": "Return key points where skyline height changes. Use a max-heap of active buildings."},
        "assumptions": ["Return list of [x, height] key points.", "Include point at x where building ends (height=0 or drops)."]
    },
    "sliding-window-maximum": {
        "example": {"input": "nums = [1,3,-1,-3,5,3,6,7], k = 3", "output": "[3,3,5,5,6,7]",
                    "explanation": "Max of each sliding window of size k. Use monotonic deque: maintain decreasing order, front is always max."},
        "assumptions": ["O(n) using deque.", "1 ≤ k ≤ nums.length."]
    },
    "sort-array-0s-1s-2s": {
        "example": {"input": "nums = [2,0,2,1,1,0]", "output": "[0,0,1,1,2,2]",
                    "explanation": "Dutch National Flag problem: three pointers (low, mid, high). One pass, O(1) space."},
        "assumptions": ["In-place, one pass (Dutch National Flag algorithm).", "No extra space."]
    },
    "sort-by-frequency": {
        "example": {"input": "nums = [1,1,2,2,2,3]", "output": "[2,2,2,1,1,3]",
                    "explanation": "Sort by frequency descending (2 appears 3×, 1 appears 2×, 3 appears 1×)."},
        "assumptions": ["Higher frequency comes first.", "Tie-break by value (smaller value first) or any consistent order."]
    },
    "spiral-matrix": {
        "example": {"input": "matrix = [[1,2,3],[4,5,6],[7,8,9]]", "output": "[1,2,3,6,9,8,7,4,5]",
                    "explanation": "Traverse: right across top, down right, left across bottom, up left, repeat inward."},
        "assumptions": ["Traverse in clockwise spiral order.", "Return elements in that order."]
    },
    "spiral-matrix-ii": {
        "example": {"input": "n = 3", "output": "[[1,2,3],[8,9,4],[7,6,5]]",
                    "explanation": "Fill 1..n² in spiral order. Same traversal as Spiral Matrix I but fill instead of read."},
        "assumptions": ["Generate n×n matrix filled with elements 1 to n² in spiral order."]
    },
    "string-to-integer": {
        "example": {"input": 's = "   -42"', "output": "-42",
                    "explanation": "Skip leading whitespace, read optional sign, read digits until non-digit. Clamp to 32-bit integer range."},
        "assumptions": ["Handle leading spaces, optional +/-, non-digit stops parsing.", "Clamp result to [-2³¹, 2³¹-1]."]
    },
    "subsets": {
        "example": {"input": "nums = [1,2,3]", "output": "[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]",
                    "explanation": "All 2ⁿ subsets. Backtracking: include or exclude each element. Or bit manipulation."},
        "assumptions": ["All nums are distinct.", "Return power set (all subsets including empty and full set)."]
    },
    "substring-concatenation-words": {
        "example": {"input": 's = "barfoothefoobarman", words = ["foo","bar"]', "output": "[0,9]",
                    "explanation": "At index 0: 'barfoo'='bar'+'foo'. At index 9: 'foobar'='foo'+'bar'. Sliding window of size totalLen."},
        "assumptions": ["All words same length.", "Each word must appear exactly once.", "Return starting indices of all valid concatenations."]
    },
    "sudoku-solver": {
        "example": {"input": "9×9 board with some cells filled", "output": "Same board filled with solution",
                    "explanation": "Backtracking: try digits 1-9 in each empty cell, check row/col/box validity, recurse, backtrack if stuck."},
        "assumptions": ["Unique solution guaranteed.", "Digits 1-9, '.' for empty cells.", "In-place modification."]
    },
    "sum-root-leaf-numbers": {
        "example": {"input": "root = [1,2,3]", "output": "25",
                    "explanation": "Root-to-leaf numbers: 12 and 13. Sum = 25. DFS passing current number as 10*cur + node.val."},
        "assumptions": ["Each root-to-leaf path represents a number.", "Return sum of all such numbers."]
    },
    "sum-two-integers": {
        "example": {"input": "a = 1, b = 2", "output": "3",
                    "explanation": "a+b without + or -. XOR gives sum without carry; AND<<1 gives carry. Repeat until no carry."},
        "assumptions": ["Cannot use + or - operators.", "Use bitwise XOR and AND with shift."]
    },
    "summary-ranges": {
        "example": {"input": "nums = [0,1,2,4,5,7]", "output": '["0->2","4->5","7"]',
                    "explanation": "Consecutive ranges: 0-2 written as '0->2', 4-5 as '4->5', lone 7 as '7'."},
        "assumptions": ["Sorted unique array.", "Consecutive numbers form a range 'a->b'; single number is just 'a'."]
    },
    "surrounded-regions": {
        "example": {"input": 'board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]',
                    "output": '[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]',
                    "explanation": "Flip surrounded 'O's to 'X'. 'O' at border and connected to border is NOT flipped. DFS from border O's first."},
        "assumptions": ["In-place modification.", "O's on border or connected to border O's are not flipped."]
    },
    "swap-nodes-pairs": {
        "example": {"input": "head = [1,2,3,4]", "output": "[2,1,4,3]",
                    "explanation": "Swap every two adjacent nodes. 1↔2, 3↔4. Iterative or recursive."},
        "assumptions": ["Swap node links, not values.", "If odd number of nodes, last node stays."]
    },
    "symmetric-tree": {
        "example": {"input": "root = [1,2,2,3,4,4,3]", "output": "true",
                    "explanation": "Tree is mirror of itself. Recursively check: left.left == right.right and left.right == right.left."},
        "assumptions": ["A tree is symmetric if left and right subtrees are mirror images.", "Can also solve iteratively with queue."]
    },
    "text-justification": {
        "example": {"input": 'words = ["This","is","an","example","of","text","justification."], maxWidth = 16',
                    "output": '["This    is    an","example  of text","justification.  "]',
                    "explanation": "Each line has exactly maxWidth chars. Distribute spaces evenly; last line is left-justified."},
        "assumptions": ["Extra spaces distributed left to right.", "Last line is left-justified (spaces only at right)."]
    },
    "tic-tac-toe": {
        "example": {"input": "n=3: move(0,0,1)→0, move(0,2,2)→0, move(2,2,1)→0, move(1,1,2)→0, move(2,0,1)→0, move(1,0,2)→0, move(2,1,1)→1",
                    "output": "0, 0, 0, 0, 0, 0, 1",
                    "explanation": "Player 1 wins by filling row 2 (2,0),(2,1),(2,2). Returns winning player, 0 if no winner yet."},
        "assumptions": ["Player 1 = 1, Player 2 = 2.", "Design for O(1) move with O(n) space."]
    },
    "top-k-frequent-elements": {
        "example": {"input": "nums = [1,1,1,2,2,3], k = 2", "output": "[1,2]",
                    "explanation": "1 appears 3×, 2 appears 2×. Top 2 frequent are [1,2]. Use heap or bucket sort."},
        "assumptions": ["Return k most frequent elements.", "Answer is unique (no tie in top-k boundary)."]
    },
    "trapping-rain-water": {
        "example": {"input": "height = [0,1,0,2,1,0,1,3,2,1,2,1]", "output": "6",
                    "explanation": "Total water trapped = 6 units. For each position: water = min(maxLeft, maxRight) - height[i]."},
        "assumptions": ["Non-negative integer heights.", "Two-pointer O(n) approach or precomputed max arrays."]
    },
    "triangle": {
        "example": {"input": "triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]", "output": "11",
                    "explanation": "Path: 2→3→5→1 = 11. Bottom-up DP: for each row, update in-place adding minimum of two children below."},
        "assumptions": ["Move to adjacent numbers on the row below.", "O(n) space bottom-up DP."]
    },
    "two-sum-ii": {
        "example": {"input": "numbers = [2,7,11,15], target = 9", "output": "[1,2]",
                    "explanation": "numbers[1] + numbers[2] = 2 + 7 = 9. 1-indexed output. Two pointers: l=0, r=end."},
        "assumptions": ["Array is sorted in non-decreasing order.", "Return 1-indexed positions.", "Exactly one solution."]
    },
    "two-sum-iii": {
        "example": {"input": "add(1), add(3), add(5), find(4)→true, find(7)→false", "output": "true, false",
                    "explanation": "find(4): 1+3=4 found. find(7): no pair sums to 7. Use HashMap storing counts."},
        "assumptions": ["Design data structure supporting add(number) and find(value).", "find: does any pair sum to value?"]
    },
    "ugly-number": {
        "example": {"input": "n = 10", "output": "12",
                    "explanation": "Ugly numbers: 1,2,3,4,5,6,8,9,10,12. 10th is 12. Use three pointers for multiples of 2,3,5."},
        "assumptions": ["Ugly numbers have only prime factors 2, 3, 5.", "1 is ugly by convention."]
    },
    "unique-binary-search-trees": {
        "example": {"input": "n = 3", "output": "5",
                    "explanation": "5 structurally unique BSTs with values 1..3. Catalan number: dp[n] = sum of dp[i-1]*dp[n-i] for i=1..n."},
        "assumptions": ["Count structurally unique BSTs (not the actual trees)."]
    },
    "unique-binary-search-trees-ii": {
        "example": {"input": "n = 3", "output": "All 5 structurally unique BSTs with nodes 1,2,3",
                    "explanation": "Generate all trees: for each root i, recursively generate all left trees from [1..i-1] and right from [i+1..n]."},
        "assumptions": ["Return all actual tree structures (not just count)."]
    },
    "unique-paths": {
        "example": {"input": "m = 3, n = 7", "output": "28",
                    "explanation": "Robot moves only right or down. Total unique paths = C(m+n-2, m-1) = C(8,2) = 28."},
        "assumptions": ["Can only move right or down.", "dp[i][j] = dp[i-1][j] + dp[i][j-1]."]
    },
    "unique-paths-ii": {
        "example": {"input": "obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]", "output": "2",
                    "explanation": "Center cell is an obstacle. Two paths remain: top-right corner and bottom-left corner routes."},
        "assumptions": ["0 = empty, 1 = obstacle.", "Set dp to 0 at obstacles."]
    },
    "valid-anagram": {
        "example": {"input": 's = "anagram", t = "nagaram"', "output": "true",
                    "explanation": "Both have same character frequencies. Sort both strings or use frequency map."},
        "assumptions": ["Only lowercase letters.", "Same length strings can still fail if frequencies differ."]
    },
    "valid-palindrome": {
        "example": {"input": 's = "A man, a plan, a canal: Panama"', "output": "true",
                    "explanation": "Ignore non-alphanumeric, case-insensitive: 'amanaplanacanalpanama'. Two pointers from both ends."},
        "assumptions": ["Consider only alphanumeric characters.", "Case-insensitive comparison."]
    },
    "valid-parentheses": {
        "example": {"input": 's = "()[]{}"', "output": "true",
                    "explanation": "Each open bracket must be closed by the same type in the correct order. Use stack."},
        "assumptions": ["Brackets: (), [], {}.", "Empty string is valid."]
    },
    "valid-sudoku": {
        "example": {"input": "9×9 board (partially filled)", "output": "true/false",
                    "explanation": "Validate: each row, each column, each 3×3 box contains digits 1-9 with no repetition. '.' is empty."},
        "assumptions": ["Only validate current state, not whether it's solvable.", "Use three sets: rows, cols, boxes."]
    },
    "validate-binary-search-tree": {
        "example": {"input": "root = [5,1,4,null,null,3,6]", "output": "false",
                    "explanation": "Node 4 is in right subtree but 4 < 5, violating BST property. Pass min/max bounds in recursion."},
        "assumptions": ["Each node's value must be strictly between its ancestor bounds.", "Duplicate values make it invalid."]
    },
    "verify-preorder-serialization": {
        "example": {"input": '"9,3,4,#,#,1,#,#,2,#,6,#,#"', "output": "true",
                    "explanation": "Track available slots: start=1, leaf adds 0 slots (uses 1), non-leaf adds 1 (uses 1, adds 2). Slots never go negative and end at 0."},
        "assumptions": ["'#' represents null.", "Valid if slots never go negative and end at exactly 0."]
    },
    "vertical-order": {
        "example": {"input": "root = [3,9,20,null,null,15,7]", "output": "[[9],[3,15],[20],[7]]",
                    "explanation": "Group nodes by horizontal distance from root (root=0, left=-1, right=+1). BFS with column tracking."},
        "assumptions": ["Nodes at same column, same level appear top to bottom.", "Use TreeMap or sort columns."]
    },
    "walls-and-gates": {
        "example": {"input": 'rooms = [[INF,-1,0,INF],[INF,INF,INF,-1],[INF,-1,INF,-1],[0,-1,INF,INF]]',
                    "output": "[[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]",
                    "explanation": "Fill each empty room with distance to nearest gate. BFS from all gates simultaneously."},
        "assumptions": ["-1=wall, 0=gate, INF=empty room.", "Multi-source BFS from all gates at once."]
    },
    "wildcard-matching": {
        "example": {"input": 's = "adceb", p = "*a*b"', "output": "true",
                    "explanation": "'*' matches 'adc', 'a' matches 'a', '*' matches empty, 'b' matches 'b'. DP or two-pointer."},
        "assumptions": ["'?' matches any single char.", "'*' matches any sequence (including empty).", "Full string must match."]
    },
    "word-break": {
        "example": {"input": 's = "leetcode", wordDict = ["leet","code"]', "output": "true",
                    "explanation": "'leet' + 'code' = 'leetcode'. DP: dp[i] = true if s[:i] can be segmented using wordDict."},
        "assumptions": ["Words in dict can be reused.", "Return true if entire string can be segmented."]
    },
    "word-ladder": {
        "example": {"input": 'beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]', "output": "5",
                    "explanation": 'hit→hot→dot→dog→cog: 5 words, 4 transformations. BFS for shortest path; each step changes 1 letter.'},
        "assumptions": ["Each intermediate word must be in wordList.", "Return length of shortest transformation sequence, or 0."]
    },
    "word-ladder-ii": {
        "example": {"input": 'beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]',
                    "output": '[["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]',
                    "explanation": "Find ALL shortest transformation sequences. BFS for levels + DFS/backtrack to reconstruct paths."},
        "assumptions": ["Return all shortest paths.", "Much harder than Word Ladder I."]
    },
    "word-pattern": {
        "example": {"input": 'pattern = "abba", s = "dog cat cat dog"', "output": "true",
                    "explanation": "a→dog, b→cat. Bijection: pattern char ↔ word. Both maps (char→word and word→char) must be consistent."},
        "assumptions": ["Bijective mapping: one pattern char = one word, and one word = one pattern char.", "No two pattern chars can map to same word."]
    },
    "word-search": {
        "example": {"input": 'board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"', "output": "true",
                    "explanation": "Path: A(0,0)→B(0,1)→C(0,2)→C(1,2)→E(2,2)→D(2,1). DFS with backtracking, mark visited."},
        "assumptions": ["Each cell used at most once per path.", "Search 4-directionally."]
    },
    "zigzag-conversion": {
        "example": {"input": 's = "PAYPALISHIRING", numRows = 3', "output": '"PAHNAPLSIIGYIR"',
                    "explanation": "Write in zigzag across numRows rows, read row by row. P-A-H-N / A-P-L-S-I-I-G / Y-I-R."},
        "assumptions": ["numRows ≥ 1.", "If numRows=1 or numRows≥len(s), return s unchanged."]
    },
    "final-problem-1": {
        "example": {"input": "See problem description", "output": "See problem description",
                    "explanation": "Practice problem — read the description carefully."},
        "assumptions": []
    },
    "final-problem-2": {
        "example": {"input": "See problem description", "output": "See problem description",
                    "explanation": "Practice problem."},
        "assumptions": []
    },
    "final-problem-3": {
        "example": {"input": "See problem description", "output": "See problem description",
                    "explanation": "Practice problem."},
        "assumptions": []
    },
    "final-problem-4": {
        "example": {"input": "See problem description", "output": "See problem description",
                    "explanation": "Practice problem."},
        "assumptions": []
    },
    "final-problem-5": {
        "example": {"input": "See problem description", "output": "See problem description",
                    "explanation": "Practice problem."},
        "assumptions": []
    },
}

def update_file(path, slug):
    with open(path) as f:
        data = json.load(f)

    ex_data = EXAMPLES.get(slug)
    if not ex_data:
        return False

    changed = False
    p = data.setdefault('problem', {})

    # Update example if missing or empty
    existing_ex = p.get('example', {})
    if not existing_ex.get('input') and ex_data.get('example'):
        p['example'] = ex_data['example']
        changed = True

    # Update assumptions if missing or empty
    if not p.get('assumptions') and ex_data.get('assumptions') is not None:
        p['assumptions'] = ex_data['assumptions']
        changed = True

    if changed:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return changed

files = sorted(glob.glob('solutions/*.json'))
updated = 0
skipped = 0
no_entry = 0

for path in files:
    slug = os.path.basename(path).replace('.json', '')
    if slug in EXAMPLES:
        if update_file(path, slug):
            updated += 1
            print(f'  UPDATED: {slug}')
        else:
            skipped += 1
    else:
        no_entry += 1
        print(f'  NO ENTRY: {slug}')

print(f'\nDone: {updated} updated, {skipped} already had examples, {no_entry} no entry in dict')
