#!/usr/bin/env python3
import os
import re

def get_more_solutions():
    """Get solutions for another batch of problems."""
    return {
        "Matrix Chain Multiplication": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use dynamic programming to find the optimal parenthesization.",
                "code": """public int matrixChainMultiplication(int[] dimensions) {
    int n = dimensions.length - 1;
    int[][] dp = new int[n][n];
    
    // Fill diagonal with 0s
    for (int i = 0; i < n; i++) {
        dp[i][i] = 0;
    }
    
    // Fill dp table
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            dp[i][j] = Integer.MAX_VALUE;
            
            for (int k = i; k < j; k++) {
                int cost = dp[i][k] + dp[k + 1][j] + 
                          dimensions[i] * dimensions[k + 1] * dimensions[j + 1];
                dp[i][j] = Math.min(dp[i][j], cost);
            }
        }
    }
    
    return dp[0][n - 1];
}""",
                "time": "O(n³)",
                "space": "O(n²)"
            }]
        },
        "Skyline Problem": {
            "solutions": [{
                "title": "Solution 1 – Line Sweep",
                "description": "Use line sweep algorithm with priority queue.",
                "code": """public List<List<Integer>> getSkyline(int[][] buildings) {
    List<List<Integer>> result = new ArrayList<>();
    List<int[]> heights = new ArrayList<>();
    
    // Convert buildings to height events
    for (int[] building : buildings) {
        heights.add(new int[]{building[0], -building[2]}); // start
        heights.add(new int[]{building[1], building[2]});  // end
    }
    
    // Sort by x-coordinate
    Collections.sort(heights, (a, b) -> {
        if (a[0] != b[0]) return a[0] - b[0];
        return a[1] - b[1];
    });
    
    PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
    pq.offer(0);
    int prevHeight = 0;
    
    for (int[] height : heights) {
        if (height[1] < 0) {
            pq.offer(-height[1]); // start of building
        } else {
            pq.remove(height[1]); // end of building
        }
        
        int currHeight = pq.peek();
        if (currHeight != prevHeight) {
            result.add(Arrays.asList(height[0], currHeight));
            prevHeight = currHeight;
        }
    }
    
    return result;
}""",
                "time": "O(n log n)",
                "space": "O(n)"
            }]
        },
        "Implement Stack using Arrays": {
            "solutions": [{
                "title": "Solution 1 – Array Implementation",
                "description": "Implement stack using a fixed-size array.",
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
            System.out.println("Stack Overflow");
            return;
        }
        arr[++top] = x;
    }
    
    public int pop() {
        if (isEmpty()) {
            System.out.println("Stack Underflow");
            return -1;
        }
        return arr[top--];
    }
    
    public int peek() {
        if (isEmpty()) {
            System.out.println("Stack is Empty");
            return -1;
        }
        return arr[top];
    }
    
    public boolean isEmpty() {
        return top == -1;
    }
    
    public boolean isFull() {
        return top == capacity - 1;
    }
}""",
                "time": "O(1) for all operations",
                "space": "O(n)"
            }]
        },
        "Implement Queue using Arrays": {
            "solutions": [{
                "title": "Solution 1 – Circular Array",
                "description": "Implement queue using circular array.",
                "code": """class Queue {
    private int[] arr;
    private int front, rear, size;
    private int capacity;
    
    public Queue(int size) {
        arr = new int[size];
        capacity = size;
        front = 0;
        rear = -1;
        size = 0;
    }
    
    public void enqueue(int x) {
        if (isFull()) {
            System.out.println("Queue Overflow");
            return;
        }
        rear = (rear + 1) % capacity;
        arr[rear] = x;
        size++;
    }
    
    public int dequeue() {
        if (isEmpty()) {
            System.out.println("Queue Underflow");
            return -1;
        }
        int x = arr[front];
        front = (front + 1) % capacity;
        size--;
        return x;
    }
    
    public int front() {
        if (isEmpty()) {
            System.out.println("Queue is Empty");
            return -1;
        }
        return arr[front];
    }
    
    public boolean isEmpty() {
        return size == 0;
    }
    
    public boolean isFull() {
        return size == capacity;
    }
}""",
                "time": "O(1) for all operations",
                "space": "O(n)"
            }]
        },
        "Sliding Window Maximum": {
            "solutions": [{
                "title": "Solution 1 – Deque",
                "description": "Use deque to maintain maximum elements in sliding window.",
                "code": """public int[] maxSlidingWindow(int[] nums, int k) {
    if (nums == null || nums.length == 0) return new int[0];
    
    int n = nums.length;
    int[] result = new int[n - k + 1];
    Deque<Integer> deque = new LinkedList<>();
    
    for (int i = 0; i < n; i++) {
        // Remove elements outside window
        while (!deque.isEmpty() && deque.peekFirst() < i - k + 1) {
            deque.pollFirst();
        }
        
        // Remove smaller elements
        while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
            deque.pollLast();
        }
        
        deque.offerLast(i);
        
        if (i >= k - 1) {
            result[i - k + 1] = nums[deque.peekFirst()];
        }
    }
    
    return result;
}""",
                "time": "O(n)",
                "space": "O(k)"
            }]
        },
        "Longest Valid Parentheses": {
            "solutions": [{
                "title": "Solution 1 – Stack",
                "description": "Use stack to track valid parentheses positions.",
                "code": """public int longestValidParentheses(String s) {
    Stack<Integer> stack = new Stack<>();
    stack.push(-1);
    int maxLen = 0;
    
    for (int i = 0; i < s.length(); i++) {
        if (s.charAt(i) == '(') {
            stack.push(i);
        } else {
            stack.pop();
            if (stack.isEmpty()) {
                stack.push(i);
            } else {
                maxLen = Math.max(maxLen, i - stack.peek());
            }
        }
    }
    
    return maxLen;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Trapping Rain Water": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to calculate trapped water.",
                "code": """public int trap(int[] height) {
    int left = 0, right = height.length - 1;
    int leftMax = 0, rightMax = 0;
    int water = 0;
    
    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) {
                leftMax = height[left];
            } else {
                water += leftMax - height[left];
            }
            left++;
        } else {
            if (height[right] >= rightMax) {
                rightMax = height[right];
            } else {
                water += rightMax - height[right];
            }
            right--;
        }
    }
    
    return water;
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Word Break": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Use DP to check if string can be segmented.",
                "code": """public boolean wordBreak(String s, List<String> wordDict) {
    Set<String> wordSet = new HashSet<>(wordDict);
    boolean[] dp = new boolean[s.length() + 1];
    dp[0] = true;
    
    for (int i = 1; i <= s.length(); i++) {
        for (int j = 0; j < i; j++) {
            if (dp[j] && wordSet.contains(s.substring(j, i))) {
                dp[i] = true;
                break;
            }
        }
    }
    
    return dp[s.length()];
}""",
                "time": "O(n³)",
                "space": "O(n)"
            }]
        },
        "Word Break II": {
            "solutions": [{
                "title": "Solution 1 – DFS with Memoization",
                "description": "Use DFS to find all possible word breaks.",
                "code": """public List<String> wordBreak(String s, List<String> wordDict) {
    Set<String> wordSet = new HashSet<>(wordDict);
    Map<String, List<String>> memo = new HashMap<>();
    return dfs(s, wordSet, memo);
}

private List<String> dfs(String s, Set<String> wordSet, Map<String, List<String>> memo) {
    if (memo.containsKey(s)) return memo.get(s);
    
    List<String> result = new ArrayList<>();
    
    if (s.isEmpty()) {
        result.add("");
        return result;
    }
    
    for (String word : wordSet) {
        if (s.startsWith(word)) {
            List<String> subList = dfs(s.substring(word.length()), wordSet, memo);
            for (String sub : subList) {
                result.add(word + (sub.isEmpty() ? "" : " " + sub));
            }
        }
    }
    
    memo.put(s, result);
    return result;
}""",
                "time": "O(n³)",
                "space": "O(n³)"
            }]
        },
        "Serialize and Deserialize Binary Tree": {
            "solutions": [{
                "title": "Solution 1 – Preorder Traversal",
                "description": "Use preorder traversal with null markers.",
                "code": """public class Codec {
    private static final String NULL = "null";
    private static final String DELIMITER = ",";
    
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serializeHelper(root, sb);
        return sb.toString();
    }
    
    private void serializeHelper(TreeNode node, StringBuilder sb) {
        if (node == null) {
            sb.append(NULL).append(DELIMITER);
            return;
        }
        sb.append(node.val).append(DELIMITER);
        serializeHelper(node.left, sb);
        serializeHelper(node.right, sb);
    }
    
    public TreeNode deserialize(String data) {
        String[] values = data.split(DELIMITER);
        int[] index = {0};
        return deserializeHelper(values, index);
    }
    
    private TreeNode deserializeHelper(String[] values, int[] index) {
        if (index[0] >= values.length || values[index[0]].equals(NULL)) {
            index[0]++;
            return null;
        }
        
        TreeNode node = new TreeNode(Integer.parseInt(values[index[0]++]));
        node.left = deserializeHelper(values, index);
        node.right = deserializeHelper(values, index);
        return node;
    }
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Alien Dictionary": {
            "solutions": [{
                "title": "Solution 1 – Topological Sort",
                "description": "Use topological sort to find order of characters.",
                "code": """public String alienOrder(String[] words) {
    Map<Character, Set<Character>> graph = new HashMap<>();
    Map<Character, Integer> inDegree = new HashMap<>();
    
    // Initialize
    for (String word : words) {
        for (char c : word.toCharArray()) {
            graph.putIfAbsent(c, new HashSet<>());
            inDegree.putIfAbsent(c, 0);
        }
    }
    
    // Build graph
    for (int i = 0; i < words.length - 1; i++) {
        String word1 = words[i], word2 = words[i + 1];
        int len = Math.min(word1.length(), word2.length());
        
        for (int j = 0; j < len; j++) {
            char c1 = word1.charAt(j), c2 = word2.charAt(j);
            if (c1 != c2) {
                if (!graph.get(c1).contains(c2)) {
                    graph.get(c1).add(c2);
                    inDegree.put(c2, inDegree.get(c2) + 1);
                }
                break;
            }
        }
    }
    
    // Topological sort
    Queue<Character> queue = new LinkedList<>();
    for (char c : inDegree.keySet()) {
        if (inDegree.get(c) == 0) {
            queue.offer(c);
        }
    }
    
    StringBuilder result = new StringBuilder();
    while (!queue.isEmpty()) {
        char c = queue.poll();
        result.append(c);
        
        for (char neighbor : graph.get(c)) {
            inDegree.put(neighbor, inDegree.get(neighbor) - 1);
            if (inDegree.get(neighbor) == 0) {
                queue.offer(neighbor);
            }
        }
    }
    
    return result.length() == graph.size() ? result.toString() : "";
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Graph Valid Tree": {
            "solutions": [{
                "title": "Solution 1 – Union Find",
                "description": "Use union find to detect cycles and ensure connectivity.",
                "code": """public boolean validTree(int n, int[][] edges) {
    if (edges.length != n - 1) return false;
    
    UnionFind uf = new UnionFind(n);
    
    for (int[] edge : edges) {
        if (!uf.union(edge[0], edge[1])) {
            return false; // Cycle detected
        }
    }
    
    return true;
}

class UnionFind {
    private int[] parent, rank;
    
    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
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
        int px = find(x), py = find(y);
        if (px == py) return false;
        
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        return true;
    }
}""",
                "time": "O(n α(n))",
                "space": "O(n)"
            }]
        },
        "Number of Connected Components": {
            "solutions": [{
                "title": "Solution 1 – Union Find",
                "description": "Use union find to count connected components.",
                "code": """public int countComponents(int n, int[][] edges) {
    UnionFind uf = new UnionFind(n);
    
    for (int[] edge : edges) {
        uf.union(edge[0], edge[1]);
    }
    
    return uf.getCount();
}

class UnionFind {
    private int[] parent, rank;
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
    
    public void union(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;
        
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        count--;
    }
    
    public int getCount() {
        return count;
    }
}""",
                "time": "O(n α(n))",
                "space": "O(n)"
            }]
        },
        "Redundant Connection": {
            "solutions": [{
                "title": "Solution 1 – Union Find",
                "description": "Use union find to find the edge that creates a cycle.",
                "code": """public int[] findRedundantConnection(int[][] edges) {
    UnionFind uf = new UnionFind(edges.length + 1);
    
    for (int[] edge : edges) {
        if (!uf.union(edge[0], edge[1])) {
            return edge;
        }
    }
    
    return new int[0];
}

class UnionFind {
    private int[] parent, rank;
    
    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
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
        int px = find(x), py = find(y);
        if (px == py) return false;
        
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        return true;
    }
}""",
                "time": "O(n α(n))",
                "space": "O(n)"
            }]
        },
        "Course Schedule": {
            "solutions": [{
                "title": "Solution 1 – Topological Sort",
                "description": "Use topological sort to detect cycles.",
                "code": """public boolean canFinish(int numCourses, int[][] prerequisites) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < numCourses; i++) {
        graph.add(new ArrayList<>());
    }
    
    int[] inDegree = new int[numCourses];
    
    for (int[] prereq : prerequisites) {
        graph.get(prereq[1]).add(prereq[0]);
        inDegree[prereq[0]]++;
    }
    
    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < numCourses; i++) {
        if (inDegree[i] == 0) {
            queue.offer(i);
        }
    }
    
    int count = 0;
    while (!queue.isEmpty()) {
        int course = queue.poll();
        count++;
        
        for (int neighbor : graph.get(course)) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                queue.offer(neighbor);
            }
        }
    }
    
    return count == numCourses;
}""",
                "time": "O(V + E)",
                "space": "O(V + E)"
            }]
        },
        "Course Schedule II": {
            "solutions": [{
                "title": "Solution 1 – Topological Sort",
                "description": "Use topological sort to find course order.",
                "code": """public int[] findOrder(int numCourses, int[][] prerequisites) {
    List<List<Integer>> graph = new ArrayList<>();
    for (int i = 0; i < numCourses; i++) {
        graph.add(new ArrayList<>());
    }
    
    int[] inDegree = new int[numCourses];
    
    for (int[] prereq : prerequisites) {
        graph.get(prereq[1]).add(prereq[0]);
        inDegree[prereq[0]]++;
    }
    
    Queue<Integer> queue = new LinkedList<>();
    for (int i = 0; i < numCourses; i++) {
        if (inDegree[i] == 0) {
            queue.offer(i);
        }
    }
    
    int[] result = new int[numCourses];
    int index = 0;
    
    while (!queue.isEmpty()) {
        int course = queue.poll();
        result[index++] = course;
        
        for (int neighbor : graph.get(course)) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                queue.offer(neighbor);
            }
        }
    }
    
    return index == numCourses ? result : new int[0];
}""",
                "time": "O(V + E)",
                "space": "O(V + E)"
            }]
        },
        "Clone Graph": {
            "solutions": [{
                "title": "Solution 1 – DFS",
                "description": "Use DFS with HashMap to clone the graph.",
                "code": """public Node cloneGraph(Node node) {
    if (node == null) return null;
    
    Map<Node, Node> visited = new HashMap<>();
    return cloneGraphHelper(node, visited);
}

private Node cloneGraphHelper(Node node, Map<Node, Node> visited) {
    if (visited.containsKey(node)) {
        return visited.get(node);
    }
    
    Node clone = new Node(node.val);
    visited.put(node, clone);
    
    for (Node neighbor : node.neighbors) {
        clone.neighbors.add(cloneGraphHelper(neighbor, visited));
    }
    
    return clone;
}""",
                "time": "O(V + E)",
                "space": "O(V)"
            }]
        },
        "Pacific Atlantic Water Flow": {
            "solutions": [{
                "title": "Solution 1 – DFS from Oceans",
                "description": "Use DFS from both oceans to find cells that can flow to both.",
                "code": """public List<List<Integer>> pacificAtlantic(int[][] heights) {
    int m = heights.length, n = heights[0].length;
    boolean[][] pacific = new boolean[m][n];
    boolean[][] atlantic = new boolean[m][n];
    
    // DFS from Pacific (top and left edges)
    for (int i = 0; i < m; i++) {
        dfs(heights, pacific, i, 0, Integer.MIN_VALUE);
    }
    for (int j = 0; j < n; j++) {
        dfs(heights, pacific, 0, j, Integer.MIN_VALUE);
    }
    
    // DFS from Atlantic (bottom and right edges)
    for (int i = 0; i < m; i++) {
        dfs(heights, atlantic, i, n - 1, Integer.MIN_VALUE);
    }
    for (int j = 0; j < n; j++) {
        dfs(heights, atlantic, m - 1, j, Integer.MIN_VALUE);
    }
    
    List<List<Integer>> result = new ArrayList<>();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (pacific[i][j] && atlantic[i][j]) {
                result.add(Arrays.asList(i, j));
            }
        }
    }
    
    return result;
}

private void dfs(int[][] heights, boolean[][] ocean, int i, int j, int prev) {
    if (i < 0 || i >= heights.length || j < 0 || j >= heights[0].length || 
        ocean[i][j] || heights[i][j] < prev) {
        return;
    }
    
    ocean[i][j] = true;
    dfs(heights, ocean, i + 1, j, heights[i][j]);
    dfs(heights, ocean, i - 1, j, heights[i][j]);
    dfs(heights, ocean, i, j + 1, heights[i][j]);
    dfs(heights, ocean, i, j - 1, heights[i][j]);
}""",
                "time": "O(m × n)",
                "space": "O(m × n)"
            }]
        },
        "Surrounded Regions": {
            "solutions": [{
                "title": "Solution 1 – DFS from Borders",
                "description": "Use DFS from borders to mark regions that should not be flipped.",
                "code": """public void solve(char[][] board) {
    if (board == null || board.length == 0) return;
    
    int m = board.length, n = board[0].length;
    
    // Mark 'O's connected to borders
    for (int i = 0; i < m; i++) {
        dfs(board, i, 0);
        dfs(board, i, n - 1);
    }
    for (int j = 0; j < n; j++) {
        dfs(board, 0, j);
        dfs(board, m - 1, j);
    }
    
    // Flip remaining 'O's to 'X' and restore marked 'O's
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
    if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || 
        board[i][j] != 'O') {
        return;
    }
    
    board[i][j] = 'E'; // Mark as escaped
    dfs(board, i + 1, j);
    dfs(board, i - 1, j);
    dfs(board, i, j + 1);
    dfs(board, i, j - 1);
}""",
                "time": "O(m × n)",
                "space": "O(m × n)"
            }]
        },
        "Walls and Gates": {
            "solutions": [{
                "title": "Solution 1 – BFS from Gates",
                "description": "Use BFS from all gates to fill distances.",
                "code": """public void wallsAndGates(int[][] rooms) {
    if (rooms == null || rooms.length == 0) return;
    
    int m = rooms.length, n = rooms[0].length;
    Queue<int[]> queue = new LinkedList<>();
    
    // Add all gates to queue
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (rooms[i][j] == 0) {
                queue.offer(new int[]{i, j});
            }
        }
    }
    
    int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    while (!queue.isEmpty()) {
        int[] cell = queue.poll();
        int row = cell[0], col = cell[1];
        
        for (int[] dir : directions) {
            int newRow = row + dir[0];
            int newCol = col + dir[1];
            
            if (newRow >= 0 && newRow < m && newCol >= 0 && newCol < n && 
                rooms[newRow][newCol] == Integer.MAX_VALUE) {
                rooms[newRow][newCol] = rooms[row][col] + 1;
                queue.offer(new int[]{newRow, newCol});
            }
        }
    }
}""",
                "time": "O(m × n)",
                "space": "O(m × n)"
            }]
        },
        "Number of Islands II": {
            "solutions": [{
                "title": "Solution 1 – Union Find",
                "description": "Use union find to track connected components as islands are added.",
                "code": """public List<Integer> numIslands2(int m, int n, int[][] positions) {
    List<Integer> result = new ArrayList<>();
    UnionFind uf = new UnionFind(m * n);
    int[][] grid = new int[m][n];
    int count = 0;
    
    int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    for (int[] pos : positions) {
        int i = pos[0], j = pos[1];
        if (grid[i][j] == 1) {
            result.add(count);
            continue;
        }
        
        grid[i][j] = 1;
        count++;
        
        for (int[] dir : directions) {
            int newI = i + dir[0], newJ = j + dir[1];
            if (newI >= 0 && newI < m && newJ >= 0 && newJ < n && grid[newI][newJ] == 1) {
                if (uf.union(i * n + j, newI * n + newJ)) {
                    count--;
                }
            }
        }
        
        result.add(count);
    }
    
    return result;
}

class UnionFind {
    private int[] parent, rank;
    
    public UnionFind(int n) {
        parent = new int[n];
        rank = new int[n];
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
        int px = find(x), py = find(y);
        if (px == py) return false;
        
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        return true;
    }
}""",
                "time": "O(k × α(m×n))",
                "space": "O(m × n)"
            }]
        },
        "Friend Circles": {
            "solutions": [{
                "title": "Solution 1 – Union Find",
                "description": "Use union find to count connected components.",
                "code": """public int findCircleNum(int[][] M) {
    int n = M.length;
    UnionFind uf = new UnionFind(n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (M[i][j] == 1) {
                uf.union(i, j);
            }
        }
    }
    
    return uf.getCount();
}

class UnionFind {
    private int[] parent, rank;
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
    
    public void union(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;
        
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
        count--;
    }
    
    public int getCount() {
        return count;
    }
}""",
                "time": "O(n² α(n))",
                "space": "O(n)"
            }]
        },
        "Accounts Merge": {
            "solutions": [{
                "title": "Solution 1 – Union Find",
                "description": "Use union find to merge accounts with common emails.",
                "code": """public List<List<String>> accountsMerge(List<List<String>> accounts) {
    Map<String, String> emailToName = new HashMap<>();
    Map<String, Integer> emailToId = new HashMap<>();
    UnionFind uf = new UnionFind(10001);
    int id = 0;
    
    for (List<String> account : accounts) {
        String name = account.get(0);
        for (int i = 1; i < account.size(); i++) {
            String email = account.get(i);
            emailToName.put(email, name);
            if (!emailToId.containsKey(email)) {
                emailToId.put(email, id++);
            }
            uf.union(emailToId.get(account.get(1)), emailToId.get(email));
        }
    }
    
    Map<Integer, List<String>> ans = new HashMap<>();
    for (String email : emailToName.keySet()) {
        int index = uf.find(emailToId.get(email));
        ans.computeIfAbsent(index, x -> new ArrayList<>()).add(email);
    }
    
    for (List<String> component : ans.values()) {
        Collections.sort(component);
        component.add(0, emailToName.get(component.get(0)));
    }
    
    return new ArrayList<>(ans.values());
}

class UnionFind {
    private int[] parent;
    
    public UnionFind(int n) {
        parent = new int[n];
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
    
    public void union(int x, int y) {
        parent[find(x)] = find(y);
    }
}""",
                "time": "O(n log n)",
                "space": "O(n)"
            }]
        },
        "Most Stones Removed": {
            "solutions": [{
                "title": "Solution 1 – Union Find",
                "description": "Use union find to count connected components.",
                "code": """public int removeStones(int[][] stones) {
    UnionFind uf = new UnionFind(20000);
    
    for (int[] stone : stones) {
        uf.union(stone[0], stone[1] + 10000);
    }
    
    Set<Integer> seen = new HashSet<>();
    for (int[] stone : stones) {
        seen.add(uf.find(stone[0]));
    }
    
    return stones.length - seen.size();
}

class UnionFind {
    private int[] parent;
    
    public UnionFind(int n) {
        parent = new int[n];
        for (int i = 0;0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    public int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    public void union(int x, int y) {
        parent[find(x)] = find(y);
    }
}""",
                "time": "O(n α(n))",
                "space": "O(n)"
            }]
        },
        "Regions Cut By Slashes": {
            "solutions": [{
                "title": "Solution 1 – Union Find",
                "description": "Use union find to count regions formed by slashes.",
                "code": """public int regionsBySlashes(String[] grid) {
    int n = grid.length;
    UnionFind uf = new UnionFind(4 * n * n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int base = 4 * (i * n + j);
            char c = grid[i].charAt(j);
            
            if (c == '/') {
                uf.union(base, base + 3);
                uf.union(base + 1, base + 2);
            } else if (c == '\\\\') {
                uf.union(base, base + 1);
                uf.union(base + 2, base + 3);
            } else {
                uf.union(base, base + 1);
                uf.union(base + 1, base + 2);
                uf.union(base + 2, base + 3);
            }
            
            // Connect to adjacent cells
            if (i > 0) {
                uf.union(base, base - 4 * n + 2);
            }
            if (j > 0) {
                uf.union(base + 3, base - 4 + 1);
            }
        }
    }
    
    Set<Integer> regions = new HashSet<>();
    for (int i = 0; i < 4 * n * n; i++) {
        regions.add(uf.find(i));
    }
    
    return regions.size();
}

class UnionFind {
    private int[] parent;
    
    public UnionFind(int n) {
        parent = new int[n];
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
    
    public void union(int x, int y) {
        parent[find(x)] = find(y);
    }
}""",
                "time": "O(n² α(n²))",
                "space": "O(n²)"
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
    solutions = get_more_solutions()
    
    problems_to_update = [
        ("matrix-chain-multiplication.html", "Matrix Chain Multiplication"),
        ("skyline-problem.html", "Skyline Problem"),
        ("implement-stack-using-arrays.html", "Implement Stack using Arrays"),
        ("implement-queue-using-arrays.html", "Implement Queue using Arrays"),
        ("sliding-window-maximum.html", "Sliding Window Maximum"),
        ("longest-valid-parentheses.html", "Longest Valid Parentheses"),
        ("trapping-rain-water.html", "Trapping Rain Water"),
        ("word-break.html", "Word Break"),
        ("word-break-ii.html", "Word Break II"),
        ("serialize-deserialize-binary-tree.html", "Serialize and Deserialize Binary Tree"),
        ("alien-dictionary.html", "Alien Dictionary"),
        ("graph-valid-tree.html", "Graph Valid Tree"),
        ("number-of-connected-components.html", "Number of Connected Components"),
        ("redundant-connection.html", "Redundant Connection"),
        ("course-schedule.html", "Course Schedule"),
        ("course-schedule-ii.html", "Course Schedule II"),
        ("clone-graph.html", "Clone Graph"),
        ("pacific-atlantic-water-flow.html", "Pacific Atlantic Water Flow"),
        ("surrounded-regions.html", "Surrounded Regions"),
        ("walls-and-gates.html", "Walls and Gates"),
        ("number-of-islands-ii.html", "Number of Islands II"),
        ("friend-circles.html", "Friend Circles"),
        ("accounts-merge.html", "Accounts Merge"),
        ("most-stones-removed.html", "Most Stones Removed"),
        ("regions-cut-by-slashes.html", "Regions Cut By Slashes")
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