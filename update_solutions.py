#!/usr/bin/env python3
import os
import re

def get_solution_for_problem(problem_name):
    """Return the actual solution for a given problem."""
    solutions = {
        "Two Sum": {
            "description": "Find two numbers in an array that add up to target",
            "solutions": [
                {
                    "title": "Solution 1 – Brute Force",
                    "description": "Check every pair of numbers in the array.",
                    "code": """public int[] twoSum(int[] nums, int target) {
    for (int i = 0; i < nums.length; i++) {
        for (int j = i + 1; j < nums.length; j++) {
            if (nums[i] + nums[j] == target) {
                return new int[]{i, j};
            }
        }
    }
    return new int[]{};
}""",
                    "time": "O(n²)",
                    "space": "O(1)"
                },
                {
                    "title": "Solution 2 – Hash Map (Optimal)",
                    "description": "Use a hash map to store complements. For each number, check if its complement (target - num) exists in the map.",
                    "code": """public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        
        if (map.containsKey(complement)) {
            return new int[]{map.get(complement), i};
        }
        
        map.put(nums[i], i);
    }
    
    return new int[]{};
}""",
                    "time": "O(n)",
                    "space": "O(n)"
                }
            ]
        },
        "Add Two Numbers": {
            "description": "Add two numbers represented by linked lists",
            "solutions": [
                {
                    "title": "Solution 1 – Iterative Approach",
                    "description": "Traverse both lists simultaneously, adding digits and handling carry.",
                    "code": """public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0);
    ListNode current = dummy;
    int carry = 0;
    
    while (l1 != null || l2 != null || carry != 0) {
        int sum = carry;
        
        if (l1 != null) {
            sum += l1.val;
            l1 = l1.next;
        }
        
        if (l2 != null) {
            sum += l2.val;
            l2 = l2.next;
        }
        
        carry = sum / 10;
        current.next = new ListNode(sum % 10);
        current = current.next;
    }
    
    return dummy.next;
}""",
                    "time": "O(max(m,n))",
                    "space": "O(max(m,n))"
                }
            ]
        },
        "Reverse Linked List": {
            "description": "Reverse a singly linked list",
            "solutions": [
                {
                    "title": "Solution 1 – Iterative",
                    "description": "Use three pointers to reverse the links.",
                    "code": """public ListNode reverseList(ListNode head) {
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
                },
                {
                    "title": "Solution 2 – Recursive",
                    "description": "Recursively reverse the rest of the list and update the head.",
                    "code": """public ListNode reverseList(ListNode head) {
    if (head == null || head.next == null) {
        return head;
    }
    
    ListNode newHead = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    
    return newHead;
}""",
                    "time": "O(n)",
                    "space": "O(n)"
                }
            ]
        },
        "Valid Parentheses": {
            "description": "Check if a string of parentheses is valid",
            "solutions": [
                {
                    "title": "Solution 1 – Stack",
                    "description": "Use a stack to keep track of opening brackets.",
                    "code": """public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    
    for (char c : s.toCharArray()) {
        if (c == '(' || c == '{' || c == '[') {
            stack.push(c);
        } else {
            if (stack.isEmpty()) return false;
            
            char top = stack.pop();
            if ((c == ')' && top != '(') ||
                (c == '}' && top != '{') ||
                (c == ']' && top != '[')) {
                return false;
            }
        }
    }
    
    return stack.isEmpty();
}""",
                    "time": "O(n)",
                    "space": "O(n)"
                }
            ]
        },
        "Merge Two Sorted Lists": {
            "description": "Merge two sorted linked lists into one sorted list",
            "solutions": [
                {
                    "title": "Solution 1 – Iterative",
                    "description": "Compare nodes from both lists and link them in sorted order.",
                    "code": """public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0);
    ListNode current = dummy;
    
    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            current.next = l1;
            l1 = l1.next;
        } else {
            current.next = l2;
            l2 = l2.next;
        }
        current = current.next;
    }
    
    current.next = (l1 != null) ? l1 : l2;
    return dummy.next;
}""",
                    "time": "O(n + m)",
                    "space": "O(1)"
                }
            ]
        },
        "Maximum Subarray": {
            "description": "Find the subarray with the largest sum",
            "solutions": [
                {
                    "title": "Solution 1 – Kadane's Algorithm",
                    "description": "Keep track of current sum and maximum sum seen so far.",
                    "code": """public int maxSubArray(int[] nums) {
    int maxSoFar = nums[0];
    int maxEndingHere = nums[0];
    
    for (int i = 1; i < nums.length; i++) {
        maxEndingHere = Math.max(nums[i], maxEndingHere + nums[i]);
        maxSoFar = Math.max(maxSoFar, maxEndingHere);
    }
    
    return maxSoFar;
}""",
                    "time": "O(n)",
                    "space": "O(1)"
                }
            ]
        },
        "Climbing Stairs": {
            "description": "Find number of ways to climb n stairs",
            "solutions": [
                {
                    "title": "Solution 1 – Dynamic Programming",
                    "description": "Use DP to build up the solution from smaller subproblems.",
                    "code": """public int climbStairs(int n) {
    if (n <= 2) return n;
    
    int[] dp = new int[n + 1];
    dp[1] = 1;
    dp[2] = 2;
    
    for (int i = 3; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];
    }
    
    return dp[n];
}""",
                    "time": "O(n)",
                    "space": "O(n)"
                },
                {
                    "title": "Solution 2 – Space Optimized",
                    "description": "Use only two variables to track previous two values.",
                    "code": """public int climbStairs(int n) {
    if (n <= 2) return n;
    
    int prev1 = 1, prev2 = 2;
    
    for (int i = 3; i <= n; i++) {
        int current = prev1 + prev2;
        prev1 = prev2;
        prev2 = current;
    }
    
    return prev2;
}""",
                    "time": "O(n)",
                    "space": "O(1)"
                }
            ]
        },
        "Best Time to Buy and Sell Stock": {
            "description": "Find maximum profit from buying and selling stock",
            "solutions": [
                {
                    "title": "Solution 1 – One Pass",
                    "description": "Keep track of minimum price and maximum profit.",
                    "code": """public int maxProfit(int[] prices) {
    int minPrice = Integer.MAX_VALUE;
    int maxProfit = 0;
    
    for (int price : prices) {
        minPrice = Math.min(minPrice, price);
        maxProfit = Math.max(maxProfit, price - minPrice);
    }
    
    return maxProfit;
}""",
                    "time": "O(n)",
                    "space": "O(1)"
                }
            ]
        },
        "Single Number": {
            "description": "Find the number that appears only once in an array",
            "solutions": [
                {
                    "title": "Solution 1 – XOR",
                    "description": "Use XOR to find the single number.",
                    "code": """public int singleNumber(int[] nums) {
    int result = 0;
    
    for (int num : nums) {
        result ^= num;
    }
    
    return result;
}""",
                    "time": "O(n)",
                    "space": "O(1)"
                }
            ]
        },
        "Happy Number": {
            "description": "Check if a number is happy (sum of squares of digits eventually becomes 1)",
            "solutions": [
                {
                    "title": "Solution 1 – Hash Set",
                    "description": "Use a hash set to detect cycles.",
                    "code": """public boolean isHappy(int n) {
    Set<Integer> seen = new HashSet<>();
    
    while (n != 1 && !seen.contains(n)) {
        seen.add(n);
        n = getNext(n);
    }
    
    return n == 1;
}

private int getNext(int n) {
    int sum = 0;
    while (n > 0) {
        int digit = n % 10;
        sum += digit * digit;
        n /= 10;
    }
    return sum;
}""",
                    "time": "O(log n)",
                    "space": "O(log n)"
                }
            ]
        }
    }
    
    return solutions.get(problem_name, None)

def update_problem_file(filename, problem_name):
    """Update a problem file with actual solutions."""
    filepath = f"problems/{filename}"
    
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file has placeholder solution
    if "TODO: Implement solution" in content:
        solution_data = get_solution_for_problem(problem_name)
        
        if solution_data:
            # Replace the placeholder solution
            new_solutions_html = ""
            for i, solution in enumerate(solution_data["solutions"]):
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
    """Update all problem files with actual solutions."""
    print("🚀 Starting Solution Update Process")
    print("=" * 70)
    
    # List of problems to update with their filenames
    problems_to_update = [
        ("two-sum.html", "Two Sum"),
        ("add-two-numbers.html", "Add Two Numbers"),
        ("reverse-linked-list.html", "Reverse Linked List"),
        ("valid-parentheses.html", "Valid Parentheses"),
        ("merge-two-sorted-lists.html", "Merge Two Sorted Lists"),
        ("maximum-subarray.html", "Maximum Subarray"),
        ("climbing-stairs.html", "Climbing Stairs"),
        ("best-time-buy-sell-stock.html", "Best Time to Buy and Sell Stock"),
        ("single-number.html", "Single Number"),
        ("happy-number.html", "Happy Number")
    ]
    
    updated_count = 0
    
    for filename, problem_name in problems_to_update:
        if update_problem_file(filename, problem_name):
            updated_count += 1
    
    print(f"\n" + "=" * 70)
    print(f"✅ Updated {updated_count} problem files with actual solutions!")
    print("🌐 You can now view the complete solutions in the problem pages.")

if __name__ == "__main__":
    main() 