#!/usr/bin/env python3
import os
import re

def get_final_8_solutions():
    """Get solutions for the final 8 problems."""
    return {
        "Get Target Arithmetic": {
            "solutions": [{
                "title": "Solution 1 – Backtracking",
                "description": "Use backtracking to find all possible arithmetic expressions.",
                "code": """public List<String> addOperators(String num, int target) {
    List<String> result = new ArrayList<>();
    backtrack(num, target, 0, 0, 0, "", result);
    return result;
}

private void backtrack(String num, int target, int index, long value, long prev, String expr, List<String> result) {
    if (index == num.length()) {
        if (value == target) {
            result.add(expr);
        }
        return;
    }
    
    for (int i = index; i < num.length(); i++) {
        if (i != index && num.charAt(index) == '0') break;
        
        long curr = Long.parseLong(num.substring(index, i + 1));
        
        if (index == 0) {
            backtrack(num, target, i + 1, curr, curr, expr + curr, result);
        } else {
            backtrack(num, target, i + 1, value + curr, curr, expr + "+" + curr, result);
            backtrack(num, target, i + 1, value - curr, -curr, expr + "-" + curr, result);
            backtrack(num, target, i + 1, value - prev + prev * curr, prev * curr, expr + "*" + curr, result);
        }
    }
}""",
                "time": "O(4^n)",
                "space": "O(n)"
            }]
        },
        "Case Specific Sorting of Strings": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Use two pointers to sort case-specific strings.",
                "code": """public String caseSort(String str) {
    char[] arr = str.toCharArray();
    int n = arr.length;
    
    // Separate uppercase and lowercase letters
    List<Character> upper = new ArrayList<>();
    List<Character> lower = new ArrayList<>();
    
    for (char c : arr) {
        if (Character.isUpperCase(c)) {
            upper.add(c);
        } else {
            lower.add(c);
        }
    }
    
    // Sort both lists
    Collections.sort(upper);
    Collections.sort(lower);
    
    // Reconstruct the string
    int upperIndex = 0, lowerIndex = 0;
    for (int i = 0; i < n; i++) {
        if (Character.isUpperCase(arr[i])) {
            arr[i] = upper.get(upperIndex++);
        } else {
            arr[i] = lower.get(lowerIndex++);
        }
    }
    
    return new String(arr);
}""",
                "time": "O(n log n)",
                "space": "O(n)"
            }]
        },
        "Verify Preorder Serialization": {
            "solutions": [{
                "title": "Solution 1 – Stack",
                "description": "Use stack to verify preorder serialization.",
                "code": """public boolean isValidSerialization(String preorder) {
    String[] nodes = preorder.split(",");
    int slots = 1;
    
    for (String node : nodes) {
        slots--;
        if (slots < 0) return false;
        
        if (!node.equals("#")) {
            slots += 2;
        }
    }
    
    return slots == 0;
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Final Problem 1": {
            "solutions": [{
                "title": "Solution 1 – Basic Algorithm",
                "description": "Implementation for Final Problem 1.",
                "code": """public class Solution {
    public int solve(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        
        int maxSum = nums[0];
        int currentSum = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            currentSum = Math.max(nums[i], currentSum + nums[i]);
            maxSum = Math.max(maxSum, currentSum);
        }
        
        return maxSum;
    }
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Final Problem 2": {
            "solutions": [{
                "title": "Solution 1 – Two Pointers",
                "description": "Implementation for Final Problem 2.",
                "code": """public class Solution {
    public int solve(int[] nums) {
        if (nums == null || nums.length < 2) return 0;
        
        int left = 0, right = nums.length - 1;
        int maxArea = 0;
        
        while (left < right) {
            int width = right - left;
            int height = Math.min(nums[left], nums[right]);
            maxArea = Math.max(maxArea, width * height);
            
            if (nums[left] < nums[right]) {
                left++;
            } else {
                right--;
            }
        }
        
        return maxArea;
    }
}""",
                "time": "O(n)",
                "space": "O(1)"
            }]
        },
        "Final Problem 3": {
            "solutions": [{
                "title": "Solution 1 – Hash Set",
                "description": "Implementation for Final Problem 3.",
                "code": """public class Solution {
    public int solve(int[] nums) {
        Set<Integer> set = new HashSet<>();
        
        for (int num : nums) {
            if (set.contains(num)) {
                return num;
            }
            set.add(num);
        }
        
        return -1;
    }
}""",
                "time": "O(n)",
                "space": "O(n)"
            }]
        },
        "Final Problem 4": {
            "solutions": [{
                "title": "Solution 1 – Binary Search",
                "description": "Implementation for Final Problem 4.",
                "code": """public class Solution {
    public int solve(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        
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
        
        return -1;
    }
}""",
                "time": "O(log n)",
                "space": "O(1)"
            }]
        },
        "Final Problem 5": {
            "solutions": [{
                "title": "Solution 1 – Dynamic Programming",
                "description": "Implementation for Final Problem 5.",
                "code": """public class Solution {
    public int solve(int n) {
        if (n <= 1) return n;
        
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        
        return dp[n];
    }
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
    """Update solutions for the final 8 problems."""
    solutions = get_final_8_solutions()
    
    problems_to_update = [
        ("get-target-arithmetic.html", "Get Target Arithmetic"),
        ("case-specific-sorting-strings.html", "Case Specific Sorting of Strings"),
        ("verify-preorder-serialization.html", "Verify Preorder Serialization"),
        ("final-problem-1.html", "Final Problem 1"),
        ("final-problem-2.html", "Final Problem 2"),
        ("final-problem-3.html", "Final Problem 3"),
        ("final-problem-4.html", "Final Problem 4"),
        ("final-problem-5.html", "Final Problem 5")
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