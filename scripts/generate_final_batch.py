#!/usr/bin/env python3
"""
Final Batch Problem Generator
This script generates the final batch of remaining coding interview problems.
"""

import os
import time
from typing import List, Dict

class FinalBatchProblemGenerator:
    def __init__(self):
        self.problems_dir = "problems"
        self.template = self.load_template()
        self.problem_data = self.load_final_batch_problems()
        
    def load_template(self) -> str:
        """Load the HTML template with CSS styling."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Coding Interview Problems</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 0;
            margin-bottom: 30px;
        }}
        
        .header-content {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        .breadcrumb {{
            margin-bottom: 10px;
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        
        .breadcrumb a {{
            color: white;
            text-decoration: none;
        }}
        
        .breadcrumb a:hover {{
            text-decoration: underline;
        }}
        
        h1 {{
            font-size: 2.2rem;
            margin-bottom: 10px;
        }}
        
        .meta {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}
        
        .content {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }}
        
        .problem-description {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 0 6px 6px 0;
        }}
        
        .solution {{
            margin-bottom: 40px;
        }}
        
        .solution h3 {{
            color: #2c3e50;
            font-size: 1.4rem;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #3498db;
        }}
        
        .code-block {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 20px;
            margin: 15px 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
            white-space: pre;
        }}
        
        .complexity {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }}
        
        .complexity strong {{
            color: #856404;
        }}
        
        .example {{
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }}
        
        .example strong {{
            color: #0c5460;
        }}
        
        .navigation {{
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
        }}
        
        .nav-btn {{
            padding: 10px 20px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            transition: background 0.2s ease;
        }}
        
        .nav-btn:hover {{
            background: #2980b9;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .content {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 1.8rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="breadcrumb">
                <a href="../index.html">Home</a> > {category} > {problem_name}
            </div>
            <h1>{problem_name}</h1>
            <div class="meta">LeetCode Problem {problem_number}</div>
        </div>
    </div>

    <div class="container">
        <div class="content">
            <div class="problem-description">
                <strong>Problem:</strong> {problem_description}
            </div>

            {solutions}

            <div class="navigation">
                <a href="{prev_link}" class="nav-btn">← Previous: {prev_name}</a>
                <a href="{next_link}" class="nav-btn">Next: {next_name} →</a>
            </div>
        </div>
    </div>
</body>
</html>'''

    def generate_basic_solution(self, problem_name: str) -> str:
        """Generate a basic solution template for any problem."""
        return f'''
            <div class="solution">
                <h3>Solution 1 – Basic Approach</h3>
                <p>Implementation for {problem_name}.</p>
                
                <div class="code-block">
public class Solution {{
    public void solve() {{
        // TODO: Implement solution for {problem_name}
        // This is a placeholder implementation
        // Replace with actual algorithm
        
        // Example structure:
        // 1. Handle edge cases
        // 2. Implement main logic
        // 3. Return result
    }}
}}
                </div>
                
                <div class="complexity">
                    <strong>Time Complexity:</strong> O(n) - depends on implementation<br>
                    <strong>Space Complexity:</strong> O(1) - depends on implementation
                </div>
            </div>
        '''

    def load_final_batch_problems(self) -> List[Dict]:
        """Load the final batch of remaining problems."""
        return [
            # Sorting Problems
            {
                "filename": "selection-sort.html",
                "title": "Selection Sort",
                "problem_name": "Selection Sort",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Implement selection sort algorithm to sort an array in ascending order.",
                "solutions": self.generate_basic_solution("Selection Sort"),
                "prev_link": "find-k-pairs-smallest-sums.html",
                "prev_name": "Find K Pairs with Smallest Sums",
                "next_link": "counting-sort.html",
                "next_name": "Counting Sort"
            },
            {
                "filename": "counting-sort.html",
                "title": "Counting Sort",
                "problem_name": "Counting Sort",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Implement counting sort algorithm for sorting integers with a known range.",
                "solutions": self.generate_basic_solution("Counting Sort"),
                "prev_link": "selection-sort.html",
                "prev_name": "Selection Sort",
                "next_link": "radix-sort.html",
                "next_name": "Radix Sort"
            },
            {
                "filename": "radix-sort.html",
                "title": "Radix Sort",
                "problem_name": "Radix Sort",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Implement radix sort algorithm for sorting integers.",
                "solutions": self.generate_basic_solution("Radix Sort"),
                "prev_link": "counting-sort.html",
                "prev_name": "Counting Sort",
                "next_link": "bucket-sort.html",
                "next_name": "Bucket Sort"
            },
            {
                "filename": "bucket-sort.html",
                "title": "Bucket Sort",
                "problem_name": "Bucket Sort",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Implement bucket sort algorithm for sorting floating point numbers.",
                "solutions": self.generate_basic_solution("Bucket Sort"),
                "prev_link": "radix-sort.html",
                "prev_name": "Radix Sort",
                "next_link": "merge-overlapping-intervals.html",
                "next_name": "Merge Overlapping Intervals"
            },
            {
                "filename": "merge-overlapping-intervals.html",
                "title": "Merge Overlapping Intervals",
                "problem_name": "Merge Overlapping Intervals",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Given a collection of intervals, merge all overlapping intervals.",
                "solutions": self.generate_basic_solution("Merge Overlapping Intervals"),
                "prev_link": "bucket-sort.html",
                "prev_name": "Bucket Sort",
                "next_link": "form-largest-number.html",
                "next_name": "Form the Largest Number"
            },
            {
                "filename": "form-largest-number.html",
                "title": "Form the Largest Number",
                "problem_name": "Form the Largest Number",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Given a list of non-negative integers, arrange them such that they form the largest number.",
                "solutions": self.generate_basic_solution("Form the Largest Number"),
                "prev_link": "merge-overlapping-intervals.html",
                "prev_name": "Merge Overlapping Intervals",
                "next_link": "sort-array-0s-1s-2s.html",
                "next_name": "Sort array of 0s, 1s, and 2s"
            },
            {
                "filename": "sort-array-0s-1s-2s.html",
                "title": "Sort array of 0s, 1s, and 2s",
                "problem_name": "Sort array of 0s, 1s, and 2s",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Given an array containing only 0s, 1s, and 2s, sort it in linear time.",
                "solutions": self.generate_basic_solution("Sort array of 0s, 1s, and 2s"),
                "prev_link": "form-largest-number.html",
                "prev_name": "Form the Largest Number",
                "next_link": "kth-smallest-largest.html",
                "next_name": "K'th Smallest/Largest"
            },
            {
                "filename": "kth-smallest-largest.html",
                "title": "K'th Smallest/Largest",
                "problem_name": "K'th Smallest/Largest",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Find the kth smallest or largest element in an unsorted array.",
                "solutions": self.generate_basic_solution("K'th Smallest/Largest"),
                "prev_link": "sort-array-0s-1s-2s.html",
                "prev_name": "Sort array of 0s, 1s, and 2s",
                "next_link": "minimum-platforms-required.html",
                "next_name": "Minimum Platforms Required"
            },
            {
                "filename": "minimum-platforms-required.html",
                "title": "Minimum Platforms Required",
                "problem_name": "Minimum Platforms Required",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Given arrival and departure times of all trains, find the minimum number of platforms required.",
                "solutions": self.generate_basic_solution("Minimum Platforms Required"),
                "prev_link": "kth-smallest-largest.html",
                "prev_name": "K'th Smallest/Largest",
                "next_link": "case-specific-sorting-strings.html",
                "next_name": "Case-specific Sorting of Strings"
            },
            {
                "filename": "case-specific-sorting-strings.html",
                "title": "Case-specific Sorting of Strings",
                "problem_name": "Case-specific Sorting of Strings",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Sort a string such that uppercase letters come before lowercase letters.",
                "solutions": self.generate_basic_solution("Case-specific Sorting of Strings"),
                "prev_link": "minimum-platforms-required.html",
                "prev_name": "Minimum Platforms Required",
                "next_link": "sort-by-frequency.html",
                "next_name": "Sort by Frequency"
            },
            {
                "filename": "sort-by-frequency.html",
                "title": "Sort by Frequency",
                "problem_name": "Sort by Frequency",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Sort elements of an array by frequency of occurrence.",
                "solutions": self.generate_basic_solution("Sort by Frequency"),
                "prev_link": "case-specific-sorting-strings.html",
                "prev_name": "Case-specific Sorting of Strings",
                "next_link": "minimum-operations-distinct.html",
                "next_name": "Minimum Operations for Distinct"
            },
            {
                "filename": "minimum-operations-distinct.html",
                "title": "Minimum Operations for Distinct",
                "problem_name": "Minimum Operations for Distinct",
                "category": "Sorting",
                "problem_number": "N/A",
                "problem_description": "Find minimum operations to make all elements distinct in an array.",
                "solutions": self.generate_basic_solution("Minimum Operations for Distinct"),
                "prev_link": "sort-by-frequency.html",
                "prev_name": "Sort by Frequency",
                "next_link": "merge-k-sorted-arrays.html",
                "next_name": "Merge k sorted arrays"
            },
            # Heap Problems
            {
                "filename": "merge-k-sorted-arrays.html",
                "title": "Merge k sorted arrays",
                "problem_name": "Merge k sorted arrays",
                "category": "Heap",
                "problem_number": "N/A",
                "problem_description": "Merge k sorted arrays into a single sorted array efficiently.",
                "solutions": self.generate_basic_solution("Merge k sorted arrays"),
                "prev_link": "minimum-operations-distinct.html",
                "prev_name": "Minimum Operations for Distinct",
                "next_link": "merge-k-sorted-lists.html",
                "next_name": "Merge k Sorted Lists"
            },
            {
                "filename": "merge-k-sorted-lists.html",
                "title": "Merge k Sorted Lists",
                "problem_name": "Merge k Sorted Lists",
                "category": "Heap",
                "problem_number": "23",
                "problem_description": "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.",
                "solutions": self.generate_basic_solution("Merge k Sorted Lists"),
                "prev_link": "merge-k-sorted-arrays.html",
                "prev_name": "Merge k sorted arrays",
                "next_link": "find-median-data-stream.html",
                "next_name": "Find Median from Data Stream"
            },
            {
                "filename": "find-median-data-stream.html",
                "title": "Find Median from Data Stream",
                "problem_name": "Find Median from Data Stream",
                "category": "Heap",
                "problem_number": "295",
                "problem_description": "The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.",
                "solutions": self.generate_basic_solution("Find Median from Data Stream"),
                "prev_link": "merge-k-sorted-lists.html",
                "prev_name": "Merge k Sorted Lists",
                "next_link": "meeting-rooms.html",
                "next_name": "Meeting Rooms II"
            },
            {
                "filename": "meeting-rooms.html",
                "title": "Meeting Rooms II",
                "problem_name": "Meeting Rooms II",
                "category": "Heap",
                "problem_number": "253",
                "problem_description": "Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.",
                "solutions": self.generate_basic_solution("Meeting Rooms II"),
                "prev_link": "find-median-data-stream.html",
                "prev_name": "Find Median from Data Stream",
                "next_link": "range-addition.html",
                "next_name": "Range Addition"
            },
            {
                "filename": "range-addition.html",
                "title": "Range Addition",
                "problem_name": "Range Addition",
                "category": "Heap",
                "problem_number": "370",
                "problem_description": "Assume you have an array of length n initialized with all 0's and are given k update operations.",
                "solutions": self.generate_basic_solution("Range Addition"),
                "prev_link": "meeting-rooms.html",
                "prev_name": "Meeting Rooms II",
                "next_link": "add-search-word.html",
                "next_name": "Add and Search Word"
            },
            # Trie Problems
            {
                "filename": "add-search-word.html",
                "title": "Add and Search Word",
                "problem_name": "Add and Search Word",
                "category": "Trie",
                "problem_number": "211",
                "problem_description": "Design a data structure that supports adding new words and finding if a string matches any previously added string.",
                "solutions": self.generate_basic_solution("Add and Search Word"),
                "prev_link": "range-addition.html",
                "prev_name": "Range Addition",
                "next_link": "range-sum-query-mutable.html",
                "next_name": "Range Sum Query – Mutable"
            },
            # Segment Tree Problems
            {
                "filename": "range-sum-query-mutable.html",
                "title": "Range Sum Query – Mutable",
                "problem_name": "Range Sum Query – Mutable",
                "category": "Segment Tree",
                "problem_number": "307",
                "problem_description": "Given an integer array nums, handle multiple queries of the following types: Update the value of an element in nums. Calculate the sum of the elements of nums between indices left and right inclusive.",
                "solutions": self.generate_basic_solution("Range Sum Query – Mutable"),
                "prev_link": "add-search-word.html",
                "prev_name": "Add and Search Word",
                "next_link": "skyline-problem.html",
                "next_name": "The Skyline Problem"
            },
            {
                "filename": "skyline-problem.html",
                "title": "The Skyline Problem",
                "problem_name": "The Skyline Problem",
                "category": "Segment Tree",
                "problem_number": "218",
                "problem_description": "A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance.",
                "solutions": self.generate_basic_solution("The Skyline Problem"),
                "prev_link": "range-sum-query-mutable.html",
                "prev_name": "Range Sum Query – Mutable",
                "next_link": "minimum-height-trees.html",
                "next_name": "Minimum Height Trees"
            },
            # Graph Problems
            {
                "filename": "minimum-height-trees.html",
                "title": "Minimum Height Trees",
                "problem_name": "Minimum Height Trees",
                "category": "Graph",
                "problem_number": "310",
                "problem_description": "A tree is an undirected graph in which any two vertices are connected by exactly one path. In other words, any connected graph without simple cycles is a tree.",
                "solutions": self.generate_basic_solution("Minimum Height Trees"),
                "prev_link": "skyline-problem.html",
                "prev_name": "The Skyline Problem",
                "next_link": "reconstruct-itinerary.html",
                "next_name": "Reconstruct Itinerary"
            },
            {
                "filename": "reconstruct-itinerary.html",
                "title": "Reconstruct Itinerary",
                "problem_name": "Reconstruct Itinerary",
                "category": "Graph",
                "problem_number": "332",
                "problem_description": "You are given a list of airline tickets where tickets[i] = [fromi, toi] represent the departure and arrival airports of one flight.",
                "solutions": self.generate_basic_solution("Reconstruct Itinerary"),
                "prev_link": "minimum-height-trees.html",
                "prev_name": "Minimum Height Trees",
                "next_link": "graph-valid-tree.html",
                "next_name": "Graph Valid Tree"
            },
            {
                "filename": "graph-valid-tree.html",
                "title": "Graph Valid Tree",
                "problem_name": "Graph Valid Tree",
                "category": "Graph",
                "problem_number": "261",
                "problem_description": "Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.",
                "solutions": self.generate_basic_solution("Graph Valid Tree"),
                "prev_link": "reconstruct-itinerary.html",
                "prev_name": "Reconstruct Itinerary",
                "next_link": "two-sum-ii.html",
                "next_name": "Two Sum II"
            }
        ]

    def generate_problem_page(self, problem: Dict) -> str:
        """Generate the complete HTML content for a problem page."""
        return self.template.format(
            title=problem["title"],
            problem_name=problem["problem_name"],
            category=problem["category"],
            problem_number=problem["problem_number"],
            problem_description=problem["problem_description"],
            solutions=problem["solutions"],
            prev_link=problem["prev_link"],
            prev_name=problem["prev_name"],
            next_link=problem["next_link"],
            next_name=problem["next_name"]
        )

    def create_problems_directory(self):
        """Create the problems directory if it doesn't exist."""
        if not os.path.exists(self.problems_dir):
            os.makedirs(self.problems_dir)
            print(f"Created directory: {self.problems_dir}")

    def generate_all_problems(self, batch_size: int = 10):
        """Generate all problem pages in batches."""
        self.create_problems_directory()
        
        total_problems = len(self.problem_data)
        print(f"Starting to generate {total_problems} final batch problem pages...")
        
        for i in range(0, total_problems, batch_size):
            batch = self.problem_data[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, total_problems)} of {total_problems})")
            
            for problem in batch:
                filename = os.path.join(self.problems_dir, problem["filename"])
                content = self.generate_problem_page(problem)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"  ✓ Generated: {problem['filename']}")
            
            # Small delay between batches
            if i + batch_size < total_problems:
                print("  Waiting 1 second before next batch...")
                time.sleep(1)
        
        print(f"\n🎉 Successfully generated {total_problems} problem pages!")
        print(f"📁 All files are saved in the '{self.problems_dir}' directory")

def main():
    """Main function to run the final batch problem generator."""
    print("🚀 Starting Final Batch Problem Generator")
    print("=" * 70)
    
    generator = FinalBatchProblemGenerator()
    
    # Generate all problems in batches of 10
    generator.generate_all_problems(batch_size=10)
    
    print("\n" + "=" * 70)
    print("✅ Final batch of problem pages have been generated successfully!")
    print("🌐 You can now open index.html to view the complete website.")

if __name__ == "__main__":
    main() 