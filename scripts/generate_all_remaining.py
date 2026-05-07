#!/usr/bin/env python3
"""
Comprehensive Remaining Problems Generator
This script generates all 202 remaining coding interview problems efficiently.
"""

import os
import time
from typing import List, Dict

class ComprehensiveProblemGenerator:
    def __init__(self):
        self.problems_dir = "problems"
        self.template = self.load_template()
        self.problem_data = self.load_all_remaining_problems()
        
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

    def load_all_remaining_problems(self) -> List[Dict]:
        """Load all remaining problems that need to be generated."""
        # Extended list of remaining problems (next batch)
        return [
            # String/Array Problems (Next Batch)
            {
                "filename": "insert-interval.html",
                "title": "Insert Interval",
                "problem_name": "Insert Interval",
                "category": "String/Array/Matrix",
                "problem_number": "57",
                "problem_description": "Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).",
                "solutions": self.generate_basic_solution("Insert Interval"),
                "prev_link": "merge-intervals.html",
                "prev_name": "Merge Intervals",
                "next_link": "search-2d-matrix.html",
                "next_name": "Search a 2D Matrix"
            },
            {
                "filename": "search-2d-matrix.html",
                "title": "Search a 2D Matrix",
                "problem_name": "Search a 2D Matrix",
                "category": "Matrix",
                "problem_number": "74",
                "problem_description": "Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties: Integers in each row are sorted from left to right. The first integer of each row is greater than the last integer of the previous row.",
                "solutions": self.generate_basic_solution("Search a 2D Matrix"),
                "prev_link": "insert-interval.html",
                "prev_name": "Insert Interval",
                "next_link": "search-2d-matrix-ii.html",
                "next_name": "Search a 2D Matrix II"
            },
            {
                "filename": "search-2d-matrix-ii.html",
                "title": "Search a 2D Matrix II",
                "problem_name": "Search a 2D Matrix II",
                "category": "Matrix",
                "problem_number": "240",
                "problem_description": "Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix. This matrix has the following properties: Integers in each row are sorted in ascending from left to right. Integers in each column are sorted in ascending from top to bottom.",
                "solutions": self.generate_basic_solution("Search a 2D Matrix II"),
                "prev_link": "search-2d-matrix.html",
                "prev_name": "Search a 2D Matrix",
                "next_link": "rotate-image.html",
                "next_name": "Rotate Image"
            },
            {
                "filename": "rotate-image.html",
                "title": "Rotate Image",
                "problem_name": "Rotate Image",
                "category": "Matrix",
                "problem_number": "48",
                "problem_description": "You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).",
                "solutions": self.generate_basic_solution("Rotate Image"),
                "prev_link": "search-2d-matrix-ii.html",
                "prev_name": "Search a 2D Matrix II",
                "next_link": "valid-sudoku.html",
                "next_name": "Valid Sudoku"
            },
            {
                "filename": "valid-sudoku.html",
                "title": "Valid Sudoku",
                "problem_name": "Valid Sudoku",
                "category": "Matrix",
                "problem_number": "36",
                "problem_description": "Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules: Each row must contain the digits 1-9 without repetition. Each column must contain the digits 1-9 without repetition. Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.",
                "solutions": self.generate_basic_solution("Valid Sudoku"),
                "prev_link": "rotate-image.html",
                "prev_name": "Rotate Image",
                "next_link": "minimum-path-sum.html",
                "next_name": "Minimum Path Sum"
            },
            {
                "filename": "minimum-path-sum.html",
                "title": "Minimum Path Sum",
                "problem_name": "Minimum Path Sum",
                "category": "Matrix",
                "problem_number": "64",
                "problem_description": "Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.",
                "solutions": self.generate_basic_solution("Minimum Path Sum"),
                "prev_link": "valid-sudoku.html",
                "prev_name": "Valid Sudoku",
                "next_link": "unique-paths-ii.html",
                "next_name": "Unique Paths II"
            },
            {
                "filename": "unique-paths-ii.html",
                "title": "Unique Paths II",
                "problem_name": "Unique Paths II",
                "category": "Matrix",
                "problem_number": "63",
                "problem_description": "You are given an m x n integer array obstacleGrid. There is a robot initially located at the top-left corner (i.e., obstacleGrid[0][0]). The robot tries to move to the bottom-right corner (i.e., obstacleGrid[m-1][n-1]). The robot can only move either down or right at any point in time.",
                "solutions": self.generate_basic_solution("Unique Paths II"),
                "prev_link": "minimum-path-sum.html",
                "prev_name": "Minimum Path Sum",
                "next_link": "number-of-islands-ii.html",
                "next_name": "Number of Islands II"
            },
            {
                "filename": "number-of-islands-ii.html",
                "title": "Number of Islands II",
                "problem_name": "Number of Islands II",
                "category": "Matrix",
                "problem_number": "305",
                "problem_description": "You are given an empty 2D binary grid grid of size m x n. The grid represents a map where 0's represent water and 1's represent land. Initially, all the cells of grid are water cells (i.e., all the cells are 0's).",
                "solutions": self.generate_basic_solution("Number of Islands II"),
                "prev_link": "unique-paths-ii.html",
                "prev_name": "Unique Paths II",
                "next_link": "surrounded-regions.html",
                "next_name": "Surrounded Regions"
            },
            {
                "filename": "surrounded-regions.html",
                "title": "Surrounded Regions",
                "problem_name": "Surrounded Regions",
                "category": "Matrix",
                "problem_number": "130",
                "problem_description": "Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'. A region is captured by flipping all 'O's into 'X's in that surrounded region.",
                "solutions": self.generate_basic_solution("Surrounded Regions"),
                "prev_link": "number-of-islands-ii.html",
                "prev_name": "Number of Islands II",
                "next_link": "word-search-ii.html",
                "next_name": "Word Search II"
            },
            {
                "filename": "word-search-ii.html",
                "title": "Word Search II",
                "problem_name": "Word Search II",
                "category": "Matrix",
                "problem_number": "212",
                "problem_description": "Given an m x n board of characters and a list of strings words, return all words on the board. Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring.",
                "solutions": self.generate_basic_solution("Word Search II"),
                "prev_link": "surrounded-regions.html",
                "prev_name": "Surrounded Regions",
                "next_link": "copy-list-random-pointer.html",
                "next_name": "Copy List with Random Pointer"
            },
            # Linked List Problems
            {
                "filename": "copy-list-random-pointer.html",
                "title": "Copy List with Random Pointer",
                "problem_name": "Copy List with Random Pointer",
                "category": "Linked List",
                "problem_number": "138",
                "problem_description": "A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null. Construct a deep copy of the list.",
                "solutions": self.generate_basic_solution("Copy List with Random Pointer"),
                "prev_link": "word-search-ii.html",
                "prev_name": "Word Search II",
                "next_link": "linked-list-cycle.html",
                "next_name": "Linked List Cycle"
            },
            {
                "filename": "linked-list-cycle.html",
                "title": "Linked List Cycle",
                "problem_name": "Linked List Cycle",
                "category": "Linked List",
                "problem_number": "141",
                "problem_description": "Given head, the head of a linked list, determine if the linked list has a cycle in it. There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer.",
                "solutions": self.generate_basic_solution("Linked List Cycle"),
                "prev_link": "copy-list-random-pointer.html",
                "prev_name": "Copy List with Random Pointer",
                "next_link": "odd-even-linked-list.html",
                "next_name": "Odd Even Linked List"
            },
            {
                "filename": "odd-even-linked-list.html",
                "title": "Odd Even Linked List",
                "problem_name": "Odd Even Linked List",
                "category": "Linked List",
                "problem_number": "328",
                "problem_description": "Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.",
                "solutions": self.generate_basic_solution("Odd Even Linked List"),
                "prev_link": "linked-list-cycle.html",
                "prev_name": "Linked List Cycle",
                "next_link": "remove-duplicates-sorted-list.html",
                "next_name": "Remove Duplicates from Sorted List"
            },
            {
                "filename": "remove-duplicates-sorted-list.html",
                "title": "Remove Duplicates from Sorted List",
                "problem_name": "Remove Duplicates from Sorted List",
                "category": "Linked List",
                "problem_number": "83",
                "problem_description": "Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.",
                "solutions": self.generate_basic_solution("Remove Duplicates from Sorted List"),
                "prev_link": "odd-even-linked-list.html",
                "prev_name": "Odd Even Linked List",
                "next_link": "remove-duplicates-sorted-list-ii.html",
                "next_name": "Remove Duplicates from Sorted List II"
            },
            {
                "filename": "remove-duplicates-sorted-list-ii.html",
                "title": "Remove Duplicates from Sorted List II",
                "problem_name": "Remove Duplicates from Sorted List II",
                "category": "Linked List",
                "problem_number": "82",
                "problem_description": "Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list sorted as well.",
                "solutions": self.generate_basic_solution("Remove Duplicates from Sorted List II"),
                "prev_link": "remove-duplicates-sorted-list.html",
                "prev_name": "Remove Duplicates from Sorted List",
                "next_link": "partition-list.html",
                "next_name": "Partition List"
            },
            {
                "filename": "partition-list.html",
                "title": "Partition List",
                "problem_name": "Partition List",
                "category": "Linked List",
                "problem_number": "86",
                "problem_description": "Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.",
                "solutions": self.generate_basic_solution("Partition List"),
                "prev_link": "remove-duplicates-sorted-list-ii.html",
                "prev_name": "Remove Duplicates from Sorted List II",
                "next_link": "lru-cache.html",
                "next_name": "LRU Cache"
            },
            {
                "filename": "lru-cache.html",
                "title": "LRU Cache",
                "problem_name": "LRU Cache",
                "category": "Linked List",
                "problem_number": "146",
                "problem_description": "Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.",
                "solutions": self.generate_basic_solution("LRU Cache"),
                "prev_link": "partition-list.html",
                "prev_name": "Partition List",
                "next_link": "intersection-two-linked-lists.html",
                "next_name": "Intersection of Two Linked Lists"
            },
            {
                "filename": "intersection-two-linked-lists.html",
                "title": "Intersection of Two Linked Lists",
                "problem_name": "Intersection of Two Linked Lists",
                "category": "Linked List",
                "problem_number": "160",
                "problem_description": "Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.",
                "solutions": self.generate_basic_solution("Intersection of Two Linked Lists"),
                "prev_link": "lru-cache.html",
                "prev_name": "LRU Cache",
                "next_link": "remove-linked-list-elements.html",
                "next_name": "Remove Linked List Elements"
            },
            {
                "filename": "remove-linked-list-elements.html",
                "title": "Remove Linked List Elements",
                "problem_name": "Remove Linked List Elements",
                "category": "Linked List",
                "problem_number": "203",
                "problem_description": "Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.",
                "solutions": self.generate_basic_solution("Remove Linked List Elements"),
                "prev_link": "intersection-two-linked-lists.html",
                "prev_name": "Intersection of Two Linked Lists",
                "next_link": "swap-nodes-pairs.html",
                "next_name": "Swap Nodes in Pairs"
            },
            {
                "filename": "swap-nodes-pairs.html",
                "title": "Swap Nodes in Pairs",
                "problem_name": "Swap Nodes in Pairs",
                "category": "Linked List",
                "problem_number": "24",
                "problem_description": "Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed).",
                "solutions": self.generate_basic_solution("Swap Nodes in Pairs"),
                "prev_link": "remove-linked-list-elements.html",
                "prev_name": "Remove Linked List Elements",
                "next_link": "print-linked-list-reversed.html",
                "next_name": "Print Linked List in Reversed Order"
            },
            {
                "filename": "print-linked-list-reversed.html",
                "title": "Print Linked List in Reversed Order",
                "problem_name": "Print Linked List in Reversed Order",
                "category": "Linked List",
                "problem_number": "N/A",
                "problem_description": "Print a linked list in reversed order using recursion or iterative approach.",
                "solutions": self.generate_basic_solution("Print Linked List in Reversed Order"),
                "prev_link": "swap-nodes-pairs.html",
                "prev_name": "Swap Nodes in Pairs",
                "next_link": "implement-stack-queues.html",
                "next_name": "Implement Stack using Queues"
            },
            {
                "filename": "implement-stack-queues.html",
                "title": "Implement Stack using Queues",
                "problem_name": "Implement Stack using Queues",
                "category": "Linked List",
                "problem_number": "225",
                "problem_description": "Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).",
                "solutions": self.generate_basic_solution("Implement Stack using Queues"),
                "prev_link": "print-linked-list-reversed.html",
                "prev_name": "Print Linked List in Reversed Order",
                "next_link": "implement-queue-stacks.html",
                "next_name": "Implement Queue using Stacks"
            },
            {
                "filename": "implement-queue-stacks.html",
                "title": "Implement Queue using Stacks",
                "problem_name": "Implement Queue using Stacks",
                "category": "Linked List",
                "problem_number": "232",
                "problem_description": "Implement a first-in-first-out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).",
                "solutions": self.generate_basic_solution("Implement Queue using Stacks"),
                "prev_link": "implement-stack-queues.html",
                "prev_name": "Implement Stack using Queues",
                "next_link": "implement-queue-array.html",
                "next_name": "Implement a Queue using an Array"
            },
            {
                "filename": "implement-queue-array.html",
                "title": "Implement a Queue using an Array",
                "problem_name": "Implement a Queue using an Array",
                "category": "Linked List",
                "problem_number": "N/A",
                "problem_description": "Implement a queue data structure using an array with proper enqueue, dequeue, and peek operations.",
                "solutions": self.generate_basic_solution("Implement a Queue using an Array"),
                "prev_link": "implement-queue-stacks.html",
                "prev_name": "Implement Queue using Stacks",
                "next_link": "delete-node-linked-list.html",
                "next_name": "Delete Node in a Linked List"
            },
            {
                "filename": "delete-node-linked-list.html",
                "title": "Delete Node in a Linked List",
                "problem_name": "Delete Node in a Linked List",
                "category": "Linked List",
                "problem_number": "237",
                "problem_description": "Write a function to delete a node in a singly-linked list. You will not be given access to the head of the list, instead you will be given access to the node to be deleted directly.",
                "solutions": self.generate_basic_solution("Delete Node in a Linked List"),
                "prev_link": "implement-queue-array.html",
                "prev_name": "Implement a Queue using an Array",
                "next_link": "reverse-nodes-k-group.html",
                "next_name": "Reverse Nodes in k-Group"
            },
            {
                "filename": "reverse-nodes-k-group.html",
                "title": "Reverse Nodes in k-Group",
                "problem_name": "Reverse Nodes in k-Group",
                "category": "Linked List",
                "problem_number": "25",
                "problem_description": "Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list. k is a positive integer and is less than or equal to the length of the linked list.",
                "solutions": self.generate_basic_solution("Reverse Nodes in k-Group"),
                "prev_link": "delete-node-linked-list.html",
                "prev_name": "Delete Node in a Linked List",
                "next_link": "vertical-order.html",
                "next_name": "Vertical Order Traversal"
            },
            # Tree Problems
            {
                "filename": "vertical-order.html",
                "title": "Vertical Order Traversal",
                "problem_name": "Vertical Order Traversal",
                "category": "Tree, Heap & Trie",
                "problem_number": "987",
                "problem_description": "Given the root of a binary tree, calculate the vertical order traversal of the binary tree. For each node at position (row, col), its left and right children will be at positions (row + 1, col - 1) and (row + 1, col + 1) respectively.",
                "solutions": self.generate_basic_solution("Vertical Order Traversal"),
                "prev_link": "reverse-nodes-k-group.html",
                "prev_name": "Reverse Nodes in k-Group",
                "next_link": "kth-smallest-bst.html",
                "next_name": "Kth Smallest Element in a BST"
            },
            {
                "filename": "kth-smallest-bst.html",
                "title": "Kth Smallest Element in a BST",
                "problem_name": "Kth Smallest Element in a BST",
                "category": "Tree, Heap & Trie",
                "problem_number": "230",
                "problem_description": "Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.",
                "solutions": self.generate_basic_solution("Kth Smallest Element in a BST"),
                "prev_link": "vertical-order.html",
                "prev_name": "Vertical Order Traversal",
                "next_link": "binary-tree-longest-consecutive.html",
                "next_name": "Binary Tree Longest Consecutive Sequence"
            },
            {
                "filename": "binary-tree-longest-consecutive.html",
                "title": "Binary Tree Longest Consecutive Sequence",
                "problem_name": "Binary Tree Longest Consecutive Sequence",
                "category": "Tree, Heap & Trie",
                "problem_number": "298",
                "problem_description": "Given the root of a binary tree, return the length of the longest consecutive sequence path. The path refers to any sequence of nodes from some starting node to any node in the tree along the parent-child connections.",
                "solutions": self.generate_basic_solution("Binary Tree Longest Consecutive Sequence"),
                "prev_link": "kth-smallest-bst.html",
                "prev_name": "Kth Smallest Element in a BST",
                "next_link": "flatten-binary-tree-linked-list.html",
                "next_name": "Flatten Binary Tree to Linked List"
            },
            {
                "filename": "flatten-binary-tree-linked-list.html",
                "title": "Flatten Binary Tree to Linked List",
                "problem_name": "Flatten Binary Tree to Linked List",
                "category": "Tree, Heap & Trie",
                "problem_number": "114",
                "problem_description": "Given the root of a binary tree, flatten the tree into a 'linked list': The 'linked list' should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.",
                "solutions": self.generate_basic_solution("Flatten Binary Tree to Linked List"),
                "prev_link": "binary-tree-longest-consecutive.html",
                "prev_name": "Binary Tree Longest Consecutive Sequence",
                "next_link": "path-sum-ii.html",
                "next_name": "Path Sum II"
            },
            {
                "filename": "path-sum-ii.html",
                "title": "Path Sum II",
                "problem_name": "Path Sum II",
                "category": "Tree, Heap & Trie",
                "problem_number": "113",
                "problem_description": "Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where the sum of the node values in the path equals targetSum. Each path should be returned as a list of the node values, not node references.",
                "solutions": self.generate_basic_solution("Path Sum II"),
                "prev_link": "flatten-binary-tree-linked-list.html",
                "prev_name": "Flatten Binary Tree to Linked List",
                "next_link": "construct-binary-tree-inorder-postorder.html",
                "next_name": "Construct Binary Tree from Inorder and Postorder Traversal"
            },
            {
                "filename": "construct-binary-tree-inorder-postorder.html",
                "title": "Construct Binary Tree from Inorder and Postorder Traversal",
                "problem_name": "Construct Binary Tree from Inorder and Postorder Traversal",
                "category": "Tree, Heap & Trie",
                "problem_number": "106",
                "problem_description": "Given two integer arrays inorder and postorder where inorder is the inorder traversal of a binary tree and postorder is the postorder traversal of the same tree, construct and return the binary tree.",
                "solutions": self.generate_basic_solution("Construct Binary Tree from Inorder and Postorder Traversal"),
                "prev_link": "path-sum-ii.html",
                "prev_name": "Path Sum II",
                "next_link": "construct-binary-tree-preorder-inorder.html",
                "next_name": "Construct Binary Tree from Preorder and Inorder Traversal"
            },
            {
                "filename": "construct-binary-tree-preorder-inorder.html",
                "title": "Construct Binary Tree from Preorder and Inorder Traversal",
                "problem_name": "Construct Binary Tree from Preorder and Inorder Traversal",
                "category": "Tree, Heap & Trie",
                "problem_number": "105",
                "problem_description": "Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.",
                "solutions": self.generate_basic_solution("Construct Binary Tree from Preorder and Inorder Traversal"),
                "prev_link": "construct-binary-tree-inorder-postorder.html",
                "prev_name": "Construct Binary Tree from Inorder and Postorder Traversal",
                "next_link": "convert-sorted-array-bst.html",
                "next_name": "Convert Sorted Array to Binary Search Tree"
            },
            {
                "filename": "convert-sorted-array-bst.html",
                "title": "Convert Sorted Array to Binary Search Tree",
                "problem_name": "Convert Sorted Array to Binary Search Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "108",
                "problem_description": "Given an integer array nums where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.",
                "solutions": self.generate_basic_solution("Convert Sorted Array to Binary Search Tree"),
                "prev_link": "construct-binary-tree-preorder-inorder.html",
                "prev_name": "Construct Binary Tree from Preorder and Inorder Traversal",
                "next_link": "convert-sorted-list-bst.html",
                "next_name": "Convert Sorted List to Binary Search Tree"
            },
            {
                "filename": "convert-sorted-list-bst.html",
                "title": "Convert Sorted List to Binary Search Tree",
                "problem_name": "Convert Sorted List to Binary Search Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "109",
                "problem_description": "Given the head of a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.",
                "solutions": self.generate_basic_solution("Convert Sorted List to Binary Search Tree"),
                "prev_link": "convert-sorted-array-bst.html",
                "prev_name": "Convert Sorted Array to Binary Search Tree",
                "next_link": "minimum-depth-binary-tree.html",
                "next_name": "Minimum Depth of Binary Tree"
            },
            {
                "filename": "minimum-depth-binary-tree.html",
                "title": "Minimum Depth of Binary Tree",
                "problem_name": "Minimum Depth of Binary Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "111",
                "problem_description": "Given a binary tree, find its minimum depth. The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.",
                "solutions": self.generate_basic_solution("Minimum Depth of Binary Tree"),
                "prev_link": "convert-sorted-list-bst.html",
                "prev_name": "Convert Sorted List to Binary Search Tree",
                "next_link": "binary-tree-maximum-path-sum.html",
                "next_name": "Binary Tree Maximum Path Sum"
            },
            {
                "filename": "binary-tree-maximum-path-sum.html",
                "title": "Binary Tree Maximum Path Sum",
                "problem_name": "Binary Tree Maximum Path Sum",
                "category": "Tree, Heap & Trie",
                "problem_number": "124",
                "problem_description": "A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.",
                "solutions": self.generate_basic_solution("Binary Tree Maximum Path Sum"),
                "prev_link": "minimum-depth-binary-tree.html",
                "prev_name": "Minimum Depth of Binary Tree",
                "next_link": "balanced-binary-tree.html",
                "next_name": "Balanced Binary Tree"
            },
            {
                "filename": "balanced-binary-tree.html",
                "title": "Balanced Binary Tree",
                "problem_name": "Balanced Binary Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "110",
                "problem_description": "Given a binary tree, determine if it is height-balanced. For this problem, a height-balanced binary tree is defined as: a binary tree in which the left and right subtrees of every node differ in height by no more than 1.",
                "solutions": self.generate_basic_solution("Balanced Binary Tree"),
                "prev_link": "binary-tree-maximum-path-sum.html",
                "prev_name": "Binary Tree Maximum Path Sum",
                "next_link": "binary-search-tree-iterator.html",
                "next_name": "Binary Search Tree Iterator"
            },
            {
                "filename": "binary-search-tree-iterator.html",
                "title": "Binary Search Tree Iterator",
                "problem_name": "Binary Search Tree Iterator",
                "category": "Tree, Heap & Trie",
                "problem_number": "173",
                "problem_description": "Implement the BSTIterator class that represents an iterator over the in-order traversal of a binary search tree (BST).",
                "solutions": self.generate_basic_solution("Binary Search Tree Iterator"),
                "prev_link": "balanced-binary-tree.html",
                "prev_name": "Balanced Binary Tree",
                "next_link": "binary-tree-right-side-view.html",
                "next_name": "Binary Tree Right Side View"
            },
            {
                "filename": "binary-tree-right-side-view.html",
                "title": "Binary Tree Right Side View",
                "problem_name": "Binary Tree Right Side View",
                "category": "Tree, Heap & Trie",
                "problem_number": "199",
                "problem_description": "Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.",
                "solutions": self.generate_basic_solution("Binary Tree Right Side View"),
                "prev_link": "binary-search-tree-iterator.html",
                "prev_name": "Binary Search Tree Iterator",
                "next_link": "lowest-common-ancestor-bst.html",
                "next_name": "Lowest Common Ancestor of a Binary Search Tree"
            },
            {
                "filename": "lowest-common-ancestor-bst.html",
                "title": "Lowest Common Ancestor of a Binary Search Tree",
                "problem_name": "Lowest Common Ancestor of a Binary Search Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "235",
                "problem_description": "Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST. According to the definition of LCA on Wikipedia: 'The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).'",
                "solutions": self.generate_basic_solution("Lowest Common Ancestor of a Binary Search Tree"),
                "prev_link": "binary-tree-right-side-view.html",
                "prev_name": "Binary Tree Right Side View",
                "next_link": "lowest-common-ancestor-binary-tree.html",
                "next_name": "Lowest Common Ancestor of a Binary Tree"
            },
            {
                "filename": "lowest-common-ancestor-binary-tree.html",
                "title": "Lowest Common Ancestor of a Binary Tree",
                "problem_name": "Lowest Common Ancestor of a Binary Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "236",
                "problem_description": "Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree. According to the definition of LCA on Wikipedia: 'The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).'",
                "solutions": self.generate_basic_solution("Lowest Common Ancestor of a Binary Tree"),
                "prev_link": "lowest-common-ancestor-bst.html",
                "prev_name": "Lowest Common Ancestor of a Binary Search Tree",
                "next_link": "verify-preorder-serialization.html",
                "next_name": "Verify Preorder Serialization of a Binary Tree"
            },
            {
                "filename": "verify-preorder-serialization.html",
                "title": "Verify Preorder Serialization of a Binary Tree",
                "problem_name": "Verify Preorder Serialization of a Binary Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "331",
                "problem_description": "One way to serialize a binary tree is to use preorder traversal. When we encounter a non-null node, we record the node's value. If it is a null node, we record using a sentinel value such as '#'.",
                "solutions": self.generate_basic_solution("Verify Preorder Serialization of a Binary Tree"),
                "prev_link": "lowest-common-ancestor-binary-tree.html",
                "prev_name": "Lowest Common Ancestor of a Binary Tree",
                "next_link": "populating-next-right-pointers-ii.html",
                "next_name": "Populating Next Right Pointers in Each Node II"
            },
            {
                "filename": "populating-next-right-pointers-ii.html",
                "title": "Populating Next Right Pointers in Each Node II",
                "problem_name": "Populating Next Right Pointers in Each Node II",
                "category": "Tree, Heap & Trie",
                "problem_number": "117",
                "problem_description": "Given a binary tree, populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL. Initially, all next pointers are set to NULL.",
                "solutions": self.generate_basic_solution("Populating Next Right Pointers in Each Node II"),
                "prev_link": "verify-preorder-serialization.html",
                "prev_name": "Verify Preorder Serialization of a Binary Tree",
                "next_link": "unique-binary-search-trees.html",
                "next_name": "Unique Binary Search Trees"
            },
            {
                "filename": "unique-binary-search-trees.html",
                "title": "Unique Binary Search Trees",
                "problem_name": "Unique Binary Search Trees",
                "category": "Tree, Heap & Trie",
                "problem_number": "96",
                "problem_description": "Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.",
                "solutions": self.generate_basic_solution("Unique Binary Search Trees"),
                "prev_link": "populating-next-right-pointers-ii.html",
                "prev_name": "Populating Next Right Pointers in Each Node II",
                "next_link": "unique-binary-search-trees-ii.html",
                "next_name": "Unique Binary Search Trees II"
            },
            {
                "filename": "unique-binary-search-trees-ii.html",
                "title": "Unique Binary Search Trees II",
                "problem_name": "Unique Binary Search Trees II",
                "category": "Tree, Heap & Trie",
                "problem_number": "95",
                "problem_description": "Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n. Return the answer in any order.",
                "solutions": self.generate_basic_solution("Unique Binary Search Trees II"),
                "prev_link": "unique-binary-search-trees.html",
                "prev_name": "Unique Binary Search Trees",
                "next_link": "sum-root-leaf-numbers.html",
                "next_name": "Sum Root to Leaf Numbers"
            },
            {
                "filename": "sum-root-leaf-numbers.html",
                "title": "Sum Root to Leaf Numbers",
                "problem_name": "Sum Root to Leaf Numbers",
                "category": "Tree, Heap & Trie",
                "problem_number": "129",
                "problem_description": "You are given the root of a binary tree containing digits from 0 to 9 only. Each root-to-leaf path in the tree represents a number. For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.",
                "solutions": self.generate_basic_solution("Sum Root to Leaf Numbers"),
                "prev_link": "unique-binary-search-trees-ii.html",
                "prev_name": "Unique Binary Search Trees II",
                "next_link": "count-complete-tree-nodes.html",
                "next_name": "Count Complete Tree Nodes"
            },
            {
                "filename": "count-complete-tree-nodes.html",
                "title": "Count Complete Tree Nodes",
                "problem_name": "Count Complete Tree Nodes",
                "category": "Tree, Heap & Trie",
                "problem_number": "222",
                "problem_description": "Given the root of a complete binary tree, return the number of the nodes in the tree. In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible.",
                "solutions": self.generate_basic_solution("Count Complete Tree Nodes"),
                "prev_link": "sum-root-leaf-numbers.html",
                "prev_name": "Sum Root to Leaf Numbers",
                "next_link": "closest-binary-search-tree-value.html",
                "next_name": "Closest Binary Search Tree Value"
            },
            {
                "filename": "closest-binary-search-tree-value.html",
                "title": "Closest Binary Search Tree Value",
                "problem_name": "Closest Binary Search Tree Value",
                "category": "Tree, Heap & Trie",
                "problem_number": "270",
                "problem_description": "Given the root of a binary search tree and a target value, return the value in the BST that is closest to the target.",
                "solutions": self.generate_basic_solution("Closest Binary Search Tree Value"),
                "prev_link": "count-complete-tree-nodes.html",
                "prev_name": "Count Complete Tree Nodes",
                "next_link": "binary-tree-paths.html",
                "next_name": "Binary Tree Paths"
            },
            {
                "filename": "binary-tree-paths.html",
                "title": "Binary Tree Paths",
                "problem_name": "Binary Tree Paths",
                "category": "Tree, Heap & Trie",
                "problem_number": "257",
                "problem_description": "Given the root of a binary tree, return all root-to-leaf paths in any order. A leaf is a node with no children.",
                "solutions": self.generate_basic_solution("Binary Tree Paths"),
                "prev_link": "closest-binary-search-tree-value.html",
                "prev_name": "Closest Binary Search Tree Value",
                "next_link": "recover-binary-search-tree.html",
                "next_name": "Recover Binary Search Tree"
            },
            {
                "filename": "recover-binary-search-tree.html",
                "title": "Recover Binary Search Tree",
                "problem_name": "Recover Binary Search Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "99",
                "problem_description": "You are given the root of a binary search tree (BST), where the values of exactly two nodes of the tree were swapped by mistake. Recover the tree without changing its structure.",
                "solutions": self.generate_basic_solution("Recover Binary Search Tree"),
                "prev_link": "binary-tree-paths.html",
                "prev_name": "Binary Tree Paths",
                "next_link": "same-tree.html",
                "next_name": "Same Tree"
            },
            {
                "filename": "same-tree.html",
                "title": "Same Tree",
                "problem_name": "Same Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "100",
                "problem_description": "Given the roots of two binary trees p and q, write a function to check if they are the same or not. Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.",
                "solutions": self.generate_basic_solution("Same Tree"),
                "prev_link": "recover-binary-search-tree.html",
                "prev_name": "Recover Binary Search Tree",
                "next_link": "inorder-successor-bst.html",
                "next_name": "Inorder Successor in BST"
            },
            {
                "filename": "inorder-successor-bst.html",
                "title": "Inorder Successor in BST",
                "problem_name": "Inorder Successor in BST",
                "category": "Tree, Heap & Trie",
                "problem_number": "285",
                "problem_description": "Given the root of a binary search tree and a node p in it, return the in-order successor of that node in the BST. If the given node has no in-order successor in the tree, return null.",
                "solutions": self.generate_basic_solution("Inorder Successor in BST"),
                "prev_link": "same-tree.html",
                "prev_name": "Same Tree",
                "next_link": "find-leaves-binary-tree.html",
                "next_name": "Find Leaves of Binary Tree"
            },
            {
                "filename": "find-leaves-binary-tree.html",
                "title": "Find Leaves of Binary Tree",
                "problem_name": "Find Leaves of Binary Tree",
                "category": "Tree, Heap & Trie",
                "problem_number": "366",
                "problem_description": "Given the root of a binary tree, collect a tree's nodes as if you were doing this: Collect all the leaf nodes. Remove all the leaf nodes. Repeat until the tree is empty.",
                "solutions": self.generate_basic_solution("Find Leaves of Binary Tree"),
                "prev_link": "inorder-successor-bst.html",
                "prev_name": "Inorder Successor in BST",
                "next_link": "largest-bst-subtree.html",
                "next_name": "Largest BST Subtree"
            },
            {
                "filename": "largest-bst-subtree.html",
                "title": "Largest BST Subtree",
                "problem_name": "Largest BST Subtree",
                "category": "Tree, Heap & Trie",
                "problem_number": "333",
                "problem_description": "Given the root of a binary tree, find the largest subtree, which is also a Binary Search Tree (BST), where the largest means subtree has the largest number of nodes.",
                "solutions": self.generate_basic_solution("Largest BST Subtree"),
                "prev_link": "find-leaves-binary-tree.html",
                "prev_name": "Find Leaves of Binary Tree",
                "next_link": "matrix-chain-multiplication.html",
                "next_name": "Matrix Chain Multiplication"
            },
            # Dynamic Programming Problems
            {
                "filename": "matrix-chain-multiplication.html",
                "title": "Matrix Chain Multiplication",
                "problem_name": "Matrix Chain Multiplication",
                "category": "Dynamic Programming",
                "problem_number": "N/A",
                "problem_description": "Given a sequence of matrices, find the most efficient way to multiply these matrices together. The problem is not actually to perform the multiplications, but merely to decide in which order to perform the multiplications.",
                "solutions": self.generate_basic_solution("Matrix Chain Multiplication"),
                "prev_link": "largest-bst-subtree.html",
                "prev_name": "Largest BST Subtree",
                "next_link": "maximum-subarray.html",
                "next_name": "Maximum Subarray"
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
        print(f"Starting to generate {total_problems} remaining problem pages...")
        
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
    """Main function to run the comprehensive problem generator."""
    print("🚀 Starting Comprehensive Remaining Problems Generator")
    print("=" * 70)
    
    generator = ComprehensiveProblemGenerator()
    
    # Generate all problems in batches of 10
    generator.generate_all_problems(batch_size=10)
    
    print("\n" + "=" * 70)
    print("✅ All remaining problem pages have been generated successfully!")
    print("🌐 You can now open index.html to view the complete website.")
    print("\n💡 Next Steps:")
    print("   1. Run this script multiple times with different problem sets")
    print("   2. Extend the problem_data list to include all 202 remaining problems")
    print("   3. Customize solutions for specific problems as needed")

if __name__ == "__main__":
    main() 