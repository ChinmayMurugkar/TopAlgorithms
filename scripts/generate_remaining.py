#!/usr/bin/env python3
import os

def generate_problem(filename, title, description):
    template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Coding Interview Problems</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa; }}
        .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px 0; margin-bottom: 30px; }}
        .header-content {{ max-width: 1000px; margin: 0 auto; padding: 0 20px; }}
        .breadcrumb {{ margin-bottom: 10px; font-size: 0.9rem; opacity: 0.8; }}
        .breadcrumb a {{ color: white; text-decoration: none; }}
        .breadcrumb a:hover {{ text-decoration: underline; }}
        h1 {{ font-size: 2.2rem; margin-bottom: 10px; }}
        .meta {{ font-size: 0.9rem; opacity: 0.8; }}
        .content {{ background: white; border-radius: 10px; padding: 30px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 30px; }}
        .problem-description {{ background: #e3f2fd; border-left: 4px solid #2196f3; padding: 20px; margin-bottom: 30px; border-radius: 0 6px 6px 0; }}
        .solution {{ margin-bottom: 40px; }}
        .solution h3 {{ color: #2c3e50; font-size: 1.4rem; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #3498db; }}
        .code-block {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 20px; margin: 15px 0; overflow-x: auto; font-family: 'Courier New', monospace; font-size: 0.9rem; line-height: 1.4; white-space: pre; }}
        .complexity {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 6px; padding: 15px; margin: 15px 0; }}
        .complexity strong {{ color: #856404; }}
        .navigation {{ display: flex; justify-content: space-between; margin-top: 30px; }}
        .nav-btn {{ padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 6px; transition: background 0.2s ease; }}
        .nav-btn:hover {{ background: #2980b9; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="breadcrumb">
                <a href="../index.html">Home</a> > String/Array/Matrix > {title}
            </div>
            <h1>{title}</h1>
            <div class="meta">LeetCode Problem N/A</div>
        </div>
    </div>

    <div class="container">
        <div class="content">
            <div class="problem-description">
                <strong>Problem:</strong> {description}
            </div>

            <div class="solution">
                <h3>Solution 1 – Basic Approach</h3>
                <p>Implementation for {title}.</p>
                
                <div class="code-block">
public class Solution {{
    public void solve() {{
        // TODO: Implement solution for {title}
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

            <div class="navigation">
                <a href="two-sum.html" class="nav-btn">← Previous: Two Sum</a>
                <a href="two-sum-iii.html" class="nav-btn">Next: Two Sum III →</a>
            </div>
        </div>
    </div>
</body>
</html>'''
    
    content = template.format(title=title, description=description)
    
    with open(f"problems/{filename}", 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Generated: {filename}")

def main():
    problems = [
        ("two-sum-ii.html", "Two Sum II", "Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number."),
        ("two-sum-iii.html", "Two Sum III", "Design and implement a TwoSum class. It should support the following operations: add and find."),
        ("3sum.html", "3Sum", "Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0."),
        ("4sum.html", "4Sum", "Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that: 0 <= a, b, c, d < n, a, b, c, and d are distinct, nums[a] + nums[b] + nums[c] + nums[d] == target"),
        ("3sum-closest.html", "3Sum Closest", "Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target."),
        ("string-to-integer.html", "String to Integer", "Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer."),
        ("merge-sorted-array.html", "Merge Sorted Array", "You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively."),
        ("longest-valid-parentheses.html", "Longest Valid Parentheses", "Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring."),
        ("implement-strstr.html", "Implement strStr()", "Implement strStr(). Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack."),
        ("minimum-size-subarray-sum.html", "Minimum Size Subarray Sum", "Given an array of positive integers nums and a positive integer target, return the minimal length of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr] of which the sum is greater than or equal to target."),
        ("search-insert-position.html", "Search Insert Position", "Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order."),
        ("longest-consecutive-sequence.html", "Longest Consecutive Sequence", "Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence."),
        ("valid-palindrome.html", "Valid Palindrome", "A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward."),
        ("zigzag-conversion.html", "ZigZag Conversion", "The string 'PAYPALISHIRING' is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)"),
        ("add-binary.html", "Add Binary", "Given two binary strings a and b, return their sum as a binary string."),
        ("length-last-word.html", "Length of Last Word", "Given a string s consisting of words and spaces, return the length of the last word in the string."),
        ("triangle.html", "Triangle", "Given a triangle array, return the minimum path sum from top to bottom."),
        ("contains-duplicate.html", "Contains Duplicate", "Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct."),
        ("remove-duplicates-sorted-array.html", "Remove Duplicates from Sorted Array", "Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once."),
        ("remove-element.html", "Remove Element", "Given an integer array nums and an integer val, remove all occurrences of val in nums in-place."),
        ("move-zeroes.html", "Move Zeroes", "Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements."),
        ("longest-substring-2-unique.html", "Longest Substring that contains 2 unique characters", "Given a string s, find the length of the longest substring that contains at most two distinct characters."),
        ("substring-concatenation-words.html", "Substring with Concatenation of All Words", "You are given a string s and an array of strings words. All the strings of words are of the same length."),
        ("minimum-window-substring.html", "Minimum Window Substring", "Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window."),
        ("find-minimum-rotated-sorted-array.html", "Find Minimum in Rotated Sorted Array", "Suppose an array of length n sorted in ascending order is rotated between 1 and n times."),
        ("search-rotated-array.html", "Search in Rotated Array", "There is an integer array nums sorted in ascending order (with distinct values)."),
        ("min-stack.html", "Min Stack", "Design a stack that supports push, pop, top, and retrieving the minimum element in constant time."),
        ("majority-element.html", "Majority Element", "Given an array nums of size n, return the majority element."),
        ("bulls-and-cows.html", "Bulls and Cows", "You are playing the Bulls and Cows game with your friend."),
        ("largest-rectangle-histogram.html", "Largest Rectangle in Histogram", "Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram."),
        ("longest-common-prefix.html", "Longest Common Prefix", "Write a function to find the longest common prefix string amongst an array of strings."),
        ("largest-number.html", "Largest Number", "Given a list of non-negative integers nums, arrange them such that they form the largest number and return it."),
        ("simplify-path.html", "Simplify Path", "Given a string path, which is an absolute path (starting with a slash '/') to a file or directory in a Unix-style file system, convert it to the simplified canonical path."),
        ("compare-version-numbers.html", "Compare Version Numbers", "Given two version numbers, version1 and version2, compare them."),
        ("gas-station.html", "Gas Station", "There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i]."),
        ("pascals-triangle.html", "Pascal's Triangle", "Given an integer numRows, return the first numRows of Pascal's triangle."),
        ("candy.html", "Candy", "There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings."),
        ("trapping-rain-water.html", "Trapping Rain Water", "Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining."),
        ("count-and-say.html", "Count and Say", "The count-and-say sequence is a sequence of digit strings defined by the recursive formula:"),
        ("search-range.html", "Search for a Range", "Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value."),
        ("basic-calculator.html", "Basic Calculator", "Given a string s representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation."),
        ("group-anagrams.html", "Group Anagrams", "Given an array of strings strs, group the anagrams together. You can return the answer in any order."),
        ("shortest-palindrome.html", "Shortest Palindrome", "You are given a string s. You can convert s to a palindrome by adding characters in front of it."),
        ("rectangle-area.html", "Rectangle Area", "Given the coordinates of two rectilinear rectangles in a 2D plane, return the total area covered by the two rectangles."),
        ("summary-ranges.html", "Summary Ranges", "You are given a sorted unique integer array nums."),
        ("increasing-triplet-subsequence.html", "Increasing Triplet Subsequence", "Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k]."),
        ("get-target-arithmetic.html", "Get Target Using Number List And Arithmetic Operations", "Given a list of numbers and a target, find if you can reach the target using arithmetic operations."),
        ("reverse-vowels-string.html", "Reverse Vowels of a String", "Given a string s, reverse only all the vowels in the string and return it."),
        ("flip-game.html", "Flip Game", "You are playing the following Flip Game with your friend: Given a string that contains only these two characters: + and -, you and your friend take turns to flip two consecutive '++' into '--'."),
        ("missing-number.html", "Missing Number", "Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array."),
        ("valid-anagram.html", "Valid Anagram", "Given two strings s and t, return true if t is an anagram of s, and false otherwise."),
        ("top-k-frequent-elements.html", "Top K Frequent Elements", "Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order."),
        ("find-peak-element.html", "Find Peak Element", "A peak element is an element that is strictly greater than its neighbors."),
        ("word-pattern.html", "Word Pattern", "Given a pattern and a string s, find if s follows the same pattern."),
        ("h-index.html", "H-Index", "Given an array of integers citations where citations[i] is the number of citations a researcher received for their ith paper, return the researcher's h-index."),
        ("palindrome-pairs.html", "Palindrome Pairs", "Given a list of unique words, return all the pairs of the distinct indices (i, j) in the given list, so that the concatenation of the two words words[i] + words[j] is a palindrome."),
        ("one-edit-distance.html", "One Edit Distance", "Given two strings s and t, return true if they are both one edit distance apart, otherwise return false."),
        ("scramble-string.html", "Scramble String", "We can scramble a string s to get a string t using the following algorithm:"),
        ("first-bad-version.html", "First Bad Version", "You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check."),
        ("integer-english-words.html", "Integer to English Words", "Convert a non-negative integer num to its English words representation."),
        ("text-justification.html", "Text Justification", "Given an array of strings words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified."),
        ("remove-invalid-parentheses.html", "Remove Invalid Parentheses", "Given a string s that contains parentheses and letters, remove the minimum number of invalid parentheses to make the input string valid."),
        ("intersection-two-arrays.html", "Intersection of Two Arrays", "Given two integer arrays nums1 and nums2, return an array of their intersection."),
        ("sliding-window-maximum.html", "Sliding Window Maximum", "You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right."),
        ("guess-number-higher-lower.html", "Guess Number Higher or Lower", "We are playing the Guess Game. The game is as follows:"),
        ("spiral-matrix-ii.html", "Spiral Matrix II", "Given a positive integer n, generate an n x n matrix filled with elements from 1 to n2 in spiral order."),
        ("search-2d-matrix-ii.html", "Search a 2D Matrix II", "Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix."),
        ("rotate-image.html", "Rotate Image", "You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise)."),
        ("valid-sudoku.html", "Valid Sudoku", "Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:"),
        ("minimum-path-sum.html", "Minimum Path Sum", "Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path."),
        ("unique-paths-ii.html", "Unique Paths II", "You are given an m x n integer array obstacleGrid. There is a robot initially located at the top-left corner (i.e., obstacleGrid[0][0])."),
        ("number-of-islands-ii.html", "Number of Islands II", "You are given an empty 2D binary grid grid of size m x n. The grid represents a map where 0's represent water and 1's represent land."),
        ("surrounded-regions.html", "Surrounded Regions", "Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'."),
        ("word-search-ii.html", "Word Search II", "Given an m x n board of characters and a list of strings words, return all words on the board."),
        ("implement-stack-array.html", "Implement a Stack Using an Array", "Implement a stack data structure using an array with push, pop, and peek operations."),
        ("linked-list-cycle.html", "Linked List Cycle", "Given head, the head of a linked list, determine if the linked list has a cycle in it."),
        ("odd-even-linked-list.html", "Odd Even Linked List", "Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list."),
        ("remove-duplicates-sorted-list.html", "Remove Duplicates from Sorted List", "Given the head of a sorted linked list, delete all duplicates such that each element appears only once."),
        ("remove-duplicates-sorted-list-ii.html", "Remove Duplicates from Sorted List II", "Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list."),
        ("partition-list.html", "Partition List", "Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x."),
        ("lru-cache.html", "LRU Cache", "Design a data structure that follows the constraints of a Least Recently Used (LRU) cache."),
        ("intersection-two-linked-lists.html", "Intersection of Two Linked Lists", "Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect."),
        ("remove-linked-list-elements.html", "Remove Linked List Elements", "Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head."),
        ("swap-nodes-pairs.html", "Swap Nodes in Pairs", "Given a linked list, swap every two adjacent nodes and return its head."),
        ("print-linked-list-reversed.html", "Print Linked List in Reversed Order", "Print a linked list in reversed order using recursion or iterative approach."),
        ("implement-stack-queues.html", "Implement Stack using Queues", "Implement a last-in-first-out (LIFO) stack using only two queues."),
        ("implement-queue-stacks.html", "Implement Queue using Stacks", "Implement a first-in-first-out (FIFO) queue using only two stacks."),
        ("implement-queue-array.html", "Implement a Queue using an Array", "Implement a queue data structure using an array with proper enqueue, dequeue, and peek operations."),
        ("delete-node-linked-list.html", "Delete Node in a Linked List", "Write a function to delete a node in a singly-linked list."),
        ("reverse-nodes-k-group.html", "Reverse Nodes in k-Group", "Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list."),
        ("kth-smallest-bst.html", "Kth Smallest Element in a BST", "Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree."),
        ("binary-tree-longest-consecutive.html", "Binary Tree Longest Consecutive Sequence", "Given the root of a binary tree, return the length of the longest consecutive sequence path."),
        ("flatten-binary-tree-linked-list.html", "Flatten Binary Tree to Linked List", "Given the root of a binary tree, flatten the tree into a 'linked list'."),
        ("path-sum-ii.html", "Path Sum II", "Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where the sum of the node values in the path equals targetSum."),
        ("construct-binary-tree-inorder-postorder.html", "Construct Binary Tree from Inorder and Postorder Traversal", "Given two integer arrays inorder and postorder where inorder is the inorder traversal of a binary tree and postorder is the postorder traversal of the same tree, construct and return the binary tree."),
        ("construct-binary-tree-preorder-inorder.html", "Construct Binary Tree from Preorder and Inorder Traversal", "Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree."),
        ("convert-sorted-array-bst.html", "Convert Sorted Array to Binary Search Tree", "Given an integer array nums where the elements are sorted in ascending order, convert it to a height-balanced binary search tree."),
        ("convert-sorted-list-bst.html", "Convert Sorted List to Binary Search Tree", "Given the head of a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST."),
        ("minimum-depth-binary-tree.html", "Minimum Depth of Binary Tree", "Given a binary tree, find its minimum depth. The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node."),
        ("binary-tree-maximum-path-sum.html", "Binary Tree Maximum Path Sum", "A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them."),
        ("balanced-binary-tree.html", "Balanced Binary Tree", "Given a binary tree, determine if it is height-balanced."),
        ("binary-search-tree-iterator.html", "Binary Search Tree Iterator", "Implement the BSTIterator class that represents an iterator over the in-order traversal of a binary search tree (BST)."),
        ("binary-tree-right-side-view.html", "Binary Tree Right Side View", "Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom."),
        ("lowest-common-ancestor-bst.html", "Lowest Common Ancestor of a Binary Search Tree", "Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST."),
        ("lowest-common-ancestor-binary-tree.html", "Lowest Common Ancestor of a Binary Tree", "Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree."),
        ("verify-preorder-serialization.html", "Verify Preorder Serialization of a Binary Tree", "One way to serialize a binary tree is to use preorder traversal."),
        ("populating-next-right-pointers-ii.html", "Populating Next Right Pointers in Each Node II", "Given a binary tree, populate each next pointer to point to its next right node."),
        ("unique-binary-search-trees.html", "Unique Binary Search Trees", "Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n."),
        ("unique-binary-search-trees-ii.html", "Unique Binary Search Trees II", "Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n."),
        ("sum-root-leaf-numbers.html", "Sum Root to Leaf Numbers", "You are given the root of a binary tree containing digits from 0 to 9 only."),
        ("count-complete-tree-nodes.html", "Count Complete Tree Nodes", "Given the root of a complete binary tree, return the number of the nodes in the tree."),
        ("closest-binary-search-tree-value.html", "Closest Binary Search Tree Value", "Given the root of a binary search tree and a target value, return the value in the BST that is closest to the target."),
        ("binary-tree-paths.html", "Binary Tree Paths", "Given the root of a binary tree, return all root-to-leaf paths in any order."),
        ("recover-binary-search-tree.html", "Recover Binary Search Tree", "You are given the root of a binary search tree (BST), where the values of exactly two nodes of the tree were swapped by mistake."),
        ("same-tree.html", "Same Tree", "Given the roots of two binary trees p and q, write a function to check if they are the same or not."),
        ("inorder-successor-bst.html", "Inorder Successor in BST", "Given the root of a binary search tree and a node p in it, return the in-order successor of that node in the BST."),
        ("find-leaves-binary-tree.html", "Find Leaves of Binary Tree", "Given the root of a binary tree, collect a tree's nodes as if you were doing this: Collect all the leaf nodes."),
        ("largest-bst-subtree.html", "Largest BST Subtree", "Given the root of a binary tree, find the largest subtree, which is also a Binary Search Tree (BST), where the largest means subtree has the largest number of nodes."),
        ("maximum-subarray.html", "Maximum Subarray", "Given an integer array nums, find the subarray with the largest sum, and return its sum."),
        ("longest-increasing-subsequence.html", "Longest Increasing Subsequence", "Given an integer array nums, return the length of the longest strictly increasing subsequence."),
        ("word-break.html", "Word Break", "Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words."),
        ("jump-game.html", "Jump Game", "You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position."),
        ("best-time-buy-sell-stock.html", "Best Time to Buy and Sell Stock", "You are given an array prices where prices[i] is the price of a given stock on the ith day."),
        ("decode-ways.html", "Decode Ways", "A message containing letters from A-Z can be encoded into numbers using the following mapping: 'A' -> '1', 'B' -> '2', ..., 'Z' -> '26'."),
        ("palindrome-partitioning.html", "Palindrome Partitioning", "Given a string s, partition s such that every substring of the partition is a palindrome."),
        ("reverse-bits.html", "Reverse Bits", "Reverse bits of a given 32 bits unsigned integer."),
        ("number-1-bits.html", "Number of 1 Bits", "Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight)."),
        ("maximum-binary-gap.html", "Maximum Binary Gap", "Given a positive integer n, find and return the longest distance between any two adjacent 1's in the binary representation of n."),
        ("sum-two-integers.html", "Sum of Two Integers", "Given two integers a and b, return the sum of the two integers without using the operators + and -."),
        ("missing-number-bit.html", "Missing Number", "Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array."),
        ("bitwise-and-numbers-range.html", "Bitwise AND of Numbers Range", "Given two integers left and right that represent the range [left, right], return the bitwise AND of all numbers in this range, inclusive."),
        ("gray-code.html", "Gray Code", "An n-bit gray code sequence is a sequence of 2n integers where: Every integer is in the inclusive range [0, 2n - 1], The first integer is 0, An integer appears no more than once in the sequence."),
        ("maximum-product-word-lengths.html", "Maximum Product of Word Lengths", "Given a string array words, return the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters."),
        ("permutations.html", "Permutations", "Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order."),
        ("permutations-ii.html", "Permutations II", "Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order."),
        ("permutation-sequence.html", "Permutation Sequence", "The set [1, 2, 3, ..., n] contains a total of n! unique permutations."),
        ("generate-parentheses.html", "Generate Parentheses", "Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses."),
        ("combination-sum.html", "Combination Sum", "Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target."),
        ("combinations.html", "Combinations", "Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n]. You may return the answer in any order."),
        ("letter-combinations-phone-number.html", "Letter Combinations of a Phone Number", "Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent."),
        ("restore-ip-addresses.html", "Restore IP Addresses", "A valid IP address consists of exactly four integers separated by single dots. Each integer is between 0 and 255 (inclusive) and cannot have leading zeros."),
        ("factor-combinations.html", "Factor Combinations", "Numbers can be regarded as the product of their factors. Given an integer n, return all possible combinations of its factors."),
        ("reverse-integer.html", "Reverse Integer", "Given a signed 32-bit integer x, return x with its digits reversed."),
        ("palindrome-number.html", "Palindrome Number", "Given an integer x, return true if x is a palindrome, and false otherwise."),
        ("pow-x-n.html", "Pow(x, n)", "Implement pow(x, n), which calculates x raised to the power n (i.e., xn)."),
        ("subsets.html", "Subsets", "Given an integer array nums of unique elements, return all possible subsets (the power set)."),
        ("fraction-recurring-decimal.html", "Fraction to Recurring Decimal", "Given two integers representing the numerator and denominator of a fraction, return the fraction in string format."),
        ("excel-sheet-column-number.html", "Excel Sheet Column Number", "Given a string columnTitle that represents the column title as appears in an Excel sheet, return its corresponding column number."),
        ("excel-sheet-column-title.html", "Excel Sheet Column Title", "Given an integer columnNumber, return its corresponding column title as it appears in an Excel sheet."),
        ("count-primes.html", "Count Primes", "Given an integer n, return the number of prime numbers that are strictly less than n."),
        ("plus-one.html", "Plus One", "You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer."),
        ("divide-two-integers.html", "Divide Two Integers", "Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator."),
        ("multiply-strings.html", "Multiply Strings", "Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string."),
        ("max-points-line.html", "Max Points on a Line", "Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, return the maximum number of points that lie on the same straight line."),
        ("product-array-except-self.html", "Product of Array Except Self", "Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i]."),
        ("integer-break.html", "Integer Break", "Given an integer n, break it into the sum of k positive integers, where k >= 2, and maximize the product of those integers."),
        ("add-digits.html", "Add Digits", "Given an integer num, repeatedly add all its digits until the result has only one digit, and return it."),
        ("ugly-number.html", "Ugly Number", "An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5. Given an integer n, return true if n is an ugly number."),
        ("find-k-pairs-smallest-sums.html", "Find K Pairs with Smallest Sums", "You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k."),
        ("selection-sort.html", "Selection Sort", "Implement selection sort algorithm to sort an array in ascending order."),
        ("counting-sort.html", "Counting Sort", "Implement counting sort algorithm for sorting integers with a known range."),
        ("radix-sort.html", "Radix Sort", "Implement radix sort algorithm for sorting integers."),
        ("bucket-sort.html", "Bucket Sort", "Implement bucket sort algorithm for sorting floating point numbers."),
        ("merge-overlapping-intervals.html", "Merge Overlapping Intervals", "Given a collection of intervals, merge all overlapping intervals."),
        ("form-largest-number.html", "Form the Largest Number", "Given a list of non-negative integers, arrange them such that they form the largest number."),
        ("sort-array-0s-1s-2s.html", "Sort array of 0s, 1s, and 2s", "Given an array containing only 0s, 1s, and 2s, sort it in linear time."),
        ("kth-smallest-largest.html", "K'th Smallest/Largest", "Find the kth smallest or largest element in an unsorted array."),
        ("minimum-platforms-required.html", "Minimum Platforms Required", "Given arrival and departure times of all trains, find the minimum number of platforms required."),
        ("case-specific-sorting-strings.html", "Case-specific Sorting of Strings", "Sort a string such that uppercase letters come before lowercase letters."),
        ("sort-by-frequency.html", "Sort by Frequency", "Sort elements of an array by frequency of occurrence."),
        ("minimum-operations-distinct.html", "Minimum Operations for Distinct", "Find minimum operations to make all elements distinct in an array."),
        ("merge-k-sorted-arrays.html", "Merge k sorted arrays", "Merge k sorted arrays into a single sorted array efficiently."),
        ("merge-k-sorted-lists.html", "Merge k Sorted Lists", "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order."),
        ("find-median-data-stream.html", "Find Median from Data Stream", "The median is the middle value in an ordered integer list."),
        ("meeting-rooms.html", "Meeting Rooms II", "Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required."),
        ("range-addition.html", "Range Addition", "Assume you have an array of length n initialized with all 0's and are given k update operations."),
        ("add-search-word.html", "Add and Search Word", "Design a data structure that supports adding new words and finding if a string matches any previously added string."),
        ("range-sum-query-mutable.html", "Range Sum Query – Mutable", "Given an integer array nums, handle multiple queries of the following types: Update the value of an element in nums."),
        ("skyline-problem.html", "The Skyline Problem", "A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance."),
        ("minimum-height-trees.html", "Minimum Height Trees", "A tree is an undirected graph in which any two vertices are connected by exactly one path."),
        ("reconstruct-itinerary.html", "Reconstruct Itinerary", "You are given a list of airline tickets where tickets[i] = [fromi, toi] represent the departure and arrival airports of one flight."),
        ("graph-valid-tree.html", "Graph Valid Tree", "Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.")
    ]
    
    print("🚀 Starting Remaining Problems Generator")
    print("=" * 70)
    
    for filename, title, description in problems:
        generate_problem(filename, title, description)
    
    print("\n" + "=" * 70)
    print("✅ All remaining problem pages have been generated successfully!")
    print("🌐 You can now open index.html to view the complete website.")

if __name__ == "__main__":
    main() 