#!/usr/bin/env python3
"""
Next Batch Problem Generator
This script generates the next batch of remaining coding interview problems.
"""

import os
import time
from typing import List, Dict

class NextBatchProblemGenerator:
    def __init__(self):
        self.problems_dir = "problems"
        self.template = self.load_template()
        self.problem_data = self.load_next_batch_problems()
        
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

    def load_next_batch_problems(self) -> List[Dict]:
        """Load the next batch of remaining problems."""
        return [
            # Dynamic Programming Problems
            {
                "filename": "maximum-subarray.html",
                "title": "Maximum Subarray",
                "problem_name": "Maximum Subarray",
                "category": "Dynamic Programming",
                "problem_number": "53",
                "problem_description": "Given an integer array nums, find the subarray with the largest sum, and return its sum.",
                "solutions": self.generate_basic_solution("Maximum Subarray"),
                "prev_link": "matrix-chain-multiplication.html",
                "prev_name": "Matrix Chain Multiplication",
                "next_link": "longest-increasing-subsequence.html",
                "next_name": "Longest Increasing Subsequence"
            },
            {
                "filename": "longest-increasing-subsequence.html",
                "title": "Longest Increasing Subsequence",
                "problem_name": "Longest Increasing Subsequence",
                "category": "Dynamic Programming",
                "problem_number": "300",
                "problem_description": "Given an integer array nums, return the length of the longest strictly increasing subsequence.",
                "solutions": self.generate_basic_solution("Longest Increasing Subsequence"),
                "prev_link": "maximum-subarray.html",
                "prev_name": "Maximum Subarray",
                "next_link": "word-break.html",
                "next_name": "Word Break"
            },
            {
                "filename": "word-break.html",
                "title": "Word Break",
                "problem_name": "Word Break",
                "category": "Dynamic Programming",
                "problem_number": "139",
                "problem_description": "Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.",
                "solutions": self.generate_basic_solution("Word Break"),
                "prev_link": "longest-increasing-subsequence.html",
                "prev_name": "Longest Increasing Subsequence",
                "next_link": "jump-game.html",
                "next_name": "Jump Game"
            },
            {
                "filename": "jump-game.html",
                "title": "Jump Game",
                "problem_name": "Jump Game",
                "category": "Dynamic Programming",
                "problem_number": "55",
                "problem_description": "You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.",
                "solutions": self.generate_basic_solution("Jump Game"),
                "prev_link": "word-break.html",
                "prev_name": "Word Break",
                "next_link": "best-time-buy-sell-stock.html",
                "next_name": "Best Time to Buy and Sell Stock"
            },
            {
                "filename": "best-time-buy-sell-stock.html",
                "title": "Best Time to Buy and Sell Stock",
                "problem_name": "Best Time to Buy and Sell Stock",
                "category": "Dynamic Programming",
                "problem_number": "121",
                "problem_description": "You are given an array prices where prices[i] is the price of a given stock on the ith day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.",
                "solutions": self.generate_basic_solution("Best Time to Buy and Sell Stock"),
                "prev_link": "jump-game.html",
                "prev_name": "Jump Game",
                "next_link": "decode-ways.html",
                "next_name": "Decode Ways"
            },
            {
                "filename": "decode-ways.html",
                "title": "Decode Ways",
                "problem_name": "Decode Ways",
                "category": "Dynamic Programming",
                "problem_number": "91",
                "problem_description": "A message containing letters from A-Z can be encoded into numbers using the following mapping: 'A' -> '1', 'B' -> '2', ..., 'Z' -> '26'.",
                "solutions": self.generate_basic_solution("Decode Ways"),
                "prev_link": "best-time-buy-sell-stock.html",
                "prev_name": "Best Time to Buy and Sell Stock",
                "next_link": "palindrome-partitioning.html",
                "next_name": "Palindrome Partitioning"
            },
            {
                "filename": "palindrome-partitioning.html",
                "title": "Palindrome Partitioning",
                "problem_name": "Palindrome Partitioning",
                "category": "Dynamic Programming",
                "problem_number": "131",
                "problem_description": "Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.",
                "solutions": self.generate_basic_solution("Palindrome Partitioning"),
                "prev_link": "decode-ways.html",
                "prev_name": "Decode Ways",
                "next_link": "reverse-bits.html",
                "next_name": "Reverse Bits"
            },
            # Bit Manipulation Problems
            {
                "filename": "reverse-bits.html",
                "title": "Reverse Bits",
                "problem_name": "Reverse Bits",
                "category": "Bit Manipulation",
                "problem_number": "190",
                "problem_description": "Reverse bits of a given 32 bits unsigned integer.",
                "solutions": self.generate_basic_solution("Reverse Bits"),
                "prev_link": "palindrome-partitioning.html",
                "prev_name": "Palindrome Partitioning",
                "next_link": "number-1-bits.html",
                "next_name": "Number of 1 Bits"
            },
            {
                "filename": "number-1-bits.html",
                "title": "Number of 1 Bits",
                "problem_name": "Number of 1 Bits",
                "category": "Bit Manipulation",
                "problem_number": "191",
                "problem_description": "Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).",
                "solutions": self.generate_basic_solution("Number of 1 Bits"),
                "prev_link": "reverse-bits.html",
                "prev_name": "Reverse Bits",
                "next_link": "maximum-binary-gap.html",
                "next_name": "Maximum Binary Gap"
            },
            {
                "filename": "maximum-binary-gap.html",
                "title": "Maximum Binary Gap",
                "problem_name": "Maximum Binary Gap",
                "category": "Bit Manipulation",
                "problem_number": "868",
                "problem_description": "Given a positive integer n, find and return the longest distance between any two adjacent 1's in the binary representation of n. If there are no two adjacent 1's, return 0.",
                "solutions": self.generate_basic_solution("Maximum Binary Gap"),
                "prev_link": "number-1-bits.html",
                "prev_name": "Number of 1 Bits",
                "next_link": "sum-two-integers.html",
                "next_name": "Sum of Two Integers"
            },
            {
                "filename": "sum-two-integers.html",
                "title": "Sum of Two Integers",
                "problem_name": "Sum of Two Integers",
                "category": "Bit Manipulation",
                "problem_number": "371",
                "problem_description": "Given two integers a and b, return the sum of the two integers without using the operators + and -.",
                "solutions": self.generate_basic_solution("Sum of Two Integers"),
                "prev_link": "maximum-binary-gap.html",
                "prev_name": "Maximum Binary Gap",
                "next_link": "missing-number-bit.html",
                "next_name": "Missing Number"
            },
            {
                "filename": "missing-number-bit.html",
                "title": "Missing Number",
                "problem_name": "Missing Number",
                "category": "Bit Manipulation",
                "problem_number": "268",
                "problem_description": "Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.",
                "solutions": self.generate_basic_solution("Missing Number"),
                "prev_link": "sum-two-integers.html",
                "prev_name": "Sum of Two Integers",
                "next_link": "bitwise-and-numbers-range.html",
                "next_name": "Bitwise AND of Numbers Range"
            },
            {
                "filename": "bitwise-and-numbers-range.html",
                "title": "Bitwise AND of Numbers Range",
                "problem_name": "Bitwise AND of Numbers Range",
                "category": "Bit Manipulation",
                "problem_number": "201",
                "problem_description": "Given two integers left and right that represent the range [left, right], return the bitwise AND of all numbers in this range, inclusive.",
                "solutions": self.generate_basic_solution("Bitwise AND of Numbers Range"),
                "prev_link": "missing-number-bit.html",
                "prev_name": "Missing Number",
                "next_link": "gray-code.html",
                "next_name": "Gray Code"
            },
            {
                "filename": "gray-code.html",
                "title": "Gray Code",
                "problem_name": "Gray Code",
                "category": "Bit Manipulation",
                "problem_number": "89",
                "problem_description": "An n-bit gray code sequence is a sequence of 2n integers where: Every integer is in the inclusive range [0, 2n - 1], The first integer is 0, An integer appears no more than once in the sequence.",
                "solutions": self.generate_basic_solution("Gray Code"),
                "prev_link": "bitwise-and-numbers-range.html",
                "prev_name": "Bitwise AND of Numbers Range",
                "next_link": "maximum-product-word-lengths.html",
                "next_name": "Maximum Product of Word Lengths"
            },
            {
                "filename": "maximum-product-word-lengths.html",
                "title": "Maximum Product of Word Lengths",
                "problem_name": "Maximum Product of Word Lengths",
                "category": "Bit Manipulation",
                "problem_number": "318",
                "problem_description": "Given a string array words, return the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters. If no such two words exist, return 0.",
                "solutions": self.generate_basic_solution("Maximum Product of Word Lengths"),
                "prev_link": "gray-code.html",
                "prev_name": "Gray Code",
                "next_link": "permutations.html",
                "next_name": "Permutations"
            },
            # Combinations and Permutations Problems
            {
                "filename": "permutations.html",
                "title": "Permutations",
                "problem_name": "Permutations",
                "category": "Combinations & Permutations",
                "problem_number": "46",
                "problem_description": "Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.",
                "solutions": self.generate_basic_solution("Permutations"),
                "prev_link": "maximum-product-word-lengths.html",
                "prev_name": "Maximum Product of Word Lengths",
                "next_link": "permutations-ii.html",
                "next_name": "Permutations II"
            },
            {
                "filename": "permutations-ii.html",
                "title": "Permutations II",
                "problem_name": "Permutations II",
                "category": "Combinations & Permutations",
                "problem_number": "47",
                "problem_description": "Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.",
                "solutions": self.generate_basic_solution("Permutations II"),
                "prev_link": "permutations.html",
                "prev_name": "Permutations",
                "next_link": "permutation-sequence.html",
                "next_name": "Permutation Sequence"
            },
            {
                "filename": "permutation-sequence.html",
                "title": "Permutation Sequence",
                "problem_name": "Permutation Sequence",
                "category": "Combinations & Permutations",
                "problem_number": "60",
                "problem_description": "The set [1, 2, 3, ..., n] contains a total of n! unique permutations. By listing and labeling all of the permutations in order, we get the following sequence for n = 3: '123', '132', '213', '231', '312', '321'.",
                "solutions": self.generate_basic_solution("Permutation Sequence"),
                "prev_link": "permutations-ii.html",
                "prev_name": "Permutations II",
                "next_link": "generate-parentheses.html",
                "next_name": "Generate Parentheses"
            },
            {
                "filename": "generate-parentheses.html",
                "title": "Generate Parentheses",
                "problem_name": "Generate Parentheses",
                "category": "Combinations & Permutations",
                "problem_number": "22",
                "problem_description": "Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.",
                "solutions": self.generate_basic_solution("Generate Parentheses"),
                "prev_link": "permutation-sequence.html",
                "prev_name": "Permutation Sequence",
                "next_link": "combination-sum.html",
                "next_name": "Combination Sum"
            },
            {
                "filename": "combination-sum.html",
                "title": "Combination Sum",
                "problem_name": "Combination Sum",
                "category": "Combinations & Permutations",
                "problem_number": "39",
                "problem_description": "Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target.",
                "solutions": self.generate_basic_solution("Combination Sum"),
                "prev_link": "generate-parentheses.html",
                "prev_name": "Generate Parentheses",
                "next_link": "combinations.html",
                "next_name": "Combinations"
            },
            {
                "filename": "combinations.html",
                "title": "Combinations",
                "problem_name": "Combinations",
                "category": "Combinations & Permutations",
                "problem_number": "77",
                "problem_description": "Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n]. You may return the answer in any order.",
                "solutions": self.generate_basic_solution("Combinations"),
                "prev_link": "combination-sum.html",
                "prev_name": "Combination Sum",
                "next_link": "letter-combinations-phone-number.html",
                "next_name": "Letter Combinations of a Phone Number"
            },
            {
                "filename": "letter-combinations-phone-number.html",
                "title": "Letter Combinations of a Phone Number",
                "problem_name": "Letter Combinations of a Phone Number",
                "category": "Combinations & Permutations",
                "problem_number": "17",
                "problem_description": "Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.",
                "solutions": self.generate_basic_solution("Letter Combinations of a Phone Number"),
                "prev_link": "combinations.html",
                "prev_name": "Combinations",
                "next_link": "restore-ip-addresses.html",
                "next_name": "Restore IP Addresses"
            },
            {
                "filename": "restore-ip-addresses.html",
                "title": "Restore IP Addresses",
                "problem_name": "Restore IP Addresses",
                "category": "Combinations & Permutations",
                "problem_number": "93",
                "problem_description": "A valid IP address consists of exactly four integers separated by single dots. Each integer is between 0 and 255 (inclusive) and cannot have leading zeros.",
                "solutions": self.generate_basic_solution("Restore IP Addresses"),
                "prev_link": "letter-combinations-phone-number.html",
                "prev_name": "Letter Combinations of a Phone Number",
                "next_link": "factor-combinations.html",
                "next_name": "Factor Combinations"
            },
            {
                "filename": "factor-combinations.html",
                "title": "Factor Combinations",
                "problem_name": "Factor Combinations",
                "category": "Combinations & Permutations",
                "problem_number": "254",
                "problem_description": "Numbers can be regarded as the product of their factors. Given an integer n, return all possible combinations of its factors. You may return the answer in any order.",
                "solutions": self.generate_basic_solution("Factor Combinations"),
                "prev_link": "restore-ip-addresses.html",
                "prev_name": "Restore IP Addresses",
                "next_link": "reverse-integer.html",
                "next_name": "Reverse Integer"
            },
            # Math Problems
            {
                "filename": "reverse-integer.html",
                "title": "Reverse Integer",
                "problem_name": "Reverse Integer",
                "category": "Math",
                "problem_number": "7",
                "problem_description": "Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.",
                "solutions": self.generate_basic_solution("Reverse Integer"),
                "prev_link": "factor-combinations.html",
                "prev_name": "Factor Combinations",
                "next_link": "palindrome-number.html",
                "next_name": "Palindrome Number"
            },
            {
                "filename": "palindrome-number.html",
                "title": "Palindrome Number",
                "problem_name": "Palindrome Number",
                "category": "Math",
                "problem_number": "9",
                "problem_description": "Given an integer x, return true if x is a palindrome, and false otherwise.",
                "solutions": self.generate_basic_solution("Palindrome Number"),
                "prev_link": "reverse-integer.html",
                "prev_name": "Reverse Integer",
                "next_link": "pow-x-n.html",
                "next_name": "Pow(x, n)"
            },
            {
                "filename": "pow-x-n.html",
                "title": "Pow(x, n)",
                "problem_name": "Pow(x, n)",
                "category": "Math",
                "problem_number": "50",
                "problem_description": "Implement pow(x, n), which calculates x raised to the power n (i.e., xn).",
                "solutions": self.generate_basic_solution("Pow(x, n)"),
                "prev_link": "palindrome-number.html",
                "prev_name": "Palindrome Number",
                "next_link": "subsets.html",
                "next_name": "Subsets"
            },
            {
                "filename": "subsets.html",
                "title": "Subsets",
                "problem_name": "Subsets",
                "category": "Math",
                "problem_number": "78",
                "problem_description": "Given an integer array nums of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets. Return the solution in any order.",
                "solutions": self.generate_basic_solution("Subsets"),
                "prev_link": "pow-x-n.html",
                "prev_name": "Pow(x, n)",
                "next_link": "fraction-recurring-decimal.html",
                "next_name": "Fraction to Recurring Decimal"
            },
            {
                "filename": "fraction-recurring-decimal.html",
                "title": "Fraction to Recurring Decimal",
                "problem_name": "Fraction to Recurring Decimal",
                "category": "Math",
                "problem_number": "166",
                "problem_description": "Given two integers representing the numerator and denominator of a fraction, return the fraction in string format. If the fractional part is repeating, enclose the repeating part in parentheses.",
                "solutions": self.generate_basic_solution("Fraction to Recurring Decimal"),
                "prev_link": "subsets.html",
                "prev_name": "Subsets",
                "next_link": "excel-sheet-column-number.html",
                "next_name": "Excel Sheet Column Number"
            },
            {
                "filename": "excel-sheet-column-number.html",
                "title": "Excel Sheet Column Number",
                "problem_name": "Excel Sheet Column Number",
                "category": "Math",
                "problem_number": "171",
                "problem_description": "Given a string columnTitle that represents the column title as appears in an Excel sheet, return its corresponding column number.",
                "solutions": self.generate_basic_solution("Excel Sheet Column Number"),
                "prev_link": "fraction-recurring-decimal.html",
                "prev_name": "Fraction to Recurring Decimal",
                "next_link": "excel-sheet-column-title.html",
                "next_name": "Excel Sheet Column Title"
            },
            {
                "filename": "excel-sheet-column-title.html",
                "title": "Excel Sheet Column Title",
                "problem_name": "Excel Sheet Column Title",
                "category": "Math",
                "problem_number": "168",
                "problem_description": "Given an integer columnNumber, return its corresponding column title as it appears in an Excel sheet.",
                "solutions": self.generate_basic_solution("Excel Sheet Column Title"),
                "prev_link": "excel-sheet-column-number.html",
                "prev_name": "Excel Sheet Column Number",
                "next_link": "count-primes.html",
                "next_name": "Count Primes"
            },
            {
                "filename": "count-primes.html",
                "title": "Count Primes",
                "problem_name": "Count Primes",
                "category": "Math",
                "problem_number": "204",
                "problem_description": "Given an integer n, return the number of prime numbers that are strictly less than n.",
                "solutions": self.generate_basic_solution("Count Primes"),
                "prev_link": "excel-sheet-column-title.html",
                "prev_name": "Excel Sheet Column Title",
                "next_link": "plus-one.html",
                "next_name": "Plus One"
            },
            {
                "filename": "plus-one.html",
                "title": "Plus One",
                "problem_name": "Plus One",
                "category": "Math",
                "problem_number": "66",
                "problem_description": "You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order.",
                "solutions": self.generate_basic_solution("Plus One"),
                "prev_link": "count-primes.html",
                "prev_name": "Count Primes",
                "next_link": "divide-two-integers.html",
                "next_name": "Divide Two Integers"
            },
            {
                "filename": "divide-two-integers.html",
                "title": "Divide Two Integers",
                "problem_name": "Divide Two Integers",
                "category": "Math",
                "problem_number": "29",
                "problem_description": "Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator. The integer division should truncate toward zero, which means losing its fractional part.",
                "solutions": self.generate_basic_solution("Divide Two Integers"),
                "prev_link": "plus-one.html",
                "prev_name": "Plus One",
                "next_link": "multiply-strings.html",
                "next_name": "Multiply Strings"
            },
            {
                "filename": "multiply-strings.html",
                "title": "Multiply Strings",
                "problem_name": "Multiply Strings",
                "category": "Math",
                "problem_number": "43",
                "problem_description": "Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.",
                "solutions": self.generate_basic_solution("Multiply Strings"),
                "prev_link": "divide-two-integers.html",
                "prev_name": "Divide Two Integers",
                "next_link": "max-points-line.html",
                "next_name": "Max Points on a Line"
            },
            {
                "filename": "max-points-line.html",
                "title": "Max Points on a Line",
                "problem_name": "Max Points on a Line",
                "category": "Math",
                "problem_number": "149",
                "problem_description": "Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane, return the maximum number of points that lie on the same straight line.",
                "solutions": self.generate_basic_solution("Max Points on a Line"),
                "prev_link": "multiply-strings.html",
                "prev_name": "Multiply Strings",
                "next_link": "product-array-except-self.html",
                "next_name": "Product of Array Except Self"
            },
            {
                "filename": "product-array-except-self.html",
                "title": "Product of Array Except Self",
                "problem_name": "Product of Array Except Self",
                "category": "Math",
                "problem_number": "238",
                "problem_description": "Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].",
                "solutions": self.generate_basic_solution("Product of Array Except Self"),
                "prev_link": "max-points-line.html",
                "prev_name": "Max Points on a Line",
                "next_link": "integer-break.html",
                "next_name": "Integer Break"
            },
            {
                "filename": "integer-break.html",
                "title": "Integer Break",
                "problem_name": "Integer Break",
                "category": "Math",
                "problem_number": "343",
                "problem_description": "Given an integer n, break it into the sum of k positive integers, where k >= 2, and maximize the product of those integers. Return the maximum product you can get.",
                "solutions": self.generate_basic_solution("Integer Break"),
                "prev_link": "product-array-except-self.html",
                "prev_name": "Product of Array Except Self",
                "next_link": "add-digits.html",
                "next_name": "Add Digits"
            },
            {
                "filename": "add-digits.html",
                "title": "Add Digits",
                "problem_name": "Add Digits",
                "category": "Math",
                "problem_number": "258",
                "problem_description": "Given an integer num, repeatedly add all its digits until the result has only one digit, and return it.",
                "solutions": self.generate_basic_solution("Add Digits"),
                "prev_link": "integer-break.html",
                "prev_name": "Integer Break",
                "next_link": "ugly-number.html",
                "next_name": "Ugly Number"
            },
            {
                "filename": "ugly-number.html",
                "title": "Ugly Number",
                "problem_name": "Ugly Number",
                "category": "Math",
                "problem_number": "263",
                "problem_description": "An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5. Given an integer n, return true if n is an ugly number.",
                "solutions": self.generate_basic_solution("Ugly Number"),
                "prev_link": "add-digits.html",
                "prev_name": "Add Digits",
                "next_link": "find-k-pairs-smallest-sums.html",
                "next_name": "Find K Pairs with Smallest Sums"
            },
            {
                "filename": "find-k-pairs-smallest-sums.html",
                "title": "Find K Pairs with Smallest Sums",
                "problem_name": "Find K Pairs with Smallest Sums",
                "category": "Math",
                "problem_number": "373",
                "problem_description": "You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k. Define a pair (u, v) which consists of one element from the first array and one element from the second array.",
                "solutions": self.generate_basic_solution("Find K Pairs with Smallest Sums"),
                "prev_link": "ugly-number.html",
                "prev_name": "Ugly Number",
                "next_link": "selection-sort.html",
                "next_name": "Selection Sort"
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
        print(f"Starting to generate {total_problems} next batch problem pages...")
        
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
    """Main function to run the next batch problem generator."""
    print("🚀 Starting Next Batch Problem Generator")
    print("=" * 70)
    
    generator = NextBatchProblemGenerator()
    
    # Generate all problems in batches of 10
    generator.generate_all_problems(batch_size=10)
    
    print("\n" + "=" * 70)
    print("✅ Next batch of problem pages have been generated successfully!")
    print("🌐 You can now open index.html to view the complete website.")

if __name__ == "__main__":
    main() 