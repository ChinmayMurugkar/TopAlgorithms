#!/usr/bin/env python3
"""
Script to extract solutions from HTML problem files and convert them to JSON format.
This allows developers to edit solution files without touching HTML.
"""

import os
import re
import json
from bs4 import BeautifulSoup
from pathlib import Path

def extract_problem_info(html_content):
    """Extract basic problem information from HTML."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract title
    title_elem = soup.find('h1')
    title = title_elem.text.strip() if title_elem else "Unknown Problem"
    
    # Extract LeetCode ID if available
    meta_elem = soup.find('div', class_='meta')
    leetcode_id = None
    if meta_elem:
        id_match = re.search(r'LeetCode Problem (\d+)', meta_elem.text)
        if id_match:
            leetcode_id = int(id_match.group(1))
    
    # Extract category from breadcrumb
    breadcrumb = soup.find('div', class_='breadcrumb')
    category = "Unknown"
    if breadcrumb:
        category_match = re.search(r'> ([^>]+) >', breadcrumb.text)
        if category_match:
            category = category_match.group(1).strip()
    
    # Extract problem description
    problem_desc = soup.find('div', class_='problem-description')
    description = ""
    assumptions = []
    example = {}
    
    if problem_desc:
        # Extract main description
        desc_text = problem_desc.get_text()
        desc_match = re.search(r'Problem:(.*?)(?:Assumptions:|Example:|$)', desc_text, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()
        
        # Extract assumptions
        assumptions_match = re.search(r'Assumptions:(.*?)(?:Example:|$)', desc_text, re.DOTALL)
        if assumptions_match:
            assumptions_text = assumptions_match.group(1).strip()
            assumptions = [assumption.strip() for assumption in assumptions_text.split('\n') if assumption.strip()]
        
        # Extract example
        example_match = re.search(r'Example:(.*?)(?:$)', desc_text, re.DOTALL)
        if example_match:
            example_text = example_match.group(1).strip()
            input_match = re.search(r'Input:(.*?)(?:Output:|$)', example_text, re.DOTALL)
            output_match = re.search(r'Output:(.*?)(?:Explanation:|$)', example_text, re.DOTALL)
            explanation_match = re.search(r'Explanation:(.*?)(?:$)', example_text, re.DOTALL)
            
            if input_match:
                example['input'] = input_match.group(1).strip()
            if output_match:
                example['output'] = output_match.group(1).strip()
            if explanation_match:
                example['explanation'] = explanation_match.group(1).strip()
    
    return {
        'title': title,
        'leetcode_id': leetcode_id,
        'category': category,
        'description': description,
        'assumptions': assumptions,
        'example': example
    }

def extract_solutions(html_content):
    """Extract solutions from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    solutions = []
    
    solution_divs = soup.find_all('div', class_='solution')
    
    for solution_div in solution_divs:
        solution = {}
        
        # Extract title
        title_elem = solution_div.find('h3')
        if title_elem:
            solution['title'] = title_elem.text.strip()
        
        # Extract description
        desc_elem = solution_div.find('p')
        if desc_elem:
            solution['description'] = desc_elem.text.strip()
        
        # Extract code
        code_elem = solution_div.find('div', class_='code-block')
        if code_elem:
            solution['code'] = code_elem.get_text().strip()
            solution['language'] = 'java'  # Default assumption
        
        # Extract complexity
        complexity_elem = solution_div.find('div', class_='complexity')
        if complexity_elem:
            complexity = {}
            complexity_text = complexity_elem.get_text()
            
            time_match = re.search(r'Time Complexity:\s*([^\n]+)', complexity_text)
            if time_match:
                complexity['time'] = time_match.group(1).strip()
            
            space_match = re.search(r'Space Complexity:\s*([^\n]+)', complexity_text)
            if space_match:
                complexity['space'] = space_match.group(1).strip()
            
            if complexity:
                solution['complexity'] = complexity
        
        # Extract example walkthrough
        example_elem = solution_div.find('div', class_='example')
        if example_elem:
            example_text = example_elem.get_text()
            if 'Example Walkthrough:' in example_text:
                solution['example_walkthrough'] = {
                    'text': example_text.replace('Example Walkthrough:', '').strip()
                }
        
        if solution:  # Only add if we have at least some content
            solutions.append(solution)
    
    return solutions

def extract_variations(html_content):
    """Extract variations from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    variations = []
    
    # Look for variations section
    variations_section = soup.find('div', class_='solution')
    if variations_section and 'Variations' in variations_section.get_text():
        # Find all h4 elements (variation titles)
        h4_elements = variations_section.find_all('h4')
        for h4 in h4_elements:
            variation = {'title': h4.text.strip()}
            
            # Get description from next sibling
            next_elem = h4.find_next_sibling()
            if next_elem and next_elem.name == 'p':
                variation['description'] = next_elem.text.strip()
            
            # Get code if available
            code_elem = next_elem.find_next_sibling('div', class_='code-block') if next_elem else None
            if code_elem:
                variation['code'] = code_elem.get_text().strip()
                variation['language'] = 'java'
            
            variations.append(variation)
    
    return variations

def extract_navigation(html_content):
    """Extract navigation links."""
    soup = BeautifulSoup(html_content, 'html.parser')
    navigation = {}
    
    nav_div = soup.find('div', class_='navigation')
    if nav_div:
        prev_link = nav_div.find('a', string=re.compile(r'← Previous:'))
        next_link = nav_div.find('a', string=re.compile(r'Next:.*→'))
        
        if prev_link:
            navigation['previous'] = prev_link['href'].split('/')[-1]
        if next_link:
            navigation['next'] = next_link['href'].split('/')[-1]
    
    return navigation

def convert_html_to_json(html_file_path):
    """Convert a single HTML file to JSON format."""
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract all components
    problem_info = extract_problem_info(html_content)
    solutions = extract_solutions(html_content)
    variations = extract_variations(html_content)
    navigation = extract_navigation(html_content)
    
    # Create JSON structure
    json_data = {
        'problem': problem_info,
        'solutions': solutions,
        'variations': variations,
        'navigation': navigation
    }
    
    return json_data

def main():
    """Main function to process all HTML files."""
    problems_dir = Path('problems')
    solutions_dir = Path('solutions')
    
    # Create solutions directory if it doesn't exist
    solutions_dir.mkdir(exist_ok=True)
    
    # Process each HTML file
    html_files = list(problems_dir.glob('*.html'))
    
    print(f"Found {len(html_files)} HTML files to process...")
    
    for html_file in html_files:
        print(f"Processing {html_file.name}...")
        
        try:
            json_data = convert_html_to_json(html_file)
            
            # Create JSON filename
            json_filename = html_file.stem + '.json'
            json_file_path = solutions_dir / json_filename
            
            # Write JSON file
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Created {json_filename}")
            
        except Exception as e:
            print(f"  ✗ Error processing {html_file.name}: {e}")
    
    print(f"\nConversion complete! Created {len(html_files)} JSON solution files.")
    print("Developers can now edit these JSON files to update solutions.")

if __name__ == "__main__":
    main() 