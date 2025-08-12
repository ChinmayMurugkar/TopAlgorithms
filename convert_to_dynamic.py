#!/usr/bin/env python3
"""
Script to convert all existing HTML problem files to use the dynamic template system.
This will replace the hardcoded content with dynamic loading from JSON files.
"""

import os
import shutil
from pathlib import Path

def convert_html_to_dynamic():
    """Convert all HTML problem files to use the dynamic template."""
    problems_dir = Path('problems')
    template_file = Path('dynamic_problem_template.html')
    
    if not template_file.exists():
        print("Error: dynamic_problem_template.html not found!")
        return
    
    # Read the template
    with open(template_file, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Process each HTML file
    html_files = list(problems_dir.glob('*.html'))
    
    print(f"Found {len(html_files)} HTML files to convert...")
    
    for html_file in html_files:
        print(f"Converting {html_file.name}...")
        
        try:
            # Create backup
            backup_file = html_file.with_suffix('.html.backup')
            shutil.copy2(html_file, backup_file)
            
            # Replace with dynamic template
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            print(f"  ✓ Converted {html_file.name} (backup: {backup_file.name})")
            
        except Exception as e:
            print(f"  ✗ Error converting {html_file.name}: {e}")
    
    print(f"\nConversion complete! Converted {len(html_files)} HTML files.")
    print("All files now use the dynamic template system.")
    print("Backup files have been created with .backup extension.")

def create_solutions_directory():
    """Create the solutions directory if it doesn't exist."""
    solutions_dir = Path('solutions')
    solutions_dir.mkdir(exist_ok=True)
    print(f"Created solutions directory: {solutions_dir}")

def main():
    """Main function."""
    print("Converting HTML files to dynamic template system...")
    
    # Create solutions directory
    create_solutions_directory()
    
    # Convert HTML files
    convert_html_to_dynamic()
    
    print("\nNext steps:")
    print("1. Run extract_solutions.py to create JSON solution files")
    print("2. Test the dynamic loading by opening any problem HTML file")
    print("3. Developers can now edit JSON files to update solutions")

if __name__ == "__main__":
    main() 