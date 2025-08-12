#!/usr/bin/env python3
"""
Test script to verify the dynamic solution system works correctly.
"""

import json
import os
from pathlib import Path

def test_json_files():
    """Test that JSON solution files are valid and contain required fields."""
    solutions_dir = Path('solutions')
    
    if not solutions_dir.exists():
        print("❌ Solutions directory not found!")
        return False
    
    json_files = list(solutions_dir.glob('*.json'))
    
    if not json_files:
        print("❌ No JSON solution files found!")
        return False
    
    print(f"Found {len(json_files)} JSON solution files")
    
    required_fields = {
        'problem': ['title', 'description'],
        'solutions': ['title', 'code'],
        'navigation': []
    }
    
    all_valid = True
    
    for json_file in json_files:
        print(f"\nTesting {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required fields
            for section, fields in required_fields.items():
                if section not in data:
                    print(f"  ❌ Missing section: {section}")
                    all_valid = False
                    continue
                
                if section == 'problem':
                    for field in fields:
                        if field not in data[section]:
                            print(f"  ❌ Missing problem field: {field}")
                            all_valid = False
                
                elif section == 'solutions':
                    if not isinstance(data[section], list):
                        print(f"  ❌ Solutions should be a list")
                        all_valid = False
                    else:
                        for i, solution in enumerate(data[section]):
                            for field in fields:
                                if field not in solution:
                                    print(f"  ❌ Missing solution {i} field: {field}")
                                    all_valid = False
            
            print(f"  ✅ Valid JSON structure")
            
        except json.JSONDecodeError as e:
            print(f"  ❌ Invalid JSON: {e}")
            all_valid = False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_valid = False
    
    return all_valid

def test_html_files():
    """Test that HTML files exist and can be opened."""
    problems_dir = Path('problems')
    
    if not problems_dir.exists():
        print("❌ Problems directory not found!")
        return False
    
    html_files = list(problems_dir.glob('*.html'))
    
    if not html_files:
        print("❌ No HTML files found!")
        return False
    
    print(f"\nFound {len(html_files)} HTML files")
    
    # Check if they're using the dynamic template
    dynamic_template_marker = "ProblemLoader"
    
    all_valid = True
    
    for html_file in html_files:
        print(f"\nTesting {html_file.name}...")
        
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if dynamic_template_marker in content:
                print(f"  ✅ Using dynamic template")
            else:
                print(f"  ❌ Not using dynamic template")
                all_valid = False
                
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
            all_valid = False
    
    return all_valid

def test_file_correspondence():
    """Test that each HTML file has a corresponding JSON file."""
    problems_dir = Path('problems')
    solutions_dir = Path('solutions')
    
    html_files = set(f.stem for f in problems_dir.glob('*.html'))
    json_files = set(f.stem for f in solutions_dir.glob('*.json'))
    
    missing_json = html_files - json_files
    missing_html = json_files - html_files
    
    print(f"\nFile correspondence test:")
    
    if missing_json:
        print(f"❌ HTML files missing JSON: {missing_json}")
        return False
    
    if missing_html:
        print(f"⚠️  JSON files without HTML: {missing_html}")
    
    print(f"✅ All HTML files have corresponding JSON files")
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Dynamic Solution System")
    print("=" * 40)
    
    tests = [
        ("JSON Files", test_json_files),
        ("HTML Files", test_html_files),
        ("File Correspondence", test_file_correspondence)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 20)
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n📊 Test Results")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The dynamic system is working correctly.")
        print("\nNext steps:")
        print("1. Open any problem HTML file in a browser")
        print("2. Verify content loads from JSON files")
        print("3. Edit a JSON file and refresh to see changes")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main() 