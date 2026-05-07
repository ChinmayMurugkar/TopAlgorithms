#!/usr/bin/env python3
import os

def check_placeholder_solutions():
    """Check how many problems still have placeholder solutions."""
    problems_dir = "problems"
    placeholder_count = 0
    total_count = 0
    
    for filename in os.listdir(problems_dir):
        if filename.endswith('.html'):
            total_count += 1
            filepath = os.path.join(problems_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if "TODO: Implement solution" in content:
                    placeholder_count += 1
                    print(f"⚠️  Still has placeholder: {filename}")
    
    print(f"\n📊 Summary:")
    print(f"Total problems: {total_count}")
    print(f"Problems with placeholders: {placeholder_count}")
    print(f"Problems with real solutions: {total_count - placeholder_count}")
    print(f"Completion rate: {((total_count - placeholder_count) / total_count * 100):.1f}%")
    
    return placeholder_count

if __name__ == "__main__":
    check_placeholder_solutions() 