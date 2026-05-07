#!/usr/bin/env python3
"""
Setup script for the dynamic solution system.
This script automates the entire setup process.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies. Please install manually:")
        print("pip install beautifulsoup4 lxml")
        return False
    
    return True

def create_solutions_directory():
    """Create the solutions directory."""
    solutions_dir = Path('solutions')
    solutions_dir.mkdir(exist_ok=True)
    print(f"✅ Created solutions directory: {solutions_dir}")
    return True

def extract_solutions():
    """Extract solutions from existing HTML files."""
    if not run_command("python extract_solutions.py", "Extracting solutions from HTML files"):
        return False
    
    return True

def convert_html_files():
    """Convert HTML files to use dynamic template."""
    if not run_command("python convert_to_dynamic.py", "Converting HTML files to dynamic template"):
        return False
    
    return True

def test_system():
    """Test the dynamic system."""
    if not run_command("python test_dynamic_system.py", "Testing dynamic system"):
        return False
    
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Dynamic Solution System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create solutions directory
    if not create_solutions_directory():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Extract solutions
    if not extract_solutions():
        print("❌ Failed to extract solutions. Please check the errors above.")
        sys.exit(1)
    
    # Convert HTML files
    if not convert_html_files():
        print("❌ Failed to convert HTML files. Please check the errors above.")
        sys.exit(1)
    
    # Test the system
    if not test_system():
        print("❌ System test failed. Please check the errors above.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 What was accomplished:")
    print("1. ✅ Created solutions directory")
    print("2. ✅ Installed required dependencies")
    print("3. ✅ Extracted solutions from HTML files")
    print("4. ✅ Converted HTML files to dynamic template")
    print("5. ✅ Tested the system")
    
    print("\n🚀 Next steps:")
    print("1. Open any problem HTML file in a browser")
    print("2. Verify that content loads from JSON files")
    print("3. Edit a JSON file in the solutions/ directory")
    print("4. Refresh the page to see your changes")
    
    print("\n💡 Benefits:")
    print("- Developers can now edit solutions without touching HTML")
    print("- All changes are automatically reflected on the website")
    print("- Simple JSON format for easy editing")
    print("- No HTML knowledge required")
    
    print("\n📚 Documentation:")
    print("- Read DYNAMIC_SOLUTIONS_README.md for detailed instructions")
    print("- Check the solutions/ directory for example JSON files")
    print("- Use test_dynamic_system.py to verify everything works")

if __name__ == "__main__":
    main() 