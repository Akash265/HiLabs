#!/usr/bin/env python
"""
Setup script to create project directory structure
Run this to set up all directories and create __init__.py files
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    
    # Define directory structure
    directories = [
        "Contracts/TN",
        "Contracts/WA",
        "Standard_Templates",
        "src/utils",
        "src/data_processing",
        "src/classification",
        "src/reporting",
        "output",
        "cache",
        "logs",
        "temp",
        "tests"
    ]
    
    # Create directories
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "src/utils/__init__.py",
        "src/data_processing/__init__.py",
        "src/classification/__init__.py",
        "src/reporting/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Created file: {init_file}")
    
    print("\n" + "="*50)
    print("PROJECT STRUCTURE CREATED SUCCESSFULLY!")
    print("="*50)
    print("\nNext steps:")
    print("1. Copy all the generated Python files to their respective directories")
    print("2. Place your contract PDFs in Contracts/TN/ and Contracts/WA/")
    print("3. Place template PDFs in Standard_Templates/")
    print("4. Place Attribute_Dictionary.xlsx in the root directory")
    print("5. Run: pip install -r requirements.txt")
    print("6. Run: python main.py")
    print("\n" + "="*50)

if __name__ == "__main__":
    create_project_structure()