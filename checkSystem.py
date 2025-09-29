#!/usr/bin/env python
"""
System check and verification script
"""

import sys
import os
import yaml

def check_system():
    """Check system configuration and dependencies"""
    
    print("="*60)
    print("SYSTEM CHECK - HiLabs Contract Classification")
    print("="*60)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    
    # Check essential dependencies
    print("\n[Checking Dependencies]")
    dependencies = {
        'pdfplumber': 'PDF Processing',
        'pandas': 'Data Processing',
        'fuzzywuzzy': 'Text Matching',
        'streamlit': 'Web Interface',
        'plotly': 'Visualizations'
    }
    
    all_good = True
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"  ✓ {package}: {description}")
        except ImportError:
            print(f"  ✗ {package}: Missing - {description}")
            all_good = False
    
    # Check optional ML dependencies
    print("\n[Optional ML Dependencies]")
    ml_available = False
    try:
        import sentence_transformers
        print("  ✓ sentence-transformers: Available (Semantic models enabled)")
        ml_available = True
    except ImportError:
        print("  ℹ sentence-transformers: Not installed (Using fuzzy mode only)")
    
    # Check configuration
    print("\n[Configuration]")
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        model_type = config.get('classification', {}).get('model_type', 'unknown')
        print(f"  ✓ config.yaml exists")
        print(f"  - Model type: {model_type}")
        
        if model_type in ['semantic', 'hybrid'] and not ml_available:
            print("  ⚠ WARNING: Config uses semantic/hybrid but ML packages not available")
            print("    System will fall back to fuzzy mode")
        
        # Check weights
        weights = config.get('classification', {})
        fuzzy_weight = weights.get('fuzzy', {}).get('weight', 0)
        semantic_weight = weights.get('semantic', {}).get('weight', 0)
        exact_weight = weights.get('exact', {}).get('weight', 0)
        
        print(f"  - Weights: Fuzzy={fuzzy_weight}, Semantic={semantic_weight}, Exact={exact_weight}")
        
        # Verify weights sum to approximately 1
        total_weight = fuzzy_weight + semantic_weight + exact_weight
        if abs(total_weight - 1.0) > 0.01:
            print(f"  ⚠ WARNING: Weights sum to {total_weight}, should be 1.0")
    else:
        print("  ✗ config.yaml not found")
        all_good = False
    
    # Check directories
    print("\n[Directory Structure]")
    dirs = {
        'Contracts': 'Contract PDFs',
        'Standard Templates': 'Template PDFs',
        'output': 'Results',
        'cache': 'Cache files'
    }
    
    for dir_name, description in dirs.items():
        if os.path.exists(dir_name):
            files = len(os.listdir(dir_name))
            print(f"  ✓ {dir_name}: {files} files")
        else:
            print(f"  ✗ {dir_name}: Missing - {description}")
    
    # Performance recommendation
    print("\n[Performance Settings]")
    if ml_available:
        print("  ℹ ML models available - Processing time: 30-60 sec/contract")
        print("  Recommendation: Use 'fuzzy' mode for demos (2-5 sec/contract)")
    else:
        print("  ✓ Fuzzy-only mode - Processing time: 2-5 sec/contract")
        print("  Perfect for demos and rapid processing!")
    
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("SYSTEM READY!")
        print("Run: streamlit run app.py")
    else:
        print("ISSUES FOUND - Please install missing dependencies")
        print("Run: pip install pdfplumber pandas fuzzywuzzy streamlit plotly")
    print("="*60)

if __name__ == "__main__":
    check_system()