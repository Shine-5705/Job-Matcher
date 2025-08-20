#!/usr/bin/env python3
"""
Build script for Render deployment
Downloads required NLTK data during build process
"""

import nltk
import ssl
import os

def download_nltk_data():
    """Download required NLTK data"""
    try:
        # Handle SSL certificate issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        print("📦 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    
    return True

def verify_installation():
    """Verify that all required packages are working"""
    try:
        # Test imports
        import flask
        import pandas
        import numpy
        import sklearn
        import textstat
        import textblob
        from ats_parser import SmartATSMatcher
        
        print("✅ All packages imported successfully!")
        
        # Test ATS parser
        matcher = SmartATSMatcher()
        print("✅ ATS Parser initialized successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Setting up ATS Resume Matcher for production...")
    
    # Download NLTK data
    if not download_nltk_data():
        exit(1)
    
    # Verify installation
    if not verify_installation():
        exit(1)
    
    print("🎉 Build completed successfully!")
