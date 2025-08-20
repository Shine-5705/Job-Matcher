#!/usr/bin/env python
"""
Test script for ATS Resume Matcher Web Application
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'pandas', 
        'numpy',
        'scikit-learn',
        'nltk',
        'PyPDF2',
        'textstat',
        'textblob'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("✅ All packages installed successfully!")
    else:
        print("✅ All required packages are installed!")

def setup_nltk_data():
    """Download required NLTK data"""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        print("✅ NLTK data already available")
    except LookupError:
        print("📦 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully!")

def test_ats_parser():
    """Test the ATS parser functionality"""
    try:
        from ats_parser import SmartATSMatcher
        matcher = SmartATSMatcher()
        print("✅ ATS Parser loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ ATS Parser error: {e}")
        return False

def start_web_app():
    """Start the Flask web application"""
    print("\n🚀 Starting ATS Resume Matcher Web Application...")
    print("📍 The application will be available at: http://localhost:5000")
    print("🔗 Opening browser automatically...")
    
    # Wait a moment then open browser
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Application error: {e}")

def main():
    """Main test function"""
    print("🔍 ATS Resume Matcher - Web Application Setup")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check for required files
    required_files = ['app.py', 'ats_parser.py', 'requirements.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return
    
    print("✅ All required files found!")
    
    # Step 1: Check dependencies
    print("\n1️⃣ Checking dependencies...")
    check_dependencies()
    
    # Step 2: Setup NLTK data
    print("\n2️⃣ Setting up NLTK data...")
    setup_nltk_data()
    
    # Step 3: Test ATS parser
    print("\n3️⃣ Testing ATS parser...")
    if not test_ats_parser():
        print("❌ Cannot proceed with web application due to ATS parser issues")
        return
    
    # Step 4: Start web application
    print("\n4️⃣ Starting web application...")
    start_web_app()

if __name__ == "__main__":
    main()
