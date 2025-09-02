#!/usr/bin/env python3
"""
Test script for the accurate roof calculator setup
"""

import os
import sys
from dotenv import load_dotenv

def test_environment():
    """Test environment setup"""
    print("🔧 Testing Environment Setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check API keys
    google_key = os.getenv('GOOGLE_MAPS_API_KEY')
    mapbox_key = os.getenv('MAPBOX_API_KEY')
    bing_key = os.getenv('BING_MAPS_API_KEY')
    
    print(f"Google Maps API Key: {'✅ Set' if google_key else '❌ Not set'}")
    print(f"Mapbox API Key: {'✅ Set' if mapbox_key else '❌ Not set'}")
    print(f"Bing Maps API Key: {'✅ Set' if bing_key else '❌ Not set'}")
    
    if not any([google_key, mapbox_key, bing_key]):
        print("⚠️  No API keys found! Please configure at least one API key in your .env file")
        return False
    
    return True

def test_imports():
    """Test required imports"""
    print("\n📦 Testing Required Imports...")
    
    try:
        import streamlit
        print("✅ Streamlit")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import pandas
        print("✅ Pandas")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        import numpy
        print("✅ NumPy")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import requests
        print("✅ Requests")
    except ImportError as e:
        print(f"❌ Requests: {e}")
        return False
    
    try:
        from geopy.geocoders import Nominatim
        print("✅ Geopy")
    except ImportError as e:
        print(f"❌ Geopy: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow")
    except ImportError as e:
        print(f"❌ Pillow: {e}")
        return False
    
    return True

def test_roof_calculator():
    """Test roof calculator initialization"""
    print("\n🏠 Testing Roof Calculator...")
    
    try:
        from accurate_roof_calculator import AccurateRoofCalculator
        calculator = AccurateRoofCalculator()
        print("✅ Roof Calculator initialized successfully")
        
        # Test geocoding
        print("🌍 Testing geocoding...")
        coords = calculator.geocode_address("1600 Amphitheatre Parkway, Mountain View, CA")
        if coords:
            print(f"✅ Geocoding successful: {coords}")
        else:
            print("⚠️  Geocoding failed (this might be normal if no internet connection)")
        
        return True
        
    except Exception as e:
        print(f"❌ Roof Calculator error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Accurate Roof Calculator Setup Test")
    print("=" * 50)
    
    # Test environment
    env_ok = test_environment()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test roof calculator
    calculator_ok = test_roof_calculator()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Environment: {'✅ Pass' if env_ok else '❌ Fail'}")
    print(f"Imports: {'✅ Pass' if imports_ok else '❌ Fail'}")
    print(f"Calculator: {'✅ Pass' if calculator_ok else '❌ Fail'}")
    
    if env_ok and imports_ok and calculator_ok:
        print("\n🎉 All tests passed! You're ready to use the accurate roof calculator.")
        print("\nTo run the application:")
        print("streamlit run streamlit_app_accurate.py")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        print("\nTo install missing dependencies:")
        print("pip install -r requirements_accurate.txt")

if __name__ == "__main__":
    main()
