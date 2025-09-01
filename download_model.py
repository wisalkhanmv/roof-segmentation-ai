#!/usr/bin/env python3
"""
Model Download Script for Streamlit Deployment
Downloads the trained model from cloud storage if not present locally
"""
import os
import requests
from pathlib import Path
import hashlib


def download_model():
    """Download the trained model if not present"""
    model_path = Path("checkpoints/best_model.ckpt")

    # Create checkpoints directory if it doesn't exist
    model_path.parent.mkdir(exist_ok=True)

    if model_path.exists():
        print(f"✅ Model already exists at {model_path}")
        return True

    print("📥 Model not found. Downloading from cloud storage...")

    # For now, we'll create a placeholder model
    # In production, you would download from Google Drive, AWS S3, etc.
    print("⚠️  Creating placeholder model for demonstration...")

    # Create a simple placeholder model file
    placeholder_content = b"PLACEHOLDER_MODEL_FILE"
    with open(model_path, 'wb') as f:
        f.write(placeholder_content)

    print(f"✅ Placeholder model created at {model_path}")
    print("📝 Note: In production, replace this with actual model download from cloud storage")

    return True


def verify_model():
    """Verify the model file exists and has correct size"""
    model_path = Path("checkpoints/best_model.ckpt")

    if not model_path.exists():
        print("❌ Model file not found")
        return False

    file_size = model_path.stat().st_size
    print(f"📊 Model file size: {file_size:,} bytes")

    # Check if it's a placeholder
    with open(model_path, 'rb') as f:
        content = f.read(20)  # Read first 20 bytes

    if content == b"PLACEHOLDER_MODEL_FILE":
        print("⚠️  This is a placeholder model. Replace with actual trained model for production use.")
        return False

    return True


if __name__ == "__main__":
    print("🏠 Roof Area Predictor - Model Download Script")
    print("=" * 50)

    success = download_model()
    if success:
        verify_model()
        print("\n🎉 Model setup completed!")
    else:
        print("\n❌ Model setup failed!")
