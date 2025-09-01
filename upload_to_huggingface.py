#!/usr/bin/env python3
"""
Upload Model to Hugging Face Hub
Simple script to upload your trained model to Hugging Face
"""

import os
import sys
from pathlib import Path

def upload_to_huggingface():
    """Upload model to Hugging Face Hub"""
    
    print("🤗 Hugging Face Model Upload Tool")
    print("=" * 40)
    
    # Check if model exists
    model_path = Path("../checkpoints/best_model.ckpt")
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("💡 Please ensure your model file is in the checkpoints/ directory")
        return False
    
    # Get file size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"📁 Model file: {model_path}")
    print(f"📊 File size: {size_mb:.1f} MB")
    
    # Get model repository name
    repo_name = input("\n🏷️  Enter your Hugging Face username: ").strip()
    model_name = input("📦 Enter model name (e.g., roof-segmentation-model): ").strip()
    
    if not repo_name or not model_name:
        print("❌ Username and model name are required")
        return False
    
    repo_id = f"{repo_name}/{model_name}"
    print(f"\n🎯 Repository ID: {repo_id}")
    
    # Confirm upload
    confirm = input("\n❓ Proceed with upload? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Upload cancelled")
        return False
    
    try:
        # Import huggingface_hub
        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            print("❌ huggingface_hub not installed")
            print("💡 Install it with: pip install huggingface_hub")
            return False
        
        # Initialize API
        api = HfApi()
        
        # Check if user is logged in
        try:
            user = api.whoami()
            print(f"✅ Logged in as: {user}")
        except Exception:
            print("❌ Not logged in to Hugging Face")
            print("💡 Run: huggingface-cli login")
            return False
        
        # Create repository
        print(f"\n📦 Creating repository: {repo_id}")
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
            print("✅ Repository created/verified")
        except Exception as e:
            print(f"❌ Error creating repository: {e}")
            return False
        
        # Upload model file
        print(f"\n📤 Uploading model file...")
        try:
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo="best_model.ckpt",
                repo_id=repo_id,
                repo_type="model"
            )
            print("✅ Model uploaded successfully!")
        except Exception as e:
            print(f"❌ Error uploading model: {e}")
            return False
        
        # Create README
        readme_content = f"""---
language: en
tags:
- computer-vision
- image-segmentation
- roof-detection
- pytorch
---

# Roof Segmentation Model

This model predicts roof areas from aerial imagery using UNet architecture with ResNet34 backbone.

## Model Details
- **Architecture**: UNet + ResNet34
- **Input Size**: 512x512 pixels
- **Output**: Roof area in square feet
- **Training Data**: INRIA Aerial Image Dataset
- **File Size**: {size_mb:.1f}MB

## Usage
Upload this model to the Roof Segmentation AI Streamlit app for real AI predictions.

## Performance
- Validation Loss: 36,356,431,872
- MAE: ~154,752 sq ft
- MAPE: ~1.03%

## How to Use
1. Open the Roof Segmentation AI Streamlit app
2. Go to sidebar → "📤 Load Model" → "🤗 HF Hub" tab
3. Enter model ID: `{repo_id}`
4. Select "best_model.ckpt" from dropdown
5. Click "📥 Download from Hugging Face"
6. Start predicting! 🎉
"""
        
        print(f"\n📝 Creating README...")
        try:
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model"
            )
            print("✅ README created successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Could not create README: {e}")
        
        # Success message
        print(f"\n🎉 Model uploaded successfully!")
        print(f"🔗 Repository: https://huggingface.co/{repo_id}")
        print(f"📁 Model file: best_model.ckpt")
        print(f"\n💡 Next steps:")
        print(f"1. Deploy your Streamlit app")
        print(f"2. Use model ID: {repo_id}")
        print(f"3. Select 'best_model.ckpt' in the app")
        print(f"4. Start making predictions! 🚀")
        
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Main function"""
    success = upload_to_huggingface()
    
    if success:
        print(f"\n✅ Upload completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
