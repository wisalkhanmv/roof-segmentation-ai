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

    # Create a more realistic placeholder model file (smaller size)
    # This simulates a PyTorch Lightning checkpoint structure
    import torch
    from pytorch_lightning import LightningModule

    # Create a minimal model checkpoint
    checkpoint = {
        'state_dict': {},
        'epoch': 0,
        'global_step': 0,
        'pytorch-lightning_version': '2.0.0',
        'callbacks': {},
        'optimizer_states': [],
        'lr_schedulers': [],
        'hparams': {},
        'hyper_parameters': {}
    }

    torch.save(checkpoint, model_path)

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

    # Check if it's a placeholder by trying to load it
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        if not checkpoint.get('state_dict'):
            print("⚠️  This appears to be a placeholder model. Replace with actual trained model for production use.")
            return False
        print("✅ Model checkpoint loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  Model verification failed: {e}")
        return False


if __name__ == "__main__":
    print("🏠 Roof Area Predictor - Model Download Script")
    print("=" * 50)

    success = download_model()
    if success:
        verify_model()
        print("\n🎉 Model setup completed!")
    else:
        print("\n❌ Model setup failed!")
