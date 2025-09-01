#!/usr/bin/env python3
"""
Model Compression Script
Reduces model file size to fit within Streamlit Cloud's 200MB limit
"""

import torch
import os
from pathlib import Path


def compress_model(input_path, output_path=None):
    """Compress model by removing unnecessary data"""
    
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_compressed{input_file.suffix}"
    
    print(f"📥 Loading model from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    print(f"📊 Original size: {original_size:.1f} MB")
    
    # Extract only the state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("✅ Found state_dict in checkpoint")
    else:
        state_dict = checkpoint
        print("⚠️ No state_dict found, using checkpoint directly")
    
    # Save only the state_dict
    torch.save(state_dict, output_path)
    
    compressed_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"📊 Compressed size: {compressed_size:.1f} MB")
    print(f"📉 Size reduction: {((original_size - compressed_size) / original_size * 100):.1f}%")
    
    if compressed_size <= 200:
        print("✅ Compressed model fits within 200MB limit!")
        print(f"💾 Saved to: {output_path}")
        return True
    else:
        print("❌ Still too large for direct upload")
        print("💡 Use cloud storage method instead")
        return False


def quantize_model(input_path, output_path=None):
    """Quantize model to reduce size further"""
    
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_quantized{input_file.suffix}"
    
    print(f"🔧 Quantizing model...")
    
    # Load model architecture (you'll need to import your model)
    try:
        from models.segmentation_models import create_model
        
        model_config = {
            'model_name': 'unet',
            'backbone': 'resnet34',
            'classes': 1,
            'encoder_weights': 'imagenet'
        }
        model = create_model(model_config)
        
        # Load state dict
        state_dict = torch.load(input_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        # Quantize
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), output_path)
        
        original_size = os.path.getsize(input_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"📊 Original: {original_size:.1f} MB")
        print(f"📊 Quantized: {quantized_size:.1f} MB")
        print(f"📉 Reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
        
        if quantized_size <= 200:
            print("✅ Quantized model fits within 200MB limit!")
            return True
        else:
            print("❌ Still too large, use cloud storage")
            return False
            
    except ImportError:
        print("❌ Could not import model architecture")
        print("💡 Use cloud storage method instead")
        return False


def main():
    """Main compression function"""
    print("🔧 Model Compression Tool")
    print("=" * 40)
    
    # Check if model exists
    model_path = "checkpoints/best_model.ckpt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("💡 Please place your model file in the checkpoints/ directory")
        return
    
    # Get file size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📁 Model file: {model_path}")
    print(f"📊 File size: {size_mb:.1f} MB")
    
    if size_mb <= 200:
        print("✅ Model already fits within 200MB limit!")
        return
    
    print("\n🔧 Attempting compression...")
    
    # Try basic compression first
    if compress_model(model_path):
        print("\n🎉 Success! You can now use direct upload.")
        return
    
    # Try quantization
    print("\n🔧 Attempting quantization...")
    if quantize_model(model_path):
        print("\n🎉 Success! You can now use direct upload.")
        return
    
    # If all else fails, recommend cloud storage
    print("\n💡 Recommendation: Use Cloud Storage")
    print("Since compression didn't work, use the cloud storage method:")
    print("1. Upload your model to Google Drive")
    print("2. Get a direct download link")
    print("3. Use the '☁️ Cloud URL' tab in the app")
    print("\n📚 See LARGE_MODEL_GUIDE.md for detailed instructions")


if __name__ == "__main__":
    main()
