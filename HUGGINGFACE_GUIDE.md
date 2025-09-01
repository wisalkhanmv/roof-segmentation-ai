# 🤗 Hugging Face Model Hosting Guide

## Why Hugging Face?

Hugging Face is the **best solution** for hosting large ML models because:

- ✅ **No file size limits** (supports models up to 100GB+)
- ✅ **Fast, reliable downloads** with CDN
- ✅ **Version control** for model iterations
- ✅ **Easy sharing** with public/private repositories
- ✅ **Built for ML** - specifically designed for model hosting
- ✅ **Free hosting** for public models

## 🚀 Quick Start

### Step 1: Upload Your Model to Hugging Face

#### Option A: Using Hugging Face CLI (Recommended)

1. **Install Hugging Face CLI**:
   ```bash
   pip install huggingface_hub
   ```

2. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```

3. **Create a new model repository**:
   ```bash
   huggingface-cli repo create roof-segmentation-model
   ```

4. **Upload your model**:
   ```bash
   huggingface-cli upload wisalkhanmv/roof-segmentation-model checkpoints/best_model.ckpt
   ```

#### Option B: Using the Web Interface

1. Go to [huggingface.co](https://huggingface.co)
2. Click "New Model"
3. Choose repository name (e.g., `roof-segmentation-model`)
4. Set visibility (Public or Private)
5. Upload your `best_model.ckpt` file
6. Add a README with model description

### Step 2: Use in Your Streamlit App

1. **Open your Streamlit app**
2. **Go to sidebar** → "📤 Load Model" → "🤗 HF Hub" tab
3. **Enter your model ID**: `wisalkhanmv/roof-segmentation-model`
4. **Select the model file** from the dropdown
5. **Click "📥 Download from Hugging Face"**
6. **Wait for download** (progress bar will show)
7. **Start predicting!** 🎉

## 📋 Detailed Instructions

### Creating a Model Repository

#### 1. Repository Structure
```
roof-segmentation-model/
├── README.md              # Model description
├── best_model.ckpt        # Your trained model
├── config.json            # Model configuration (optional)
└── .gitattributes         # File attributes (auto-generated)
```

#### 2. README.md Template
```markdown
---
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
- **File Size**: 280MB

## Usage
Upload this model to the Roof Segmentation AI Streamlit app for real AI predictions.

## Performance
- Validation Loss: 36,356,431,872
- MAE: ~154,752 sq ft
- MAPE: ~1.03%
```

### Uploading Your Model

#### Using Python Script
```python
from huggingface_hub import HfApi

# Initialize API
api = HfApi()

# Upload model file
api.upload_file(
    path_or_fileobj="checkpoints/best_model.ckpt",
    path_in_repo="best_model.ckpt",
    repo_id="wisalkhanmv/roof-segmentation-model",
    repo_type="model"
)
```

#### Using Git (Advanced)
```bash
# Clone the repository
git clone https://huggingface.co/wisalkhanmv/roof-segmentation-model

# Copy your model file
cp checkpoints/best_model.ckpt roof-segmentation-model/

# Commit and push
cd roof-segmentation-model
git add .
git commit -m "Add trained roof segmentation model"
git push
```

## 🔧 Advanced Features

### Version Control
Hugging Face supports model versioning:
```bash
# Tag a specific version
huggingface-cli upload wisalkhanmv/roof-segmentation-model@v1.0 checkpoints/best_model.ckpt

# Download specific version
huggingface-cli download wisalkhanmv/roof-segmentation-model@v1.0 best_model.ckpt
```

### Private Models
For private models, you'll need authentication:
```python
from huggingface_hub import login
login("your_token_here")
```

### Multiple Model Files
You can upload multiple files to the same repository:
```
roof-segmentation-model/
├── best_model.ckpt
├── best_model_quantized.ckpt
├── config.json
└── training_logs.txt
```

## 🎯 Benefits Over Other Methods

| Feature | Hugging Face | Google Drive | Dropbox | GitHub |
|---------|-------------|--------------|---------|---------|
| File Size Limit | 100GB+ | 15GB | 2GB | 100MB |
| Download Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Reliability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| ML-Specific | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐ |
| Version Control | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| Free Hosting | ✅ | ✅ | ✅ | ✅ |

## 🚨 Troubleshooting

### Model Won't Download
- **Check repository name**: Ensure it matches exactly
- **Verify file exists**: Check the repository on huggingface.co
- **Authentication**: For private models, ensure you're logged in
- **Network issues**: Try again, large files may timeout

### Import Error
```bash
# Install huggingface_hub
pip install huggingface_hub

# Or add to requirements.txt
echo "huggingface_hub>=0.16.0" >> requirements.txt
```

### Slow Download
- **Large file**: 280MB will take 1-3 minutes
- **Progress tracking**: Watch the progress bar in the app
- **CDN**: Hugging Face uses global CDN for fast downloads

## 📝 Example Workflow

1. **Train your model** → `best_model.ckpt` (280MB)
2. **Create HF repository** → `wisalkhanmv/roof-segmentation-model`
3. **Upload model** → `huggingface-cli upload ...`
4. **Deploy Streamlit app** → No large files in repository
5. **Load model in app** → Enter model ID, select file, download
6. **Start predicting** → Real AI predictions! 🎉

## 🔗 Useful Links

- [Hugging Face Hub](https://huggingface.co)
- [Model Hub Documentation](https://huggingface.co/docs/hub/index)
- [CLI Documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [Python API Documentation](https://huggingface.co/docs/huggingface_hub/guides/upload)

This is the **recommended approach** for hosting large ML models! 🚀
