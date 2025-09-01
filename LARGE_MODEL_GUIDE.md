# 🚀 Large Model File Guide (>200MB)

## Problem
Streamlit Cloud has a **200MB file upload limit**. Your model file is **280MB**, which exceeds this limit.

## ✅ Solution: Cloud Storage Integration

The app now supports downloading large model files from cloud storage services.

## 📋 Step-by-Step Instructions

### Method 1: Google Drive (Recommended)

#### Step 1: Upload to Google Drive
1. Go to [drive.google.com](https://drive.google.com)
2. Upload your `best_model.ckpt` file
3. Right-click the file → "Share" → "Copy link"

#### Step 2: Create Direct Download Link
1. The link will look like: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
2. Replace it with: `https://drive.google.com/uc?export=download&id=FILE_ID`
3. **Example**: If your FILE_ID is `1ABC123xyz`, use:
   ```
   https://drive.google.com/uc?export=download&id=1ABC123xyz
   ```

#### Step 3: Use in App
1. Open your Streamlit app
2. Go to sidebar → "📤 Load Model" → "☁️ Cloud URL" tab
3. Click "🔗 Load from Cloud Storage"
4. Paste your direct download link
5. Wait for download to complete

### Method 2: Dropbox

#### Step 1: Upload to Dropbox
1. Go to [dropbox.com](https://dropbox.com)
2. Upload your `best_model.ckpt` file
3. Right-click → "Share" → "Copy link"

#### Step 2: Create Direct Download Link
1. The link will look like: `https://www.dropbox.com/s/FILE_ID/filename.ckpt?dl=0`
2. Change `dl=0` to `dl=1` for direct download:
   ```
   https://www.dropbox.com/s/FILE_ID/filename.ckpt?dl=1
   ```

### Method 3: GitHub Releases

#### Step 1: Create GitHub Release
1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Upload your `best_model.ckpt` file
4. Publish the release

#### Step 2: Get Download Link
1. After publishing, click on your model file
2. Copy the download URL (looks like):
   ```
   https://github.com/username/repo/releases/download/v1.0/best_model.ckpt
   ```

## 🔧 Alternative: Model Compression

If you prefer to keep using direct upload, you can compress your model:

### Option A: Quantization
```python
# Load your model
model = YourModel()
model.load_state_dict(torch.load('best_model.ckpt'))

# Quantize to reduce size
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'best_model_quantized.ckpt')
```

### Option B: Remove Unnecessary Data
```python
# Load checkpoint
checkpoint = torch.load('best_model.ckpt', map_location='cpu')

# Keep only state_dict (remove optimizer states, etc.)
torch.save(checkpoint['state_dict'], 'best_model_clean.ckpt')
```

## 📊 File Size Comparison

| Method | Original Size | Compressed Size | Upload Method |
|--------|---------------|-----------------|---------------|
| Original | 280MB | - | ❌ Too large |
| Quantized | 280MB | ~140MB | ✅ Direct upload |
| Clean state_dict | 280MB | ~200MB | ✅ Direct upload |
| Cloud storage | 280MB | 280MB | ✅ Cloud download |

## 🚨 Troubleshooting

### Download Fails
- **Check URL format**: Ensure it's a direct download link
- **File permissions**: Make sure the file is publicly accessible
- **Network issues**: Try again, large files may timeout

### Model Won't Load
- **Architecture mismatch**: Ensure model matches UNet+ResNet34
- **File corruption**: Re-upload to cloud storage
- **Memory issues**: Close other browser tabs

### Slow Download
- **Large file**: 280MB will take 2-5 minutes depending on connection
- **Progress bar**: Watch the download progress in the app
- **Patience**: Don't refresh during download

## 🎯 Recommended Workflow

1. **Upload to Google Drive** (easiest method)
2. **Get direct download link**
3. **Use Cloud URL tab** in the app
4. **Download once per session** (model stays loaded)
5. **Process multiple CSVs** with the same model

## 📝 Example URLs

### Google Drive
```
https://drive.google.com/uc?export=download&id=1ABC123xyz789
```

### Dropbox
```
https://www.dropbox.com/s/abc123def456/best_model.ckpt?dl=1
```

### GitHub Releases
```
https://github.com/username/repo/releases/download/v1.0/best_model.ckpt
```

This method allows you to use your full 280MB model without any size restrictions!
