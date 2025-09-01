# 🚀 Model Upload Guide for Streamlit Cloud

## Problem Solved
Streamlit Cloud clones your repository fresh on every deployment, which means any files you upload through the Streamlit Cloud interface get overwritten when the repository is cloned again.

## ✅ Solution: Model Upload Feature

The app now includes a **Model Upload** feature that allows you to upload your trained model file directly through the web interface. This model will persist during your session and won't be lost when Streamlit Cloud redeploys.

## 📋 How to Use

### Step 1: Access the App
1. Go to your Streamlit Cloud app URL
2. Wait for the app to load

### Step 2: Upload Your Model
1. Look at the **sidebar on the left**
2. Find the **"📤 Upload Model"** section
3. Click **"Choose model file"**
4. Select your trained PyTorch Lightning checkpoint file (`.ckpt` extension)
5. Wait for the model to load

### Step 3: Verify Model Status
- **✅ AI Mode Active**: Your model is loaded and ready for real predictions
- **⚠️ Demo Mode Active**: No model loaded, using synthetic predictions

## 🔧 Technical Details

### Supported Model Format
- **File Type**: PyTorch Lightning checkpoint (`.ckpt`)
- **Architecture**: UNet + ResNet34
- **Input Size**: 512x512 pixels
- **Classes**: 1 (binary segmentation)

### Model Loading Process
1. The app creates the model architecture
2. Loads your checkpoint file
3. Applies the trained weights
4. Stores the model in session state for persistence

### Session Persistence
- The uploaded model stays loaded during your entire session
- No need to re-upload for each CSV file
- Model is automatically used for all predictions

## 🎯 Benefits

1. **No Repository Cloning Issues**: Model uploads aren't affected by Streamlit Cloud's repository cloning
2. **Session Persistence**: Model stays loaded throughout your session
3. **Easy to Use**: Simple file upload interface
4. **Real AI Predictions**: Get actual AI predictions instead of demo mode
5. **No File Size Limits**: Upload large model files without repository size concerns

## 🚨 Troubleshooting

### Model Won't Load
- Ensure your model file is a valid PyTorch Lightning checkpoint
- Check that the model architecture matches (UNet + ResNet34)
- Try refreshing the page and uploading again

### Demo Mode Still Active
- Check the sidebar for any error messages
- Verify your model file format
- Make sure the upload completed successfully

### App Performance
- Large model files may take a moment to load
- The app will show a loading spinner during model upload
- Once loaded, predictions will be fast

## 📝 Example Workflow

1. **Open the app** → See "Demo Mode Active"
2. **Upload model** → Click "Choose model file" in sidebar
3. **Select .ckpt file** → Wait for "✅ Model uploaded and loaded successfully!"
4. **Upload CSV** → Process your address data
5. **Generate predictions** → Get real AI predictions for all addresses
6. **Download results** → Get CSV with actual AI predictions

## 🔄 Migration from Old Method

If you previously uploaded model files to the `checkpoints/` folder in Streamlit Cloud:
1. **Download** your model file from the old location
2. **Upload** it using the new sidebar feature
3. **Delete** the old model file from the repository (already done in this update)

This new method is more reliable and user-friendly!
