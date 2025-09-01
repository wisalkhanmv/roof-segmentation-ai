#!/bin/bash

# 🚀 Streamlit App Deployment Script
# This script prepares and pushes the roof area predictor app to GitHub

echo "🏠 Deploying Roof Area Predictor App to GitHub..."

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found. Please run this script from the streamlit_deployment directory."
    exit 1
fi

# Check if model file exists
if [ ! -f "checkpoints/best_model.ckpt" ]; then
    echo "❌ Error: Best model not found. Please ensure the model is copied to checkpoints/."
    exit 1
fi

echo "✅ Model file found"

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "🚀 Deploy roof area predictor app with trained model

- Added trained UNet+ResNet34 model (epoch 9, val_loss=36356431872)
- Updated Streamlit app to use best model
- Added comprehensive documentation
- Configured Git LFS for large model files
- Ready for Streamlit Cloud deployment"

# Check if remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "🌐 No remote repository configured."
    echo "Please add your GitHub repository:"
    echo "git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
    echo ""
    echo "Then run: git push -u origin main"
else
    echo "🚀 Pushing to GitHub..."
    git push origin main
fi

echo ""
echo "🎉 Deployment script completed!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://share.streamlit.io"
echo "2. Connect your GitHub repository"
echo "3. Set main file path: streamlit_app.py"
echo "4. Deploy!"
echo ""
echo "🌐 Your app will be live at: https://YOUR_APP_NAME.streamlit.app"
