# 🏠 Address-Based Roof Area Predictor

AI-powered roof square footage prediction from addresses using aerial imagery and deep learning.

## 🎯 Features

- **Address Input**: Upload CSV files with addresses
- **AI Prediction**: Uses trained UNet+ResNet34 model to predict roof areas
- **Geocoding**: Automatic address-to-coordinates conversion
- **Aerial Imagery**: Downloads satellite imagery for analysis
- **CSV Export**: Download results with predictions

## 🚀 Quick Start

### Local Development

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**:

   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open browser**: Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub**:

   ```bash
   git add .
   git commit -m "Add roof area predictor app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path: `streamlit_app.py`
   - Deploy!

## 🎯 Demo Mode

This deployment runs in **Demo Mode** with synthetic predictions. For production use:

1. **Add Your Model**: Place your trained `best_model.ckpt` file in the `checkpoints/` folder
2. **Redeploy**: The app will automatically use real AI predictions
3. **Model Format**: Use the model trained with `train_roof_area_predictor.py`

## 📁 Project Structure

```
streamlit_deployment/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── checkpoints/             # Trained model files
│   └── best_model.ckpt
├── models/                  # Model architecture definitions
├── datasets/                # Dataset loading utilities
├── utils/                   # Utility functions
├── config.py               # Configuration settings
└── README.md               # This file
```

## 🧠 Model Information

- **Architecture**: UNet with ResNet34 backbone
- **Training Data**: INRIA Aerial Image Dataset (451 images)
- **Input**: 512x512 aerial imagery
- **Output**: Roof area in square feet
- **Performance**:
  - Validation Loss: 36,356,431,872
  - MAE: ~154,752 sq ft
  - MAPE: ~1.03%

## 📊 Input Format

Upload a CSV file with these columns:

- `Name`: Company/Property name
- `Full_Address`: Complete address
- `City`: City name
- `State`: State abbreviation
- `Roof 10k` (optional): Existing roof area in 10k units

## 📈 Output Format

The app generates a CSV with:

- `Address`: Original address
- `Latitude`: Geocoded latitude
- `Longitude`: Geocoded longitude
- `Predicted_Roof_Area_SqFt`: AI-predicted roof area
- `Predicted_Roof_Area_Formatted`: Formatted area with commas

## 🔧 Configuration

Key settings in `config.py`:

- Model architecture and backbone
- Image processing parameters
- Training hyperparameters

## 🛠️ Development

### Training the Model

To retrain the model:

```bash
python train_roof_area_predictor.py --epochs 50 --batch-size 4 --lr 1e-4
```

### Model Performance

The current model was trained for 16 epochs with early stopping:

- Best epoch: 9
- Validation loss: 36,356,431,872
- Training samples: 72
- Validation samples: 19

## 🌐 Deployment Notes

- **Streamlit Cloud**: Recommended for easy deployment
- **Model Size**: ~97MB (compressed)
- **Memory**: Requires ~2GB RAM
- **CPU**: Compatible with CPU-only environments

## 📝 License

This project is for educational and commercial use.

## 🤝 Support

For issues or questions, please check the main repository documentation.
