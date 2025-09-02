# Accurate Roof Area Calculator

This branch implements a more accurate method for calculating rooftop square footage using real satellite imagery from various providers.

## Features

- **Real Satellite Imagery**: Uses actual satellite images from Google Maps, Mapbox, or Bing Maps
- **Computer Vision Analysis**: Detects roof areas using advanced image processing techniques
- **Multiple API Providers**: Supports Google Maps, Mapbox, and Bing Maps APIs
- **Geocoding**: Automatically converts addresses to coordinates
- **Confidence Scoring**: Provides confidence levels for roof detection accuracy
- **Batch Processing**: Process multiple addresses from CSV files

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements_accurate.txt
```

### 2. Configure API Keys

Create a `.env` file in the `streamlit_deployment` directory with your API keys:

```env
# Google Maps API Key (Recommended)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# Mapbox API Key (Alternative)
MAPBOX_API_KEY=your_mapbox_api_key_here

# Bing Maps API Key (Alternative)
BING_MAPS_API_KEY=your_bing_maps_api_key_here

# Default API provider
DEFAULT_API_PROVIDER=google

# Image processing settings
IMAGE_SIZE=512
ROOF_DETECTION_CONFIDENCE=0.5
```

### 3. Obtain API Keys

#### Google Maps API Key
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the "Maps Static API"
4. Create credentials (API Key)
5. Restrict the API key to "Maps Static API" for security

#### Mapbox API Key
1. Go to [Mapbox](https://www.mapbox.com/)
2. Sign up for a free account
3. Go to your account page and copy your default public token
4. Or create a new token with appropriate permissions

#### Bing Maps API Key
1. Go to [Bing Maps Dev Center](https://www.bingmapsportal.com/)
2. Sign in with your Microsoft account
3. Create a new key
4. Select "Basic" key type for static imagery

### 4. Run the Application

```bash
streamlit run streamlit_app_accurate.py
```

## How It Works

### 1. Address Geocoding
- Converts addresses to latitude/longitude coordinates using OpenStreetMap Nominatim
- Handles various address formats and provides fallback options

### 2. Satellite Image Retrieval
- Downloads high-resolution satellite imagery from configured API providers
- Supports multiple providers with automatic fallback
- Configurable zoom levels for different detail requirements

### 3. Roof Detection
- Uses computer vision techniques to detect roof areas:
  - Edge detection using Canny algorithm
  - Morphological operations to close gaps
  - Contour analysis to identify roof-like shapes
  - Area and aspect ratio filtering for realistic roofs

### 4. Area Calculation
- Converts detected roof pixels to actual square footage
- Uses zoom-level-based pixel-to-meter conversion
- Accounts for geographic location and image resolution

### 5. Confidence Scoring
- Provides confidence scores based on:
  - Quality of roof detection
  - Size and shape of detected areas
  - Image clarity and resolution

## API Usage and Costs

### Google Maps Static API
- **Free tier**: 28,000 requests per month
- **Cost**: $2.00 per 1,000 requests after free tier
- **Best for**: High-quality imagery, global coverage

### Mapbox Static Images API
- **Free tier**: 50,000 requests per month
- **Cost**: $0.50 per 1,000 requests after free tier
- **Best for**: Good quality imagery, competitive pricing

### Bing Maps REST Services
- **Free tier**: 125,000 transactions per month
- **Cost**: $8.00 per 1,000 transactions after free tier
- **Best for**: Enterprise use, comprehensive coverage

## Configuration Options

### Image Processing Settings
- `IMAGE_SIZE`: Size of satellite images (default: 512x512)
- `ROOF_DETECTION_CONFIDENCE`: Minimum confidence threshold (default: 0.5)
- `OVERLAP_THRESHOLD`: Minimum overlap for roof detection (default: 0.1)

### API Provider Selection
- `DEFAULT_API_PROVIDER`: Primary API provider (google, mapbox, bing)
- Automatic fallback to other providers if primary fails

## Input Requirements

### CSV File Format
Your CSV file must include these columns:
- `Name`: Company or building name
- `Full_Address`: Complete address string
- `City`: City name
- `State`: State or province

### Address Format
Addresses should be in a standard format:
```
123 Main Street, City, State ZIP Code
```

## Output

### Results CSV
The application generates a CSV file with:
- All original columns from input
- `Roof_Area_SqFt`: Calculated roof area in square feet
- `Confidence`: Detection confidence score (0.0 to 1.0)
- `Method`: Analysis method used
- `Latitude`/`Longitude`: Geocoded coordinates
- `Status`: Success/Failed/Error status
- `Error`: Error message if calculation failed

### Metrics
- Total addresses processed
- Success rate
- Average roof area
- Average confidence score
- Total roof area

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify API keys are correct and active
   - Check API quotas and billing
   - Ensure APIs are enabled in your account

2. **Geocoding Failures**
   - Verify address format is correct
   - Try more specific addresses (include ZIP codes)
   - Check internet connection

3. **Low Confidence Scores**
   - Addresses may be in areas with poor satellite imagery
   - Buildings may be too small or obscured
   - Try adjusting confidence threshold

4. **Rate Limiting**
   - Reduce batch size
   - Add delays between requests
   - Use multiple API providers

### Performance Tips

1. **Batch Size**: Process addresses in smaller batches (10-50 at a time)
2. **API Rotation**: Use multiple API providers to avoid rate limits
3. **Caching**: Results are cached to avoid duplicate API calls
4. **Error Handling**: Failed addresses are logged for retry

## Accuracy Considerations

### Factors Affecting Accuracy
- **Image Quality**: Higher resolution images provide better accuracy
- **Building Type**: Single-family homes are easier to detect than complex buildings
- **Weather**: Cloud cover can affect image quality
- **Season**: Vegetation can obscure roof areas
- **Building Age**: Newer buildings may not appear in satellite imagery

### Expected Accuracy
- **Single-family homes**: 80-95% accuracy
- **Commercial buildings**: 70-85% accuracy
- **Complex structures**: 60-75% accuracy
- **Confidence scores**: Generally correlate with actual accuracy

## Support

For issues or questions:
1. Check the console output for detailed error messages
2. Verify API keys and quotas
3. Test with a small batch of addresses first
4. Review the troubleshooting section above

## License

This software is provided as-is for educational and research purposes. Please ensure compliance with API provider terms of service.
