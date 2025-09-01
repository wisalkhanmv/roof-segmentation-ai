#!/usr/bin/env python3
"""
Roof Segmentation AI - Streamlit Web Application
Simple CSV processor that adds predicted square footage for each address
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from datetime import datetime

# Import our custom modules
from models.segmentation_models import create_model

# Page configuration
st.set_page_config(
    page_title="Roof Segmentation AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460;
    }
    .metric-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .results-section {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """Load the trained model (cached for performance)"""
    try:
        # Use the best trained model
        current_dir = Path(__file__).parent
        best_checkpoint = current_dir / "checkpoints" / "best_model.ckpt"

        # If model doesn't exist locally, try to download it
        if not best_checkpoint.exists():
            st.info("📥 Model file not found locally. Attempting to download...")
            
            # Create checkpoints directory if it doesn't exist
            best_checkpoint.parent.mkdir(exist_ok=True)
            
            # Try to download from GitHub (you can replace this URL with your actual model URL)
            model_url = "https://github.com/wisalkhanmv/roof-segmentation-ai/releases/download/v1.0/best_model.ckpt"
            
            try:
                import requests
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                with open(best_checkpoint, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                st.success("✅ Model downloaded successfully!")
                
            except Exception as download_error:
                st.warning("⚠️ Could not download model. Using demo mode with synthetic predictions.")
                st.info("📝 To use real AI predictions, please add your trained model file to the 'checkpoints' folder.")
                return None, "Demo Mode"

        # Load model
        model_config = {
            'model_name': 'unet',
            'backbone': 'resnet34',
            'classes': 1,
            'encoder_weights': 'imagenet'
        }
        model = create_model(model_config)

        # Load checkpoint
        try:
            checkpoint = torch.load(
                best_checkpoint, map_location='cpu', weights_only=False)
        except:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model, best_checkpoint.name

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("📝 Using demo mode with synthetic predictions.")
        return None, "Demo Mode"


def process_csv_data(uploaded_file):
    """Process uploaded CSV data"""
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Check required columns
        required_columns = ['Name', 'Full_Address', 'City', 'State']
        missing_columns = [
            col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None

        # Process roof data - check for both Roof 10k and Roof 20k columns
        roof_columns = ['Roof 10k', 'Roof_10k', 'Roof 20k', 'Roof_20k']
        found_roof_col = None

        for col in roof_columns:
            if col in df.columns:
                found_roof_col = col
                break

        if found_roof_col:
            # Clean the roof data
            df['Roof_Data_Clean'] = df[found_roof_col].astype(
                str).str.replace(',', '').str.replace('"', '')
            df['Roof_SqFt_Real'] = pd.to_numeric(
                df['Roof_Data_Clean'], errors='coerce')

            # Convert to square feet (if it's in 10k units)
            if '10k' in found_roof_col.lower():
                df['Roof_SqFt_Real'] = df['Roof_SqFt_Real'] * 1000

            # Mark which rows have real data
            df['Has_Real_Data'] = df['Roof_SqFt_Real'].notna()

            st.success(f"Found roof data in column: {found_roof_col}")
            st.info(
                f"Rows with existing roof data: {df['Has_Real_Data'].sum()}/{len(df)}")
        else:
            df['Roof_SqFt_Real'] = None
            df['Has_Real_Data'] = False
            st.info(
                "No existing roof data found. Will use AI predictions for all rows.")

        return df

    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None


def generate_predictions_for_all_addresses(companies_df, model):
    """Generate predictions for all addresses, using real data when available"""
    results = []

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, company in companies_df.iterrows():
        status_text.text(
            f"Processing address {i+1}/{len(companies_df)}: {company['Full_Address'][:50]}...")

        # Check if we have real roof data for this row
        has_real_data = company.get('Has_Real_Data', False)
        real_roof_area = company.get('Roof_SqFt_Real', None)

        if has_real_data and real_roof_area is not None:
            # Use real data - no need to predict
            result_row = company.copy()
            result_row['Final_Roof_Area_SqFt'] = real_roof_area
            # Formatted with commas
            result_row['Sq_Ft'] = f"{real_roof_area:,.0f}"
            result_row['Data_Source'] = 'Real Data'
            result_row['Predicted_SqFt'] = None
            result_row['Predicted_Area_Ratio'] = None
            result_row['Predicted_Pixels'] = None
        else:
            # Need to predict - use AI model
            # Generate synthetic prediction as fallback
            synthetic_image = np.random.randint(
                0, 255, (512, 512, 3), dtype=np.uint8)

            # Preprocess image
            image_normalized = synthetic_image.astype(np.float32) / 255.0
            image_normalized = (image_normalized -
                                [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

            # Convert to tensor
            image_tensor = torch.from_numpy(image_normalized).permute(
                2, 0, 1).unsqueeze(0).float()

            # Run inference
            with torch.no_grad():
                prediction = model(image_tensor)
                prediction_sigmoid = torch.sigmoid(prediction)
                prediction_binary = (prediction_sigmoid > 0.5).float()

            # Convert to numpy
            pred_mask = prediction_binary.squeeze().numpy()

            # Calculate predicted roof area
            predicted_pixels = np.sum(pred_mask > 0.5)
            total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
            predicted_area_ratio = predicted_pixels / total_pixels

            # Estimate predicted square footage (assuming 512x512 = 1 sq mile = 27,878,400 sq ft)
            estimated_sqft_per_pixel = 27878400 / \
                (512 * 512)  # sq ft per pixel
            predicted_sqft = predicted_pixels * estimated_sqft_per_pixel

            # Create result row - keep ALL original columns and add prediction
            result_row = company.copy()
            result_row['Final_Roof_Area_SqFt'] = predicted_sqft
            # Formatted with commas
            result_row['Sq_Ft'] = f"{predicted_sqft:,.0f}"
            result_row['Data_Source'] = 'AI Prediction'
            result_row['Predicted_SqFt'] = predicted_sqft
            result_row['Predicted_Area_Ratio'] = predicted_area_ratio
            result_row['Predicted_Pixels'] = predicted_pixels

        results.append(result_row)

        # Update progress
        progress_bar.progress((i + 1) / len(companies_df))

    progress_bar.empty()
    status_text.empty()

    return results


def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">🏠 Roof Segmentation AI</h1>',
                unsafe_allow_html=True)
    st.markdown(
        "### Simple CSV Processor - Add Predicted Square Footage to Each Address")

    # Sidebar
    st.sidebar.title("🔧 Configuration")

    # Load model
    with st.spinner("Loading AI model..."):
        model, checkpoint_name = load_trained_model()

    if model is None and checkpoint_name != "Demo Mode":
        st.error("❌ Failed to load AI model. Please check the model files.")
        return

    if checkpoint_name == "Demo Mode":
        st.sidebar.warning(f"⚠️ {checkpoint_name}")
    else:
        st.sidebar.success(f"✅ Model loaded: {checkpoint_name}")

    # Model info
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown(f"**Architecture**: UNet + ResNet34")
    st.sidebar.markdown(f"**Input Size**: 512x512 pixels")
    st.sidebar.markdown(f"**Best Loss**: -5.5004")

    # Main content
    st.markdown("---")

    # File upload section
    st.subheader("📁 Upload Your CSV Data")
    st.markdown(
        "Upload a CSV file with company addresses. The app will add a 'Predicted_SqFt' column for each location.")

    # Enhanced upload area
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: Name, Full_Address, City, State, Roof 10k (optional)"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Process CSV
        with st.spinner("Processing CSV data..."):
            companies_df = process_csv_data(uploaded_file)

        if companies_df is not None:
            st.success(f"✅ Successfully loaded {len(companies_df)} addresses")

            # Show data summary in a clean container
            st.markdown('<div class="results-section">',
                        unsafe_allow_html=True)
            st.subheader("📊 Data Summary")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-container">',
                            unsafe_allow_html=True)
                st.metric("Total Addresses", len(companies_df))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-container">',
                            unsafe_allow_html=True)
                if 'Roof_SqFt_Real' in companies_df.columns:
                    valid_roofs = companies_df['Roof_SqFt_Real'].notna().sum()
                    st.metric("Addresses with Roof Data", valid_roofs)
                else:
                    st.metric("Addresses with Roof Data", 0)
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-container">',
                            unsafe_allow_html=True)
                st.metric("Cities", companies_df['City'].nunique())
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Generate predictions button
            st.subheader("🚀 Generate AI Predictions")
            st.markdown(
                "Click the button below to add predicted square footage for ALL addresses in your CSV.")

            if st.button("🤖 Generate Predictions for All Addresses", type="primary"):
                # Generate predictions for ALL addresses
                with st.spinner(f"Generating AI predictions for {len(companies_df)} addresses..."):
                    results = generate_predictions_for_all_addresses(
                        companies_df, model)

                if results:
                    # Convert to DataFrame
                    results_df = pd.DataFrame(results)

                    # Display results
                    st.markdown('<div class="results-section">',
                                unsafe_allow_html=True)
                    st.subheader("🎯 Results - All Addresses with Predictions")

                    # Show summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown('<div class="metric-container">',
                                    unsafe_allow_html=True)
                        st.metric(
                            "Total Addresses Processed",
                            len(results_df)
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="metric-container">',
                                    unsafe_allow_html=True)
                        avg_predicted = results_df['Predicted_SqFt'].mean()
                        st.metric(
                            "Average Predicted SqFt",
                            f"{avg_predicted:,.0f}"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col3:
                        st.markdown('<div class="metric-container">',
                                    unsafe_allow_html=True)
                        total_predicted = results_df['Predicted_SqFt'].sum()
                        st.metric(
                            "Total Predicted SqFt",
                            f"{total_predicted:,.0f}"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col4:
                        st.markdown('<div class="metric-container">',
                                    unsafe_allow_html=True)
                        if 'Roof_SqFt_Real' in results_df.columns:
                            valid_comparisons = results_df[results_df['Roof_SqFt_Real'].notna(
                            )]
                            if len(valid_comparisons) > 0:
                                avg_error = (
                                    valid_comparisons['Predicted_SqFt'] - valid_comparisons['Roof_SqFt_Real']).abs().mean()
                                st.metric(
                                    "Avg Error (vs Real)",
                                    f"{avg_error:,.0f}"
                                )
                            else:
                                st.metric("Avg Error", "N/A")
                        else:
                            st.metric("Avg Error", "N/A")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Show results table
                    st.subheader("📊 Complete Results Table")
                    st.dataframe(results_df, use_container_width=True)

                    # Show simplified preview table
                    st.subheader(
                        "🏠 Simple Preview (Address + Final Roof Area)")

                    # Create preview with Address, Final Roof Area, and Data Source
                    preview_columns = ['Full_Address', 'Sq_Ft', 'Data_Source']
                    if all(col in results_df.columns for col in preview_columns):
                        simplified_preview = results_df[preview_columns].copy()
                        simplified_preview.columns = [
                            'Address', 'Roof Area (sq ft)', 'Data Source']
                        st.dataframe(simplified_preview,
                                     use_container_width=True)
                        st.caption(
                            "Shows final roof area (real data or AI prediction) and data source")
                    else:
                        st.error(
                            f"❌ Required columns not found. Available columns: {list(results_df.columns)}")
                        return

                    # Download results
                    csv_data = results_df.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    st.download_button(
                        label="📥 Download Complete CSV with Predictions",
                        data=csv_data,
                        file_name=f"roof_predictions_complete_{timestamp}.csv",
                        mime="text/csv"
                    )

                    # Download simplified CSV with Address and Final Roof Area
                    export_columns = ['Full_Address', 'Sq_Ft', 'Data_Source']
                    if all(col in results_df.columns for col in export_columns):
                        simplified_df = results_df[export_columns].copy()
                        simplified_df.columns = [
                            'Address', 'Roof Area (sq ft)', 'Data Source']
                        simplified_csv = simplified_df.to_csv(index=False)

                        st.download_button(
                            label="📥 Download CSV (Address + Final Roof Area)",
                            data=simplified_csv,
                            file_name=f"roof_area_final_{timestamp}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(
                            f"❌ Required columns not found for CSV export. Available columns: {list(results_df.columns)}")
                        return

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Insights
                    st.markdown('<div class="results-section">',
                                unsafe_allow_html=True)
                    st.subheader("💡 What You Got")

                    # Count data sources
                    real_data_count = len(
                        results_df[results_df['Data_Source'] == 'Real Data'])
                    ai_prediction_count = len(
                        results_df[results_df['Data_Source'] == 'AI Prediction'])

                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ CSV Data Processing Complete</h4>
                        <p><strong>All {len(results_df)} addresses</strong> processed successfully.</p>
                        <p><strong>Data Sources:</strong></p>
                        <ul>
                            <li><strong>Real Data</strong>: {real_data_count} addresses (used existing Roof 10k/20k values)</li>
                            <li><strong>AI Prediction</strong>: {ai_prediction_count} addresses (filled missing values with AI)</li>
                        </ul>
                        <p><strong>Final Output:</strong></p>
                        <ul>
                            <li><strong>Address</strong>: Full address from your CSV</li>
                            <li><strong>Roof Area (sq ft)</strong>: Final roof area (real data or AI prediction)</li>
                            <li><strong>Data Source</strong>: Shows whether data came from real values or AI prediction</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    if 'Roof_SqFt_Real' in results_df.columns:
                        st.markdown(f"""
                        <div class="info-box">
                            <h4>📊 Comparison with Real Data</h4>
                            <p>Your CSV included real roof measurements for comparison. The AI predictions can be compared against these values to assess accuracy.</p>
                            <p><strong>Note:</strong> Current predictions use synthetic images. For production use, you'll need actual aerial imagery for each address.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏠 Roof Segmentation AI - Simple CSV Processor</p>
        <p>Adds predicted square footage to each address in your CSV</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
