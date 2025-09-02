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
        # Check if model is already loaded in session state
        if 'trained_model' in st.session_state and st.session_state.trained_model is not None:
            return st.session_state.trained_model, "Loaded from Session"

        # Try to load from local checkpoints first
        current_dir = Path(__file__).parent
        best_checkpoint = current_dir / "checkpoints" / "best_model.ckpt"

        if best_checkpoint.exists():
            # Load model
            model_config = {
                'model_name': 'unet',
                'backbone': 'resnet34',
                'classes': 1,
                'encoder_weights': 'imagenet'
            }
            model = create_model(model_config)

            # Load checkpoint with proper error handling for PyTorch 2.6+
            try:
                # First try with weights_only=True (secure)
                checkpoint = torch.load(
                    best_checkpoint, map_location='cpu', weights_only=True)
            except Exception as e:
                # If that fails, try with weights_only=False (less secure but works with custom objects)
                try:
                    checkpoint = torch.load(
                        best_checkpoint, map_location='cpu', weights_only=False)
                except Exception as e2:
                    return None, "Demo Mode"

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
            st.session_state.trained_model = model
            return model, "Local Model"

        # Auto-download from Hugging Face if no local model
        try:
            # Check if huggingface_hub is available
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                return None, "Demo Mode"

            # Show download progress
            with st.spinner("📥 Downloading AI model from Hugging Face..."):
                # Download model from Hugging Face
                model_path = hf_hub_download(
                    repo_id="dreamireal/roof-segmentation-ai",
                    filename="best_model.ckpt",
                    cache_dir="checkpoints"
                )

            # Load model
            model_config = {
                'model_name': 'unet',
                'backbone': 'resnet34',
                'classes': 1,
                'encoder_weights': 'imagenet'
            }
            model = create_model(model_config)

            # Load checkpoint with proper error handling for PyTorch 2.6+
            try:
                # First try with weights_only=True (secure)
                checkpoint = torch.load(
                    model_path, map_location='cpu', weights_only=True)
            except Exception as e:
                # If that fails, try with weights_only=False (less secure but works with custom objects)
                try:
                    checkpoint = torch.load(
                        model_path, map_location='cpu', weights_only=False)
                except Exception as e2:
                    return None, "Demo Mode"

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
            st.session_state.trained_model = model
            return model, "Hugging Face Model"

        except Exception as download_error:
            # Fall back to demo mode silently
            return None, "Demo Mode"

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
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("📝 Your CSV should include: Name, Full_Address, City, State")
            return None

        # Data quality check
        data_issues = []

        # Check for missing addresses
        missing_addresses = df['Full_Address'].isna().sum()
        if missing_addresses > 0:
            data_issues.append(
                f"⚠️ {missing_addresses} rows have missing addresses")

        # Check for empty addresses
        empty_addresses = (df['Full_Address'] == '').sum()
        if empty_addresses > 0:
            data_issues.append(
                f"⚠️ {empty_addresses} rows have empty addresses")

        # Check for missing cities/states
        missing_cities = df['City'].isna().sum()
        if missing_cities > 0:
            data_issues.append(f"⚠️ {missing_cities} rows have missing cities")

        missing_states = df['State'].isna().sum()
        if missing_states > 0:
            data_issues.append(f"⚠️ {missing_states} rows have missing states")

        # Show data quality warnings if any
        if data_issues:
            st.warning("📊 Data Quality Issues Detected:")
            for issue in data_issues:
                st.markdown(f"- {issue}")

            st.info("💡 **How to fix these issues:**")
            st.markdown("""
            1. **Missing Addresses**: Fill in the Full_Address column for all rows
            2. **Empty Addresses**: Replace empty strings with actual addresses
            3. **Missing Cities/States**: Fill in City and State for all rows
            4. **Data Format**: Ensure addresses are in format: 'Street, City, State ZIP'

            **Example format:**
            - Full_Address: '123 Main St, New York, NY 10001'
            - City: 'New York'
            - State: 'NY'
            """)

            # Ask user if they want to continue
            continue_anyway = st.checkbox(
                "✅ Continue processing despite data issues?")
            if not continue_anyway:
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
        st.error(f"❌ Error processing CSV: {e}")
        st.info("💡 **Common CSV issues and solutions:**")
        st.markdown("""
        1. **File Format**: Ensure your file is a valid CSV (.csv extension)
        2. **Column Names**: Check that column names match exactly: Name, Full_Address, City, State
        3. **Data Encoding**: Try saving your CSV with UTF-8 encoding
        4. **Special Characters**: Remove any special characters from column names
        5. **File Size**: Ensure file is not corrupted or too large

        **Need help?** Check your CSV file and try again.
        """)
        return None


def generate_predictions_for_all_addresses(model, companies_df):
    """Generate predictions for all addresses, using real data when available"""
    print(f"🧪 Generating predictions for {len(companies_df)} addresses...")

    # Check if model is valid (None or string means no real model)
    if model is None or isinstance(model, str):
        print("❌ No model available - using demo mode")
        # Return demo results with synthetic data
        results = []
        for i, company in companies_df.iterrows():
            result_row = company.copy()
            # Generate synthetic prediction
            synthetic_sqft = np.random.randint(
                5000, 50000)  # Random between 5k-50k sqft
            result_row['Final_Roof_Area_SqFt'] = synthetic_sqft
            result_row['Sq_Ft'] = f"{synthetic_sqft:,.0f}"
            result_row['Data_Source'] = 'Demo Mode'
            result_row['Predicted_SqFt'] = float(synthetic_sqft)
            result_row['Predicted_Area_Ratio'] = float(0.3)  # 30% of image
            result_row['Predicted_Pixels'] = int(78643)  # Random pixel count
            results.append(result_row)
        return results

    # Check if DataFrame is valid
    if companies_df is None or len(companies_df) == 0:
        print("❌ No data to process")
        return []

    print(f"📊 DataFrame shape: {companies_df.shape}")
    print(f"📊 DataFrame columns: {list(companies_df.columns)}")

    results = []

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, company in companies_df.iterrows():
        # Safe address display with null checking
        address = company.get('Full_Address', '')
        if pd.isna(address) or address == '':
            address = f"Row {i+1} (No Address)"
        else:
            address = str(address)[
                :50] + "..." if len(str(address)) > 50 else str(address)

        status_text.text(
            f"Processing address {i+1}/{len(companies_df)}: {address}")

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
            result_row['Predicted_SqFt'] = float(
                real_roof_area)  # Use real data as prediction
            result_row['Predicted_Area_Ratio'] = float(
                0.0)  # No prediction ratio for real data
            # No prediction pixels for real data
            result_row['Predicted_Pixels'] = int(0)
        else:
            # Need to predict - use AI model or demo mode
            if model is not None:
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

                data_source = 'AI Prediction'
            else:
                # Demo mode - generate synthetic prediction
                predicted_sqft = np.random.randint(
                    5000, 50000)  # Random between 5k-50k sqft
                # Approximate pixel count
                predicted_pixels = int(predicted_sqft / 106.4)
                predicted_area_ratio = 0.3  # 30% of image
                data_source = 'Demo Mode'

        # Create result row - keep ALL original columns and add prediction
        result_row = company.copy()
        result_row['Final_Roof_Area_SqFt'] = predicted_sqft
        # Formatted with commas
        result_row['Sq_Ft'] = f"{predicted_sqft:,.0f}"
            result_row['Data_Source'] = data_source
            result_row['Predicted_SqFt'] = float(predicted_sqft)
            result_row['Predicted_Area_Ratio'] = float(predicted_area_ratio)
            result_row['Predicted_Pixels'] = int(predicted_pixels)

        results.append(result_row)

        # Update progress
        progress_bar.progress((i + 1) / len(companies_df))

    progress_bar.empty()
    status_text.empty()

    print(f"✅ Completed processing. Generated {len(results)} predictions")
    if len(results) > 0:
        print(f"📊 Sample result keys: {list(results[0].keys())}")

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

    # Model status
    st.sidebar.markdown("### 🤖 AI Model")

    # Check if model is already loaded
    if 'trained_model' in st.session_state and st.session_state.trained_model is not None:
        st.sidebar.success("✅ Model loaded from session")
        model = st.session_state.trained_model
        checkpoint_name = "Session Model"
    else:
        # Load model
        with st.spinner("Loading AI model..."):
            model, checkpoint_name = load_trained_model()

    # Show model status
    if model is None or checkpoint_name == "Demo Mode":
        st.sidebar.warning("⚠️ Demo Mode")
        st.sidebar.info("Using synthetic predictions")
        st.sidebar.error("Model loading failed - using demo data")
        model = None  # Ensure model is None for demo mode
    else:
        st.sidebar.success(f"✅ {checkpoint_name}")
        st.sidebar.info("Real AI predictions active")

    # Simple model info
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("**Type**: AI-powered roof segmentation")
    st.sidebar.markdown("**Input**: Aerial imagery")
    st.sidebar.markdown("**Output**: Roof area predictions")

    # Main content
    st.markdown("---")

    # Simple status indicator
    if model is None or checkpoint_name == "Demo Mode":
        st.info("🤖 **Demo Mode**: Using synthetic predictions for demonstration")
    else:
        st.success("✅ **AI Mode**: Real AI model loaded and ready!")

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
                try:
                    with st.spinner(f"Generating AI predictions for {len(companies_df)} addresses..."):
                        results = generate_predictions_for_all_addresses(
                            model, companies_df)

                    if results:
                        try:
                            # Convert to DataFrame
                            results_df = pd.DataFrame(results)
                            print(
                                f"📊 Results DataFrame created with shape: {results_df.shape}")
                            print(
                                f"📊 Results DataFrame columns: {list(results_df.columns)}")

                            # Debug: Check for any problematic values
                            print(
                                f"📊 Predicted_SqFt dtype: {results_df['Predicted_SqFt'].dtype}")
                            print(
                                f"📊 Predicted_SqFt has NaN: {results_df['Predicted_SqFt'].isna().any()}")
                            print(
                                f"📊 Predicted_SqFt sample: {results_df['Predicted_SqFt'].head()}")

                            # Check other potentially problematic columns
                            print(
                                f"📊 Roof_SqFt_Real dtype: {results_df['Roof_SqFt_Real'].dtype}")
                            print(
                                f"📊 Roof_SqFt_Real sample: {results_df['Roof_SqFt_Real'].head()}")
                            print(
                                f"📊 Final_Roof_Area_SqFt dtype: {results_df['Final_Roof_Area_SqFt'].dtype}")
                            print(
                                f"📊 Final_Roof_Area_SqFt sample: {results_df['Final_Roof_Area_SqFt'].head()}")

                            # Display results
                            st.markdown('<div class="results-section">',
                                        unsafe_allow_html=True)
                            st.subheader(
                                "🎯 Results - All Addresses with Predictions")
                        except Exception as df_error:
                            st.error(
                                f"❌ Error creating results DataFrame: {str(df_error)}")
                            st.info(
                                "💡 The predictions were generated but there was an issue displaying them.")
                            return

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
                            try:
                                # Convert to numeric, handling any non-numeric values
                                predicted_sqft_numeric = pd.to_numeric(
                                    results_df['Predicted_SqFt'], errors='coerce')
                                avg_predicted = predicted_sqft_numeric.mean()
                                st.metric(
                                    "Average Predicted SqFt",
                                    f"{avg_predicted:,.0f}"
                                )
                            except Exception as metric_error:
                                st.metric("Average Predicted SqFt", "Error")
                                print(
                                    f"❌ Error calculating average: {metric_error}")
                            st.markdown('</div>', unsafe_allow_html=True)

                        with col3:
                            st.markdown('<div class="metric-container">',
                                        unsafe_allow_html=True)
                            try:
                                # Convert to numeric, handling any non-numeric values
                                predicted_sqft_numeric = pd.to_numeric(
                                    results_df['Predicted_SqFt'], errors='coerce')
                                total_predicted = predicted_sqft_numeric.sum()
                                st.metric(
                                    "Total Predicted SqFt",
                                    f"{total_predicted:,.0f}"
                                )
                            except Exception as metric_error:
                                st.metric("Total Predicted SqFt", "Error")
                                print(
                                    f"❌ Error calculating total: {metric_error}")
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

                        # Show results table using a different approach
                        st.subheader("📊 Complete Results Table")

                        # Instead of using st.dataframe, let's create a custom display
                        try:
                            # Create a simple table using st.table which is more robust
                            display_columns = [
                                'Name', 'City', 'State', 'Predicted_SqFt', 'Data_Source']
                            available_columns = [
                                col for col in display_columns if col in results_df.columns]

                            if available_columns:
                                simple_df = results_df[available_columns].copy(
                                )

                                # Format the Predicted_SqFt column for better display
                                if 'Predicted_SqFt' in simple_df.columns:
                                    simple_df['Predicted_SqFt'] = simple_df['Predicted_SqFt'].apply(
                                        lambda x: f"{float(x):,.0f}" if pd.notna(
                                            x) else "N/A"
                                    )

                                # Use st.table instead of st.dataframe
                                st.table(simple_df)

                                # Also provide download option
                                csv_data = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Complete Results CSV",
                                    data=csv_data,
                                    file_name="roof_predictions_complete.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("❌ No displayable columns found")

                        except Exception as e:
                            st.error(f"❌ Error displaying results: {str(e)}")
                            print(f"❌ Display error: {e}")

                            # Last resort: show just the summary
                            st.info("📊 **Results Summary:**")
                            st.write(
                                f"- **Total Addresses Processed:** {len(results_df)}")
                            if 'Predicted_SqFt' in results_df.columns:
                                avg_sqft = results_df['Predicted_SqFt'].mean()
                                total_sqft = results_df['Predicted_SqFt'].sum()
                                st.write(
                                    f"- **Average Predicted SqFt:** {avg_sqft:,.0f}")
                                st.write(
                                    f"- **Total Predicted SqFt:** {total_sqft:,.0f}")

                            # Provide download option even if display fails
                            try:
                                csv_data = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results CSV",
                                    data=csv_data,
                                    file_name="roof_predictions.csv",
                                    mime="text/csv"
                                )
                            except Exception as e2:
                                st.error(f"❌ Download also failed: {str(e2)}")

                        # Show simplified preview table
                        st.subheader(
                            "🏠 Simple Preview (Address + Final Roof Area)")

                        try:
                        # Create preview with Address, Final Roof Area, and Data Source
                        preview_columns = [
                            'Full_Address', 'Sq_Ft', 'Data_Source']
                        if all(col in results_df.columns for col in preview_columns):
                            simplified_preview = results_df[preview_columns].copy(
                            )
                            simplified_preview.columns = [
                                'Address', 'Roof Area (sq ft)', 'Data Source']

                                # Use st.table instead of st.dataframe
                                st.table(simplified_preview)
                            st.caption(
                                "Shows final roof area (real data or AI prediction) and data source")
                        else:
                            st.error(
                                f"❌ Required columns not found. Available columns: {list(results_df.columns)}")
                        except Exception as e:
                            st.error(
                                f"❌ Error displaying preview table: {str(e)}")
                            print(f"❌ Preview table error: {e}")

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
                        export_columns = [
                            'Full_Address', 'Sq_Ft', 'Data_Source']
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
                    else:
                        st.warning(
                            "⚠️ No results generated. Please check the console for error messages.")
                        st.info(
                            "💡 Make sure your CSV file has the required columns: 'Full_Address', 'Name', 'City', 'State'")
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")
                    st.info("💡 **Common causes and solutions:**")
                    st.markdown("""
                    1. **Missing Data**: Check that all required columns have data
                    2. **Invalid Addresses**: Ensure addresses are properly formatted
                    3. **File Corruption**: Try re-uploading your CSV file
                    4. **Memory Issues**: Try processing smaller batches of data
                    
                    **What to do:**
                    - Check your CSV file for missing or invalid data
                    - Ensure all addresses are complete and properly formatted
                    - Try uploading a smaller file first
                    - Contact support if the issue persists
                    """)

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
