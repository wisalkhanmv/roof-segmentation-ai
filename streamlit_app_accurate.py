#!/usr/bin/env python3
"""
Accurate Roof Area Calculator - Streamlit Web Application
Uses real satellite imagery to calculate roof area for addresses
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
from accurate_roof_calculator import AccurateRoofCalculator

# Page configuration
st.set_page_config(
    page_title="Accurate Roof Area Calculator",
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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
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
def load_roof_calculator():
    """Load the roof calculator (cached for performance)"""
    return AccurateRoofCalculator()


def process_csv_data(uploaded_file):
    """Process uploaded CSV data"""
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Check required columns
        required_columns = ['Name', 'Full_Address', 'City', 'State']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("📝 Your CSV should include: Name, Full_Address, City, State")
            return None

        # Data quality check
        data_issues = []

        # Check for missing addresses
        missing_addresses = df['Full_Address'].isna().sum()
        if missing_addresses > 0:
            data_issues.append(f"⚠️ {missing_addresses} rows have missing addresses")

        # Check for empty addresses
        empty_addresses = (df['Full_Address'] == '').sum()
        if empty_addresses > 0:
            data_issues.append(f"⚠️ {empty_addresses} rows have empty addresses")

        # Show data quality warnings if any
        if data_issues:
            st.warning("📊 Data Quality Issues Detected:")
            for issue in data_issues:
                st.markdown(f"- {issue}")

            st.info("💡 **How to fix these issues:**")
            st.markdown("""
            1. **Missing Addresses**: Fill in the Full_Address column for all rows
            2. **Empty Addresses**: Replace empty strings with actual addresses
            3. **Data Format**: Ensure addresses are in format: 'Street, City, State ZIP'

            **Example format:**
            - Full_Address: '123 Main St, New York, NY 10001'
            - City: 'New York'
            - State: 'NY'
            """)

            # Ask user if they want to continue
            continue_anyway = st.checkbox("✅ Continue processing despite data issues?")
            if not continue_anyway:
                return None

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


def calculate_roof_areas_for_addresses(calculator, companies_df, max_addresses=None):
    """Calculate roof areas for all addresses using satellite imagery"""
    print(f"🧪 Calculating roof areas for {len(companies_df)} addresses...")
    
    results = []
    
    # Limit number of addresses if specified
    if max_addresses and len(companies_df) > max_addresses:
        companies_df = companies_df.head(max_addresses)
        st.warning(f"⚠️ Processing only first {max_addresses} addresses for demo purposes")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (_, company) in enumerate(companies_df.iterrows()):
        # Get address info safely
        address = company.get('Full_Address', '')
        name = company.get('Name', 'Unknown')
        city = company.get('City', 'Unknown')
        state = company.get('State', 'Unknown')
        
        # Display progress
        status_text.text(f"Processing {i+1}/{len(companies_df)}: {name} - {city}, {state}")
        
        # Calculate roof area
        try:
            result = calculator.calculate_roof_area_for_address(address)
            
            # Create result row
            result_row = company.copy()
            
            if result['success']:
                result_row['Roof_Area_SqFt'] = result['roof_area_sqft']
                result_row['Confidence'] = result['confidence']
                result_row['Method'] = result['method']
                result_row['Latitude'] = result.get('latitude', None)
                result_row['Longitude'] = result.get('longitude', None)
                result_row['Status'] = 'Success'
                result_row['Error'] = None
            else:
                result_row['Roof_Area_SqFt'] = 0
                result_row['Confidence'] = 0.0
                result_row['Method'] = result['method']
                result_row['Latitude'] = result.get('latitude', None)
                result_row['Longitude'] = result.get('longitude', None)
                result_row['Status'] = 'Failed'
                result_row['Error'] = result.get('error', 'Unknown error')
            
            results.append(result_row)
            
        except Exception as e:
            # Handle any unexpected errors
            result_row = company.copy()
            result_row['Roof_Area_SqFt'] = 0
            result_row['Confidence'] = 0.0
            result_row['Method'] = 'satellite_imagery'
            result_row['Latitude'] = None
            result_row['Longitude'] = None
            result_row['Status'] = 'Error'
            result_row['Error'] = str(e)
            results.append(result_row)
        
        # Update progress
        progress_bar.progress((i + 1) / len(companies_df))
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()
    
    print(f"✅ Completed processing. Generated {len(results)} results")
    return results


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Accurate Roof Area Calculator</h1>', unsafe_allow_html=True)
    st.markdown("### Real Satellite Imagery Analysis - Calculate Actual Roof Square Footage")
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # API Status
    st.sidebar.markdown("### 🌐 API Status")
    
    # Load calculator
    with st.spinner("Loading roof calculator..."):
        calculator = load_roof_calculator()
    
    # Check API keys
    api_status = {
        'Google Maps': bool(calculator.google_api_key),
        'Mapbox': bool(calculator.mapbox_api_key),
        'Bing Maps': bool(calculator.bing_api_key)
    }
    
    for provider, status in api_status.items():
        if status:
            st.sidebar.success(f"✅ {provider}")
        else:
            st.sidebar.warning(f"⚠️ {provider} (No API key)")
    
    if not any(api_status.values()):
        st.sidebar.error("❌ No API keys configured!")
        st.sidebar.info("Please add API keys to your .env file")
    
    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    
    max_addresses = st.sidebar.number_input(
        "Max addresses to process",
        min_value=1,
        max_value=1000,
        value=50,
        help="Limit the number of addresses to process (useful for testing)"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Minimum confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Minimum confidence required for roof detection"
    )
    
    calculator.confidence_threshold = confidence_threshold
    
    # Main content
    st.markdown("---")
    
    # API Key Warning
    if not any(api_status.values()):
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ API Keys Required</h4>
            <p>To use this accurate roof area calculator, you need to configure API keys for satellite imagery providers.</p>
            <p><strong>Required:</strong> At least one of the following API keys in your .env file:</p>
            <ul>
                <li><strong>Google Maps API Key</strong> - For Google satellite imagery</li>
                <li><strong>Mapbox API Key</strong> - For Mapbox satellite imagery</li>
                <li><strong>Bing Maps API Key</strong> - For Bing satellite imagery</li>
            </ul>
            <p>See the README for instructions on obtaining these API keys.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # File upload section
    st.subheader("📁 Upload Your CSV Data")
    st.markdown("Upload a CSV file with company addresses. The app will calculate actual roof area using satellite imagery.")
    
    # Enhanced upload area
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: Name, Full_Address, City, State"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Process CSV
        with st.spinner("Processing CSV data..."):
            companies_df = process_csv_data(uploaded_file)
        
        if companies_df is not None:
            st.success(f"✅ Successfully loaded {len(companies_df)} addresses")
            
            # Show data summary
            st.markdown('<div class="results-section">', unsafe_allow_html=True)
            st.subheader("📊 Data Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Total Addresses", len(companies_df))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Cities", companies_df['City'].nunique())
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("States", companies_df['State'].nunique())
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Calculate roof areas button
            st.subheader("🛰️ Calculate Roof Areas from Satellite Imagery")
            st.markdown("Click the button below to calculate actual roof area for ALL addresses using satellite imagery.")
            
            if st.button("🛰️ Calculate Roof Areas from Satellite Imagery", type="primary"):
                # Calculate roof areas
                try:
                    with st.spinner(f"Calculating roof areas for {len(companies_df)} addresses using satellite imagery..."):
                        results = calculate_roof_areas_for_addresses(calculator, companies_df, max_addresses)
                    
                    if results:
                        # Convert to DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.markdown('<div class="results-section">', unsafe_allow_html=True)
                        st.subheader("🎯 Results - Roof Areas from Satellite Imagery")
                        
                        # Show summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            st.metric("Total Addresses Processed", len(results_df))
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            successful = len(results_df[results_df['Status'] == 'Success'])
                            st.metric("Successful Calculations", successful)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            if successful > 0:
                                avg_area = results_df[results_df['Status'] == 'Success']['Roof_Area_SqFt'].mean()
                                st.metric("Average Roof Area (sq ft)", f"{avg_area:,.0f}")
                            else:
                                st.metric("Average Roof Area (sq ft)", "N/A")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            if successful > 0:
                                avg_confidence = results_df[results_df['Status'] == 'Success']['Confidence'].mean()
                                st.metric("Average Confidence", f"{avg_confidence:.2f}")
                            else:
                                st.metric("Average Confidence", "N/A")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show results table
                        st.subheader("📊 Results Table")
                        
                        # Create display table
                        display_columns = ['Name', 'City', 'State', 'Roof_Area_SqFt', 'Confidence', 'Status']
                        available_columns = [col for col in display_columns if col in results_df.columns]
                        
                        if available_columns:
                            display_df = results_df[available_columns].copy()
                            
                            # Format the Roof_Area_SqFt column
                            if 'Roof_Area_SqFt' in display_df.columns:
                                display_df['Roof_Area_SqFt'] = display_df['Roof_Area_SqFt'].apply(
                                    lambda x: f"{float(x):,.0f}" if pd.notna(x) and x > 0 else "N/A"
                                )
                            
                            # Format confidence
                            if 'Confidence' in display_df.columns:
                                display_df['Confidence'] = display_df['Confidence'].apply(
                                    lambda x: f"{float(x):.2f}" if pd.notna(x) else "N/A"
                                )
                            
                            st.table(display_df)
                        
                        # Download results
                        csv_data = results_df.to_csv(index=False)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        st.download_button(
                            label="📥 Download Complete Results CSV",
                            data=csv_data,
                            file_name=f"accurate_roof_areas_{timestamp}.csv",
                            mime="text/csv"
                        )
                        
                        # Show insights
                        st.markdown('<div class="results-section">', unsafe_allow_html=True)
                        st.subheader("💡 What You Got")
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>✅ Satellite Imagery Analysis Complete</h4>
                            <p><strong>Total addresses processed:</strong> {len(results_df)}</p>
                            <p><strong>Successful calculations:</strong> {successful}</p>
                            <p><strong>Failed calculations:</strong> {len(results_df) - successful}</p>
                            <p><strong>Method used:</strong> Real satellite imagery analysis</p>
                            <p><strong>Data source:</strong> Satellite imagery from configured API providers</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if successful > 0:
                            st.markdown(f"""
                            <div class="info-box">
                                <h4>📊 Analysis Results</h4>
                                <p><strong>Average roof area:</strong> {results_df[results_df['Status'] == 'Success']['Roof_Area_SqFt'].mean():,.0f} sq ft</p>
                                <p><strong>Average confidence:</strong> {results_df[results_df['Status'] == 'Success']['Confidence'].mean():.2f}</p>
                                <p><strong>Total roof area:</strong> {results_df[results_df['Status'] == 'Success']['Roof_Area_SqFt'].sum():,.0f} sq ft</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No results generated. Please check the console for error messages.")
                        
                except Exception as e:
                    st.error(f"❌ Error calculating roof areas: {str(e)}")
                    st.info("💡 **Common causes and solutions:**")
                    st.markdown("""
                    1. **API Key Issues**: Check that your API keys are valid and have sufficient quota
                    2. **Network Issues**: Ensure you have a stable internet connection
                    3. **Address Format**: Make sure addresses are properly formatted
                    4. **Rate Limiting**: Try reducing the number of addresses or adding delays
                    
                    **What to do:**
                    - Check your API keys in the .env file
                    - Verify your internet connection
                    - Try with a smaller number of addresses first
                    - Check the console for detailed error messages
                    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏠 Accurate Roof Area Calculator - Real Satellite Imagery Analysis</p>
        <p>Uses actual satellite imagery to calculate precise roof square footage</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
