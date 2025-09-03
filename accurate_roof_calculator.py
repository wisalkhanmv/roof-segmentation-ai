"""
Accurate Roof Area Calculator using Real Aerial Imagery
This module provides methods to calculate roof area using actual satellite imagery
from various providers like Google Maps, Mapbox, etc.
"""

import os
import requests
import numpy as np
import cv2
from PIL import Image
import io
import json
from typing import Tuple, Optional, Dict, Any
# Removed Nominatim imports - using Mapbox only
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccurateRoofCalculator:
    """
    Calculate roof area using real aerial imagery from various providers
    """

    def __init__(self):
        self.mapbox_api_key = os.getenv('MAPBOX_API_KEY')
        self.default_provider = 'mapbox'
        self.image_size = int(os.getenv('IMAGE_SIZE', 512))
        self.confidence_threshold = float(
            os.getenv('ROOF_DETECTION_CONFIDENCE', 0.5))

        # Debug logging for API keys
        logger.info(
            f"Mapbox API key loaded: {'Yes' if self.mapbox_api_key else 'No'}")
        if self.mapbox_api_key:
            logger.info(
                f"Mapbox API key starts with: {self.mapbox_api_key[:10]}...")

        # No fallback geocoder needed - using Mapbox only

    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Convert address to latitude and longitude coordinates using Mapbox only

        Args:
            address: Full address string

        Returns:
            Tuple of (latitude, longitude) or None if geocoding fails
        """
        # Check if Mapbox API key is available
        if not self.mapbox_api_key:
            logger.error("Mapbox API key not provided - cannot geocode address")
            return None

        try:
            import requests
            url = "https://api.mapbox.com/geocoding/v5/mapbox.places"
            params = {
                'access_token': self.mapbox_api_key,
                'query': address,
                'limit': 1
            }
            logger.info(f"Attempting to geocode with Mapbox: {address}")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()
            if data['features']:
                coords = data['features'][0]['center']
                # Mapbox returns [lng, lat]
                lng, lat = coords[0], coords[1]
                logger.info(f"Successfully geocoded with Mapbox: {address} -> ({lat}, {lng})")
                return (lat, lng)
            else:
                logger.warning(f"No results found for address: {address}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Mapbox geocoding request failed for {address}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Mapbox geocoding for {address}: {e}")
            return None

    def get_mapbox_satellite_image(self, lat: float, lon: float, zoom: int = 20) -> Optional[np.ndarray]:
        """
        Get satellite image from Mapbox API

        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level

        Returns:
            Image as numpy array or None if failed
        """
        if not self.mapbox_api_key:
            logger.error("Mapbox API key not provided")
            return None

        try:
            # Convert lat/lon to Mapbox tile coordinates
            import math

            # Convert to tile coordinates
            n = 2.0 ** zoom
            x = int((lon + 180.0) / 360.0 * n)
            y = int(
                (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)

            # Mapbox Static Images API
            url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom},0/{self.image_size}x{self.image_size}@2x"
            params = {
                'access_token': self.mapbox_api_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Convert to numpy array
            image = Image.open(io.BytesIO(response.content))
            image_array = np.array(image)

            logger.info(
                f"Successfully retrieved Mapbox satellite image for {lat}, {lon}")
            return image_array

        except Exception as e:
            logger.error(f"Error getting Mapbox satellite image: {e}")
            return None

    def get_satellite_image(self, lat: float, lon: float, provider: str = None) -> Optional[np.ndarray]:
        """
        Get satellite image from Mapbox API

        Args:
            lat: Latitude
            lon: Longitude
            provider: API provider (only 'mapbox' supported)

        Returns:
            Image as numpy array or None if failed
        """
        # Only use Mapbox
        if not self.mapbox_api_key:
            logger.error("Mapbox API key not provided")
            return None

        image = self.get_mapbox_satellite_image(lat, lon)
        if image is not None:
            return image

        logger.error("Mapbox satellite image provider failed")
        return None

    def detect_roof_areas(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect roof areas in satellite image using computer vision techniques

        Args:
            image: Satellite image as numpy array

        Returns:
            Tuple of (binary mask, confidence score)
        """
        try:
            # Convert to grayscale - handle both RGB and grayscale images
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection using Canny
            edges = cv2.Canny(blurred, 50, 150)

            # Morphological operations to close gaps
            kernel = np.ones((3, 3), np.uint8)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create binary mask
            mask = np.zeros_like(gray)

            # Filter contours by area and fill them
            min_area = (self.image_size * 0.05) ** 2  # Minimum 5% of image
            max_area = (self.image_size * 0.8) ** 2   # Maximum 80% of image

            roof_pixels = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    # Check if contour is roughly rectangular (roof-like)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.3 < aspect_ratio < 3.0:  # Reasonable roof aspect ratio
                        cv2.fillPoly(mask, [contour], 255)
                        roof_pixels += area

            # Calculate confidence based on detected roof area
            total_pixels = self.image_size * self.image_size
            roof_ratio = roof_pixels / total_pixels
            confidence = min(roof_ratio * 2, 1.0)  # Scale confidence

            logger.info(
                f"Detected roof area: {roof_pixels} pixels ({roof_ratio:.2%} of image)")

            return mask, confidence

        except Exception as e:
            logger.error(f"Error in roof detection: {e}")
            # Handle both RGB and grayscale images for error case
            if len(image.shape) == 3:
                return np.zeros_like(image[:, :, 0]), 0.0
            else:
                return np.zeros_like(image), 0.0

    def calculate_roof_area_sqft(self, lat: float, lon: float, zoom: int = 20) -> Dict[str, Any]:
        """
        Calculate roof area in square feet for a given location

        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level for satellite imagery

        Returns:
            Dictionary with roof area information
        """
        try:
            # Get satellite image
            image = self.get_satellite_image(lat, lon)
            if image is None:
                return {
                    'success': False,
                    'error': 'Could not retrieve satellite image',
                    'roof_area_sqft': 0,
                    'confidence': 0.0,
                    'method': 'satellite_imagery'
                }

            # Detect roof areas
            roof_mask, confidence = self.detect_roof_areas(image)

            # Accept all results regardless of confidence (user requested this)
            # if confidence < self.confidence_threshold:
            #     return {
            #         'success': False,
            #         'error': f'Low confidence in roof detection: {confidence:.2f}',
            #         'roof_area_sqft': 0,
            #         'confidence': confidence,
            #         'method': 'satellite_imagery'
            #     }

            # Calculate roof area in pixels
            roof_pixels = np.sum(roof_mask > 0)

            # Convert pixels to square feet
            # At zoom level 20, each pixel represents approximately 0.6 meters
            # This is an approximation and may vary by location
            meters_per_pixel = 0.6 / (2 ** (20 - zoom))
            sq_meters_per_pixel = meters_per_pixel ** 2
            sq_feet_per_pixel = sq_meters_per_pixel * 10.764  # Convert m² to ft²

            roof_area_sqft = roof_pixels * sq_feet_per_pixel

            logger.info(f"Calculated roof area: {roof_area_sqft:.0f} sq ft")

            return {
                'success': True,
                'roof_area_sqft': roof_area_sqft,
                'confidence': confidence,
                'roof_pixels': int(roof_pixels),
                'method': 'satellite_imagery',
                'zoom_level': zoom,
                'meters_per_pixel': meters_per_pixel
            }

        except Exception as e:
            logger.error(f"Error calculating roof area: {e}")
            return {
                'success': False,
                'error': str(e),
                'roof_area_sqft': 0,
                'confidence': 0.0,
                'method': 'satellite_imagery'
            }

    def calculate_roof_area_for_address(self, address: str) -> Dict[str, Any]:
        """
        Calculate roof area for a given address

        Args:
            address: Full address string

        Returns:
            Dictionary with roof area information
        """
        try:
            # Geocode the address
            coords = self.geocode_address(address)
            if coords is None:
                return {
                    'success': False,
                    'error': 'Could not geocode address',
                    'roof_area_sqft': 0,
                    'confidence': 0.0,
                    'method': 'satellite_imagery'
                }

            lat, lon = coords
            logger.info(f"Geocoded {address} to {lat}, {lon}")

            # Calculate roof area
            result = self.calculate_roof_area_sqft(lat, lon)
            result['address'] = address
            result['latitude'] = lat
            result['longitude'] = lon

            return result

        except Exception as e:
            logger.error(f"Error processing address {address}: {e}")
            return {
                'success': False,
                'error': str(e),
                'roof_area_sqft': 0,
                'confidence': 0.0,
                'method': 'satellite_imagery',
                'address': address
            }


def test_roof_calculator():
    """
    Test function for the roof calculator
    """
    calculator = AccurateRoofCalculator()

    # Test with a sample address
    test_address = "1600 Amphitheatre Parkway, Mountain View, CA"
    result = calculator.calculate_roof_area_for_address(test_address)

    print(f"Test result for {test_address}:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_roof_calculator()
