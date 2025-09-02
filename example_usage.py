#!/usr/bin/env python3
"""
Example usage of the accurate roof calculator
"""

from accurate_roof_calculator import AccurateRoofCalculator
import json

def main():
    """Example usage of the roof calculator"""
    
    # Initialize the calculator
    print("🏠 Initializing Accurate Roof Calculator...")
    calculator = AccurateRoofCalculator()
    
    # Example addresses to test
    test_addresses = [
        "1600 Amphitheatre Parkway, Mountain View, CA",
        "1 Apple Park Way, Cupertino, CA",
        "410 Terry Avenue North, Seattle, WA",
        "1 Hacker Way, Menlo Park, CA"
    ]
    
    print(f"\n🧪 Testing with {len(test_addresses)} addresses...")
    
    results = []
    
    for i, address in enumerate(test_addresses, 1):
        print(f"\n📍 Processing {i}/{len(test_addresses)}: {address}")
        
        # Calculate roof area
        result = calculator.calculate_roof_area_for_address(address)
        results.append(result)
        
        # Display result
        if result['success']:
            print(f"✅ Success!")
            print(f"   Roof Area: {result['roof_area_sqft']:,.0f} sq ft")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Coordinates: {result['latitude']:.4f}, {result['longitude']:.4f}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Summary
    print(f"\n📊 Summary:")
    successful = sum(1 for r in results if r['success'])
    print(f"Successful calculations: {successful}/{len(results)}")
    
    if successful > 0:
        successful_results = [r for r in results if r['success']]
        avg_area = sum(r['roof_area_sqft'] for r in successful_results) / len(successful_results)
        avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
        
        print(f"Average roof area: {avg_area:,.0f} sq ft")
        print(f"Average confidence: {avg_confidence:.2f}")
    
    # Save results to file
    with open('example_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to 'example_results.json'")

if __name__ == "__main__":
    main()
