#!/usr/bin/env python3
"""
Test script for Team PDF Report Generation
Tests the team report PDF generation with mock data
"""

import base64
import json
import requests
from datetime import datetime
import random

# Mock individual assessment results
# Each result has ordered_traits and ranks for one team member
def generate_mock_individual_results(num_members=5):
    """Generate mock assessment results for N team members"""
    
    traits = [
        "Courage", "Practicality", "Curiosity", "Prudence", "Confidence",
        "Discernment", "Fairness", "Tenacity", "Foresight", "Discipline",
        "Objectivity", "Empathy"
    ]
    
    individual_results = []
    
    for i in range(num_members):
        # Randomize the order for each person (with some correlation for realism)
        shuffled_traits = traits.copy()
        random.shuffle(shuffled_traits)
        
        # Assign ranks (with some ties possible)
        ranks = {}
        for idx, trait in enumerate(shuffled_traits):
            # Add some variation - occasionally create ties
            rank = idx + 1
            if random.random() < 0.1:  # 10% chance of tie
                rank = max(1, rank - 0.5)
            ranks[trait] = rank
        
        result = {
            "ordered_traits": shuffled_traits,
            "ranks": ranks
        }
        
        individual_results.append(result)
        print(f"✓ Generated results for team member {i+1}")
    
    return individual_results


def test_team_pdf_generation(
    service_url="http://127.0.0.1:8080",
    company_name="Acme Corporation",
    team_name="Sales Team",
    num_members=5,
    save_pdf=True
):
    """
    Test team PDF generation
    
    Args:
        service_url: URL of the PDF generation service
        company_name: Company name for the report
        team_name: Team name for the report
        num_members: Number of team members to simulate
        save_pdf: Whether to save the generated PDF to a file
    """
    
    print("=" * 60)
    print("🧪 TEAM REPORT PDF GENERATION TEST")
    print("=" * 60)
    print(f"📍 Service URL: {service_url}")
    print(f"🏢 Company: {company_name}")
    print(f"👥 Team: {team_name}")
    print(f"🔢 Members: {num_members}")
    print("=" * 60)
    
    # Generate mock individual results
    print("\n📊 Generating mock assessment results...")
    individual_results = generate_mock_individual_results(num_members)
    print(f"✅ Generated {len(individual_results)} individual results\n")
    
    # Prepare request data
    request_data = {
        "company_name": company_name,
        "team_name": team_name,
        "num_members": num_members,
        "date_str": datetime.now().strftime("%Y-%m-%d"),
        "individual_results": individual_results
    }
    
    # Call the team PDF generation endpoint
    print("📡 Calling team PDF generation endpoint...")
    try:
        response = requests.post(
            f"{service_url}/generate-team-pdf",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        result = response.json()
        
        if not result.get("success"):
            print(f"❌ PDF generation failed: {result.get('message', 'Unknown error')}")
            return False
        
        print("✅ Team PDF generated successfully!\n")
        
        # Get the base64 PDF
        pdf_base64 = result.get("pdf_base64")
        filename = result.get("filename", "team_report.pdf")
        
        if not pdf_base64:
            print("❌ No PDF data in response")
            return False
        
        # Decode base64 to bytes
        pdf_bytes = base64.b64decode(pdf_base64)
        print(f"📄 PDF size: {len(pdf_bytes):,} bytes ({len(pdf_bytes)/1024:.1f} KB)")
        print(f"📝 Filename: {filename}")
        
        # Save to file if requested
        if save_pdf:
            with open(filename, "wb") as f:
                f.write(pdf_bytes)
            print(f"💾 Saved to: {filename}")
        
        print("\n" + "=" * 60)
        print("✅ TEST PASSED - Team report generated successfully!")
        print("=" * 60)
        
        # Print sample data from one team member
        print("\n📊 Sample Individual Result (Member 1):")
        print(f"  Top 3 Strengths: {individual_results[0]['ordered_traits'][:3]}")
        print(f"  Bottom 3 Strengths: {individual_results[0]['ordered_traits'][-3:]}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to PDF service at {service_url}")
        print("   Make sure the service is running:")
        print("   python main.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out (PDF generation took too long)")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_health_check(service_url="http://127.0.0.1:8080"):
    """Test the health check endpoint"""
    print("\n🏥 Testing health check endpoint...")
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service is healthy: {data}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test team PDF report generation")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="PDF service URL")
    parser.add_argument("--company", default="Acme Corporation", help="Company name")
    parser.add_argument("--team", default="Sales Team", help="Team name")
    parser.add_argument("--members", type=int, default=5, help="Number of team members")
    parser.add_argument("--no-save", action="store_true", help="Don't save PDF to file")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Run health check first
    if not test_health_check(args.url):
        print("\n⚠️  Service may not be running. Starting test anyway...\n")
    
    # Run the test
    success = test_team_pdf_generation(
        service_url=args.url,
        company_name=args.company,
        team_name=args.team,
        num_members=args.members,
        save_pdf=not args.no_save
    )
    
    exit(0 if success else 1)
