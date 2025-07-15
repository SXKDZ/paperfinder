#!/usr/bin/env python3
"""
Test script for Google Custom Search JSON API
"""

import os
import asyncio
import sys
from dotenv import load_dotenv
from search_tools import GoogleSearchTool

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_SEARCH_CX")
    
    print("=== Google Custom Search API Configuration ===")
    print(f"API Key present: {'Yes' if api_key else 'No'}")
    print(f"Search Engine ID present: {'Yes' if cx else 'No'}")
    
    if api_key:
        print(f"API Key (first 10 chars): {api_key[:10]}...")
    if cx:
        print(f"Search Engine ID: {cx}")
    
    return api_key, cx

async def test_google_search():
    """Test Google Custom Search API"""
    print("\n=== Testing Google Custom Search ===")
    
    tool = GoogleSearchTool()
    test_query = "attention is all you need transformer"
    
    print(f"Testing query: '{test_query}'")
    print("Searching...")
    
    try:
        results = await tool.search(test_query, max_results=3)
        
        if results:
            print(f"\n✅ Success! Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.get('title', 'No title')}")
                print(f"   URL: {result.get('url', 'No URL')}")
                print(f"   Source: {result.get('source', 'Unknown')}")
                print(f"   API Used: {result.get('api_used', 'Unknown')}")
                if result.get('snippet'):
                    snippet = result['snippet'][:100] + "..." if len(result['snippet']) > 100 else result['snippet']
                    print(f"   Snippet: {snippet}")
        else:
            print("❌ No results found")
            
    except Exception as e:
        print(f"❌ Error during search: {e}")

async def test_api_directly():
    """Test the API directly with a simple request"""
    print("\n=== Direct API Test ===")
    
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_SEARCH_CX")
    
    if not api_key or not cx:
        print("❌ Missing API credentials - skipping direct test")
        return
    
    import requests
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": "test query",
        "num": 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            total_results = data.get("searchInformation", {}).get("totalResults", "0")
            print(f"✅ API working! Total results available: {total_results}")
        elif response.status_code == 403:
            print("❌ Access denied - check API key and permissions")
        else:
            print(f"❌ API error: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def main():
    """Main test function"""
    print("Google Custom Search API Test")
    print("=" * 40)
    
    # Load environment
    api_key, cx = load_environment()
    
    if not api_key or not cx:
        print("\n❌ Missing required environment variables:")
        if not api_key:
            print("  - GOOGLE_SEARCH_API_KEY")
        if not cx:
            print("  - GOOGLE_SEARCH_CX")
        print("\nPlease set these in your .env file and try again.")
        return
    
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(test_api_directly())
        loop.run_until_complete(test_google_search())
    finally:
        loop.close()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()