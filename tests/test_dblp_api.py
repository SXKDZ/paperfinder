#!/usr/bin/env python3
"""
Test script for DBLP API endpoints
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import DBLPSearchTool

async def test_dblp_publication_search():
    """Test DBLP publication search"""
    print("=== Testing DBLP Publication Search ===")
    
    tool = DBLPSearchTool()
    test_queries = [
        "attention mechanism transformer",
        "neural networks deep learning",
        "machine learning"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        
        try:
            results = await tool.search_publications(query, max_results=3)
            
            if results:
                print(f"✅ Found {len(results)} publications:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.get('title', 'No title')}")
                    print(f"     Authors: {result.get('authors', [])}")
                    print(f"     Year: {result.get('year', 'Unknown')}")
                    print(f"     Venue: {result.get('venue', 'Unknown')}")
                    print(f"     URL: {result.get('url', 'No URL')}")
                    print(f"     Type: {result.get('type', 'Unknown')}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error during search: {e}")

async def test_dblp_author_search():
    """Test DBLP author search"""
    print("\n=== Testing DBLP Author Search ===")
    
    tool = DBLPSearchTool()
    test_queries = [
        "Yoshua Bengio",
        "Geoffrey Hinton",
        "Yann LeCun"
    ]
    
    for query in test_queries:
        print(f"\nTesting author: '{query}'")
        
        try:
            results = await tool.search_authors(query, max_results=2)
            
            if results:
                print(f"✅ Found {len(results)} authors:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.get('author', 'No name')}")
                    print(f"     URL: {result.get('url', 'No URL')}")
                    print(f"     Affiliations: {result.get('affiliations', [])}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error during search: {e}")

async def test_dblp_venue_search():
    """Test DBLP venue search"""
    print("\n=== Testing DBLP Venue Search ===")
    
    tool = DBLPSearchTool()
    test_queries = [
        "ICML",
        "NeurIPS", 
        "ICLR"
    ]
    
    for query in test_queries:
        print(f"\nTesting venue: '{query}'")
        
        try:
            results = await tool.search_venues(query, max_results=2)
            
            if results:
                print(f"✅ Found {len(results)} venues:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.get('venue', 'No name')}")
                    print(f"     Acronym: {result.get('acronym', 'No acronym')}")
                    print(f"     URL: {result.get('url', 'No URL')}")
                    print(f"     Type: {result.get('type', 'Unknown')}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error during search: {e}")

async def test_dblp_api_limits():
    """Test DBLP API limits and pagination"""
    print("\n=== Testing DBLP API Limits ===")
    
    tool = DBLPSearchTool()
    query = "machine learning"
    
    # Test different result limits
    test_limits = [1, 10, 50, 100]
    
    for limit in test_limits:
        print(f"\nTesting limit: {limit}")
        try:
            results = await tool.search_publications(query, max_results=limit)
            print(f"✅ Requested {limit}, got {len(results)} results")
        except Exception as e:
            print(f"❌ Error with limit {limit}: {e}")

def main():
    """Main test function"""
    print("DBLP API Test Suite")
    print("=" * 40)
    
    # Run all tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(test_dblp_publication_search())
        loop.run_until_complete(test_dblp_author_search())
        loop.run_until_complete(test_dblp_venue_search())
        loop.run_until_complete(test_dblp_api_limits())
    finally:
        loop.close()
    
    print("\n=== DBLP Tests Complete ===")

if __name__ == "__main__":
    main()