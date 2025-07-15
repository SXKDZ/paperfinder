#!/usr/bin/env python3
"""
Test script for Semantic Scholar API endpoints
"""

import asyncio
import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import SemanticScholarTool

def load_environment():
    """Load environment variables"""
    load_dotenv()
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    
    print("=== Semantic Scholar API Configuration ===")
    print(f"API Key present: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API Key (first 10 chars): {api_key[:10]}...")
    
    return api_key

async def test_semantic_scholar_paper_search():
    """Test Semantic Scholar paper search"""
    print("\n=== Testing Semantic Scholar Paper Search ===")
    
    tool = SemanticScholarTool()
    test_queries = [
        "attention is all you need",
        "BERT language model",
        "graph neural networks"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        
        try:
            results = await tool.search_papers(query, max_results=3)
            
            if results:
                print(f"✅ Found {len(results)} papers:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.get('title', 'No title')}")
                    print(f"     Authors: {result.get('authors', [])}")
                    print(f"     Year: {result.get('year', 'Unknown')}")
                    print(f"     Venue: {result.get('venue', 'Unknown')}")
                    print(f"     Citations: {result.get('citations', 0)}")
                    print(f"     DOI: {result.get('doi', 'No DOI')}")
                    print(f"     ArXiv ID: {result.get('arxiv_id', 'No ArXiv ID')}")
                    if result.get('pdf_url'):
                        print(f"     PDF Available: Yes")
                    print(f"     Fields: {result.get('fields_of_study', [])}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error during search: {e}")

async def test_semantic_scholar_author_search():
    """Test Semantic Scholar author search"""
    print("\n=== Testing Semantic Scholar Author Search ===")
    
    tool = SemanticScholarTool()
    test_queries = [
        "Ashish Vaswani",
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
                    print(f"  {i}. {result.get('name', 'No name')}")
                    print(f"     Author ID: {result.get('author_id', 'No ID')}")
                    print(f"     Paper Count: {result.get('paper_count', 0)}")
                    print(f"     Citation Count: {result.get('citation_count', 0)}")
                    print(f"     H-Index: {result.get('h_index', 0)}")
                    print(f"     Affiliations: {result.get('affiliations', [])}")
                    print(f"     URL: {result.get('url', 'No URL')}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error during search: {e}")

async def test_semantic_scholar_filtered_search():
    """Test Semantic Scholar filtered search"""
    print("\n=== Testing Semantic Scholar Filtered Search ===")
    
    tool = SemanticScholarTool()
    
    # Test with filters
    test_cases = [
        {
            "query": "transformer",
            "filters": {"year": "2017"},
            "description": "Transformer papers from 2017"
        },
        {
            "query": "neural networks",
            "filters": {"minCitationCount": 100},
            "description": "Neural network papers with 100+ citations"
        },
        {
            "query": "machine learning",
            "filters": {"publicationTypes": ["JournalArticle"]},
            "description": "Machine learning journal articles"
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['description']}")
        print(f"Query: '{case['query']}' with filters: {case['filters']}")
        
        try:
            results = await tool.search_papers(case['query'], max_results=3, **case['filters'])
            
            if results:
                print(f"✅ Found {len(results)} papers:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.get('title', 'No title')}")
                    print(f"     Year: {result.get('year', 'Unknown')}")
                    print(f"     Citations: {result.get('citations', 0)}")
                    print(f"     Types: {result.get('publication_types', [])}")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error during search: {e}")

async def test_semantic_scholar_paper_details():
    """Test getting paper details by ID"""
    print("\n=== Testing Semantic Scholar Paper Details ===")
    
    tool = SemanticScholarTool()
    
    # First search for a paper to get ID
    print("First searching for 'attention is all you need' to get paper ID...")
    try:
        results = await tool.search_papers("attention is all you need", max_results=1)
        if results:
            paper_id = results[0].get('paper_id')
            print(f"Found paper ID: {paper_id}")
            
            # Now get detailed information
            print("\nGetting detailed paper information...")
            details = await tool.get_paper_details(paper_id)
            
            if details:
                print("✅ Paper details retrieved:")
                print(f"  Title: {details.get('title', 'No title')}")
                print(f"  Citation Count: {details.get('citationCount', 0)}")
                print(f"  Reference Count: {details.get('referenceCount', 0)}")
                print(f"  Fields of Study: {details.get('fieldsOfStudy', [])}")
                print(f"  Publication Types: {details.get('publicationTypes', [])}")
            else:
                print("❌ No details found")
        else:
            print("❌ No papers found to test details")
            
    except Exception as e:
        print(f"❌ Error during paper details test: {e}")

async def test_semantic_scholar_author_details():
    """Test getting author details by ID"""
    print("\n=== Testing Semantic Scholar Author Details ===")
    
    tool = SemanticScholarTool()
    
    # First search for an author to get ID
    print("First searching for 'Ashish Vaswani' to get author ID...")
    try:
        results = await tool.search_authors("Ashish Vaswani", max_results=1)
        if results:
            author_id = results[0].get('author_id')
            print(f"Found author ID: {author_id}")
            
            # Now get detailed information
            print("\nGetting detailed author information...")
            details = await tool.get_author_details(author_id)
            
            if details:
                print("✅ Author details retrieved:")
                print(f"  Name: {details.get('name', 'No name')}")
                print(f"  Paper Count: {details.get('paperCount', 0)}")
                print(f"  Citation Count: {details.get('citationCount', 0)}")
                print(f"  H-Index: {details.get('hIndex', 0)}")
                print(f"  Affiliations: {details.get('affiliations', [])}")
                
                # Show some papers
                papers = details.get('papers', [])
                if papers:
                    print(f"  Recent Papers ({len(papers[:3])} of {len(papers)}):")
                    for i, paper in enumerate(papers[:3], 1):
                        print(f"    {i}. {paper.get('title', 'No title')} ({paper.get('year', 'Unknown')})")
            else:
                print("❌ No details found")
        else:
            print("❌ No authors found to test details")
            
    except Exception as e:
        print(f"❌ Error during author details test: {e}")

def main():
    """Main test function"""
    print("Semantic Scholar API Test Suite")
    print("=" * 40)
    
    # Load environment
    api_key = load_environment()
    
    # Run all tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(test_semantic_scholar_paper_search())
        loop.run_until_complete(test_semantic_scholar_author_search())
        loop.run_until_complete(test_semantic_scholar_filtered_search())
        loop.run_until_complete(test_semantic_scholar_paper_details())
        loop.run_until_complete(test_semantic_scholar_author_details())
    finally:
        loop.close()
    
    print("\n=== Semantic Scholar Tests Complete ===")

if __name__ == "__main__":
    main()