"""
Search tools for different academic databases and search engines
"""

import asyncio
import requests
from typing import List, Dict, Any, Optional
import arxiv

try:
    from scholarly import scholarly
except ImportError:
    scholarly = None
from bs4 import BeautifulSoup
import json
import re


class SearchTool:
    """Base class for search tools"""

    def __init__(self, name: str):
        self.name = name

    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Perform search and return results"""
        raise NotImplementedError


class ArxivSearchTool(SearchTool):
    """Search arXiv papers"""

    def __init__(self):
        super().__init__("arxiv_search")
        self.client = arxiv.Client()

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search arXiv papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            results = []
            for paper in self.client.results(search):
                result = {
                    "title": paper.title,
                    "authors": [str(author) for author in paper.authors],
                    "abstract": paper.summary,
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "published": (
                        paper.published.strftime("%Y-%m-%d")
                        if paper.published
                        else None
                    ),
                    "arxiv_id": paper.entry_id.split("/")[-1],
                    "source": "arxiv",
                }
                results.append(result)

            return results
        except Exception as e:
            print(f"ArXiv search error: {e}")
            return []


class ArxivDirectTool(SearchTool):
    """Get paper directly by arXiv ID"""

    def __init__(self):
        super().__init__("arxiv_direct")
        self.client = arxiv.Client()

    async def search(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Get paper by arXiv ID"""
        try:
            search = arxiv.Search(id_list=[arxiv_id])

            results = []
            for paper in self.client.results(search):
                result = {
                    "title": paper.title,
                    "authors": [str(author) for author in paper.authors],
                    "abstract": paper.summary,
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "published": (
                        paper.published.strftime("%Y-%m-%d")
                        if paper.published
                        else None
                    ),
                    "arxiv_id": arxiv_id,
                    "source": "arxiv",
                }
                results.append(result)

            return results
        except Exception as e:
            print(f"ArXiv direct error: {e}")
            return []


class GoogleScholarTool(SearchTool):
    """Search Google Scholar"""

    def __init__(self):
        super().__init__("google_scholar")

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search Google Scholar"""
        if scholarly is None:
            print("Scholarly library not available")
            return []

        try:
            search_query = scholarly.search_pubs(query)
            results = []

            count = 0
            for paper in search_query:
                if count >= max_results:
                    break

                result = {
                    "title": paper.get("bib", {}).get("title", ""),
                    "authors": paper.get("bib", {}).get("author", []),
                    "abstract": paper.get("bib", {}).get("abstract", ""),
                    "url": paper.get("pub_url", ""),
                    "year": paper.get("bib", {}).get("pub_year", ""),
                    "venue": paper.get("bib", {}).get("venue", ""),
                    "citations": paper.get("num_citations", 0),
                    "source": "google_scholar",
                }
                results.append(result)
                count += 1

            return results
        except Exception as e:
            print(f"Google Scholar search error: {e}")
            return []


class DBLPSearchTool(SearchTool):
    """Search DBLP database with publication, author, and venue search capabilities"""

    def __init__(self):
        super().__init__("dblp_search")
        self.endpoints = {
            "publication": "https://dblp.org/search/publ/api",
            "author": "https://dblp.org/search/author/api", 
            "venue": "https://dblp.org/search/venue/api"
        }

    async def search(self, query: str, max_results: int = 5, search_type: str = "publication") -> List[Dict[str, Any]]:
        """Search DBLP database
        
        Args:
            query: Search query
            max_results: Maximum number of results (max 1000, default 30)
            search_type: Type of search - "publication", "author", or "venue"
        """
        try:
            if search_type not in self.endpoints:
                search_type = "publication"
                
            base_url = self.endpoints[search_type]
            params = {
                "q": query, 
                "format": "json", 
                "h": min(max_results, 1000),  # DBLP max is 1000
                "c": 10  # Completion terms
            }

            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()

            results = []
            hits = data.get("result", {}).get("hits", {}).get("hit", [])

            for hit in hits:
                info = hit.get("info", {})
                
                if search_type == "publication":
                    result = {
                        "title": info.get("title", ""),
                        "authors": info.get("authors", {}).get("author", []),
                        "year": info.get("year", ""),
                        "venue": info.get("venue", ""),
                        "url": info.get("url", ""),
                        "type": info.get("type", ""),
                        "doi": info.get("doi", ""),
                        "source": "dblp",
                        "search_type": "publication"
                    }
                elif search_type == "author":
                    result = {
                        "author": info.get("author", ""),
                        "url": info.get("url", ""),
                        "affiliations": info.get("note", []),
                        "source": "dblp",
                        "search_type": "author"
                    }
                elif search_type == "venue":
                    result = {
                        "venue": info.get("venue", ""),
                        "acronym": info.get("acronym", ""),
                        "url": info.get("url", ""),
                        "type": info.get("type", ""),
                        "source": "dblp", 
                        "search_type": "venue"
                    }
                    
                results.append(result)

            return results
        except Exception as e:
            print(f"DBLP search error: {e}")
            return []

    async def search_publications(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for publications"""
        return await self.search(query, max_results, "publication")
        
    async def search_authors(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for authors"""
        return await self.search(query, max_results, "author")
        
    async def search_venues(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for venues (conferences, journals)"""
        return await self.search(query, max_results, "venue")


class SemanticScholarTool(SearchTool):
    """Search Semantic Scholar with comprehensive API endpoints"""

    def __init__(self):
        super().__init__("semantic_scholar")
        import os
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.endpoints = {
            "paper_search": f"{self.base_url}/paper/search",
            "paper_batch": f"{self.base_url}/paper/batch",
            "paper_details": f"{self.base_url}/paper",
            "author_search": f"{self.base_url}/author/search",
            "author_batch": f"{self.base_url}/author/batch",
            "author_details": f"{self.base_url}/author",
            "snippet_search": f"{self.base_url}/snippet/search"
        }

    def _get_headers(self):
        """Get headers with optional API key"""
        headers = {"User-Agent": "PaperFinder/1.0 (research@example.com)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search Semantic Scholar papers"""
        return await self.search_papers(query, max_results)

    async def search_papers(self, query: str, max_results: int = 5, **filters) -> List[Dict[str, Any]]:
        """Search for papers with optional filters"""
        try:
            params = {
                "query": query,
                "limit": min(max_results, 100),  # API max is 100
                "fields": "paperId,title,authors,abstract,year,venue,externalIds,citationCount,publicationDate,referenceCount,influentialCitationCount,fieldsOfStudy,publicationTypes,publicationVenue,openAccessPdf",
            }
            
            # Add filters if provided
            if filters.get("year"):
                params["year"] = filters["year"]
            if filters.get("venue"):
                params["venue"] = filters["venue"]
            if filters.get("fieldsOfStudy"):
                params["fieldsOfStudy"] = filters["fieldsOfStudy"]
            if filters.get("publicationTypes"):
                params["publicationTypes"] = filters["publicationTypes"]
            if filters.get("minCitationCount"):
                params["minCitationCount"] = filters["minCitationCount"]
            if filters.get("publicationDateOrYear"):
                params["publicationDateOrYear"] = filters["publicationDateOrYear"]
            if filters.get("openAccessPdf"):
                params["openAccessPdf"] = filters["openAccessPdf"]

            response = requests.get(
                self.endpoints["paper_search"], 
                params=params, 
                headers=self._get_headers(), 
                timeout=15
            )

            if response.status_code == 429:
                print("Semantic Scholar API rate limited (429). Waiting before retry...")
                import time
                time.sleep(2)
                return []
            elif response.status_code != 200:
                print(f"Semantic Scholar API error: {response.status_code} - {response.text}")
                return []

            data = response.json()
            results = []
            papers = data.get("data", [])

            if not papers:
                print(f"Semantic Scholar: No papers found for query '{query}'")
                return []

            for paper in papers:
                # Extract external IDs
                external_ids = paper.get("externalIds", {}) or {}
                arxiv_id = external_ids.get("ArXiv", "")
                doi = external_ids.get("DOI", "")

                result = {
                    "title": paper.get("title", ""),
                    "authors": [
                        author.get("name", "") for author in paper.get("authors", [])
                    ],
                    "abstract": paper.get("abstract", ""),
                    "year": str(paper.get("year", "")),
                    "venue": paper.get("venue", ""),
                    "url": f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}",
                    "citations": paper.get("citationCount", 0),
                    "paper_id": paper.get("paperId", ""),
                    "doi": doi,
                    "arxiv_id": arxiv_id,
                    "publication_date": paper.get("publicationDate", ""),
                    "reference_count": paper.get("referenceCount", 0),
                    "influential_citations": paper.get("influentialCitationCount", 0),
                    "fields_of_study": paper.get("fieldsOfStudy", []),
                    "publication_types": paper.get("publicationTypes", []),
                    "pdf_url": paper.get("openAccessPdf", {}).get("url", "") if paper.get("openAccessPdf") else "",
                    "source": "semantic_scholar",
                }
                results.append(result)

            return results
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []

    async def search_authors(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for authors"""
        try:
            params = {
                "query": query,
                "limit": min(max_results, 100),
                "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex,papers.paperId,papers.title,papers.year"
            }

            response = requests.get(
                self.endpoints["author_search"], 
                params=params, 
                headers=self._get_headers(), 
                timeout=15
            )

            if response.status_code != 200:
                print(f"Semantic Scholar author search error: {response.status_code}")
                return []

            data = response.json()
            results = []
            authors = data.get("data", [])

            for author in authors:
                result = {
                    "author_id": author.get("authorId", ""),
                    "name": author.get("name", ""),
                    "affiliations": author.get("affiliations", []),
                    "paper_count": author.get("paperCount", 0),
                    "citation_count": author.get("citationCount", 0),
                    "h_index": author.get("hIndex", 0),
                    "papers": author.get("papers", []),
                    "url": f"https://www.semanticscholar.org/author/{author.get('authorId', '')}",
                    "source": "semantic_scholar",
                    "search_type": "author"
                }
                results.append(result)

            return results
        except Exception as e:
            print(f"Semantic Scholar author search error: {e}")
            return []

    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """Get detailed paper information by ID"""
        try:
            fields = "paperId,title,authors,abstract,year,venue,externalIds,citationCount,referenceCount,citations,references,fieldsOfStudy,publicationTypes,publicationVenue,openAccessPdf"
            
            response = requests.get(
                f"{self.endpoints['paper_details']}/{paper_id}",
                params={"fields": fields},
                headers=self._get_headers(),
                timeout=15
            )

            if response.status_code != 200:
                print(f"Paper details error: {response.status_code}")
                return {}

            return response.json()
        except Exception as e:
            print(f"Paper details error: {e}")
            return {}

    async def get_author_details(self, author_id: str) -> Dict[str, Any]:
        """Get detailed author information by ID"""
        try:
            fields = "authorId,name,affiliations,paperCount,citationCount,hIndex,papers.paperId,papers.title,papers.year,papers.citationCount"
            
            response = requests.get(
                f"{self.endpoints['author_details']}/{author_id}",
                params={"fields": fields},
                headers=self._get_headers(),
                timeout=15
            )

            if response.status_code != 200:
                print(f"Author details error: {response.status_code}")
                return {}

            return response.json()
        except Exception as e:
            print(f"Author details error: {e}")
            return {}


class GoogleSearchTool(SearchTool):
    """Google search for academic papers using Custom Search JSON API with fallback to web scraping"""

    def __init__(self):
        super().__init__("google_search")
        import os
        self.api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
        self.cx = os.getenv("GOOGLE_SEARCH_CX")  # Custom Search Engine ID
        self.api_base = "https://www.googleapis.com/customsearch/v1"

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search Google for academic content using Custom Search API with fallback"""
        
        # Try Custom Search API first if credentials are available
        if self.api_key and self.cx:
            try:
                return await self._search_custom_api(query, max_results)
            except Exception as e:
                print(f"Custom Search API failed: {e}. Falling back to web scraping...")
        
        # Fallback to web scraping
        return await self._search_web_scraping(query, max_results)

    async def _search_custom_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search JSON API"""
        academic_query = f"{query} filetype:pdf OR site:arxiv.org OR site:scholar.google.com"
        
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": academic_query,
            "num": min(max_results, 10),  # API max is 10 per request
            "fields": "items(title,link,snippet,displayLink,formattedUrl)"
        }
        
        response = requests.get(self.api_base, params=params, timeout=10)
        
        if response.status_code == 403:
            raise Exception("API quota exceeded or access denied")
        elif response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        data = response.json()
        results = []
        
        for item in data.get("items", [])[:max_results]:
            result_data = {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "display_link": item.get("displayLink", ""),
                "source": "google_custom_search",
                "api_used": True
            }
            results.append(result_data)
        
        return results

    async def _search_web_scraping(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback web scraping method"""
        try:
            # Add academic keywords to improve results
            academic_query = (
                f"{query} filetype:pdf OR site:arxiv.org OR site:scholar.google.com"
            )

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            params = {"q": academic_query, "num": max_results}

            response = requests.get(
                "https://www.google.com/search",
                params=params,
                headers=headers,
                timeout=10,
            )
            soup = BeautifulSoup(response.content, "html.parser")

            results = []
            search_results = soup.find_all("div", class_="g")

            for result in search_results[:max_results]:
                title_elem = result.find("h3")
                link_elem = result.find("a")
                snippet_elem = result.find("span", class_=["aCOpRe", "st"])

                if title_elem and link_elem:
                    result_data = {
                        "title": title_elem.get_text(),
                        "url": link_elem.get("href", ""),
                        "snippet": snippet_elem.get_text() if snippet_elem else "",
                        "source": "google_search",
                        "api_used": False
                    }
                    results.append(result_data)

            return results
        except Exception as e:
            print(f"Google web scraping error: {e}")
            return []


class ACLAnthologyTool(SearchTool):
    """Search ACL Anthology for computational linguistics and NLP papers"""

    def __init__(self):
        super().__init__("acl_anthology")
        self.base_url = "https://aclanthology.org"

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search ACL Anthology"""
        try:
            # ACL Anthology search endpoint
            search_url = f"{self.base_url}/search/"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            params = {"q": query}

            response = requests.get(
                search_url, params=params, headers=headers, timeout=15
            )

            if response.status_code != 200:
                print(f"ACL Anthology search failed: {response.status_code}")
                return []

            soup = BeautifulSoup(response.content, "html.parser")
            results = []

            # Find paper results in the search page
            paper_items = soup.find_all("div", class_="paper-item") or soup.find_all(
                "div", class_="search-result"
            )

            if not paper_items:
                # Fallback: look for any links to papers
                paper_items = soup.find_all("a", href=re.compile(r"/\d{4}\.\w+"))

            for item in paper_items[:max_results]:
                try:
                    # Extract title
                    title_elem = (
                        item.find("h3")
                        or item.find("h4")
                        or item.find("strong")
                        or item
                    )
                    title = title_elem.get_text().strip() if title_elem else ""

                    # Extract URL
                    link_elem = item.find("a") if item.name != "a" else item
                    url = ""
                    if link_elem and link_elem.get("href"):
                        href = link_elem.get("href")
                        url = (
                            href
                            if href.startswith("http")
                            else f"{self.base_url}{href}"
                        )

                    # Extract authors
                    authors = []
                    author_elem = item.find("span", class_="author") or item.find(
                        "div", class_="authors"
                    )
                    if author_elem:
                        authors = [
                            author.strip()
                            for author in author_elem.get_text().split(",")
                        ]

                    # Extract venue/year
                    venue = ""
                    year = ""
                    venue_elem = item.find("span", class_="venue") or item.find(
                        "div", class_="venue"
                    )
                    if venue_elem:
                        venue_text = venue_elem.get_text().strip()
                        venue = venue_text
                        # Extract year from venue
                        year_match = re.search(r"(\d{4})", venue_text)
                        if year_match:
                            year = year_match.group(1)

                    # Extract paper ID from URL
                    paper_id = ""
                    if url:
                        id_match = re.search(r"/(\d{4}\.\w+)", url)
                        if id_match:
                            paper_id = id_match.group(1)

                    if title and url:
                        result = {
                            "title": title,
                            "authors": authors,
                            "url": url,
                            "venue": venue,
                            "year": year,
                            "paper_id": paper_id,
                            "pdf_url": (
                                f"{url}.pdf"
                                if url and not url.endswith(".pdf")
                                else url
                            ),
                            "source": "acl_anthology",
                        }
                        results.append(result)

                except Exception as e:
                    print(f"Error parsing ACL item: {e}")
                    continue

            return results

        except Exception as e:
            print(f"ACL Anthology search error: {e}")
            return []


# Collection of all search tools
SEARCH_TOOLS = {
    "arxiv_search": ArxivSearchTool(),
    "arxiv_direct": ArxivDirectTool(),
    "google_scholar": GoogleScholarTool(),
    "dblp_search": DBLPSearchTool(),
    "semantic_scholar": SemanticScholarTool(),
    "google_search": GoogleSearchTool(),
    "acl_anthology": ACLAnthologyTool(),
}
