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
    """Search DBLP database"""

    def __init__(self):
        super().__init__("dblp_search")
        self.base_url = "https://dblp.org/search/publ/api"

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search DBLP database"""
        try:
            params = {"q": query, "format": "json", "h": max_results}

            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()

            results = []
            hits = data.get("result", {}).get("hits", {}).get("hit", [])

            for hit in hits:
                info = hit.get("info", {})
                result = {
                    "title": info.get("title", ""),
                    "authors": info.get("authors", {}).get("author", []),
                    "year": info.get("year", ""),
                    "venue": info.get("venue", ""),
                    "url": info.get("url", ""),
                    "type": info.get("type", ""),
                    "source": "dblp",
                }
                results.append(result)

            return results
        except Exception as e:
            print(f"DBLP search error: {e}")
            return []


class SemanticScholarTool(SearchTool):
    """Search Semantic Scholar"""

    def __init__(self):
        super().__init__("semantic_scholar")
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search Semantic Scholar"""
        try:
            # Updated API endpoint and parameters
            params = {
                "query": query,
                "limit": max_results,
                "fields": "paperId,title,authors,abstract,year,venue,externalIds,citationCount,publicationDate",
            }

            headers = {"User-Agent": "PaperFinder/1.0 (research@example.com)"}

            response = requests.get(
                self.base_url, params=params, headers=headers, timeout=15
            )

            if response.status_code == 429:
                print(
                    f"Semantic Scholar API rate limited (429). Waiting before retry..."
                )
                import time

                time.sleep(2)  # Wait 2 seconds before returning empty results
                return []
            elif response.status_code != 200:
                print(
                    f"Semantic Scholar API error: {response.status_code} - {response.text}"
                )
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
                    "source": "semantic_scholar",
                }
                results.append(result)

            return results
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []


class GoogleSearchTool(SearchTool):
    """Google search for academic papers"""

    def __init__(self):
        super().__init__("google_search")

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search Google for academic content"""
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
                    }
                    results.append(result_data)

            return results
        except Exception as e:
            print(f"Google search error: {e}")
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
