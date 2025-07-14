"""
Interactive tools for file operations, web browsing, and PDF processing
Similar to Claude Code capabilities
"""

import os
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import PyPDF2
import pdfplumber
import re
import tempfile
from urllib.parse import urlparse, urljoin

class FileManager:
    """File operations manager"""
    
    def __init__(self, download_dir: str = "./downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
    
    async def download_file(self, url: str, filename: Optional[str] = None) -> str:
        """Download file from URL"""
        try:
            if not filename:
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path) or "downloaded_file"
            
            filepath = self.download_dir / filename
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        async with aiofiles.open(filepath, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        return str(filepath)
                    else:
                        raise Exception(f"HTTP {response.status}: Failed to download {url}")
        except Exception as e:
            raise Exception(f"Download failed: {e}")
    
    async def read_text_file(self, filepath: str) -> str:
        """Read text file content"""
        try:
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception as e:
            raise Exception(f"Failed to read file {filepath}: {e}")
    
    def list_files(self, directory: str = None) -> List[str]:
        """List files in directory"""
        target_dir = Path(directory) if directory else self.download_dir
        return [str(f) for f in target_dir.glob('*') if f.is_file()]

class PDFProcessor:
    """PDF processing and text extraction"""
    
    @staticmethod
    async def extract_text_pypdf2(pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            import warnings
            # Suppress PDF parsing warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*invalid float value.*")
            
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_error:
                        # Skip problematic pages but continue
                        print(f"Warning: Could not extract text from page: {page_error}")
                        continue
            return text
        except Exception as e:
            raise Exception(f"PyPDF2 extraction failed: {e}")
    
    @staticmethod
    async def extract_text_pdfplumber(pdf_path: str) -> str:
        """Extract text using pdfplumber (better for complex layouts)"""
        try:
            import warnings
            import logging
            # Suppress PDF parsing warnings and errors
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*invalid float value.*")
            logging.getLogger("pdfplumber").setLevel(logging.ERROR)
            
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_error:
                        # Skip problematic pages but continue
                        print(f"Warning: Could not extract text from page {i+1}: {page_error}")
                        continue
            return text
        except Exception as e:
            raise Exception(f"pdfplumber extraction failed: {e}")
    
    @staticmethod
    async def extract_metadata(pdf_path: str) -> Dict[str, Any]:
        """Extract PDF metadata"""
        try:
            import warnings
            # Suppress PDF parsing warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*invalid float value.*")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata or {}
                
                return {
                    'title': metadata.get('/Title', '') if metadata else '',
                    'author': metadata.get('/Author', '') if metadata else '',
                    'subject': metadata.get('/Subject', '') if metadata else '',
                    'creator': metadata.get('/Creator', '') if metadata else '',
                    'producer': metadata.get('/Producer', '') if metadata else '',
                    'creation_date': str(metadata.get('/CreationDate', '')) if metadata else '',
                    'modification_date': str(metadata.get('/ModDate', '')) if metadata else '',
                    'pages': len(pdf_reader.pages) if pdf_reader.pages else 0
                }
        except Exception as e:
            return {'error': f"Metadata extraction failed: {e}"}
    
    @staticmethod
    async def extract_references(pdf_text: str) -> List[str]:
        """Extract references from PDF text"""
        try:
            # Look for references section
            text_lower = pdf_text.lower()
            
            # Find references section
            ref_patterns = [
                r'references\s*\n(.*?)(?:\n\s*appendix|\n\s*acknowledgment|\Z)',
                r'bibliography\s*\n(.*?)(?:\n\s*appendix|\n\s*acknowledgment|\Z)',
                r'\[\d+\][^\n]*(?:\n[^\n\[\]]*)*'
            ]
            
            references = []
            for pattern in ref_patterns:
                matches = re.findall(pattern, pdf_text, re.DOTALL | re.IGNORECASE)
                if matches:
                    # Split references by common patterns
                    ref_text = matches[0]
                    refs = re.split(r'\n(?=\[\d+\]|\d+\.)', ref_text)
                    references.extend([ref.strip() for ref in refs if ref.strip()])
                    break
            
            return references[:20]  # Limit to first 20 references
        except Exception as e:
            return [f"Reference extraction failed: {e}"]

class WebBrowser:
    """Web browsing and content extraction"""
    
    @staticmethod
    async def fetch_webpage(url: str) -> Dict[str, Any]:
        """Fetch and parse webpage content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Extract metadata
                        title = soup.find('title')
                        title = title.get_text().strip() if title else ""
                        
                        # Remove script and style elements
                        for element in soup(["script", "style", "nav", "header", "footer"]):
                            element.decompose()
                        
                        # Get main content
                        main_content = soup.get_text()
                        # Clean up whitespace
                        main_content = re.sub(r'\n\s*\n', '\n\n', main_content)
                        main_content = re.sub(r' +', ' ', main_content)
                        
                        return {
                            'url': url,
                            'title': title,
                            'content': main_content,
                            'status': response.status,
                            'content_type': response.headers.get('content-type', '')
                        }
                    else:
                        return {
                            'url': url,
                            'error': f"HTTP {response.status}",
                            'status': response.status
                        }
        except Exception as e:
            return {
                'url': url,
                'error': f"Failed to fetch webpage: {e}"
            }
    
    @staticmethod
    async def extract_paper_info_from_webpage(url: str) -> Dict[str, Any]:
        """Extract paper information from academic webpages"""
        try:
            webpage_data = await WebBrowser.fetch_webpage(url)
            
            if 'error' in webpage_data:
                return webpage_data
            
            content = webpage_data['content']
            title = webpage_data['title']
            
            # Extract paper information using regex patterns
            info = {
                'url': url,
                'title': title,
                'content_preview': content[:500] + "..." if len(content) > 500 else content
            }
            
            # Look for arXiv ID in content
            arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5})', content)
            if arxiv_match:
                info['arxiv_id'] = arxiv_match.group(1)
            
            # Look for DOI
            doi_match = re.search(r'doi:?\s*([10]\.\d+/[^\s]+)', content, re.IGNORECASE)
            if doi_match:
                info['doi'] = doi_match.group(1)
            
            # Look for author information
            author_patterns = [
                r'(?:authors?|by)\s*:?\s*([^.]+)',
                r'<meta name="author" content="([^"]+)"',
                r'<meta property="article:author" content="([^"]+)"'
            ]
            
            for pattern in author_patterns:
                author_match = re.search(pattern, content, re.IGNORECASE)
                if author_match:
                    info['authors'] = author_match.group(1).strip()
                    break
            
            # Look for publication year
            year_match = re.search(r'(?:20\d{2})', title + " " + content)
            if year_match:
                info['year'] = year_match.group()
            
            return info
            
        except Exception as e:
            return {'url': url, 'error': f"Paper info extraction failed: {e}"}

class URLExtractor:
    """Extract and process URLs from various sources"""
    
    @staticmethod
    def extract_urls_from_text(text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?]'
        urls = re.findall(url_pattern, text)
        return list(set(urls))  # Remove duplicates
    
    @staticmethod
    def is_pdf_url(url: str) -> bool:
        """Check if URL points to a PDF"""
        return url.lower().endswith('.pdf') or 'pdf' in url.lower()
    
    @staticmethod
    def is_arxiv_url(url: str) -> bool:
        """Check if URL is from arXiv"""
        return 'arxiv.org' in url.lower()
    
    @staticmethod
    def extract_arxiv_id_from_url(url: str) -> Optional[str]:
        """Extract arXiv ID from URL"""
        match = re.search(r'arxiv\.org/(?:abs/|pdf/)?(\d{4}\.\d{4,5})', url)
        return match.group(1) if match else None
    
    @staticmethod
    def is_acl_anthology_url(url: str) -> bool:
        """Check if URL is from ACL Anthology"""
        return 'aclanthology.org' in url.lower()
    
    @staticmethod
    def get_acl_pdf_url(url: str) -> str:
        """Convert ACL Anthology paper URL to PDF URL"""
        if url.endswith('.pdf'):
            return url
        elif url.endswith('/'):
            return url[:-1] + '.pdf'
        else:
            return url + '.pdf'

# Global instances
file_manager = FileManager()
pdf_processor = PDFProcessor()
web_browser = WebBrowser()
url_extractor = URLExtractor()