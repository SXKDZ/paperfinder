"""
BibTeX formatter for academic papers
"""

import re
from typing import Dict, Any, Optional

def format_to_bibtex(paper_data: Dict[str, Any]) -> Optional[str]:
    """Convert paper data to BibTeX format"""
    
    if not paper_data:
        return None
    
    # Extract key information
    title = paper_data.get('title', '').strip()
    if not title:
        return None
    
    authors = paper_data.get('authors', [])
    if isinstance(authors, str):
        authors = [authors]
    
    year = paper_data.get('year', paper_data.get('published', ''))
    if isinstance(year, str) and year:
        year = re.search(r'\d{4}', year)
        year = year.group() if year else ''
    
    venue = paper_data.get('venue', paper_data.get('booktitle', paper_data.get('journal', '')))
    url = paper_data.get('url', paper_data.get('pdf_url', ''))
    arxiv_id = paper_data.get('arxiv_id', '')
    abstract = paper_data.get('abstract', paper_data.get('summary', ''))
    
    # Generate citation key
    first_author = ""
    if authors:
        first_author = authors[0].split()[-1] if isinstance(authors[0], str) else ""
    
    citation_key = f"{first_author}{year}" if first_author and year else "paper"
    
    # Determine entry type
    entry_type = "article"
    if venue:
        venue_lower = venue.lower()
        if any(conf in venue_lower for conf in ['conference', 'proceedings', 'workshop', 'symposium']):
            entry_type = "inproceedings"
        elif 'arxiv' in venue_lower:
            entry_type = "misc"
    elif arxiv_id:
        entry_type = "misc"
    
    # Build BibTeX entry
    bibtex_lines = [f"@{entry_type}{{{citation_key},"]
    
    # Add title
    clean_title = title.replace('{', '').replace('}', '').strip()
    bibtex_lines.append(f"  title = {{{clean_title}}},")
    
    # Add authors
    if authors:
        if isinstance(authors, list):
            author_str = " and ".join(str(author) for author in authors)
        else:
            author_str = str(authors)
        bibtex_lines.append(f"  author = {{{author_str}}},")
    
    # Add venue/journal/booktitle
    if venue:
        clean_venue = venue.replace('{', '').replace('}', '').strip()
        if entry_type == "inproceedings":
            bibtex_lines.append(f"  booktitle = {{{clean_venue}}},")
        elif entry_type == "article":
            bibtex_lines.append(f"  journal = {{{clean_venue}}},")
        else:
            bibtex_lines.append(f"  venue = {{{clean_venue}}},")
    
    # Add year
    if year:
        bibtex_lines.append(f"  year = {{{year}}},")
    
    # Add URL
    if url:
        bibtex_lines.append(f"  url = {{{url}}},")
    
    # Add arXiv ID if available
    if arxiv_id:
        bibtex_lines.append(f"  eprint = {{{arxiv_id}}},")
        bibtex_lines.append(f"  archivePrefix = {{arXiv}},")
    
    # Add abstract (truncated if too long)
    if abstract:
        clean_abstract = abstract.replace('{', '').replace('}', '').strip()
        if len(clean_abstract) > 500:
            clean_abstract = clean_abstract[:500] + "..."
        bibtex_lines.append(f"  abstract = {{{clean_abstract}}},")
    
    # Close the entry (remove trailing comma from last line)
    if bibtex_lines[-1].endswith(','):
        bibtex_lines[-1] = bibtex_lines[-1][:-1]
    bibtex_lines.append("}")
    
    return "\n".join(bibtex_lines)

def clean_bibtex_string(text: str) -> str:
    """Clean text for use in BibTeX entries"""
    if not text:
        return ""
    
    # Remove problematic characters
    text = text.replace('{', '').replace('}', '')
    text = text.replace('\\', '')
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    return text.strip()