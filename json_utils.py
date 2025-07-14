"""
JSON utilities for handling academic paper data with special characters
"""

import json
import re
from typing import Any, Dict, List

def clean_text_for_json(text: str) -> str:
    """Clean text content to be JSON-safe"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove or replace problematic characters
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = text.replace('\r', ' ')  # Replace carriage returns
    text = text.replace('\t', ' ')  # Replace tabs
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.replace('"', "'")  # Replace double quotes with single quotes
    text = text.replace('\\', '/')  # Replace backslashes
    text = text.strip()
    
    return text

def clean_paper_data(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a paper data dictionary for JSON serialization"""
    cleaned = {}
    
    for key, value in paper.items():
        if isinstance(value, str):
            cleaned[key] = clean_text_for_json(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_text_for_json(str(item)) if isinstance(item, str) else item for item in value]
        else:
            cleaned[key] = value
    
    return cleaned

def safe_json_dumps(data: Any, **kwargs) -> str:
    """Safely serialize data to JSON with proper error handling"""
    try:
        # Clean the data if it's a list of papers
        if isinstance(data, list) and data and isinstance(data[0], dict):
            cleaned_data = [clean_paper_data(paper) for paper in data]
        else:
            cleaned_data = data
        
        return json.dumps(cleaned_data, ensure_ascii=False, **kwargs)
    except (TypeError, ValueError) as e:
        print(f"JSON serialization error: {e}")
        return "[]"  # Return empty array as fallback