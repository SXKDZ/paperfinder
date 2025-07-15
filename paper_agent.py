"""
PaperFinder LLM Agent using LangGraph with ReAct reasoning
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from search_tools import SEARCH_TOOLS
from json_utils import safe_json_dumps


class AgentState(TypedDict):
    """State for the paper search agent"""

    messages: Annotated[List, add_messages]
    search_results: List[Dict[str, Any]]
    final_answer: str
    reasoning_steps: List[str]
    console: Any  # For displaying progress
    query_id: str  # For logging
    logger: Any  # Logger instance
    iteration_count: int  # To prevent infinite loops
    tools_used: List[str]  # Track which tools have been called


@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers. Use for general paper searches."""
    try:
        results = asyncio.run(SEARCH_TOOLS["arxiv_search"].search(query, max_results))
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error searching arXiv: {e}"


@tool
def arxiv_direct(arxiv_id: str) -> str:
    """Get paper directly by arXiv ID (e.g., '2505.15134'). Use when you identify an arXiv ID."""
    try:
        results = asyncio.run(SEARCH_TOOLS["arxiv_direct"].search(arxiv_id))
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error getting arXiv paper: {e}"


@tool
def google_scholar_search(query: str, max_results: int = 5) -> str:
    """Search Google Scholar for academic papers. Good for finding highly cited papers."""
    try:
        results = asyncio.run(SEARCH_TOOLS["google_scholar"].search(query, max_results))
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error searching Google Scholar: {e}"


@tool
def dblp_search(query: str, max_results: int = 5) -> str:
    """Search DBLP computer science bibliography. Best for CS papers and conference proceedings."""
    try:
        results = asyncio.run(SEARCH_TOOLS["dblp_search"].search(query, max_results))
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error searching DBLP: {e}"


@tool
def dblp_search_authors(query: str, max_results: int = 5) -> str:
    """Search DBLP for authors. Good for finding author profiles and affiliations."""
    try:
        results = asyncio.run(SEARCH_TOOLS["dblp_search"].search_authors(query, max_results))
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error searching DBLP authors: {e}"


@tool
def dblp_search_venues(query: str, max_results: int = 5) -> str:
    """Search DBLP for venues (conferences, journals). Good for finding venue information."""
    try:
        results = asyncio.run(SEARCH_TOOLS["dblp_search"].search_venues(query, max_results))
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error searching DBLP venues: {e}"


@tool
def semantic_scholar_search(query: str, max_results: int = 5) -> str:
    """Search Semantic Scholar for papers with rich metadata and citations."""
    try:
        results = asyncio.run(
            SEARCH_TOOLS["semantic_scholar"].search(query, max_results)
        )
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error searching Semantic Scholar: {e}"


@tool
def semantic_scholar_search_authors(query: str, max_results: int = 5) -> str:
    """Search Semantic Scholar for authors with detailed metrics (H-index, citations, etc.)."""
    try:
        results = asyncio.run(
            SEARCH_TOOLS["semantic_scholar"].search_authors(query, max_results)
        )
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error searching Semantic Scholar authors: {e}"


@tool
def semantic_scholar_paper_details(paper_id: str) -> str:
    """Get detailed information about a specific paper using Semantic Scholar paper ID."""
    try:
        result = asyncio.run(
            SEARCH_TOOLS["semantic_scholar"].get_paper_details(paper_id)
        )
        return safe_json_dumps(result, indent=2)
    except Exception as e:
        return f"Error getting paper details: {e}"


@tool
def semantic_scholar_author_details(author_id: str) -> str:
    """Get detailed information about a specific author using Semantic Scholar author ID."""
    try:
        result = asyncio.run(
            SEARCH_TOOLS["semantic_scholar"].get_author_details(author_id)
        )
        return safe_json_dumps(result, indent=2)
    except Exception as e:
        return f"Error getting author details: {e}"


@tool
def semantic_scholar_paper_batch(paper_ids: str) -> str:
    """Get details for multiple papers using comma-separated Semantic Scholar paper IDs (max 500)."""
    try:
        # Split the comma-separated string into a list
        id_list = [pid.strip() for pid in paper_ids.split(',')]
        
        import requests
        headers = SEARCH_TOOLS["semantic_scholar"]._get_headers()
        
        response = requests.post(
            SEARCH_TOOLS["semantic_scholar"].endpoints["paper_batch"],
            json={"ids": id_list},
            headers=headers,
            timeout=15
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        results = response.json()
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error in batch paper lookup: {e}"


@tool
def semantic_scholar_author_batch(author_ids: str) -> str:
    """Get details for multiple authors using comma-separated Semantic Scholar author IDs (max 1000)."""
    try:
        # Split the comma-separated string into a list
        id_list = [aid.strip() for aid in author_ids.split(',')]
        
        import requests
        headers = SEARCH_TOOLS["semantic_scholar"]._get_headers()
        
        response = requests.post(
            SEARCH_TOOLS["semantic_scholar"].endpoints["author_batch"],
            json={"ids": id_list},
            headers=headers,
            timeout=15
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        results = response.json()
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error in batch author lookup: {e}"


@tool
def semantic_scholar_snippet_search(query: str, max_results: int = 5) -> str:
    """Search Semantic Scholar for text snippets from papers. Good for finding specific content or quotes."""
    try:
        import requests
        headers = SEARCH_TOOLS["semantic_scholar"]._get_headers()
        
        params = {
            "query": query,
            "limit": min(max_results, 100)
        }
        
        response = requests.get(
            SEARCH_TOOLS["semantic_scholar"].endpoints["snippet_search"],
            params=params,
            headers=headers,
            timeout=15
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        results = response.json()
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error in snippet search: {e}"


@tool
def google_search(query: str, max_results: int = 5) -> str:
    """General Google search for academic content. Use as fallback or for specific URLs."""
    try:
        results = asyncio.run(SEARCH_TOOLS["google_search"].search(query, max_results))
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error searching Google: {e}"


@tool
def acl_anthology_search(query: str, max_results: int = 5) -> str:
    """Search ACL Anthology for computational linguistics and NLP papers. Best for linguistics, NLP, and language research."""
    try:
        results = asyncio.run(SEARCH_TOOLS["acl_anthology"].search(query, max_results))
        return safe_json_dumps(results, indent=2)
    except Exception as e:
        return f"Error searching ACL Anthology: {e}"


@tool
def download_file(url: str, filename: str = None) -> str:
    """Download a file from URL (PDF, document, etc.). Returns the local file path."""
    try:
        from interactive_tools import file_manager
        import asyncio

        filepath = asyncio.run(file_manager.download_file(url, filename))
        return f"File downloaded successfully to: {filepath}"
    except Exception as e:
        return f"Download failed: {e}"


@tool
def read_webpage(url: str) -> str:
    """Fetch and read content from a webpage. Good for academic paper pages. Look for DOI, arXiv ID, authors, and other metadata in the content."""
    try:
        from interactive_tools import web_browser

        result = asyncio.run(web_browser.fetch_webpage(url))
        if "error" in result:
            return f"Error: {result['error']}"
        return safe_json_dumps(
            {
                "title": result.get("title", ""),
                "content": (
                    result.get("content", "")[:5000] + "..."
                    if len(result.get("content", "")) > 5000
                    else result.get("content", "")
                ),
                "url": result.get("url", ""),
                "status": result.get("status", ""),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error reading webpage: {e}"


@tool
def read_pdf_text(pdf_path: str) -> str:
    """Extract and read text content from a PDF file."""
    try:
        from interactive_tools import pdf_processor
        import asyncio

        text = asyncio.run(pdf_processor.extract_text_pdfplumber(pdf_path))
        if not text.strip():
            text = asyncio.run(pdf_processor.extract_text_pypdf2(pdf_path))

        # Limit output size
        if len(text) > 5000:
            text = text[:5000] + "...\n[TEXT TRUNCATED - Full text available in file]"

        return f"PDF Text Content:\n{text}"
    except Exception as e:
        return f"Error reading PDF: {e}"


@tool
def extract_pdf_metadata(pdf_path: str) -> str:
    """Extract metadata from PDF (title, author, creation date, etc.)."""
    try:
        from interactive_tools import pdf_processor

        metadata = asyncio.run(pdf_processor.extract_metadata(pdf_path))
        return safe_json_dumps(metadata, indent=2)
    except Exception as e:
        return f"Error extracting PDF metadata: {e}"


@tool
def extract_references_from_pdf(pdf_path: str) -> str:
    """Extract references/bibliography from a PDF file."""
    try:
        from interactive_tools import pdf_processor

        # First extract text
        text = asyncio.run(pdf_processor.extract_text_pdfplumber(pdf_path))
        if not text.strip():
            text = asyncio.run(pdf_processor.extract_text_pypdf2(pdf_path))

        # Then extract references
        references = asyncio.run(pdf_processor.extract_references(text))
        return safe_json_dumps(references, indent=2)
    except Exception as e:
        return f"Error extracting references: {e}"


@tool
def list_downloaded_files() -> str:
    """List all downloaded files in the downloads directory."""
    try:
        from interactive_tools import file_manager

        files = file_manager.list_files()
        return safe_json_dumps(files, indent=2)
    except Exception as e:
        return f"Error listing files: {e}"


@tool
def read_text_file(filepath: str) -> str:
    """Read content from a text file."""
    try:
        from interactive_tools import file_manager

        content = asyncio.run(file_manager.read_text_file(filepath))

        # Limit output size
        if len(content) > 5000:
            content = content[:5000] + "...\n[FILE TRUNCATED]"

        return f"File content:\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def extract_urls_from_text(text: str) -> str:
    """Extract all URLs from given text."""
    try:
        from interactive_tools import url_extractor

        urls = url_extractor.extract_urls_from_text(text)
        return safe_json_dumps(urls, indent=2)
    except Exception as e:
        return f"Error extracting URLs: {e}"


@tool
def doi_search(doi: str) -> str:
    """Search for a paper by DOI using CrossRef API."""
    try:
        import requests
        import json

        # Clean DOI
        doi = doi.strip()
        if doi.startswith("doi:"):
            doi = doi[4:]
        if doi.startswith("http"):
            # Extract DOI from URL
            import re

            match = re.search(r"10\.\d+/[^\s]+", doi)
            if match:
                doi = match.group()

        # Search CrossRef
        url = f"https://api.crossref.org/works/{doi}"
        headers = {"User-Agent": "PaperFinder/1.0 (mailto:research@example.com)"}

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            work = data.get("message", {})

            # Extract paper information
            result = {
                "title": " ".join(work.get("title", [""])),
                "authors": [
                    f"{author.get('given', '')} {author.get('family', '')}"
                    for author in work.get("author", [])
                ],
                "doi": work.get("DOI", ""),
                "year": str(
                    work.get("published-print", {}).get("date-parts", [[]])[0][0]
                    if work.get("published-print")
                    else ""
                ),
                "venue": (
                    work.get("container-title", [""])[0]
                    if work.get("container-title")
                    else ""
                ),
                "publisher": work.get("publisher", ""),
                "type": work.get("type", ""),
                "url": f"https://doi.org/{work.get('DOI', '')}",
                "source": "crossref",
            }

            return safe_json_dumps([result], indent=2)
        else:
            return f"DOI not found: {doi}"

    except Exception as e:
        return f"Error searching DOI: {e}"


class PaperAgent:
    """LLM agent for searching academic papers"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.1, api_key=os.getenv("OPENAI_API_KEY")
        )

        self.tools = [
            arxiv_search,
            arxiv_direct,
            google_scholar_search,
            dblp_search,
            dblp_search_authors,
            dblp_search_venues,
            semantic_scholar_search,
            semantic_scholar_search_authors,
            semantic_scholar_paper_details,
            semantic_scholar_author_details,
            semantic_scholar_paper_batch,
            semantic_scholar_author_batch,
            semantic_scholar_snippet_search,
            google_search,
            acl_anthology_search,
            doi_search,
            download_file,
            read_webpage,
            read_pdf_text,
            extract_pdf_metadata,
            extract_references_from_pdf,
            list_downloaded_files,
            read_text_file,
            extract_urls_from_text,
        ]

        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)

        self.system_prompt = """You are a research assistant that finds academic papers and returns them in BibTeX format.

WORKFLOW:
1. Choose appropriate search tools based on domain
2. Search, deduplicate, rank by relevance  
3. Prioritize formal publications over preprints
4. Provide initial results as JSON for BibTeX formatting
5. Download PDFs and refine BibTeX based on actual content

REQUIRED OUTPUT FORMAT: Always end your response with filtered results as JSON:
```json
[
  {
    "title": "Paper Title Here",
    "authors": ["Author One", "Author Two"],
    "year": "2024",
    "venue": "Conference Name",
    "url": "https://example.com",
    "abstract": "Paper abstract here..."
  }
]
```

SEARCH TOOLS (Priority Order):
CS/AI: dblp_search → semantic_scholar_search → arxiv_search
NLP: acl_anthology_search → dblp_search → semantic_scholar_search  
Physics/Math: semantic_scholar_search → arxiv_search
Bio/Med: semantic_scholar_search

AVAILABLE TOOLS:
- dblp_search(query, max_results=5): CS conferences/journals (highest quality)
- dblp_search_authors(query, max_results=5): Search DBLP for authors
- dblp_search_venues(query, max_results=5): Search DBLP for venues/conferences
- semantic_scholar_search(query, max_results=5): Rich metadata, all domains ⚠️ RATE LIMITED - use sparingly
- semantic_scholar_search_authors(query, max_results=5): Search authors with metrics ⚠️ RATE LIMITED
- semantic_scholar_paper_details(paper_id): Get detailed paper info by ID ⚠️ RATE LIMITED
- semantic_scholar_author_details(author_id): Get detailed author info by ID ⚠️ RATE LIMITED
- semantic_scholar_paper_batch(paper_ids): Get multiple papers by IDs (comma-separated) ⚠️ RATE LIMITED
- semantic_scholar_author_batch(author_ids): Get multiple authors by IDs (comma-separated) ⚠️ RATE LIMITED
- semantic_scholar_snippet_search(query, max_results=5): Search text snippets from papers ⚠️ RATE LIMITED
- acl_anthology_search(query, max_results=5): NLP/linguistics papers
- arxiv_search(query, max_results=5): Preprints (use sparingly)
- arxiv_direct(arxiv_id): Get paper by arXiv ID
- doi_search(doi): Search by DOI
- download_file(url, filename=None): Download PDFs
- read_pdf_text(pdf_path): Extract text from LOCAL PDFs
- read_webpage(url): Extract metadata from webpages

RATE LIMITING GUIDANCE:
- Semantic Scholar APIs are strictly rate limited and may be slow during heavy usage
- Prefer DBLP, ACL Anthology, or arXiv when possible to avoid rate limits
- If you get rate limited (429 errors), wait and try fewer semantic_scholar calls
- Use semantic_scholar_paper_batch() and semantic_scholar_author_batch() for efficiency when you have multiple IDs

SEMANTIC SCHOLAR ID SYSTEM:
- Semantic Scholar uses multiple ID types for papers:
  * Primary: Semantic Scholar SHA (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
  * Secondary: CorpusId (e.g., "CorpusId:215416146")
  * External: DOI, ArXiv ID, MAG, ACL, PMID, PMCID, URLs
- Author IDs: Numeric strings (e.g., "1741101", "40348417")
- Detail functions (semantic_scholar_paper_details, semantic_scholar_author_details) accept:
  * Semantic Scholar IDs (primary/secondary)
  * External IDs (DOI, ArXiv, etc.)
- Batch functions require comma-separated Semantic Scholar IDs only
- To get Semantic Scholar IDs: Use search results' paper_id or author_id fields
- Example: semantic_scholar_search("query") → extract paper_id → semantic_scholar_paper_details(paper_id)

CRITICAL: You must REASON through the deduplication, ranking, and filtering process yourself:
1. DEDUPLICATE: Remove duplicate papers by comparing titles (even if from different sources/years). If same paper appears as both conference and preprint, KEEP ONLY the formal publication.
2. RANK: Order papers by relevance to the user's query
3. FILTER: Remove papers that are clearly not relevant - if initial search results are not closely related to the user's query, return empty results and try different search strategies or escalate to google_search
4. CHOOSE: Select only the most appropriate ones for the user's query

Example: If you find "SciBench" in both ICML 2024 and arXiv 2023, keep ONLY the ICML 2024 version (formal > preprint).

AFTER READING WEBPAGE: Always analyze the webpage content to identify and extract structured metadata like DOI, arXiv ID, authors, title, venue, publication year. Use this metadata for follow-up searches if needed.

QUALITY CHECK: Before finalizing, ensure results aren't just arXiv preprints. For CS/AI papers, try dblp_search to find formal venues (ICML, NeurIPS, etc.).
Example: "I notice my current results are all arXiv preprints from arxiv_search. Let me try dblp_search first to find these papers in formal venues like ICML, NeurIPS, or ICLR."

PDF tools require download_file() first. Always format final results as proper BibTeX entries."""

        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow"""

        def should_continue(state: AgentState) -> str:
            """Decide whether to continue with tools or finish"""
            messages = state["messages"]
            last_message = messages[-1]

            # Check iteration count to prevent infinite loops
            iteration_count = state.get("iteration_count", 0)
            if iteration_count > 15:  # Increased to allow for PDF refinement
                console = state.get("console")
                if console:
                    console.print(
                        "⚠️ [yellow]Max iterations reached, finishing...[/yellow]"
                    )
                return "end"

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # Check if this is a BibTeX confirmation/refinement prompt that should continue
            if (hasattr(last_message, "content") and last_message.content and 
                last_message.__class__.__name__ == "AIMessage"):
                if ("JSON results have been converted to BibTeX format" in last_message.content or
                    "Please review the BibTeX entries" in last_message.content or
                    "Please review and refine these entries" in last_message.content):
                    return "continue_refinement"

            return "end"

        def call_model(state: AgentState):
            """Call the LLM with current state"""
            messages = state["messages"]
            console = state.get("console")
            logger = state.get("logger")
            query_id = state.get("query_id")
            tools_used = state.get("tools_used", [])

            # Display reasoning step
            last_message = messages[-1] if messages else None
            if console and last_message:
                if hasattr(last_message, "content") and last_message.content:
                    thinking_text = last_message.content[:100] + "..."
                    console.print(f"🤔 [yellow]Thinking: {thinking_text}[/yellow]")

                    # Log thinking step
                    if logger and query_id:
                        logger.log_llm_interaction(
                            query_id, "thinking", last_message.content
                        )

            # Add tools used context to the messages if available
            if tools_used:
                tools_context = f"\n\nTOOLS USED SO FAR: {', '.join(tools_used)}\nConsider if you need to try additional search tools for better results."
                # Add this context to the last message if it's from assistant
                if messages and hasattr(messages[-1], "content"):
                    messages[-1].content += tools_context

            response = self.llm_with_tools.invoke(messages)

            # Log raw LLM interaction
            if logger and query_id and response:
                # Convert messages to dict format for logging
                messages_dict = []
                for msg in messages:
                    if hasattr(msg, "content"):
                        msg_dict = {
                            "role": (
                                "system"
                                if msg.__class__.__name__ == "SystemMessage"
                                else "user"
                            ),
                            "content": msg.content,
                        }
                        messages_dict.append(msg_dict)

                # Log raw prompt/response
                response_content = ""
                if hasattr(response, "content") and response.content:
                    response_content = response.content
                elif hasattr(response, "tool_calls") and response.tool_calls:
                    response_content = f"Tool calls: {response.tool_calls}"

                logger.log_raw_llm_interaction(
                    query_id, messages_dict, response_content
                )

                # Log processed LLM response
                if hasattr(response, "content") and response.content:
                    logger.log_llm_interaction(
                        query_id, "llm_response", response.content
                    )

            # Display tool calls and add them to the message history for LLM awareness
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_summary = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "Unknown")
                    args = tool_call.get("args", {})
                    
                    if console:
                        console.print(f"🔧 [cyan]Using tool: {tool_name}[/cyan]")
                        if "query" in args:
                            console.print(f"   Query: {args['query']}")
                        elif "url" in args:
                            console.print(f"   URL: {args['url']}")
                        elif "arxiv_id" in args:
                            console.print(f"   arXiv ID: {args['arxiv_id']}")
                        elif "doi" in args:
                            console.print(f"   DOI: {args['doi']}")
                        elif "pdf_path" in args:
                            console.print(f"   PDF Path: {args['pdf_path']}")
                        elif "filepath" in args:
                            console.print(f"   File Path: {args['filepath']}")
                        elif "filename" in args:
                            console.print(f"   Filename: {args['filename']}")
                        elif "text" in args:
                            console.print(f"   Text (preview): {str(args['text'])[:50]}...")
                    
                    # Always add to tool summary regardless of args structure
                    if "query" in args:
                        tool_summary.append(f"{tool_name}(query='{args['query']}')")
                    elif "url" in args:
                        tool_summary.append(f"{tool_name}(url='{args['url']}')")
                    elif "arxiv_id" in args:
                        tool_summary.append(f"{tool_name}(arxiv_id='{args['arxiv_id']}')")
                    elif "doi" in args:
                        tool_summary.append(f"{tool_name}(doi='{args['doi']}')")
                    elif "pdf_path" in args:
                        tool_summary.append(f"{tool_name}(pdf_path='{args['pdf_path']}')")
                    elif "filepath" in args:
                        tool_summary.append(f"{tool_name}(filepath='{args['filepath']}')")
                    else:
                        # Fallback: always include tool name even if args don't match patterns
                        tool_summary.append(f"{tool_name}")

                    # Log tool call
                    if logger and query_id:
                        logger.log_llm_interaction(
                            query_id, "tool_call", f"{tool_name}: {args}"
                        )

                # Always add tool call summary to state for LLM awareness
                state["tools_used"] = state.get("tools_used", []) + tool_summary

            # Increment iteration count
            iteration_count = state.get("iteration_count", 0) + 1

            return {"messages": [response], "iteration_count": iteration_count}

        def format_final_answer(state: AgentState):
            """Process JSON results and convert to BibTeX"""
            messages = state["messages"]
            console = state.get("console")

            # Look for reranked results first (these are the filtered/relevant ones)
            reranked_results = []
            search_results = []

            for message in reversed(messages):  # Start from most recent messages
                if hasattr(message, "content") and message.content and message.__class__.__name__ == "AIMessage":
                    content = message.content


                    # Look for BibTeX entries and extract them (final refined output)
                    import re
                    bibtex_pattern = (
                        r"```bibtex\s*((?:@\w+\{[^}]+,(?:[^@])*?\}\s*)+)\s*```"
                    )
                    bibtex_match = re.search(bibtex_pattern, content, re.DOTALL)
                    if bibtex_match:
                        # If we found refined BibTeX, use it directly
                        final_answer = bibtex_match.group(1).strip()
                        return {
                            "messages": [AIMessage(content=final_answer)],
                            "final_answer": final_answer,
                        }

                    # Look for JSON data in markdown code blocks (most recent first)
                    json_match = re.search(
                        r"```json\s*(\[.*?\])\s*```", content, re.DOTALL
                    )
                    if json_match:
                        try:
                            # Clean up common JSON issues
                            json_content = json_match.group(1)
                            # Remove trailing commas before closing brackets/braces
                            json_content = re.sub(r',\s*}', '}', json_content)
                            json_content = re.sub(r',\s*]', ']', json_content)
                            
                            results = json.loads(json_content)
                            if isinstance(results, list) and all(
                                isinstance(r, dict) for r in results
                            ):
                                reranked_results = results
                                break
                        except json.JSONDecodeError:
                            pass

                    # Fallback: look for direct JSON at start of message
                    try:
                        if content.strip().startswith(
                            "["
                        ) or content.strip().startswith("{"):
                            results = json.loads(content.strip())
                            if isinstance(results, list):
                                if len(results) <= 10 and all(
                                    isinstance(r, dict) for r in results
                                ):
                                    reranked_results = results
                                else:
                                    search_results.extend(results)
                            elif isinstance(results, dict):
                                search_results.append(results)
                    except json.JSONDecodeError:
                        continue

            # Use reranked results if available, otherwise use all search results
            final_results = reranked_results if reranked_results else search_results
            

            # Remove duplicates based on title with priority for formal publications
            seen_titles = {}  # title -> best_result
            for result in final_results:
                title = result.get("title", "").strip().lower()
                if not title or len(title) <= 10:
                    continue

                # Normalize title for better matching
                import re

                normalized_title = re.sub(r"[^\w\s]", "", title).strip()
                if not normalized_title:
                    continue

                # If we haven't seen this title, add it
                if normalized_title not in seen_titles:
                    seen_titles[normalized_title] = result
                else:
                    # We have a duplicate - keep the better one
                    existing = seen_titles[normalized_title]
                    current = result

                    # Priority: conference > journal > workshop > preprint
                    def get_publication_priority(paper):
                        venue = paper.get("venue", "").lower()
                        booktitle = paper.get("booktitle", "").lower()
                        journal = paper.get("journal", "").lower()

                        if any(
                            conf in venue or conf in booktitle
                            for conf in [
                                "icml",
                                "neurips",
                                "iclr",
                                "aaai",
                                "ijcai",
                                "acl",
                                "emnlp",
                            ]
                        ):
                            return 4  # Major conference
                        elif "proceedings" in venue or "proceedings" in booktitle:
                            return 3  # Conference
                        elif (
                            journal and "arxiv" not in journal and "corr" not in journal
                        ):
                            return 3  # Journal
                        elif "workshop" in venue:
                            return 2  # Workshop
                        elif "arxiv" in venue or "corr" in journal:
                            return 1  # Preprint
                        else:
                            return 0  # Unknown

                    if get_publication_priority(current) > get_publication_priority(
                        existing
                    ):
                        seen_titles[normalized_title] = current

            unique_results = list(seen_titles.values())

            # Limit to top 5 results
            unique_results = unique_results[:5]

            # Convert JSON to BibTeX and ask LLM to confirm/refine
            if unique_results:
                from bibtex_formatter import format_to_bibtex

                bibtex_entries = []
                for result in unique_results:
                    bibtex = format_to_bibtex(result)
                    if bibtex:
                        bibtex_entries.append(bibtex)

                if bibtex_entries:
                    initial_bibtex = "\n\n".join(bibtex_entries)

                    # Check if any papers have PDFs available for refinement
                    has_pdfs = any(
                        result.get("pdf_url")
                        or (result.get("url", "").endswith(".pdf"))
                        or ("arxiv.org" in result.get("url", ""))
                        for result in unique_results
                    )

                    # Create confirmation/refinement prompt
                    if has_pdfs:
                        if console:
                            console.print(
                                "🔍 [yellow]Converting to BibTeX and checking PDFs for refinement...[/yellow]"
                            )

                        confirmation_prompt = f"""
JSON results have been converted to BibTeX format. Please review and refine these entries:

```bibtex
{initial_bibtex}
```

MANDATORY TASKS:
1. Review the BibTeX entries for accuracy and completeness
2. For papers with available PDFs, download and extract content to refine metadata:

Papers with potential PDFs:
{json.dumps(unique_results, indent=2)}

For each paper with PDF URLs:
1. Use download_file() to download the PDF
2. Use read_pdf_text() to extract text content
3. Use extract_pdf_metadata() to get PDF metadata
4. Refine the BibTeX entry based on actual PDF content (title, authors, venue, year, etc.)

After refinement, output the final BibTeX entries in a ```bibtex``` code block.
"""

                        return {
                            "messages": [AIMessage(content=confirmation_prompt)],
                        }
                    else:
                        if console:
                            console.print(
                                "📄 [yellow]Converting to BibTeX format...[/yellow]"
                            )

                        confirmation_prompt = f"""
JSON results have been converted to BibTeX format. Please review these entries:

```bibtex
{initial_bibtex}
```

Please review the BibTeX entries for accuracy and completeness. If they look correct, output the final BibTeX entries in a ```bibtex``` code block.
"""

                        return {
                            "messages": [AIMessage(content=confirmation_prompt)],
                        }
                else:
                    final_answer = "No papers found or could not format results."
            else:
                final_answer = "No papers found."

            return {
                "messages": [AIMessage(content=final_answer)],
                "final_answer": final_answer,
            }

        def tool_execution_wrapper(state: AgentState):
            """Wrapper to show tool execution results"""
            console = state.get("console")
            logger = state.get("logger")
            query_id = state.get("query_id")

            result = self.tool_node.invoke(state)

            if console:
                # Show completion of tool calls
                messages = result.get("messages", [])
                for message in messages:
                    if hasattr(message, "content") and message.content:
                        content = message.content[:200]
                        if len(message.content) > 200:
                            content += "..."
                        console.print(f"✅ [green]Tool completed[/green]: {content}")

                        # Log tool response
                        if logger and query_id:
                            logger.log_llm_interaction(
                                query_id, "tool_response", message.content
                            )

            return result

        # Build graph
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_execution_wrapper)
        workflow.add_node("format", format_final_answer)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": "format",
                "continue_refinement": "agent",  # Continue to agent for refinement
            },
        )
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges(
            "format",
            should_continue,
            {
                "tools": "tools",
                "end": END,
                "continue_refinement": "agent",  # Continue to agent for refinement
            },
        )

        return workflow.compile()

    async def search(self, query: str, console=None) -> str:
        """Search for papers based on user query"""
        from logger import get_logger

        # Initialize logging
        logger = get_logger()
        query_id = logger.start_query(query)

        if console:
            console.print(f"🔍 [bold]Analyzing query:[/bold] {query}")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Find academic papers for this query: {query}"),
        ]

        initial_state = {
            "messages": messages,
            "search_results": [],
            "final_answer": "",
            "reasoning_steps": [],
            "console": console,
            "query_id": query_id,
            "logger": logger,
            "iteration_count": 0,
            "tools_used": [],
        }
        result = await self.graph.ainvoke(initial_state)

        final_answer = result.get("final_answer", "No results found.")

        # Log final result
        logger.log_final_result(query_id, final_answer)

        return final_answer
