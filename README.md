# PaperFinder

An intelligent LLM-powered agent for searching academic papers and returning them in properly formatted BibTeX entries. PaperFinder uses advanced reasoning and multiple academic databases to find high-quality research papers and automatically generate publication-ready citations.

## Features

- **Multi-source search**: Integrates with DBLP, Semantic Scholar, ACL Anthology, arXiv, and more
- **Intelligent tool selection**: Automatically chooses the best search tools based on query domain
- **Quality prioritization**: Prefers formal publications (conferences/journals) over preprints
- **Smart deduplication**: Automatically removes duplicates and keeps the highest quality version
- **PDF analysis**: Downloads and analyzes PDFs to refine BibTeX accuracy
- **Interactive capabilities**: Can download files, read webpages, and extract metadata
- **Comprehensive logging**: Tracks all interactions and tool usage for transparency

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd paperfinder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys (see Configuration section)
```

## Usage

### Basic Usage

```bash
python main.py
```

Then enter your query when prompted:
```
📝 Query: SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models
```

### Query Types Supported

PaperFinder supports **arbitrary natural language queries**, making it incredibly flexible:

- **Paper titles**: "Attention Is All You Need"
- **Author queries**: "papers by Yanqiao Zhu" or "find the highly cited papers of author X"
- **Research topics**: "graph neural networks for recommendation" or "recent papers on large language models with verifiable rewards"
- **Complex queries**: "transformer models for computer vision in 2023" or "survey papers on federated learning privacy"
- **Specific requirements**: "ICML papers on reinforcement learning" or "NeurIPS 2024 papers about diffusion models"
- **arXiv URLs/IDs**: "https://arxiv.org/abs/2307.10635" or "2307.10635"
- **ACL Anthology URLs**: "https://aclanthology.org/2024.emnlp-main.1241/"
- **DOIs**: "10.1145/3442381.3449802"

## Search Strategy

PaperFinder automatically selects the best search tools based on your query domain:

### Computer Science & AI/ML
1. **DBLP** (formal CS publications - ICML, NeurIPS, ICLR, etc.)
2. **Semantic Scholar** (comprehensive metadata)
3. **arXiv** (recent preprints)

### Natural Language Processing
1. **ACL Anthology** (specialized NLP venues)
2. **DBLP** (broader CS conferences)
3. **Semantic Scholar** (comprehensive coverage)

### Physics/Mathematics
1. **Semantic Scholar** (better metadata than arXiv)
2. **arXiv** (physics/math preprints)

### Other Domains
1. **Semantic Scholar** (cross-disciplinary coverage)

## Architecture

PaperFinder is built using:

- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and tool calling
- **ReAct Pattern**: Reasoning + Acting for intelligent tool selection
- **Rich/Prompt-toolkit**: Beautiful terminal interface
- **Multiple APIs**: DBLP, Semantic Scholar, ACL Anthology, arXiv, CrossRef

### Core Components

- `main.py`: CLI interface with Rich console
- `paper_agent.py`: Core LLM agent with LangGraph workflow
- `search_tools.py`: Academic search API integrations
- `interactive_tools.py`: File operations, PDF processing, web browsing
- `bibtex_formatter.py`: BibTeX entry generation
- `json_utils.py`: Safe JSON serialization for academic data
- `logger.py`: Comprehensive logging system

## Configuration

### Environment Variables

#### Required
- `OPENAI_API_KEY`: Required for LLM functionality

#### Optional (for enhanced functionality)
- `GOOGLE_SEARCH_API_KEY`: Google Custom Search JSON API key
- `GOOGLE_SEARCH_CX`: Google Custom Search Engine ID
- `SEMANTIC_SCHOLAR_API_KEY`: Semantic Scholar API key for higher rate limits

#### API Setup Instructions

**Google Custom Search API (100 free queries/day):**
1. Create a project at [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Custom Search API in APIs & Services → Library
3. Create an API key in APIs & Services → Credentials → Create Credentials → API Key
4. Create a Custom Search Engine at [Google CSE](https://cse.google.com/cse/)
   - Choose "Search the entire web" or add specific sites
   - Get the Search Engine ID (CX parameter)
5. Add to `.env`:
   ```
   GOOGLE_SEARCH_API_KEY=your_api_key_here
   GOOGLE_SEARCH_CX=your_search_engine_id_here
   ```

**Semantic Scholar API:**
- **Without API key**: 1000 requests/second shared among all users (may be throttled during heavy use)
- **With API key**: 1 request/second dedicated rate limit + better support
- Request API key at [Semantic Scholar API](https://www.semanticscholar.org/product/api)
- Add to `.env` if you have one:
   ```
   SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
   ```

**DBLP API:** No authentication required, uses public endpoints

### Logging

PaperFinder creates comprehensive logs in the `./logs` directory:
- `session_YYYYMMDD_HHMMSS.txt`: Human-readable session log
- `session_YYYYMMDD_HHMMSS.json`: Structured session data
- `raw_llm_YYYYMMDD_HHMMSS.jsonl`: Raw LLM prompt/response pairs

## Features in Detail

### Smart Deduplication

PaperFinder automatically removes duplicate papers and prioritizes:
1. Major conferences (ICML, NeurIPS, ICLR, etc.)
2. Other conferences and journals
3. Workshop papers
4. Preprints (arXiv, CoRR)

### PDF Refinement

When PDFs are available, PaperFinder:
1. Downloads the PDF locally
2. Extracts text content
3. Verifies and improves BibTeX metadata
4. Provides more accurate citations

### Interactive Capabilities

- **File downloads**: Download PDFs from URLs
- **Webpage analysis**: Extract metadata from academic pages
- **PDF processing**: Extract text, metadata, and references
- **URL extraction**: Find URLs in text content

## Example Output

```bibtex
@inproceedings{Wang2024,
  title = {SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models},
  author = {Xiaoxuan Wang and Ziniu Hu and Pan Lu and Yanqiao Zhu and Jieyu Zhang and Satyen Subramaniam and Arjun R. Loomba and Shichang Zhang and Yizhou Sun and Wei Wang},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year = {2024},
  url = {https://arxiv.org/abs/2307.10635}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Some APIs (like Semantic Scholar) have rate limits. The system will automatically handle these with appropriate delays.

2. **PDF Processing Errors**: Some PDFs may have formatting issues. The system includes robust error handling to continue processing.

3. **Missing Dependencies**: Make sure all dependencies are installed with `pip install -r requirements.txt`.

### Debug Mode

Check the logs in `./logs` directory for detailed information about tool calls and LLM reasoning.