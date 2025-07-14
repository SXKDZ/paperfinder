#!/usr/bin/env python3
"""
PaperFinder: An LLM agent for searching academic papers
"""

import os
import asyncio
from dotenv import load_dotenv
from rich.console import Console
from prompt_toolkit import PromptSession

from paper_agent import PaperAgent

load_dotenv()

console = Console()

async def main():
    console.print("🔍 [bold cyan]PaperFinder[/bold cyan] - AI-powered academic paper search")
    console.print("Type your query or 'quit' to exit", style="blue")
    
    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ [red]OPENAI_API_KEY not found in environment[/red]")
        console.print("Please copy .env.example to .env and set your API key", style="magenta")
        return
    
    # Initialize logger
    from logger import init_logger
    logger = init_logger()
    summary = logger.get_session_summary()
    console.print(f"📝 [dim]Logging to:[/dim]")
    console.print(f"   [dim]Text: {summary['log_files']['text']}[/dim]")
    console.print(f"   [dim]Raw LLM: {summary['log_files']['raw_llm']}[/dim]")
    
    agent = PaperAgent()
    session = PromptSession()
    
    while True:
        try:
            query = await session.prompt_async("📝 Query: ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("👋 [green]Goodbye![/green]")
                break
            
            if not query.strip():
                continue
            
            console.print(f"\n🚀 [bold blue]Starting search for:[/bold blue] [italic]{query}[/italic]")
            result = await agent.search(query, console)
            
            if result and result.strip():
                console.print("✅ [green]Results:[/green]")
                console.print("")
                
                # Syntax highlight BibTeX
                try:
                    from pygments import highlight
                    from pygments.lexers import BibTexLexer
                    from pygments.formatters import TerminalFormatter
                    
                    highlighted = highlight(result, BibTexLexer(), TerminalFormatter())
                    console.print(highlighted, end="")
                except ImportError:
                    # Fallback if pygments not available
                    console.print(result)
                
                console.print("")
            else:
                console.print("❌ [red]No papers found[/red]")
                
        except KeyboardInterrupt:
            console.print("\n👋 [green]Goodbye![/green]")
            logger.close_session()
            break
        except Exception as e:
            console.print(f"❌ [red]Error: {e}[/red]")
    
    logger.close_session()

if __name__ == "__main__":
    asyncio.run(main())