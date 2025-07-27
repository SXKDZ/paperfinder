#!/usr/bin/env python3
"""
PaperFinder: An LLM agent for searching academic papers
"""

import asyncio
import os

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from rich.console import Console

from paper_agent import PaperAgent

load_dotenv()

console = Console()


def display_bibtex(result: str):
    """Display BibTeX results with syntax highlighting"""
    try:
        from pygments import highlight
        from pygments.formatters import TerminalFormatter
        from pygments.lexers import BibTexLexer

        highlighted = highlight(result, BibTexLexer(), TerminalFormatter())
        console.print(highlighted, end="")
    except ImportError:
        console.print(result)


async def process_query(agent, query: str, is_followup: bool = False) -> str:
    """Process a search query and return results"""
    query_type = "Follow-up search" if is_followup else "Starting search"
    console.print(
        f"\n🚀 [bold blue]{query_type} for:[/bold blue] [italic]{query}[/italic]"
    )

    result = await agent.search(query, console)

    if result and result.strip():
        result_type = "Follow-up Results" if is_followup else "Results"
        console.print(f"✅ [green]{result_type}:[/green]")
        console.print("")
        display_bibtex(result)
        console.print("")
        return result
    else:
        no_results_msg = (
            "No additional results found" if is_followup else "No papers found"
        )
        console.print(f"❌ [red]{no_results_msg}[/red]")
        return ""


async def main():
    console.print(
        "🔍 [bold cyan]PaperFinder[/bold cyan] - AI-powered academic paper search"
    )
    console.print("Type your query or 'quit' to exit", style="blue")
    console.print("Press Ctrl+D after results for new query", style="dim")

    if not os.getenv("OPENAI_API_KEY"):
        console.print("❌ [red]OPENAI_API_KEY not found in environment[/red]")
        console.print(
            "Please copy .env.example to .env and set your API key", style="magenta"
        )
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

            if query.lower() in ["quit", "exit", "q"]:
                console.print("👋 [green]Goodbye![/green]")
                break

            if not query.strip():
                continue

            result = await process_query(agent, query)

            if result:
                await handle_followup_queries(agent, result, console, session)

        except KeyboardInterrupt:
            console.print("\n👋 [green]Goodbye![/green]")
            logger.close_session()
            break
        except EOFError:
            console.print("\n👋 [green]Goodbye![/green]")
            logger.close_session()
            break
        except Exception as e:
            console.print(f"❌ [red]Error: {e}[/red]")

    logger.close_session()


async def handle_followup_queries(agent, previous_result, console, session):
    """Handle follow-up queries after BibTeX results are shown"""
    console.print(
        "💬 [cyan]Follow-up questions? Type 'done' or press Ctrl+D for new query[/cyan]"
    )

    while True:
        try:
            followup_query = await session.prompt_async("🔄 Follow-up: ")

            if followup_query.lower() in ["done", "exit", "new", "quit"]:
                break
            if not followup_query.strip():
                continue

            # Create context-aware query and process it
            context_query = f"Previous results:\n{previous_result}\n\nFollow-up question: {followup_query}"
            result = await process_query(agent, context_query, is_followup=True)

            if result:
                previous_result = result  # Update for subsequent follow-ups

        except EOFError:
            console.print("🔚 [green]Starting new query...[/green]")
            break
        except KeyboardInterrupt:
            raise


if __name__ == "__main__":
    asyncio.run(main())
