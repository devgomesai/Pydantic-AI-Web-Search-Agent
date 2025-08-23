from __future__ import annotations as _annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import logfire
import typer
from devtools import debug
from httpx import AsyncClient
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai import Agent, ModelRetry, RunContext

load_dotenv()
llm = os.getenv('LLM_MODEL')

# Initialize model based on LLM_MODEL environment variable
def initialize_model():
    if llm.lower().startswith('gpt'):
        # OpenAI models
        return OpenAIModel(llm)
    elif llm.lower().startswith('gemini'):
        # Google models
        gemini_api_key = os.getenv('GOOGLE_API_KEY', '')
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini models")
        provider = GoogleProvider(api_key=gemini_api_key)
        return GoogleModel(llm, provider=provider)
    else:
        # Ollama or other local models
        client = AsyncOpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama'
        )
        return OpenAIModel(llm, openai_client=client)

model = initialize_model()

# Settings for Google models (optional)
settings = None
if llm.lower().startswith('gemini'):
    settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})


# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

# Initialize Rich console
console = Console()
app = typer.Typer(help="AI-powered web search agent with Rich CLI interface")


class ResearchOutputParser(BaseModel):
    query: str = Field(..., description="The original research query or question.")
    key_findings: list[str] = Field(..., description="A bullet-point list of the most important findings from the search.")
    summary: str = Field(..., description="Answer to the user's Question in great detail")
    sources: list[str] = Field(..., description="List of reliable source URLs used for the answer.")
    


@dataclass
class Deps:
    client: AsyncClient
    brave_api_key: str | None

@dataclass
class TokenCost:
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    model_name: str = ""
    
    def add_usage(self, input_tokens: int, output_tokens: int, model: str):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.model_name = model
        
        # Cost calculation for common models (per 1M tokens)
        cost_table = {
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4': {'input': 30.00, 'output': 60.00},
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
            'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
            'gemini-1.5-flash': {'input': 0.075, 'output': 0.30},
            'gemini-2.0-flash-exp': {'input': 0.075, 'output': 0.30},
            'gemini-2.5-pro': {'input': 1.25, 'output': 5.00},
        }
        
        rates = cost_table.get(model.lower(), {'input': 0, 'output': 0})
        input_cost = (input_tokens / 1_000_000) * rates['input']
        output_cost = (output_tokens / 1_000_000) * rates['output']
        self.total_cost += input_cost + output_cost

# Global token tracker
session_costs = TokenCost()


# Create agent with optional settings
if settings:
    web_search_agent = Agent(
        model,
        system_prompt=f'You are an expert at researching the web to answer user questions. The current date is: {datetime.now().strftime("%Y-%m-%d")}',
        deps_type=Deps,
        retries=2,
        output_type=ResearchOutputParser,
        model_settings=settings
    )
else:
    web_search_agent = Agent(
        model,
        system_prompt=f'You are an expert at researching the web to answer user questions. The current date is: {datetime.now().strftime("%Y-%m-%d")}',
        deps_type=Deps,
        retries=2,
        output_type=ResearchOutputParser
    )


@web_search_agent.tool
async def search_web(
    ctx: RunContext[Deps], web_query: str
) -> str:
    """Search the web given a query defined to answer the user's question.

    Args:
        ctx: The context.
        web_query: The query for the web search.

    Returns:
        str: The search results as a formatted string.
    """
    if ctx.deps.brave_api_key is None:
        return "This is a test web search result. Please provide a Brave API key to get real search results."

    headers = {
        'X-Subscription-Token': ctx.deps.brave_api_key,
        'Accept': 'application/json',
    }
    
    with logfire.span('calling Brave search API', query=web_query) as span:
        r = await ctx.deps.client.get(
            'https://api.search.brave.com/res/v1/web/search',
            params={
                'q': web_query,
                'count': 5,
                'text_decorations': True,
                'search_lang': 'en'
            },
            headers=headers
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response', data)

    results = []
    # Add web results in a nice formatted way
    web_results = data.get('web', {}).get('results', [])
    for item in web_results[:3]:
        title = item.get('title', '')
        description = item.get('description', '')
        url = item.get('url', '')
        if title and description:
            results.append(f"Title: {title}\nSummary: {description}\nSource: {url}\n")

    return "\n".join(results) if results else "No results found for the query."


def display_welcome():
    """Display a welcome message with Rich formatting."""
    welcome_text = """
    🤖 AI Web Search Agent 🔎
    
    Welcome to the AI-powered web search agent! This tool helps you research topics by:
    - Searching the web with intelligent queries
    - Analyzing results with AI
    - Presenting information in a beautiful format
    
    Usage: Simply enter your question or research topic!
    """
    
    console.print(Panel(
        Markdown(welcome_text),
        title="[bold blue]Web Search Agent[/bold blue]",
        border_style="blue"
    ))


def display_search_results(raw_results: str, ai_response: str):
    """Display search results and AI analysis with Rich formatting."""
    
    # Create a table for search results
    table = Table(title="Search Results", show_header=True, header_style="bold magenta")
    table.add_column("Source", style="cyan", no_wrap=False)
    table.add_column("Summary", style="white", no_wrap=False)
    
    # Parse the raw results to extract structured data
    results_parts = raw_results.split('\n\n')
    for part in results_parts:
        if part.strip():
            lines = part.strip().split('\n')
            if len(lines) >= 3:
                title = lines[0].replace('Title: ', '')
                summary = lines[1].replace('Summary: ', '')
                url = lines[2].replace('Source: ', '')
                
                # Truncate long summaries
                if len(summary) > 100:
                    summary = summary[:100] + "..."
                
                table.add_row(f"[link={url}]{title}[/link]", summary)
    
    console.print(table)
    console.print()
    
    # Display AI analysis
    console.print(Panel(
        Markdown(f"## AI Analysis\n\n{ai_response}"),
        title="[bold green]AI Response[/bold green]",
        border_style="green"
    ))

def display_research_results(result: ResearchOutputParser):
    """Display structured research results with Rich formatting."""
    
    # Display query
    console.print("\n")
    console.print(Panel(
        f"[bold]{result.query}[/bold]",
        title="[bold blue]Research Query[/bold blue]",
        border_style="blue"
    ))
    
    # Display key findings as a table
    if result.key_findings:
        findings_table = Table(title="Key Findings", show_header=False, box=None)
        findings_table.add_column("", style="cyan", no_wrap=False)
        
        for i, finding in enumerate(result.key_findings, 1):
            findings_table.add_row(f"• {finding}")
        
        console.print(findings_table)
        console.print()
    
    # Display summary
    console.print(Panel(
        Markdown(f"## Summary\n\n{result.summary}"),
        title="[bold green]AI Analysis Summary[/bold green]",
        border_style="green"
    ))
    
    # Display sources
    if result.sources:
        sources_table = Table(title="Sources", show_header=False, box=None)
        sources_table.add_column("", style="blue", no_wrap=False)
        
        for i, source in enumerate(result.sources, 1):
            sources_table.add_row(f"{i}. [link={source}]{source}[/link]")
        
        console.print(sources_table)
        console.print()
    


def display_session_costs():

    """Display session token costs."""
    if session_costs.total_cost > 0:
        cost_info = f"""
        Session Token Costs
        
        Model: {session_costs.model_name}
        Input Tokens: {session_costs.input_tokens:,}
        Output Tokens: {session_costs.output_tokens:,}
        Total Cost: ${session_costs.total_cost:.6f} USD
        """
        
        console.print(Panel(
            Markdown(cost_info),
            title="[bold yellow]Token Usage[/bold yellow]",
            border_style="yellow"
        ))


async def search_and_analyze(query: str) -> tuple[str, str]:
    """Perform web search and AI analysis."""
    async with AsyncClient() as client:
        brave_api_key = os.getenv('BRAVE_API_KEY', None)
        deps = Deps(client=client, brave_api_key=brave_api_key)

        console.print("[dim]Searching and analyzing...[/dim]")
        
        result = await web_search_agent.run(query, deps=deps)
        
        # Track token usage if available
        try:
            usage = result.usage()
            input_tokens = getattr(usage, 'input_tokens', getattr(usage, 'request_tokens', 0))
            output_tokens = getattr(usage, 'output_tokens', getattr(usage, 'response_tokens', 0))
            if input_tokens > 0 or output_tokens > 0:
                session_costs.add_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=llm
                )
        except (AttributeError, Exception):
            # If usage is not available, continue without tracking
            pass
        
        console.print("[green]Search completed![/green]")
        
        return result.output, ""


@app.command()
def search(
    query: str = typer.Argument(None, help="Your search query or question"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Run in interactive mode")
):
    """Search the web and get AI-powered analysis."""
    
    display_welcome()
    
    async def run_search(search_query: str):
        try:
            result_output, _ = await search_and_analyze(search_query)
            
            # Format the structured ResearchOutputParser output
            display_research_results(result_output)
            
        except Exception as e:
            console.print(Panel(
                f"Error occurred: {str(e)}",
                title="[bold red]Error[/bold red]",
                border_style="red"
            ))
    
    if interactive or not query:
        # Interactive mode
        while True:
            try:
                if not query:
                    query = typer.prompt("\nWhat would you like to search for?")
                
                console.print(f"\n[dim]Searching for: {query}[/dim]")
                asyncio.run(run_search(query))
                console.print("\nThanks for using Web Search Agent!")
                display_session_costs()
                break
                
            except (KeyboardInterrupt, typer.Abort):
                console.print("\nThanks for using Web Search Agent!")
                display_session_costs()
                break
    else:
        # Single search mode
        console.print(f"\n[dim]Searching for: {query}[/dim]")
        asyncio.run(run_search(query))
        display_session_costs()


@app.command()
def config():
    """Show configuration information."""
    # Determine base URL based on model type
    base_url = "OpenAI API"
    if llm.lower().startswith('gemini'):
        base_url = "Google AI API"
    elif not llm.lower().startswith('gpt'):
        base_url = "http://localhost:11434/v1 (Ollama)"
    
    config_info = f"""
    # Configuration Status
    
    **LLM Model**: `{llm}`
    **Model Type**: {'OpenAI' if llm.lower().startswith('gpt') else 'Google AI' if llm.lower().startswith('gemini') else 'Local/Ollama'}
    **Brave API Key**: {'[green]Configured[/green]' if os.getenv('BRAVE_API_KEY') else '[red]Not configured[/red]'}
    **OpenAI API Key**: {'[green]Configured[/green]' if os.getenv('OPENAI_API_KEY') else '[red]Not configured[/red]'}
    **Google API Key**: {'[green]Configured[/green]' if os.getenv('GOOGLE_API_KEY') else '[red]Not configured[/red]'}
    **Base URL**: `{base_url}`
    
    ## Environment Variables
    - `LLM_MODEL`: Set your preferred model (default: gpt-4o)
    - `BRAVE_API_KEY`: Required for real web search results
    - `OPENAI_API_KEY`: Required if using OpenAI models (gpt-*)
    - `GOOGLE_API_KEY`: Required if using Google models (gemini-*)
    
    ## Supported Model Formats
    - **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5-turbo
    - **Google AI**: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp, gemini-2.5-pro
    - **Local/Ollama**: Any model name (will use localhost:11434)
    """
    
    console.print(Panel(
        Markdown(config_info),
        title="[bold blue]Configuration[/bold blue]",
        border_style="blue"
    ))


if __name__ == '__main__':
    app()