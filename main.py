"""Main entry point for the Agentic RAG system."""

import sys
import os
import subprocess

# Always run relative to the project root (where this file lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# Ensure src is on path
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table

from src.vector_store import build_vector_store
from src.agent import build_rag_graph, run_query, ModelConfig
from src.summarizer import summarize_documents

console = Console()

BANNER = """
[bold cyan]Agentic RAG[/bold cyan] — powered by [bold green]LangGraph[/bold green] + [bold yellow]Ollama[/bold yellow]

• Drop PDF / TXT / MD files into the [bold]docs/[/bold] folder to add them to the knowledge base.
• [bold]/models[/bold]         List available Ollama models.
• [bold]/model <name>[/bold]   Switch to a different Ollama LLM model.
• [bold]/summarize[/bold]      Summarize all loaded documents.
• [bold]rebuild[/bold]         Re-index all documents in docs/.
• [bold]exit / quit[/bold]     Stop the session.
"""


def show_models(model_config: ModelConfig) -> None:
    """List all available Ollama models in a Rich table."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            console.print("[yellow]No Ollama models found.[/yellow]")
            return

        table = Table(title="Available Ollama Models", border_style="cyan")
        table.add_column("Model", style="bold green")
        table.add_column("ID")
        table.add_column("Size")
        table.add_column("Modified")

        for line in lines[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 4:
                name = parts[0]
                model_id = parts[1]
                size = parts[2] + " " + parts[3]
                modified = " ".join(parts[4:]) if len(parts) > 4 else ""
                label = f"{name} [bold yellow](active)[/bold yellow]" if name == model_config.model_name else name
                table.add_row(label, model_id, size, modified)

        console.print(table)
    except subprocess.TimeoutExpired:
        console.print("[red]Ollama timed out. Is Ollama running?[/red]")
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")


def switch_model(new_model: str, model_config: ModelConfig, vector_store, graph_ref: list) -> None:
    """Switch the active LLM model and rebuild the graph."""
    if new_model == model_config.model_name:
        console.print(f"[dim]Already using {new_model}[/dim]")
        return

    # Validate model exists
    try:
        result = subprocess.run(
            ["ollama", "show", new_model, "--modelfile"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            console.print(f"[red]Model '{new_model}' not found. Run /models to see available models.[/red]")
            return
    except subprocess.TimeoutExpired:
        console.print("[red]Ollama timed out. Is Ollama running?[/red]")
        return

    model_config.model_name = new_model
    with console.status(f"[bold green]Rebuilding graph with model '{new_model}'..."):
        graph_ref[0] = build_rag_graph(vector_store, model_config)
    console.print(f"[bold green]Switched to model: {new_model}[/bold green]")


def main():
    console.print(Panel(BANNER, border_style="cyan"))

    # Runtime model config (mutable)
    model_config = ModelConfig()

    # Build vector store
    with console.status("[bold green]Loading knowledge base..."):
        vector_store = build_vector_store()

    # Build LangGraph RAG graph (wrapped in list so switch_model can mutate it)
    with console.status("[bold green]Compiling LangGraph agent..."):
        graph_ref = [build_rag_graph(vector_store, model_config)]

    console.print(f"[bold green]Ready![/bold green] Model: [bold yellow]{model_config.model_name}[/bold yellow]. Ask me anything.\n")

    conversation_history = []   # persists across turns within a session
    HISTORY_WINDOW = 6          # last 3 turns (3 Human + 3 AI messages)

    while True:
        try:
            question = Prompt.ask("[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Goodbye![/bold red]")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            console.print("[bold red]Goodbye![/bold red]")
            break

        if question.lower() == "rebuild":
            import shutil
            shutil.rmtree(os.path.join(PROJECT_ROOT, "chroma_db"), ignore_errors=True)
            with console.status("[bold green]Re-indexing documents..."):
                vector_store = build_vector_store()
                graph_ref[0] = build_rag_graph(vector_store, model_config)
            conversation_history.clear()
            console.print("[bold green]Knowledge base rebuilt![/bold green]\n")
            continue

        if question.lower() == "/models":
            show_models(model_config)
            continue

        if question.lower().startswith("/model "):
            new_model = question[7:].strip()
            if new_model:
                switch_model(new_model, model_config, vector_store, graph_ref)
                conversation_history.clear()
                console.print("[dim]Conversation history cleared.[/dim]")
            else:
                console.print("[yellow]Usage: /model <model-name>  (e.g. /model deepseek-r1:8b)[/yellow]")
            continue

        if question.lower() == "/summarize":
            with console.status("[bold green]Summarizing documents..."):
                results = summarize_documents(vector_store, model_config.model_name)
            if not results:
                console.print("[yellow]No documents found in the knowledge base. Drop files into docs/ and run rebuild.[/yellow]")
            else:
                for display_name, summary in results:
                    console.print()
                    console.print(Panel(
                        Markdown(summary),
                        title=f"[bold magenta]{display_name}[/bold magenta]",
                        border_style="magenta",
                    ))
            console.print()
            continue

        with console.status("[bold green]Thinking..."):
            answer, new_messages = run_query(
                graph_ref[0],
                question,
                history=conversation_history[-HISTORY_WINDOW:],
                model_config=model_config,
            )
        conversation_history.extend(new_messages)

        console.print()
        console.print(Panel(
            Markdown(answer),
            title=f"[bold green]Assistant[/bold green] [dim]({model_config.model_name})[/dim]",
            border_style="green",
        ))
        console.print()


if __name__ == "__main__":
    main()
