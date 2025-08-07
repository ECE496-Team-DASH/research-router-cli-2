"""
Elia CLI
"""

import asyncio
import pathlib
from textwrap import dedent
import tomllib
from typing import Any

import click
from click_default_group import DefaultGroup

from rich.console import Console
from rich.table import Table

from elia_chat.app import Elia
from elia_chat.config import LaunchConfig
from elia_chat.database.import_chatgpt import import_chatgpt_data
from elia_chat.database.database import create_database, sqlite_file_name
from elia_chat.locations import config_file
from elia_chat.graphrag_manager import get_graphrag_manager, is_graphrag_available

console = Console()

def create_db_if_not_exists() -> None:
    if not sqlite_file_name.exists():
        click.echo(f"Creating database at {sqlite_file_name!r}")
        asyncio.run(create_database())

def load_or_create_config_file() -> dict[str, Any]:
    config = config_file()

    try:
        file_config = tomllib.loads(config.read_text())
    except FileNotFoundError:
        file_config = {}
        try:
            config.touch()
        except OSError:
            pass

    return file_config


def save_config_file(config_data: dict[str, Any]) -> None:
    """Save configuration to the config file."""
    config = config_file()
    try:
        # Try to use tomli_w for writing TOML
        import tomli_w
        config.write_text(tomli_w.dumps(config_data))
    except ImportError:
        try:
            # Try to use toml for writing
            import toml
            config.write_text(toml.dump(config_data))
        except ImportError:
            # Fallback - write a simple TOML-like format
            lines = []
            for key, value in config_data.items():
                if isinstance(value, dict):
                    lines.append(f"[{key}]")
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str):
                            lines.append(f'{subkey} = "{subvalue}"')
                        elif isinstance(subvalue, bool):
                            lines.append(f'{subkey} = {str(subvalue).lower()}')
                        else:
                            lines.append(f'{subkey} = {subvalue}')
                    lines.append("")
                else:
                    if isinstance(value, str):
                        lines.append(f'{key} = "{value}"')
                    elif isinstance(value, bool):
                        lines.append(f'{key} = {str(value).lower()}')
                    else:
                        lines.append(f'{key} = {value}')
            config.write_text("\n".join(lines))

@click.group(cls=DefaultGroup, default="default", default_if_no_args=True)
def cli() -> None:
    """Interact with large language models using your terminal."""

@cli.command()
@click.argument("prompt", nargs=-1, type=str, required=False)
@click.option(
    "-m",
    "--model",
    type=str,
    default="",
    help="The model to use for the chat",
)
@click.option(
    "-i",
    "--inline",
    is_flag=True,
    help="Run in inline mode, without launching full TUI.",
    default=False,
)
def default(prompt: tuple[str, ...], model: str, inline: bool) -> None:
    prompt = prompt or ("",)
    joined_prompt = " ".join(prompt)
    create_db_if_not_exists()
    file_config = load_or_create_config_file()
    cli_config = {}
    if model:
        cli_config["default_model"] = model

    launch_config: dict[str, Any] = {**file_config, **cli_config}
    app = Elia(LaunchConfig(**launch_config), startup_prompt=joined_prompt)
    app.run(inline=inline)

@cli.command()
def reset() -> None:
    """
    Reset the database

    This command will delete the database file and recreate it.
    Previously saved conversations and data will be lost.
    """
    from rich.padding import Padding
    from rich.text import Text

    console.print(
        Padding(
            Text.from_markup(
                dedent(f"""\
[u b red]Warning![/]

[b red]This will delete all messages and chats.[/]

You may wish to create a backup of \
"[bold blue u]{str(sqlite_file_name.resolve().absolute())}[/]" before continuing.
            """)
            ),
            pad=(1, 2),
        )
    )
    if click.confirm("Delete all chats?", abort=True):
        sqlite_file_name.unlink(missing_ok=True)
        asyncio.run(create_database())
        console.print(f"♻️  Database reset @ {sqlite_file_name}")

@cli.command("import")
@click.argument(
    "file",
    type=click.Path(
        exists=True, dir_okay=False, path_type=pathlib.Path, resolve_path=True
    ),
)
def import_file_to_db(file: pathlib.Path) -> None:
    """
    Import ChatGPT Conversations

    This command will import the ChatGPT conversations from a local
    JSON file into the database.
    """
    asyncio.run(import_chatgpt_data(file=file))
    console.print(f"[green]ChatGPT data imported from {str(file)!r}")


@cli.group(name="graphrag")
def graphrag_commands() -> None:
    """GraphRAG configuration and management commands."""
    if not is_graphrag_available():
        console.print("[red]Error: nano-graphrag is not available. Please install it first.[/]")
        raise click.Abort()


@graphrag_commands.command(name="configure")
@click.option(
    "--documents-folder",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    help="Folder containing documents to index (.txt, .md, .pdf files)",
)
@click.option(
    "--storage-folder", 
    type=click.Path(path_type=pathlib.Path),
    help="Folder for GraphRAG storage and persistence",
)
@click.option(
    "--enable/--disable",
    default=None,
    help="Enable or disable GraphRAG functionality",
)
@click.option(
    "--model",
    type=str,
    help="Model to use for GraphRAG operations",
)
@click.option(
    "--query-mode",
    type=click.Choice(["global", "local", "naive"]),
    help="Default query mode for GraphRAG",
)
def configure_graphrag(
    documents_folder: pathlib.Path | None,
    storage_folder: pathlib.Path | None,
    enable: bool | None,
    model: str | None,
    query_mode: str | None,
) -> None:
    """Configure GraphRAG settings."""
    config_data = load_or_create_config_file()
    
    # Initialize graphrag section if it doesn't exist
    if "graphrag" not in config_data:
        config_data["graphrag"] = {}
    
    graphrag_config = config_data["graphrag"]
    
    # Update configuration based on provided options
    if enable is not None:
        graphrag_config["enabled"] = enable
        status = "enabled" if enable else "disabled"
        console.print(f"[green]GraphRAG {status}[/]")
    
    if documents_folder:
        graphrag_config["documents_folder"] = str(documents_folder.absolute())
        console.print(f"[green]Documents folder set to: {documents_folder.absolute()}[/]")
    
    if storage_folder:
        storage_folder.mkdir(parents=True, exist_ok=True)
        graphrag_config["storage_folder"] = str(storage_folder.absolute())
        console.print(f"[green]Storage folder set to: {storage_folder.absolute()}[/]")
    
    if model:
        graphrag_config["graphrag_model"] = model
        console.print(f"[green]GraphRAG model set to: {model}[/]")
    
    if query_mode:
        graphrag_config["query_mode"] = query_mode
        console.print(f"[green]Query mode set to: {query_mode}[/]")
    
    # Save configuration
    save_config_file(config_data)
    console.print("[green]Configuration saved[/]")


@graphrag_commands.command(name="status")
def graphrag_status() -> None:
    """Show GraphRAG configuration and status."""
    config_data = load_or_create_config_file()
    graphrag_config = config_data.get("graphrag", {})
    
    table = Table(title="GraphRAG Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Enabled", str(graphrag_config.get("enabled", False)))
    table.add_row("Documents Folder", graphrag_config.get("documents_folder", "Not set"))
    table.add_row("Storage Folder", graphrag_config.get("storage_folder", "Not set"))
    table.add_row("Model", graphrag_config.get("graphrag_model", "Default"))
    table.add_row("Query Mode", graphrag_config.get("query_mode", "global"))
    
    console.print(table)
    
    # Show additional status if enabled
    if graphrag_config.get("enabled"):
        try:
            from elia_chat.config import GraphRAGConfig
            config = GraphRAGConfig(**graphrag_config)
            manager = get_graphrag_manager(config)
            
            if manager.is_enabled:
                indexed_count = manager.get_indexed_files_count()
                console.print(f"\n[green]✓ GraphRAG is operational[/]")
                console.print(f"[blue]Indexed files: {indexed_count}[/]")
            else:
                console.print(f"\n[yellow]⚠ GraphRAG is enabled but not properly configured[/]")
        except Exception as e:
            console.print(f"\n[red]✗ Error checking GraphRAG status: {e}[/]")


@graphrag_commands.command(name="index")
@click.option(
    "--force",
    is_flag=True,
    help="Force re-indexing of all documents",
)
def index_documents(force: bool) -> None:
    """Index documents from the configured documents folder."""
    config_data = load_or_create_config_file()
    graphrag_config = config_data.get("graphrag", {})
    
    if not graphrag_config.get("enabled"):
        console.print("[red]GraphRAG is not enabled. Use 'elia graphrag configure --enable' first.[/]")
        return
    
    try:
        from elia_chat.config import GraphRAGConfig
        config = GraphRAGConfig(**graphrag_config)
        manager = get_graphrag_manager(config)
        
        if not manager.is_enabled:
            console.print("[red]GraphRAG is not properly configured. Check documents and storage folders.[/]")
            return
        
        console.print("[blue]Starting document indexing...[/]")
        
        async def run_indexing():
            return await manager.index_documents(force_reindex=force)
        
        results = asyncio.run(run_indexing())
        
        if "error" in results:
            console.print(f"[red]Error: {results['error']}[/]")
        else:
            console.print(f"[green]✓ Indexing complete![/]")
            console.print(f"[blue]Files processed: {results.get('files_processed', 0)}[/]")
            console.print(f"[blue]Files skipped: {results.get('files_skipped', 0)}[/]")
            
            if results.get('errors'):
                console.print(f"[yellow]Errors encountered:[/]")
                for error in results['errors']:
                    console.print(f"  [red]• {error}[/]")
    
    except Exception as e:
        console.print(f"[red]Error during indexing: {e}[/]")


@graphrag_commands.command(name="query")
@click.argument("query_text", nargs=-1, type=str, required=True)
@click.option(
    "--mode",
    type=click.Choice(["global", "local", "naive"]),
    help="Query mode (overrides default)",
)
def query_graphrag(query_text: tuple[str, ...], mode: str | None) -> None:
    """Query the GraphRAG knowledge base."""
    query = " ".join(query_text)
    config_data = load_or_create_config_file()
    graphrag_config = config_data.get("graphrag", {})
    
    if not graphrag_config.get("enabled"):
        console.print("[red]GraphRAG is not enabled. Use 'elia graphrag configure --enable' first.[/]")
        return
    
    try:
        from elia_chat.config import GraphRAGConfig
        config = GraphRAGConfig(**graphrag_config)
        manager = get_graphrag_manager(config)
        
        if not manager.is_enabled:
            console.print("[red]GraphRAG is not properly configured.[/]")
            return
        
        console.print(f"[blue]Querying: {query}[/]")
        if mode:
            console.print(f"[blue]Mode: {mode}[/]")
        
        async def run_query():
            return await manager.query_graphrag(query, mode)
        
        result = asyncio.run(run_query())
        console.print(f"\n[green]Response:[/]")
        console.print(result)
    
    except Exception as e:
        console.print(f"[red]Error during query: {e}[/]")


@graphrag_commands.command(name="clear")
@click.confirmation_option(prompt="Are you sure you want to clear the GraphRAG index?")
def clear_graphrag_index() -> None:
    """Clear the GraphRAG index and storage."""
    config_data = load_or_create_config_file()
    graphrag_config = config_data.get("graphrag", {})
    
    try:
        from elia_chat.config import GraphRAGConfig
        config = GraphRAGConfig(**graphrag_config)
        manager = get_graphrag_manager(config)
        
        if manager.clear_index():
            console.print("[green]✓ GraphRAG index cleared successfully[/]")
        else:
            console.print("[red]✗ Failed to clear GraphRAG index[/]")
    
    except Exception as e:
        console.print(f"[red]Error clearing index: {e}[/]")

if __name__ == "__main__":
    cli()
