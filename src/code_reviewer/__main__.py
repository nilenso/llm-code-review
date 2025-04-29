import click
from rich.console import Console
from rich.panel import Panel

from .formatter import format_review
from .indexer import CodeIndexer
from .models import CodeReviewRequest
from .ollama_client import OllamaClient
from .planner import CodeReviewer

DEFAULT_PLANNER_MODEL   = "llama3.1:latest"
DEFAULT_WORKER_MODEL    = "qwen2.5-coder:7b-instruct-q8_0"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_OLLAMA_HOST     = "http://localhost:11434"
DEFAULT_REPO_PATH       = "./"
OUTPUT_PATH = "./out"
OUTPUT_FILE = f"{OUTPUT_PATH}/test_results.html"

console = Console()

def review_code_with_rag(
    system_prompt: str,
    git_diff_path: str,
    repo_path: str = DEFAULT_REPO_PATH,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    planner_model: str = DEFAULT_PLANNER_MODEL,
    worker_model: str = DEFAULT_WORKER_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reindex: bool = False
) -> str:
    """
    Main function to run the code review process with RAG.
    Simplified synchronous version.
    
    Args:
        system_prompt: Instructions for the code review
        git_diff_path: Path to file containing git diff output
        repo_path: Path to the repository root
        ollama_host: Host of the Ollama API
        planner_model: Model to use for planning and summarization
        worker_model: Model to use for worker agents
        embedding_model: Model to use for embeddings
        reindex: Force reindexing of the codebase
        
    Returns:
        Formatted code review
    """
    with open(git_diff_path, 'r') as f:
        git_diff = f.read()
    
    request = CodeReviewRequest(
        system_prompt=system_prompt,
        git_diff=git_diff,
        repo_path=repo_path,
        models={
            "planner": planner_model,
            "worker": worker_model,
            "embedding": embedding_model
        },
        reindex=reindex
    )
    
    ollama_client = OllamaClient(host=ollama_host)
    
    code_indexer = CodeIndexer(
        repo_path=request.repo_path,
        embedding_model=request.models.get("embedding", embedding_model),
        ollama_host=ollama_host
    )

    console.print(f"[green]:arrows_counterclockwise: Indexing repository at {request.repo_path}")
    code_indexer.index_repository(force_reindex=request.reindex)
    
    planner = CodeReviewer(
        ollama_client=ollama_client,
        code_indexer=code_indexer,
        planner_model=planner_model,
        worker_model=worker_model
    )

    console.print("[green]:arrows_counterclockwise: Running Code Review")
    response = planner.plan_and_execute(request)
    
    return response, git_diff


@click.command(help="RAG-Enhanced Code Review Agent using Ollama (Simplified Version)")
@click.option("--diff", required=False, help="Path to git diff file")
@click.option("--repo", default="./", help="Path to repository root")
@click.option("--prompt", default="Perform a thorough code review focusing on best practices and code quality.", 
              help="System prompt for the code review")
@click.option("--ollama-host", default="http://localhost:11434", help="Host of the Ollama API")
@click.option("--planner-model", default=DEFAULT_PLANNER_MODEL, help="Model to use for planning and summarization")
@click.option("--worker-model", default=DEFAULT_WORKER_MODEL, help="Model to use for worker agents")
@click.option("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Model to use for embeddings")
@click.option("--reindex", is_flag=True, help="Force reindexing of the codebase")
@click.option("--format", "format_type", type=click.Choice(["markdown", "html", "comprehensive_html"]), default="markdown",
              help="Output format (markdown or html)")
def main(diff, repo, prompt, ollama_host, planner_model, worker_model, embedding_model, reindex, format_type):
    if not diff:
        console.print("[red]Error: --diff option is required unless --test-ui is specified.[/red]")
        return

    config_info = (
        f"[bold]Diff source:[/bold] {diff}\n"
        f"[bold]Repository:[/bold] {repo}\n"
        f"[bold]Ollama Host:[/bold] {ollama_host}\n"
        f"[bold]Planner Model:[/bold] {planner_model}\n"
        f"[bold]Worker Model:[/bold] {worker_model}\n"
        f"[bold]Embedding Model:[/bold] {embedding_model}"
    )

    console.print(Panel(config_info, title="Running RAG-Enhanced Code Review", border_style="bold green"))

    result, git_diff = review_code_with_rag(
        system_prompt=prompt,
        git_diff_path=diff,
        repo_path=repo,
        ollama_host=ollama_host,
        planner_model=planner_model,
        worker_model=worker_model,
        embedding_model=embedding_model,
        reindex=reindex
    )
    
    formatted_result = format_review(git_diff, result, format_type, repo)

    if format_type != "markdown":
        with open(OUTPUT_FILE, 'w') as f:
            f.write(formatted_result)
        console.print(f"[green]:white_check_mark:Review saved to [bold]{OUTPUT_FILE}[/bold]")
    else:
        console.print("Code Review Results", style="bold blue underline")
        console.print(formatted_result)


if __name__ == "__main__":
    main()
