import argparse
import os

from rich.console import Console

from .formatter import format_review
from .models import CodeReviewCategory, CodeReviewComment, CodeReviewResponse
from .ollama_client import OllamaClient
from .prompts import changelog_system_prompt, changelog_user_prompt

DEFAULT_REPO_PATH = "./test_repo_ui"
OUTPUT_FILE = "./out/test_results.html"

console = Console()

def generate_test_report(repo_path: str):
    """Generates a sample comprehensive HTML report for UI testing."""
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_client = OllamaClient(host=ollama_host)
    planner_model = "qwen2.5-coder:7b"
    sample_diff = "\n".join([
        "diff --git a/foo.py b/foo.py",
        "index 123..456 100644",
        "--- a/foo.py",
        "+++ b/foo.py",
        "@@ -1,6 +1,6 @@",
        "-def foo(x):",
        "-    # Old comment",
        "-    removed_line = x - 1",
        "-    return x + 1",
        "+def foo(y):",
        "+    # This is a new comment",
        "+    z = y * 2",
        "+    return y + z",
        "",
        "diff --git a/bar.py b/bar.py",
        "index 789..abc 100644",
        "--- a/bar.py",
        "+++ b/bar.py",
        "@@ -1,8 +1,8 @@",
        "-class Bar:",
        "-    def __init__(self, v):",
        "-        self.v = v",
        "-    def unused(self):",
        "-        pass",
        "+class Bar:",
        "+    def __init__(self, val):",
        "+        self.val = val",
        "+        print(\"Bar initialized\")",
        "+",
        "+    def double(self):",
        "+        return self.val * 2",
        "",
        "diff --git a/baz.py b/baz.py",
        "index def..123 100644",
        "--- a/baz.py",
        "+++ b/baz.py",
        "@@ -1,10 +1,8 @@",
        "-def unused_func():",
        "-    pass",
        "-",
        "-def greet_old(name):",
        "-    print(f\"Hi, {name}\")",
        "+def greet(name):",
        "+    print(f\"Hello, {name}!\")",
        "-",
        "-def add_old(a, b):",
        "-    return a - b",
        "+def add(a, b):",
        "+    return a + b",
        "",
        " def main():",
        "     greet(\"World\")"
    ])

    sample_comments = [
        # foo.py
        CodeReviewComment(
            category=CodeReviewCategory.NAMING,
            file_name="foo.py",
            line_number=1,
            comment="Consider renaming 'foo' to something more descriptive."
        ),
        CodeReviewComment(
            category=CodeReviewCategory.FUNCTIONALITY,
            file_name="foo.py",
            line_number=3,
            comment="Check if changing 'x' to 'y' affects callers."
        ),
        CodeReviewComment(
            category=CodeReviewCategory.CODING_STYLE,
            file_name="foo.py",
            line_number=2,
            comment="Avoid unnecessary comments."
        ),
        CodeReviewComment(
            category=CodeReviewCategory.FUNCTIONALITY,
            file_name="foo.py",
            line_number=0,
            comment="File-level: Review logic for correctness."
        ),
        # bar.py
        CodeReviewComment(
            category=CodeReviewCategory.NAMING,
            file_name="bar.py",
            line_number=1,
            comment="'val' could be more descriptive than 'v'."
        ),
        CodeReviewComment(
            category=CodeReviewCategory.FUNCTIONALITY,
            file_name="bar.py",
            line_number=4,
            comment="Ensure print statements are removed in production."
        ),
        # baz.py
        CodeReviewComment(
            category=CodeReviewCategory.READABILITY,
            file_name="baz.py",
            line_number=1,
            comment="Add a docstring for the 'greet' function."
        ),
        CodeReviewComment(
            category=CodeReviewCategory.TESTS,
            file_name="baz.py",
            line_number=5,
            comment="Consider adding tests for 'add'."
        ),
    ]
    
    console.print(f"[blue]Generating changelog using model '{planner_model}'...[/blue]")
    changelog_text = "Could not generate changelog."
    if sample_diff and sample_diff.strip():
        changelog_messages = [
            {"role": "system", "content": changelog_system_prompt()},
            {"role": "user", "content": changelog_user_prompt(git_diff=sample_diff)}
        ]
        try:
            changelog_text = ollama_client.chat(
                model=planner_model,
                messages=changelog_messages
            ).strip()
            console.print("[green]Changelog generated.[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate changelog via LLM: {e}[/yellow]")
            changelog_text = f"Changelog generation failed: {e}"
    else:
        changelog_text = "No changes detected in the diff."

    response = CodeReviewResponse(
        categories={
            CodeReviewCategory.NAMING: [sample_comments[0], sample_comments[4]],
            CodeReviewCategory.FUNCTIONALITY: [sample_comments[1], sample_comments[3], sample_comments[5]],
            CodeReviewCategory.CODING_STYLE: [sample_comments[2]],
            CodeReviewCategory.READABILITY: [sample_comments[6]],
            CodeReviewCategory.TESTS: [sample_comments[7]],
        },
        summary=changelog_text
    )

    files_content = {
        "foo.py": "\n".join([
            "def foo(x):",
            "# Old comment",
            "removed_line = x - 1",
            "return x + 1"
        ]),
        "bar.py": "\n".join([
            "class Bar:",
            "def __init__(self, v):",
            "    self.v = v",
            "def unused(self):",
            "    pass"
        ]),
        "baz.py": "\n".join([
            "def unused_func():",
            "pass",
            "",
            "def greet_old(name):",
            "print(f\"Hi, {name}\")",
            "",
            "def add_old(a, b):",
            "return a - b",
            "",
            "def main():",
            "greet(\"World\")"
        ])
    }

    temp_files_created = []
    os.makedirs(repo_path, exist_ok=True)
    console.print(f"[blue]Creating temporary files in: {os.path.abspath(repo_path)}[/blue]")
    for filename, content in files_content.items():
        filepath = os.path.join(repo_path, filename)
        try:
            with open(filepath, "w") as f:
                f.write(content)
            temp_files_created.append(filepath)
            console.print(f"  [green]Created: {filename}[/green]")
        except Exception as e:
            console.print(f"  [yellow]Warning: Could not create temporary file {filepath}: {e}[/yellow]")

    final_output_path = os.path.abspath(OUTPUT_FILE)
    try:
        html = format_review(sample_diff, response, "comprehensive_html", repo_path)
        output_dir = os.path.dirname(final_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(final_output_path, "w") as f:
            f.write(html)
        console.print(f"[green]:white_check_mark: Sample UI report saved to [bold]{final_output_path}[/bold]")
    except Exception as e:
         console.print(f"[red]Error generating or saving report: {e}[/red]")
    finally:
        console.print("[blue]Cleaning up temporary files...[/blue]")
        for filepath in temp_files_created:
            try:
                os.remove(filepath)
                console.print(f"  [green]Removed: {os.path.basename(filepath)}[/green]")
            except Exception as e:
                console.print(f"  [yellow]Warning: Could not remove temporary file {filepath}: {e}[/yellow]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a sample comprehensive HTML report for UI testing.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO_PATH,
        help=f"Directory to create temporary source files in (defaults to '{DEFAULT_REPO_PATH}')."
    )

    args = parser.parse_args()

    generate_test_report(repo_path=args.repo) 