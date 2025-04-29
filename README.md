# RAG-Enhanced Code Review Agent Guide

## Prerequisites

1. **Install Ollama**: Download and install from [ollama.ai](https://ollama.ai)
2. **Pull required models**:
   ```bash
   ollama pull llama3             # For planner agent
   ollama pull codellama          # For worker agents
   ollama pull nomic-embed-text   # For text embeddings
   ```
3. **Install uv (Python package manager)**:
   ```bash
   pip install uv
   ```

## Setting Up Your Environment

1. **Clone the repository**
2. **Install dependencies with uv** (see above)
3. **(Recommended) Create a virtual environment:**
   ```bash
   $ python -m venv .venv
   $ source .venv/bin/activate
   ```
4. **Install Python dependencies**:
   ```bash
   $ uv pip install -r requirements.lock
   ```

## Running a Code Review

### Basic Usage

```bash
python -m src.code_reviewer --diff patch.txt \
    --repo /path/to/repo \
    --planner-model llama3.1:latest \
    --worker-model qwen2.5-coder:7b \
    --embedding-model bge-m3:latest
```

Or, if you have a script like `ocra.py` at the project root:

```bash
python ocra.py --diff patch.txt \
    --repo /path/to/repo \
    --planner-model llama3.1:latest \
    --worker-model qwen2.5-coder:7b \
    --embedding-model bge-m3:latest
```

## Configuration Options

| Option             | Description                                | Default                             |
|--------------------|--------------------------------------------|-------------------------------------|
| `--diff`           | Path to git diff file                      | *Required*                          |
| `--repo`           | Path to repository root                    | `./` (current directory)            |
| `--prompt`         | System prompt for the code review          | "Perform a thorough code review..." |
| `--ollama-host`    | Host of the Ollama API                     | http://localhost:11434              |
| `--planner-model`  | Model for planning and summarization       | llama3                              |
| `--worker-model`   | Model for worker agents                    | codellama                           |
| `--embedding-model`| Model for text embeddings                  | nomic-embed-text                    |
| `--reindex`        | Force reindexing of the codebase           | false                               |
| `--output`         | Output file path (defaults to stdout)      | *stdout*                            |

## Testing & Evaluation

To run all tests:
  ```bash
  $ pytest
  ```

To evaluate the system, do the following:

```bash
# Install project as a package to enable running the evaluation script
$ pip install -e .

# Evaluate context retrieval
$ python -m code_reviewer.evals.code_retrieval_eval --test-cases-file src/code_reviewer/evals/test_cases.json

# Evaluate symbol lookup
$ python -m code_reviewer.evals.code_retrieval_eval --test-cases-file src/code_reviewer/evals/test_cases_symbol_lookup.json
```


### Generate Git Diff

```bash
# For all changes in staging area
git diff > patch.txt

# For changes between branches
git diff main feature-branch > patch.txt

# For changes in specific files
git diff -- path/to/file1.py path/to/file2.py > patch.txt
```

## Performance Optimization Tips

1. **Use a smaller embedding model** for faster indexing of large codebases
2. **Index once, review many times** - avoid the `--reindex` flag after the initial setup
3. **Run on a machine with good CPU/RAM** for faster embedding and LLM inference
4. **Focus diffs on smaller changes** for more targeted and helpful reviews

## Additional Notes

- The `.chroma` directory stores the code index and can be deleted to force a full reindex.
- For advanced configuration, review the CLI options and source code in `src/code_reviewer/`.

