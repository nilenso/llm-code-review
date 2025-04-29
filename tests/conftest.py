from unittest.mock import MagicMock, patch

import pytest

from code_reviewer.indexer import CodeIndexer

# Imports needed by fixtures
from code_reviewer.models import CodeReviewCategory
from code_reviewer.ollama_client import OllamaClient
from code_reviewer.planner import CodeReviewer
from code_reviewer.worker import MAX_CODE_LINES, CodeReviewWorker


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    return MagicMock(spec=OllamaClient)

@pytest.fixture
def mock_code_indexer() -> MagicMock:
    mock = MagicMock(spec=CodeIndexer)
    mock.retrieve_symbol_context.return_value = [{"file_path": "mock/symbol.py", "start_line": 1, "end_line": 5, "content": "symbol context"}]
    mock.retrieve_file_context.return_value = [{"file_path": "mock/file.py", "start_line": 1, "end_line": 10, "content": "file context"}]
    return mock

@pytest.fixture
def mock_reranker() -> MagicMock:
    with patch('code_reviewer.worker.ContextReranker') as mock_reranker_cls:
        mock_instance = mock_reranker_cls.return_value
        mock_instance.rank.side_effect = lambda query, docs: docs
        yield mock_instance

@pytest.fixture
def worker(mock_ollama_client: MagicMock, mock_code_indexer: MagicMock) -> CodeReviewWorker:
    return CodeReviewWorker(
        category=CodeReviewCategory.DESIGN,
        ollama_client=mock_ollama_client,
        code_indexer=mock_code_indexer,
        model="test-worker-model"
    )

@pytest.fixture
def sample_git_diff_simple() -> str:
    return \
        """diff --git a/file1.py b/file1.py
index 0000000..aaaaaaa 100644
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,4 @@
-def old_function():
-    print("Old greeting") # Line changed
+def new_function(): # Line changed
+    print("New greeting v2") # Line changed
+    # Added line
     pass
diff --git a/file2.txt b/file2.txt
index 0000000..bbbbbbb 100644
--- a/file2.txt
+++ b/file2.txt
@@ -1 +1,2 @@
-Old content.
+New content version 2.
+Another added line.
"""

@pytest.fixture
def sample_git_diff_long() -> str:
    num_added = MAX_CODE_LINES + 10
    added_lines_block = "\n".join([f"+    added line {i}" for i in range(1, num_added + 1)])

    first_hunk_target_lines = 1 + num_added

    return \
        f"""diff --git a/long_file.py b/long_file.py
index 0000000..ccccccc 100644
--- a/long_file.py
+++ b/long_file.py
@@ -1,1 +1,{first_hunk_target_lines} @@
 def existing_function(param=None):
{added_lines_block}
@@ -20,4 +{first_hunk_target_lines + 10},1 @@
-    removed_line_1 = True
-    removed_line_2 = False
-    removed_line_3 = None
-    existing_var = 1
+    existing_var = 2 # Line changed
"""

@pytest.fixture
def complex_diff() -> str:
    return "\n".join([
        "diff --git a/file1.py b/file1.py",
        "index 1111111..2222222 100644",
        "--- a/file1.py",
        "+++ b/file1.py",
        "@@ -1,5 +1,8 @@",
        "-def old_function():",
        "-    pass",
        "+def new_function():",
        "+    print(\"new\")",
        "+",
        "+def helper():",
        "+    print(\"helper\")",
        "-",
        "+class RenamedClass:",
        "+    def method(self): pass",
        "-",
        "-old_function()",
        "+new_function()",
    ])

@pytest.fixture
def modified_files_complex_diff() -> str:
    return "\n".join([
        "diff --git a/file1.py b/file1.py",
        "index 1111111..2222222 100644",
        "--- a/file1.py",
        "+++ b/file1.py",
        "@@ -1,1 +1,3 @@",
        "-x = 1",
        "+x = 2",
        "+y = 3",
        "+print(x)",
        "",
        "diff --git a/file2.py b/file2_renamed.py",
        "similarity index 80%",
        "rename from file2.py",
        "rename to file2_renamed.py",
        "--- a/file2.py",
        "+++ b/file2_renamed.py",
        "@@ -1,2 +1,2 @@",
        "-class ToBeRemoved:",
        "-    pass",
        "+class ToBeAdded:",
        "+    pass",
        "",
        "diff --git a/file3.py b/file3.py",
        "index 3333333..4444444 100644",
        "--- a/file3.py",
        "+++ b/file3.py",
        "@@ -10,1 +10,2 @@",
        "+def another_new_func():",
        "-    # Old comment",
        "+    \"\"\"This function returns foo\"\"\"",
        "+    return \"foo\"",
    ])

@pytest.fixture
def planner_agent(
    mock_ollama_client: MagicMock, 
    mock_code_indexer: MagicMock 
) -> CodeReviewer:
    return CodeReviewer(
        ollama_client=mock_ollama_client,
        code_indexer=mock_code_indexer,
        planner_model="test-planner-model",
        worker_model="test-worker-model"
    )