import os
from unittest.mock import MagicMock, patch

import pytest

from code_reviewer.indexer import CodeIndexer

FIXTURE_REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fixtures', 'test_repo'))

@pytest.fixture
def indexer_setup(request):
    """Sets up the CodeIndexer instance with mocks for testing."""
    repo_path = FIXTURE_REPO_PATH
    gitignore_path = os.path.join(repo_path, ".gitignore")
    app_file_path = os.path.join(repo_path, "app.py")
    requirements_path = os.path.join(repo_path, "requirements.txt")
    template_file_path = os.path.join(repo_path, "templates", "index.html")
    css_file_path = os.path.join(repo_path, "static", "style.css")
    venv_file_path = os.path.join(repo_path, ".venv", "dummy_file")

    mock_ollama_client = MagicMock()
    mock_ollama_client.embeddings.return_value = {"embedding": [0.1] * 768}
    mock_chroma_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_collection.add = MagicMock()
    mock_collection.query = MagicMock()
    mock_chroma_client.get_or_create_collection.return_value = mock_collection
    mock_chroma_client.delete_collection = MagicMock()
    mock_chroma_client.create_collection = MagicMock(return_value=mock_collection)

    patcher_ollama = patch('code_reviewer.indexer.ollama.Client', return_value=mock_ollama_client)
    patcher_chroma = patch('code_reviewer.indexer.chromadb.PersistentClient', return_value=mock_chroma_client)

    mock_ollama = patcher_ollama.start()
    mock_chroma = patcher_chroma.start()

    indexer = CodeIndexer(repo_path=repo_path)
    indexer.collection = mock_collection
    indexer.embedding_func.client = mock_ollama_client
    indexer.client = mock_chroma_client

    # Store necessary variables and mocks in the request class object
    request.cls.repo_path = repo_path
    request.cls.gitignore_path = gitignore_path
    request.cls.app_file_path = app_file_path
    request.cls.requirements_path = requirements_path
    request.cls.template_file_path = template_file_path
    request.cls.css_file_path = css_file_path
    request.cls.venv_file_path = venv_file_path
    request.cls.indexer = indexer
    request.cls.mock_collection = mock_collection
    request.cls.mock_chroma_client = mock_chroma_client
    request.cls.mock_ollama_client = mock_ollama_client

    yield # Test runs here

    patcher_ollama.stop()
    patcher_chroma.stop()


@pytest.mark.usefixtures("indexer_setup")
class TestCodeIndexer:

    def test_init(self):
         """Test CodeIndexer initialization."""
         assert self.indexer.repo_path == self.repo_path
         assert self.indexer.embedding_model == "nomic-embed-text"
         assert self.indexer.collection_name == "code_repository"
         assert self.indexer.ollama_host == "http://localhost:11434"

         expected_db_path = os.path.join(self.repo_path, ".chroma")

         assert self.indexer.db_path == expected_db_path
         assert self.indexer.client is not None
         assert self.indexer.ollama_client is not None
         assert self.indexer.gitignore_matcher is not None
         assert self.indexer.embedding_func is not None
         assert self.indexer.collection == self.mock_collection


    def test_gitignore_matcher(self):
        """Test gitignore matching with web server fixture using absolute paths."""
        assert not self.indexer.gitignore_matcher(self.app_file_path)
        assert not self.indexer.gitignore_matcher(self.requirements_path)
        assert not self.indexer.gitignore_matcher(self.template_file_path)
        assert not self.indexer.gitignore_matcher(self.css_file_path)

    def test_should_index_file(self):
        """Test file indexing decisions using absolute paths (as input) but checking relative internally."""
        assert self.indexer.should_index_file(self.app_file_path)
        assert self.indexer.should_index_file(self.requirements_path)
        assert self.indexer.should_index_file(self.template_file_path)
        assert self.indexer.should_index_file(self.css_file_path)

        assert not self.indexer.should_index_file(self.venv_file_path)

        git_config_path = os.path.join(self.repo_path, ".git", "config")
        git_dir = os.path.dirname(git_config_path)
        if not os.path.exists(git_dir): os.makedirs(git_dir)
        if not os.path.exists(git_config_path): open(git_config_path, 'a').close()
        assert not self.indexer.should_index_file(git_config_path)
        if os.path.exists(git_config_path) and "config" in git_config_path : os.remove(git_config_path)
        if os.path.exists(git_dir) and not os.listdir(git_dir): os.rmdir(git_dir)

        # Should NOT index (ignored extension)
        img_path_temp = os.path.join(self.repo_path, "temp.png")
        open(img_path_temp, 'a').close()
        assert not self.indexer.should_index_file(img_path_temp)
        if os.path.exists(img_path_temp): os.remove(img_path_temp)

    def test_extract_symbols_python_webserver(self):
        """Test symbol extraction for the fixture Flask app."""
        try:
            with open(self.app_file_path, 'r', encoding='utf-8') as f: content = f.read()
            symbols = self.indexer.extract_symbols(self.app_file_path, content)
            # Check for all expected function definitions
            assert "hello_world" in symbols
            assert "about_page" in symbols # Added check
            assert "get_data" in symbols   # Added check
            # Still expect Flask/app not to be extracted by current logic
        except FileNotFoundError:
             pytest.skip(f"Fixture file not found: {self.app_file_path}")

    def test_chunk_file_webserver(self):
        """Test file chunking logic on the Flask app file."""
        try:
            with open(self.app_file_path, 'r', encoding='utf-8') as f: content = f.read()
        except FileNotFoundError:
             pytest.skip(f"Fixture file not found: {self.app_file_path}")

        # Use a slightly larger chunk size to potentially group routes
        chunks = self.indexer.chunk_file(os.path.basename(self.app_file_path), content, chunk_size=10)

        assert len(chunks) > 0

        # Check if function definitions are found within chunks and have correct symbols
        found_hello_world = False
        found_about_page = False
        found_get_data = False

        for chunk in chunks:
            if "def hello_world():" in chunk.content:
                found_hello_world = True
                assert "hello_world" in chunk.symbols
            if "def about_page():" in chunk.content:
                found_about_page = True
                assert "about_page" in chunk.symbols
            if "def get_data():" in chunk.content:
                found_get_data = True
                assert "get_data" in chunk.symbols

        assert found_hello_world, "Chunk for 'def hello_world():' not found"
        assert found_about_page, "Chunk for 'def about_page():' not found"
        assert found_get_data, "Chunk for 'def get_data():' not found"

        # Basic check on first chunk path
        assert chunks[0].file_path == os.path.basename(self.app_file_path)

    def test_index_repository(self):
        """Test indexing the web server fixture repo."""
        self.mock_collection.count.return_value = 0
        self.indexer.index_repository()

        self.mock_collection.add.assert_called()
        calls = self.mock_collection.add.call_args_list
        added_metadatas = []
        for _, call_kwargs in calls:
            added_metadatas.extend(call_kwargs.get('metadatas', []))

        indexed_file_paths = {meta['file_path'] for meta in added_metadatas}

        assert "app.py" in indexed_file_paths
        assert "requirements.txt" in indexed_file_paths
        assert os.path.join("templates", "index.html") in indexed_file_paths
        assert os.path.join("static", "style.css") in indexed_file_paths
        assert os.path.join(".venv", "dummy_file") not in indexed_file_paths
        assert not any(".git" in p for p in indexed_file_paths)

    def test_index_repository_skip_if_exists(self):
        """Test that indexing is skipped if collection is not empty and force_reindex=False."""
        self.mock_collection.count.return_value = 10 # Simulate existing documents
        self.indexer.index_repository(force_reindex=False)
        self.mock_collection.add.assert_not_called()

    def test_index_repository_force_reindex(self):
        """Test force_reindex deletes and recreates the collection."""
        self.mock_collection.count.return_value = 10 # Simulate existing documents

        self.mock_chroma_client.reset_mock()
        self.mock_collection.reset_mock()
        # Re-establish return values for mocks after reset
        self.mock_chroma_client.get_or_create_collection.return_value = self.mock_collection
        self.mock_chroma_client.create_collection.return_value = self.mock_collection


        # Use the indexer from setup
        self.indexer.index_repository(force_reindex=True)

        # Check delete_collection was called
        self.mock_chroma_client.delete_collection.assert_called_once_with(self.indexer.collection_name)

        # Check create_collection was called *after* delete
        self.mock_chroma_client.create_collection.assert_called_once_with(
            name=self.indexer.collection_name,
            embedding_function=self.indexer.embedding_func
        )
        # Ensure add was called, signifying indexing happened after recreation
        self.mock_collection.add.assert_called()


    def test_retrieve_context(self):
        """Test retrieving context based on a query."""
        self.mock_collection.count.return_value = 1 # Simulate non-empty collection
        mock_query_results = {
            'ids': [['id1', 'id2']],
            'documents': [['doc content 1', 'doc content 2']],
            'metadatas': [[
                {'file_path': 'src/main.py', 'start_line': 1, 'end_line': 10, 'symbols': '["func1"]'},
                {'file_path': 'js/app.js', 'start_line': 5, 'end_line': 15, 'symbols': '["func2", "varA"]'}
            ]],
            'distances': [[0.1, 0.2]]
        }
        self.mock_collection.query.return_value = mock_query_results

        results = self.indexer.retrieve_context("find func1", n_results=2)

        self.mock_collection.query.assert_called_once_with(query_texts=["find func1"], n_results=2)
        assert len(results) == 2
        assert results[0]['content'] == 'doc content 1'
        assert results[0]['file_path'] == 'src/main.py' # Check relative path
        assert results[0]['start_line'] == 1
        assert results[0]['end_line'] == 10
        assert results[0]['symbols'] == ['func1']
        assert results[1]['file_path'] == 'js/app.js' # Check relative path
        assert results[1]['symbols'] == ['func2', 'varA']


    def test_retrieve_file_context(self):
        """Test retrieving context specific to a file."""
        self.mock_collection.count.return_value = 1
        mock_query_results = {
            'ids': [['id1']],
            'documents': [['doc content from main.py']],
            'metadatas': [[{'file_path': 'src/main.py', 'start_line': 1, 'end_line': 5, 'symbols': '[]'}]],
            'distances': [[0.1]]
        }
        self.mock_collection.query.return_value = mock_query_results

        # Query using a path relative to the repo root, or just the filename
        results = self.indexer.retrieve_file_context("src/main.py", n_results=1)

        # Check query uses the base filename extracted from the input path
        self.mock_collection.query.assert_called_once_with(
            query_texts=["Code from main.py"], # Uses os.path.basename()
            n_results=1
        )
        assert len(results) == 1
        assert results[0]['file_path'] == 'src/main.py'
        assert results[0]['content'] == 'doc content from main.py'


    def test_retrieve_symbol_context(self):
        """Test retrieving context specific to a symbol."""
        self.mock_collection.count.return_value = 1
        mock_query_results = {
            'ids': [['id1', 'id2', 'id3']],
            'documents': [['def process_data(): pass', 'class AnotherClass: pass', 'unrelated content']],
            'metadatas': [[
                {'file_path': 'src/main.py', 'start_line': 1, 'end_line': 2, 'symbols': '["process_data"]'},
                {'file_path': 'src/main.py', 'start_line': 3, 'end_line': 4, 'symbols': '["AnotherClass"]'},
                {'file_path': 'js/app.js', 'start_line': 1, 'end_line': 1, 'symbols': '[]'}
            ]],
            'distances': [[0.1, 0.2, 0.3]]
        }
        self.mock_collection.query.return_value = mock_query_results

        results = self.indexer.retrieve_symbol_context("process_data", n_results=1)

        self.mock_collection.query.assert_called_once_with(
            query_texts=["Code containing symbol process_data"],
            n_results=2 # n_results * 2
        )
        assert len(results) == 1
        assert results[0]['content'] == 'def process_data(): pass'
        assert results[0]['file_path'] == 'src/main.py'
        assert results[0]['symbols'] == ['process_data']
