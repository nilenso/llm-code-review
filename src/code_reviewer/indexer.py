import os
import re
import json
import hashlib
from typing import List, Dict, Any

import ollama
import chromadb
from chromadb.utils import embedding_functions
import gitignore_parser
import pygments
from pygments.lexers import get_lexer_for_filename
from pygments.token import Token
from rich.console import Console

from .models import CodeChunk

MAX_FILES_INDEXED = 100
IGNORE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.pyc', '.class', '.o', '.so', '.dll', '.exe',
    '.lock', '.log',
    '.chroma', '.DS_Store'
}
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_COLLECTION_NAME = "code_repository"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
CHROMA_DIR_NAME = ".chroma"
GITIGNORE_FILE_NAME = ".gitignore"
GIT_DIR_NAME = ".git"
VENV_DIR_NAME = ".venv"
DEFAULT_ENCODING = "utf-8"
CHUNK_ID_FORMAT = "{file_path}:{start_line}-{end_line}"
DEFAULT_SYMBOLS_JSON = "[]"
FILE_CONTEXT_QUERY_TEMPLATE = "Code from {file_name}"
SYMBOL_CONTEXT_QUERY_TEMPLATE = "Code containing symbol {symbol}"
# logical code boundaries (functions, classes, etc.)
BOUNDARY_PATTERNS = [
    r'^(?:public|private|protected|internal|fun|func|function|def|interface|class)\s+\w+',
    r'^import\s+',
    r'^from\s+\w+\s+import',
    r'^#\s*\w+',
    r'^//\s*\w+',
    r'^/\*',
    r'^\s*\*/'
]
IGNORED_DIRS = {GIT_DIR_NAME, CHROMA_DIR_NAME, VENV_DIR_NAME, '.env'}
FUNCTION_KEYWORDS = ["fun", "func", "def", "class"]


class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, client, model):
        self.client = client
        self.model = model
        
    def __call__(self, texts):
        """Generate embeddings for the given texts"""
        if not texts:
            return []
                    
        # Process in batches to avoid overwhelming Ollama
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = [self.generate_embedding(text)
                          for text in batch]            
            all_embeddings.extend(embeddings)
                    
        return all_embeddings

    def generate_embedding(self, text):
        try:
            response = self.client.embeddings(model=self.model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

        
class CodeIndexer:
    """
    Class for indexing code files in a repository into ChromaDB
    """
    def __init__(self, 
                repo_path: str,
                embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                collection_name: str = DEFAULT_COLLECTION_NAME,
                ollama_host: str = DEFAULT_OLLAMA_HOST):
        self.repo_path = os.path.abspath(repo_path)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.ollama_host = ollama_host        
        self.db_path = os.path.join(self.repo_path, CHROMA_DIR_NAME)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.ollama_client = ollama.Client(host=ollama_host)        
        self.gitignore_matcher = self._gitignore_matcher()
        self.console = Console()        
        self.embedding_func = OllamaEmbeddingFunction(
            client=self.ollama_client,
            model=self.embedding_model
        )        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_func
        )
    
    def _gitignore_matcher(self) -> callable:
        """
        Initialize the gitignore matcher by parsing .gitignore file in the repository.
        Returns a function that takes a file path and returns True if the file should be ignored.
        Raises an exception if parsing fails.
        """
        gitignore_path = os.path.join(self.repo_path, GITIGNORE_FILE_NAME)
        if os.path.exists(gitignore_path):
            try:
                matcher = gitignore_parser.parse_gitignore(gitignore_path)
                return matcher
            except Exception as e:
                print(f"Error: Failed to parse .gitignore file at {gitignore_path}: {e}")
                raise e
        
        return lambda _: False
    
    def extract_symbols(self, file_path: str, content: str) -> List[str]:
        """
        Extract function names, class names, and other symbols from code
        """
        symbols = []
        
        try:
            # Get the appropriate lexer for the file type and tokenize the content
            lexer = get_lexer_for_filename(file_path)        
            tokens = list(pygments.lex(content, lexer))
            
            # Extract symbols based on token types
            for i, (token_type, value) in enumerate(tokens):
                if token_type in Token.Name.Function or token_type in Token.Name.Class:
                    symbols.append(value)
                
                # Look for function/method declarations in various languages and extract the name
                if token_type in Token.Keyword and value in FUNCTION_KEYWORDS:
                    if i+2 < len(tokens) and tokens[i+1][0] in Token.Text:
                        if tokens[i+2][0] in Token.Name:
                            symbols.append(tokens[i+2][1])
        except Exception as e:
            # If lexing fails, fall back to regex-based extraction
            for pattern in [
                r'(?:function|def|fun)\s+(\w+)',
                r'class\s+(\w+)',
                r'(?:var|let|const)\s+(\w+)'
            ]:
                for match in re.finditer(pattern, content):
                    if match.group(1):
                        symbols.append(match.group(1))
        
        return list(set(symbols))  # Remove duplicates
    
    def should_index_file(self, file_path: str) -> bool:
        """
        Determine if a file should be indexed based on extension, gitignore rules, and other criteria
        """
        if CHROMA_DIR_NAME in file_path:
            return False
            
        if GIT_DIR_NAME in file_path:
            return False

        _, ext = os.path.splitext(file_path)
        if ext.lower() in IGNORE_EXTENSIONS:
            return False
        
        if self.gitignore_matcher(file_path):
            return False
        
        # Check for standard directory patterns using the relative path
        relative_path = os.path.relpath(file_path, self.repo_path)
        normalized_relative_path = os.path.normpath(relative_path)
        if normalized_relative_path.startswith(os.path.normpath(f"{GIT_DIR_NAME}/")) or \
           normalized_relative_path == GIT_DIR_NAME:
            return False
        if normalized_relative_path.startswith(os.path.normpath(f"{CHROMA_DIR_NAME}/")) or \
            normalized_relative_path == CHROMA_DIR_NAME:
             return False
        # Explicitly check for .venv as the parser seems inconsistent here
        if normalized_relative_path.startswith(os.path.normpath(f"{VENV_DIR_NAME}/")) or \
            normalized_relative_path == VENV_DIR_NAME:
             return False

        return True
    
    def chunk_file(self, file_path: str, content: str, chunk_size: int = 20) -> List[CodeChunk]:
        """
        Split file content into overlapping chunks based on lines and logical boundaries.
        Uses a smaller default chunk size suitable for semantic retrieval.
        """
        lines = content.split('\n')
        chunks = []
        current_chunk_lines = []
        current_chunk_start = 0

        for i, line in enumerate(lines):
            current_chunk_lines.append(line)
            
            # Check if we've reached a good chunk size and are at a logical boundary
            is_boundary = any(re.match(pattern, line) for pattern in BOUNDARY_PATTERNS)
            if (len(current_chunk_lines) >= chunk_size and is_boundary) or i == len(lines) - 1:
                chunk_content = '\n'.join(current_chunk_lines)
                symbols = self.extract_symbols(file_path, chunk_content)
                
                chunks.append(CodeChunk(
                    file_path=file_path,
                    content=chunk_content,
                    start_line=current_chunk_start,
                    end_line=current_chunk_start + len(current_chunk_lines),
                    symbols=symbols
                ))
                
                current_chunk_lines = []
                current_chunk_start = i + 1
        
        # Add any remaining lines as a chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            symbols = self.extract_symbols(file_path, chunk_content)
            
            chunks.append(CodeChunk(
                file_path=file_path,
                content=chunk_content,
                start_line=current_chunk_start,
                end_line=current_chunk_start + len(current_chunk_lines),
                symbols=symbols
            ))
        
        return chunks
    
    def _should_skip_indexing(self, force_reindex: bool) -> bool:
        """Check if indexing should be skipped based on existing collection and force_reindex flag."""
        if self.collection.count() > 0 and not force_reindex:
            self.console.print(f"[yellow]:information_source: Collection '{self.collection_name}' already contains {self.collection.count()} documents. Skipping indexing.")
            return True
        return False

    def _handle_reindex(self):
        """Delete and recreate the collection if it exists."""
        if self.collection.count() > 0:
            self.console.print(f"[yellow]:information_source: Re-indexing requested. Deleting existing collection '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_func
            )
            self.console.print(f"[green]:white_check_mark: Collection '{self.collection_name}' recreated.")

    def _process_file(self, file_path: str, relative_path: str) -> tuple[list[str], list[str], list[dict[str, any]]]:
        """Process a single file: read, chunk, generate data. Returns (ids, docs, metadatas)."""
        ids, documents, metadatas = [], [], []
        try:
            with open(file_path, 'r', encoding=DEFAULT_ENCODING, errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return ids, documents, metadatas
            
            chunks = self.chunk_file(relative_path, content)
            
            if not chunks:
                return ids, documents, metadatas

            for chunk in chunks:
                (chunk_id, chunk_content, metadata) = self.generate_chunk_data(chunk)
                ids.append(chunk_id)
                documents.append(chunk_content) 
                metadatas.append(metadata)
                            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return [], [], [] 
        
        return ids, documents, metadatas

    def _get_indexable_files(self) -> list[tuple[str, str]]:
        """Walks the repository and returns a list of (absolute_path, relative_path) for indexable files."""
        indexable_files = []
        for root, dirs, files in os.walk(self.repo_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if self.should_index_file(file_path):
                        relative_path = os.path.relpath(file_path, self.repo_path)
                        indexable_files.append((file_path, relative_path))
                except ValueError:
                    # Handles cases where file_path might not be under self.repo_path
                    # Or other potential issues with relpath
                    print(f"Warning: Could not determine relative path for {file_path}, skipping.")
                    continue
                except Exception as e:
                    print(f"Warning: Error checking if file should be indexed {file_path}: {e}, skipping.")
                    continue
                    
        return indexable_files

    def index_repository(self, force_reindex: bool = False) -> None:
        """
        Index all code files in the repository using a functional approach.
        Orchestrates file discovery, processing, and adding to the collection.
        """
        if self._should_skip_indexing(force_reindex):
            return
        
        if force_reindex:
            self._handle_reindex()
        
        self.console.print(f"Discovering files to index in '{self.repo_path}'...")
        files_to_process = self._get_indexable_files()
        
        if not files_to_process:
            self.console.print("[yellow]No indexable files found.")
            return
            
        self.console.print(f"Found {len(files_to_process)} indexable files. Processing and adding to collection...")
        
        files_indexed = 0
        chunks_indexed = 0
        
        for abs_path, rel_path in files_to_process:
            try:
                ids, documents, metadatas = self._process_file(abs_path, rel_path)
                if ids:
                    self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
                    files_indexed += 1
                    chunks_indexed += len(ids)
            except Exception as e:
                print(f"Error processing or adding file {rel_path}: {e}")
                             
        if files_indexed > 0:
            self.console.print(f"[green]:white_check_mark: Indexing complete. Indexed {files_indexed} files, {chunks_indexed} chunks.")
        else:
            self.console.print("[yellow]Indexing complete. No files were added to the collection (files might have been empty or errored).")

    def generate_chunk_data(self, chunk):
        # Create a deterministic ID for the chunk
        chunk_id_str = CHUNK_ID_FORMAT.format(
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line
        )
        chunk_id = hashlib.md5(chunk_id_str.encode()).hexdigest()        
        chunk_metadata = {
            "file_path": chunk.file_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbols": json.dumps(chunk.symbols)  # ChromaDB requires string for metadata
        }
        return (chunk_id, chunk.content, chunk_metadata)
    
    def retrieve_context(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant code chunks for a given query
        """
        if self.collection.count() == 0:
            return []
        
        documents = []
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                documents.append({
                    "content": doc,
                    "file_path": metadata.get("file_path", ""),
                    "start_line": metadata.get("start_line", 0),
                    "end_line": metadata.get("end_line", 0),
                    "symbols": json.loads(metadata.get("symbols", DEFAULT_SYMBOLS_JSON))
                })
            
            return documents
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def retrieve_file_context(self, file_path: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve chunks related to a specific file
        """
        if self.collection.count() == 0:
            return []
        
        file_name = os.path.basename(file_path)
        
        try:
            # Use where clause to filter by file_path
            results = self.collection.query(
                query_texts=[FILE_CONTEXT_QUERY_TEMPLATE.format(file_name=file_name)],
                n_results=n_results
            )
            
            documents = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                documents.append({
                    "content": doc,
                    "file_path": metadata.get("file_path", ""),
                    "start_line": metadata.get("start_line", 0),
                    "end_line": metadata.get("end_line", 0),
                    "symbols": json.loads(metadata.get("symbols", DEFAULT_SYMBOLS_JSON))
                })
            
            return documents
        except Exception as e:
            print(f"Error retrieving file context: {e}")
            return []
    
    def retrieve_symbol_context(self, symbol: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve chunks related to a specific symbol (function, class, etc.)
        """
        if self.collection.count() == 0:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[SYMBOL_CONTEXT_QUERY_TEMPLATE.format(symbol=symbol)],
                n_results=n_results*2  # Get extra results because filtering might reduce the count
            )
            
            documents = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                symbols = json.loads(metadata.get("symbols", DEFAULT_SYMBOLS_JSON))
                
                # Only include if the symbol is actually in this chunk's symbols list or in the content
                if symbol in symbols or symbol in doc:
                    documents.append({
                        "content": doc,
                        "file_path": metadata.get("file_path", ""),
                        "start_line": metadata.get("start_line", 0),
                        "end_line": metadata.get("end_line", 0),
                        "symbols": symbols
                    })
                
                if len(documents) >= n_results:
                    break
            
            return documents
        except Exception as e:
            print(f"Error retrieving symbol context: {e}")
            return []
