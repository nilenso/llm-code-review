from .diff_parser import extract_modified_symbols, extract_modified_files
from .indexer import CodeIndexer

MAX_CONTEXT_DOCS = 3
MAX_CONTEXT_LINES = 20

class ContextRetriever:
    def __init__(self, code_indexer: CodeIndexer, max_context_docs: int = MAX_CONTEXT_DOCS):
        self.code_indexer = code_indexer
        self.max_context_docs = max_context_docs

    def get_symbol_context(self, git_diff):
        modified_symbols = extract_modified_symbols(git_diff)
        symbols = set()
        for _, file_symbols in modified_symbols.items():
            symbols.update(file_symbols)
        context = [self.code_indexer.retrieve_symbol_context(symbol)
                   for symbol in symbols]
        return context

    def get_modified_files_context(self, git_diff, file_count):
        modified_files = extract_modified_files(git_diff)
        context = []
        for file_path in modified_files:
            file_context = self.code_indexer.retrieve_file_context(file_path)
            context.extend(file_context)
            if len(context) >= file_count:
                break
        return context
    
    def score_context_relevance(self, context_doc, git_diff):
        """
        Score the relevance of a context document to the current diff.
        
        Args:
            context_doc: Document containing code context
            git_diff: Git diff string
            
        Returns:
            float: Relevance score (higher is more relevant)
        """
        # Extract symbols from diff
        modified_symbols = extract_modified_symbols(git_diff)
        symbols = set()
        for _, s in modified_symbols.items():
            symbols.update(s)
        
        # Check how many symbols from diff appear in this context
        doc_content = context_doc['content'].lower()
        symbol_matches = sum(1 for symbol in symbols if symbol.lower() in doc_content)
        
        # Base score on symbol matches
        score = symbol_matches * 2
        
        # Extract symbols from context doc
        doc_symbols = context_doc.get('symbols', [])
        # Add score for direct symbol matches in the document's symbols list
        symbol_overlap = sum(1 for symbol in symbols if symbol in doc_symbols)
        score += symbol_overlap * 3  # Higher weight for explicit symbol matches
        
        # Add score for file name match
        modified_file_paths = extract_modified_files(git_diff)
        for file_path in modified_file_paths:
            if file_path.lower() in context_doc['file_path'].lower():
                score += 5
                break
        
        # Add score for recency (earlier chunks in a file might define classes/interfaces)
        if context_doc.get('start_line', 0) < 50:
            score += 2
        
        # Penalize extremely long chunks (they're less focused)
        content_length = len(doc_content.split('\n'))
        if content_length > 30:
            score -= 1
        
        return score

    def get_diff_context(self, git_diff):
        context_docs = []
        # First, get symbol context
        symbol_context = self.get_symbol_context(git_diff)
        # Next, get file context
        file_context = self.get_modified_files_context(git_diff, self.max_context_docs * 2)        
        # Combine all context
        context_docs.extend(symbol_context)
        context_docs.extend(file_context)
        
        deduped = self.dedup_context(context_docs)
        
        # Score and sort context by relevance
        scored_context = [(doc, self.score_context_relevance(doc, git_diff)) for doc in deduped]
        scored_context.sort(key=lambda x: x[1], reverse=True)  # Sort by score, descending
        
        # Take top N most relevant docs
        top_context = [doc for doc, _ in scored_context[:self.max_context_docs]]
        
        # Truncate content to avoid overwhelming the LLM
        return [self._truncate_doc_content(doc) for doc in top_context]
    
    def _truncate_doc_content(self, doc, max_lines=MAX_CONTEXT_LINES):
        lines = doc['content'].splitlines()
        if len(lines) > max_lines:
            doc['content'] = '\n'.join(lines[:max_lines]) + '\n... (truncated) ...'
        return doc
    
    def dedup_context(self, context_docs):
        unique_docs = {}
        for doc in context_docs:
            key = f"{doc['file_path']}:{doc['start_line']}-{doc['end_line']}"
            if key not in unique_docs:
                unique_docs[key] = doc
                
        return list(unique_docs.values())