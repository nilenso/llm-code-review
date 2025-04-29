from unittest import result
from rerankers import Reranker
import hashlib

DEFAULT_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

class ContextReranker():
    def __init__(self, model=DEFAULT_RERANK_MODEL):
        # TODO: Add model_type='cross-encoder'
        self.reranker = Reranker(model, verbose=0)
    
    def rank(self, query, docs):
        docs_map = self.generate_document_map(docs)
        docs = [d['content'] for d in docs]
        results = self.reranker.rank(query, docs)

        return [docs_map[self.generate_hash(r.document.text)] 
                for r in results.results]

    def generate_hash(self, data: str) -> str:
        return hashlib.md5(data.encode('utf-8')).hexdigest()

    def generate_document_map(self, docs): 
        return {self.generate_hash(d['content']): d for d in docs}
