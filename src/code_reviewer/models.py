from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CodeReviewCategory(str, Enum):
    DESIGN = "Design"
    FUNCTIONALITY = "Functionality" 
    NAMING = "Naming"
    CONSISTENCY = "Consistency"
    CODING_STYLE = "Coding Style"
    TESTS = "Tests"
    ROBUSTNESS = "Robustness"
    READABILITY = "Readability"
    ABSTRACTIONS = "Abstractions"


class CodeReviewComment(BaseModel):
    category: CodeReviewCategory
    file_name: Optional[str] = None
    line_number: Optional[int] = None
    comment: str


class CodeReviewRequest(BaseModel):
    system_prompt: str
    git_diff: str
    repo_path: str = Field(
        default="./",
        description="Path to the repository root for RAG indexing"
    )
    models: Dict[str, str] = Field(
        default_factory=lambda: {
            "planner": "llama3",
            "worker": "codellama",
            "embedding": "nomic-embed-text"
        },
        description="Models to use for the planner, worker agents, and embeddings"
    )
    max_retries: int = Field(
        default=2,
        description="Maximum number of retries if a worker fails"
    )
    sampling_rate: float = Field(
        default=0.8,
        description="Rate at which to sample code lines if there are too many (0.0-1.0)"
    )
    reindex: bool = Field(
        default=False,
        description="Force reindexing of the codebase even if an index already exists"
    )


class WorkerResponse(BaseModel):
    category: CodeReviewCategory
    comments: List[CodeReviewComment]


class CodeReviewResponse(BaseModel):
    categories: Dict[CodeReviewCategory, List[CodeReviewComment]]
    summary: str

# Code indexing and retrieval for RAG
class CodeChunk(BaseModel):
    """Model representing a chunk of code to be indexed"""
    file_path: str
    content: str
    start_line: int
    end_line: int
    symbols: List[str] = Field(default_factory=list)
