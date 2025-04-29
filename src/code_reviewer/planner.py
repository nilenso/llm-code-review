import time

from .indexer import CodeIndexer
from .models import (
    CodeReviewCategory,
    CodeReviewComment,
    CodeReviewRequest,
    CodeReviewResponse,
    WorkerResponse,
)
from .ollama_client import OllamaClient
from .prompts import changelog_system_prompt, changelog_user_prompt
from .worker import CodeReviewWorker


class CodeReviewer:
    def __init__(
        self, 
        ollama_client: OllamaClient,
        code_indexer: CodeIndexer,
        planner_model: str, 
        worker_model: str
    ):
        self.ollama_client = ollama_client
        self.code_indexer = code_indexer
        self.planner_model = planner_model
        self.worker_model = worker_model
        self.categories = list(CodeReviewCategory)
    
    def plan_and_execute(self, request: CodeReviewRequest) -> CodeReviewResponse:
        """
        Plan and execute the code review process using Ollama LLMs and RAG.
        Simplified synchronous version.
        """
        workers = [
            CodeReviewWorker(
                category=category, 
                ollama_client=self.ollama_client,
                code_indexer=self.code_indexer,
                model=request.models.get("worker", self.worker_model)
            )
            for category in self.categories
        ]
        
        # Execute workers sequentially to avoid overwhelming Ollama
        worker_responses = []
        for worker in workers:
            try:
                # Add delay between requests to give Ollama time to recover
                time.sleep(1)
                print(f"Running {worker.category} review...")
                response = worker.review(request.git_diff, request.system_prompt)
                worker_responses.append(response)
            except Exception as e:
                print(f"Error in worker {worker.category}: {str(e)}")
                import traceback
                print(traceback.format_exc()) 
                worker_responses.append(WorkerResponse(
                    category=worker.category,
                    comments=[CodeReviewComment(
                        category=worker.category,
                        comment=f"Worker encountered an error: {str(e)}"
                    )]
                ))
        
        # Collate responses by category
        categories_dict = {}
        for response in worker_responses:
            categories_dict[response.category] = response.comments
        
        # Generate changelog using Ollama based on the git diff
        try:
            changelog = self._generate_changelog(request.git_diff, request)
        except Exception as e:
            print(f"Error generating changelog: {str(e)}")
            changelog = "Could not generate changelog due to an error."
        
        return CodeReviewResponse(categories=categories_dict, summary=changelog)
    
    def _generate_changelog(self, git_diff: str, request: CodeReviewRequest) -> str:
        """
        Generate a CHANGELOG based on the git diff using Ollama.
        Simplified synchronous version.
        """
        if not git_diff or not git_diff.strip():
            return "No changes detected in the diff."

        messages = [
            {
                "role": "system", 
                "content": changelog_system_prompt()
            },
            {
                "role": "user",
                "content": changelog_user_prompt(git_diff=git_diff)
            }
        ]

        changelog = self.ollama_client.chat(
            model=request.models.get("planner", self.planner_model),
            messages=messages
        )
        
        return changelog.strip()
