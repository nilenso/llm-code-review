from unittest.mock import MagicMock, call, patch

from code_reviewer.models import (
    CodeReviewCategory,
    CodeReviewComment,
    CodeReviewRequest,
    CodeReviewResponse,
    WorkerResponse,
)
from code_reviewer.planner import CodeReviewer
from code_reviewer.prompts import changelog_system_prompt, changelog_user_prompt
from code_reviewer.worker import CodeReviewWorker


def mock_worker_side_effect(category, ollama_client, code_indexer, model):
    mock_worker = MagicMock()
    mock_worker.category = category  # Set to a valid enum string
    mock_worker.review.return_value = WorkerResponse(
        category=category,
        comments=[CodeReviewComment(category=category, comment="Mocked comment")]
    )
    return mock_worker

def test_generate_changelog_success(
    planner_agent: CodeReviewer,
    mock_ollama_client: MagicMock,
    sample_git_diff_simple: str
):
    """Tests _generate_changelog successfully calls Ollama and returns the response."""
    expected_changelog = "Generated changelog content."
    mock_ollama_client.chat.return_value = expected_changelog
    request = CodeReviewRequest(
        git_diff=sample_git_diff_simple,
        system_prompt="System Prompt",
        models={"planner": "test-planner-model"}
    )

    result = planner_agent._generate_changelog(sample_git_diff_simple, request)

    assert result == expected_changelog.strip()
    expected_messages = [
        {"role": "system", "content": changelog_system_prompt()},
        {"role": "user", "content": changelog_user_prompt(git_diff=sample_git_diff_simple)}
    ]
    mock_ollama_client.chat.assert_called_once_with(
        model="test-planner-model",
        messages=expected_messages
    )

@patch('time.sleep', return_value=None)
@patch('code_reviewer.planner.CodeReviewWorker', side_effect=mock_worker_side_effect)
def test_generate_changelog_ollama_error(mock_worker, mock_sleep, planner_agent: CodeReviewer, mock_ollama_client: MagicMock, sample_git_diff_simple: str, capsys):
    """Tests _generate_changelog handles Ollama client errors gracefully."""
    error_message = "Ollama connection failed"
    mock_ollama_client.chat.side_effect = Exception(error_message)
    request = CodeReviewRequest(
        git_diff=sample_git_diff_simple, 
        system_prompt="System Prompt", 
        models={"planner": "test-planner-model"}
    )
    expected_fallback_msg = "Could not generate changelog due to an error." 

    response = planner_agent.plan_and_execute(request)

    assert response.summary == expected_fallback_msg
    mock_ollama_client.chat.assert_called_once()
    captured = capsys.readouterr()
    assert f"Error generating changelog: {error_message}" in captured.out

@patch('time.sleep', return_value=None)
@patch('code_reviewer.planner.CodeReviewWorker', side_effect=mock_worker_side_effect)
def test_generate_changelog_empty_diff(mock_worker, mock_sleep, planner_agent: CodeReviewer, mock_ollama_client: MagicMock):
    """Tests _generate_changelog returns specific message for empty diff without calling LLM."""
    request = CodeReviewRequest(
        git_diff="", 
        system_prompt="System Prompt", 
        models={"planner": "test-planner-model"}
    )

    response = planner_agent.plan_and_execute(request)

    assert response.summary == "No changes detected in the diff."
    mock_ollama_client.chat.assert_not_called()

@patch('time.sleep', return_value=None)
@patch('code_reviewer.planner.CodeReviewer._generate_changelog')
@patch('code_reviewer.planner.CodeReviewWorker', side_effect=mock_worker_side_effect)
def test_plan_success(
    mock_worker: MagicMock,
    mock_generate_changelog: MagicMock,
    mock_sleep: MagicMock,
    planner_agent: CodeReviewer,
    sample_git_diff_simple: str
):
    """Tests plan_and_execute successfully instantiates and calls all workers and changelog."""
    request = CodeReviewRequest(
        git_diff=sample_git_diff_simple,
        system_prompt="System Prompt",
        models={"worker": "test-worker-model", "planner": "test-planner-model"}
    )
    expected_changelog = "Test Changelog"
    mock_generate_changelog.return_value = expected_changelog

    mock_worker_instances = {}
    worker_comments = {}
    for category in CodeReviewCategory:
        instance = MagicMock(spec=CodeReviewWorker)
        instance.category = category
        comment = CodeReviewComment(
            category=category,
            comment=f"{category.value} comment",
            file_name="f.py",
            line_number=1
        )
        instance.review.return_value = WorkerResponse(
            category=category, comments=[comment]
        )
        mock_worker_instances[category] = instance
        worker_comments[category] = [comment]

    def worker_side_effect(*args, **kwargs):
        category = kwargs.get('category')
        return mock_worker_instances[category]

    mock_worker.side_effect = worker_side_effect

    result = planner_agent.plan_and_execute(request)

    expected_worker_calls = []
    for category in CodeReviewCategory:
         expected_worker_calls.append(
             call(
                category=category,
                ollama_client=planner_agent.ollama_client,
                code_indexer=planner_agent.code_indexer,
                model="test-worker-model" # From request
             )
         )
    mock_worker.assert_has_calls(expected_worker_calls, any_order=False)
    assert mock_worker.call_count == len(CodeReviewCategory)

    for category, instance in mock_worker_instances.items():
        instance.review.assert_called_once_with(request.git_diff, request.system_prompt)

    mock_generate_changelog.assert_called_once_with(request.git_diff, request)

    assert mock_sleep.call_count == len(CodeReviewCategory)

    assert isinstance(result, CodeReviewResponse)
    assert result.summary == expected_changelog
    assert len(result.categories) == len(CodeReviewCategory)
    for category in CodeReviewCategory:
        assert result.categories[category] == worker_comments[category]


@patch('time.sleep', return_value=None)
@patch('code_reviewer.planner.CodeReviewer._generate_changelog')
@patch('code_reviewer.planner.CodeReviewWorker')
def test_worker_error_handling(
    mock_worker: MagicMock,
    mock_generate_changelog: MagicMock, # Still need to mock it
    mock_sleep: MagicMock,
    planner_agent: CodeReviewer,
    mock_ollama_client: MagicMock,
    mock_code_indexer: MagicMock,
    sample_git_diff_simple: str,
    capsys
):
    """Tests plan_and_execute handles a worker error gracefully and logs correctly."""
    request = CodeReviewRequest(git_diff=sample_git_diff_simple, system_prompt="System Prompt")
    mock_generate_changelog.return_value = "Mock Changelog"
    error_category = CodeReviewCategory.DESIGN
    error_message = "Worker review failed!"

    mock_worker_instances = {}
    worker_comments = {}
    for category in CodeReviewCategory:
        instance = MagicMock(spec=CodeReviewWorker)
        instance.category = category
        if category == error_category:
            instance.review.side_effect = Exception(error_message)
        else:
            comment = CodeReviewComment(category=category, comment=f"{category.value} comment")
            instance.review.return_value = WorkerResponse(category=category, comments=[comment])
            worker_comments[category] = [comment]
        mock_worker_instances[category] = instance

    mock_worker.side_effect = lambda *args, **kwargs: mock_worker_instances[kwargs.get('category')]

    result = planner_agent.plan_and_execute(request)

    # Verify all workers were instantiated and review attempted
    assert mock_worker.call_count == len(CodeReviewCategory)
    for instance in mock_worker_instances.values():
        instance.review.assert_called_once()

    # Verify Final Response structure and content (excluding summary)
    assert isinstance(result, CodeReviewResponse)
    assert len(result.categories) == len(CodeReviewCategory)

    # Check successful categories are present
    for category, comments in worker_comments.items():
         assert result.categories[category] == comments

    # Check error category fallback comment
    fallback_comment = result.categories[error_category][0]
    assert fallback_comment.category == error_category
    assert f"Worker encountered an error: {error_message}" in fallback_comment.comment
    assert fallback_comment.file_name is None
    assert fallback_comment.line_number is None

    # Verify error logging
    captured = capsys.readouterr()
    assert f"Error in worker {error_category}: {error_message}" in captured.out

@patch('time.sleep', return_value=None)
@patch('code_reviewer.planner.CodeReviewer._generate_changelog')
@patch('code_reviewer.planner.CodeReviewWorker', side_effect=mock_worker_side_effect)
def test_changelog_generated_despite_worker_error(
    mock_worker: MagicMock,
    mock_generate_changelog: MagicMock,
    mock_sleep: MagicMock,
    planner_agent: CodeReviewer,
    mock_ollama_client: MagicMock,
    mock_code_indexer: MagicMock,
    sample_git_diff_simple: str
):
    """Tests that changelog generation still occurs even if a worker fails."""
    request = CodeReviewRequest(git_diff=sample_git_diff_simple, system_prompt="System Prompt")
    expected_changelog = "Changelog despite error"
    mock_generate_changelog.return_value = expected_changelog
    error_category = CodeReviewCategory.ROBUSTNESS
    error_message = "Security worker failed!"

    def worker_side_effect(*args, **kwargs):
        instance = MagicMock(spec=CodeReviewWorker)
        instance.category = kwargs.get('category')
        if instance.category == error_category:
            instance.review.side_effect = Exception(error_message)
        else:
            instance.review.return_value = WorkerResponse(category=instance.category, comments=[])
        return instance

    mock_worker.side_effect = worker_side_effect

    result = planner_agent.plan_and_execute(request)

    mock_generate_changelog.assert_called_once_with(request.git_diff, request)
    assert result.summary == expected_changelog
