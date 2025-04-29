from unittest.mock import patch

import pytest

from code_reviewer.models import CodeReviewCategory, CodeReviewComment, WorkerResponse
from code_reviewer.prompts import CATEGORY_TO_SUBCATEGORIES, TARGETED_REVIEW_PROMPTS
from code_reviewer.reranker import ContextReranker
from code_reviewer.worker import MAX_CODE_LINES, CodeReviewWorker

SAMPLE_CONTEXT_DOC_1 = {'file_path': 'path/to/file1.py', 'start_line': 10, 'end_line': 20, 'content': 'content of file 1'}
SAMPLE_CONTEXT_DOC_2 = {'file_path': 'path/to/file2.js', 'start_line': 5, 'end_line': 15, 'content': 'content of file 2'}
SAMPLE_CONTEXT_DOC_DUPLICATE = {'file_path': 'path/to/file1.py', 'start_line': 10, 'end_line': 20, 'content': 'duplicate content'}
SAMPLE_CONTEXT_DOC_3 = {'file_path': 'another/file.py', 'start_line': 1, 'end_line': 5, 'content': 'content of file 3'}

def test_init(worker, mock_ollama_client, mock_code_indexer):
    assert worker.category == CodeReviewCategory.DESIGN
    assert worker.ollama_client == mock_ollama_client
    assert worker.code_indexer == mock_code_indexer
    assert worker.model == "test-worker-model"
    assert worker.prompt_template is not None

def test_get_sampled_code_lines_less_than_max(worker, sample_git_diff_simple):
    expected_lines = [
        "file1.py:1: + def new_function(): # Line changed",
        "file1.py:2: +     print(\"New greeting v2\") # Line changed",
        "file1.py:3: +     # Added line",
        "file2.txt:1: + New content version 2.",
        "file2.txt:2: + Another added line."
    ]
    result = worker.get_sampled_code_lines(sample_git_diff_simple)
    assert result == expected_lines
    assert "sampled" not in "\n".join(result)

def test_get_sampled_code_lines_more_than_max(worker, sample_git_diff_long):
    result = worker.get_sampled_code_lines(sample_git_diff_long)
    assert len(result) == MAX_CODE_LINES + 1 
    assert "... (sampled" in result[MAX_CODE_LINES // 2]
    assert result[0] == 'long_file.py:2: +     added line 1'

def test_get_sampled_code_lines_empty(worker, sample_git_diff_simple):
    result = worker.get_sampled_code_lines(sample_git_diff_simple)
    assert len(result) > 0 

def test_format_context(worker):
    context_docs = [SAMPLE_CONTEXT_DOC_1, SAMPLE_CONTEXT_DOC_2]
    expected = "CONTEXT:\n\n"
    expected += "--- path/to/file1.py (lines 10-20) ---\ncontent of file 1\n\n"
    expected += "--- path/to/file2.js (lines 5-15) ---\ncontent of file 2\n\n"
    result = worker.format_context(context_docs)
    assert result == expected

def test_format_context_single_doc(worker):
    context_docs = [SAMPLE_CONTEXT_DOC_1]
    expected = "CONTEXT:\n\n"
    expected += "--- path/to/file1.py (lines 10-20) ---\ncontent of file 1\n\n"
    result = worker.format_context(context_docs)
    assert result == expected

def test_format_context_empty(worker):
    assert worker.format_context([]) == ""

def test_get_prompt_with_subcategories(worker):
    worker.category = CodeReviewCategory.DESIGN  
    subcategories = CATEGORY_TO_SUBCATEGORIES.get(CodeReviewCategory.DESIGN, [])
    if not subcategories: pytest.skip("Skipping test: No subcategories defined for DESIGN")

    targeted_prompts_exist = any(sub in TARGETED_REVIEW_PROMPTS for sub in subcategories)
    if not targeted_prompts_exist: pytest.skip("Skipping test: No targeted prompts for DESIGN subcategories")

    code_to_review = "sample code"
    context_text = "sample context"
    messages = worker.get_prompt(code_to_review, context_text)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Focus specifically on these issues" in messages[0]["content"]
    assert any(TARGETED_REVIEW_PROMPTS[sub].strip() in messages[0]["content"]
               for sub in subcategories if sub in TARGETED_REVIEW_PROMPTS)

    assert messages[1]["role"] == "user"
    assert code_to_review in messages[1]["content"]
    assert context_text in messages[1]["content"]

def test_get_prompt_without_subcategories(worker):
    worker.category = CodeReviewCategory.NAMING
    code_to_review = "<sample code>"
    context_text = "<sample context>"
    messages = worker.get_prompt(code_to_review, context_text)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert code_to_review in messages[1]["content"]
    assert context_text in messages[1]["content"]

@patch.object(CodeReviewWorker, 'parse_comment')
def test_parse_llm_response_valid_json(mock_parse_comment, worker):
    mock_parse_comment.side_effect = lambda x: CodeReviewComment(category=worker.category, comment=x['c'], file_name=x['f'], line_number=x.get('l'))
    response_json = '[{"c": "Comment 1", "f": "file1.py", "l": 10}, {"c": "Comment 2", "f": "file2.py"}]'
    result = worker.parse_llm_response(response_json)

    assert len(result) == 2
    assert result[0].comment == "Comment 1"
    assert result[0].file_name == "file1.py"
    assert result[0].line_number == 10
    assert result[1].comment == "Comment 2"
    assert result[1].file_name == "file2.py"
    assert result[1].line_number is None
    assert all(c.category == worker.category for c in result)

@patch.object(CodeReviewWorker, 'parse_comment')
def test_parse_llm_response_valid_json_with_extra_text(mock_parse_comment, worker):
    mock_parse_comment.side_effect = lambda x: CodeReviewComment(
        category=worker.category, 
        comment=x['c'], 
        file_name=x['f']
    )
    response_json_only = '[{"c": "Comment 1", "f": "file1.py"}]'
    result = worker.parse_llm_response(response_json_only) 

    assert len(result) == 1
    assert result[0].comment == "Comment 1"
    assert result[0].file_name == "file1.py"

def test_parse_llm_response_invalid_json(worker, capsys):
    response = "This is not valid JSON"
    result = worker.parse_llm_response(response)

    assert len(result) == 1
    assert result[0].category == worker.category
    assert "Unable to parse LLM response" in result[0].comment
    assert result[0].file_name is None
    assert result[0].line_number is None
    captured = capsys.readouterr()
    assert "JSON parsing error" in captured.out 

@patch.object(CodeReviewWorker, 'parse_comment')
def test_parse_llm_response_empty_list(mock_parse_comment, worker):
    mock_parse_comment.return_value = None 
    response_json = '[]'
    result = worker.parse_llm_response(response_json)

    assert len(result) == 1
    assert result[0].category == worker.category
    assert f"No {worker.category.value.lower()} issues detected" in result[0].comment
    assert result[0].file_name is None
    assert result[0].line_number is None

@patch.object(CodeReviewWorker, 'parse_comment')
def test_parse_llm_response_partial_invalid_comments(mock_parse_comment, worker):
    valid_comment = CodeReviewComment(category=worker.category, comment="Valid", file_name="a.py", line_number=1)
    mock_parse_comment.side_effect = [valid_comment, None]
    response_json = '[{"c": "Valid", "f": "a.py", "l": 1}, {"invalid": "data"}]' 
    result = worker.parse_llm_response(response_json)

    assert len(result) == 1
    assert result[0] == valid_comment

@patch.object(CodeReviewWorker, 'get_sampled_code_lines')
@patch.object(ContextReranker, 'rank') 
@patch.object(CodeReviewWorker, 'format_context')
@patch.object(CodeReviewWorker, 'get_prompt')
@patch.object(CodeReviewWorker, 'parse_llm_response')
def test_review_success(
    mock_parse,
    mock_get_prompt,
    mock_format,
    mock_rank,
    mock_get_lines,
    worker,
    mock_ollama_client,
    sample_git_diff_simple: str 
):
    git_diff = sample_git_diff_simple 
    dummy_system_prompt = "System prompt" 
    sampled_lines = ["file1.py:2: + print(\"New greeting v2\")" , "file1.py:3: + # Added line"] 
    code_to_review = "\n".join(sampled_lines)
    context_docs = [SAMPLE_CONTEXT_DOC_1]
    reranked_context = [SAMPLE_CONTEXT_DOC_1]
    formatted_context = "Formatted context"
    prompt_messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"}
    ]
    llm_response = '[{"comment": "Good job", "file_name": "file1.py", "line_number": 2}]'
    expected_comments = [CodeReviewComment(category=worker.category, comment="Good job", file_name="file1.py", line_number=2)]

    mock_get_lines.return_value = sampled_lines
    mock_rank.return_value = reranked_context
    mock_format.return_value = formatted_context
    mock_get_prompt.return_value = prompt_messages
    mock_ollama_client.chat.return_value = llm_response
    mock_parse.return_value = expected_comments

    with patch.object(worker.context_retriever, 'get_diff_context', return_value=context_docs) as mock_get_context:
        result = worker.review(git_diff, dummy_system_prompt)
        mock_get_lines.assert_called_once_with(git_diff)
        mock_get_context.assert_called_once_with(git_diff)
        mock_rank.assert_called_once_with(code_to_review, context_docs)
        mock_format.assert_called_once_with(reranked_context)
        mock_get_prompt.assert_called_once_with(code_to_review, formatted_context)
        mock_ollama_client.chat.assert_called_once_with(model=worker.model, messages=prompt_messages, temperature=0.3)
        mock_parse.assert_called_once_with(llm_response)

        assert isinstance(result, WorkerResponse)
        assert result.category == worker.category
        assert result.comments == expected_comments

@patch.object(CodeReviewWorker, 'get_sampled_code_lines')
@patch.object(ContextReranker, 'rank')
@patch.object(CodeReviewWorker, 'format_context')
@patch.object(CodeReviewWorker, 'get_prompt')
@patch.object(CodeReviewWorker, 'parse_llm_response')
def test_review_ollama_error(
    mock_parse,
    mock_get_prompt,
    mock_format,
    mock_rank,
    mock_get_lines,
    worker,
    mock_ollama_client,
    capsys,
    sample_git_diff_simple: str 
):
    git_diff = sample_git_diff_simple 
    dummy_system_prompt = "System prompt" 
    sampled_lines = ["file1.py:2: + print(\"New greeting v2\")" , "file1.py:3: + # Added line"] 
    context_docs = [SAMPLE_CONTEXT_DOC_1]
    reranked_context = [SAMPLE_CONTEXT_DOC_1]
    formatted_context = "Formatted context"
    prompt_messages = [
        {"role": "system", "content": "sys-prompt"}, 
        {"role": "user", "content": "prompt"}
    ]
    error_message = "Ollama connection failed"
    mock_ollama_client.chat.side_effect = Exception(error_message)

    mock_get_lines.return_value = sampled_lines
    mock_rank.return_value = reranked_context
    mock_format.return_value = formatted_context
    mock_get_prompt.return_value = prompt_messages
    mock_parse.return_value = []

    with patch.object(worker.context_retriever, 'get_diff_context', return_value=context_docs) as mock_get_context:
        result = worker.review(git_diff, dummy_system_prompt)
        mock_ollama_client.chat.assert_called_once()
        mock_parse.assert_not_called()

        assert isinstance(result, WorkerResponse)
        assert result.category == worker.category
        assert len(result.comments) == 1
        assert result.comments[0].category == worker.category
        assert f"Error during {worker.category.value} review: {error_message}" in result.comments[0].comment
        assert result.comments[0].file_name is None
        assert result.comments[0].line_number is None

        captured = capsys.readouterr()
        assert f"Error during review: {error_message}" in captured.out 

@pytest.mark.parametrize("comment_data, expected_comment, expected_file, expected_line", [
    ({"comment": "Test comment", "file_name": "a.py", "line_number": 10}, "Test comment", "a.py", 10),
    ({"issue": "Issue found", "suggestion": "Fix it", "file": "b/b.py", "line": "20"}, "Issue found Fix it", "b.py", 20),
    ({"description": "Desc", "improvement": "Imp", "filename": "c.py", "line_number": "30"}, "Desc Imp", "c.py", 30),
    ({"message": "Just msg", "file_name": "d.py"}, "Just msg", "d.py", None), 
    ({"comment": "Test comment", "file_name": "e.py", "line_number": "invalid"}, "Test comment", "e.py", None), 
    ({"comment": "Test comment", "file_name": "f.py", "line_number": None}, "Test comment", "f.py", None), 
])
def test_parse_comment_valid(worker, comment_data, expected_comment, expected_file, expected_line):
    worker.category = CodeReviewCategory.CODING_STYLE 
    result = worker.parse_comment(comment_data)
    assert isinstance(result, CodeReviewComment)
    assert result.category == CodeReviewCategory.CODING_STYLE
    assert result.comment == expected_comment
    assert result.file_name == expected_file
    assert result.line_number == expected_line

@pytest.mark.parametrize("invalid_data", [
    {}, 
    {"comment": "Test comment"}, 
    {"file_name": "a.py"}, 
    {"issue": None, "file": "b.py"}, 
])
def test_parse_comment_invalid(worker, invalid_data):
    result = worker.parse_comment(invalid_data)
    assert result is None
