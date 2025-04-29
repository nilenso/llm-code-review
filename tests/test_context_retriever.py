from unittest.mock import MagicMock, call

from code_reviewer.context_retriever import ContextRetriever
from code_reviewer.indexer import CodeIndexer

SAMPLE_CONTEXT_DOC_1 = {'file_path': 'path/to/file1.py', 'start_line': 10, 'end_line': 20, 'content': 'content of file 1'}
SAMPLE_CONTEXT_DOC_2 = {'file_path': 'path/to/file2.js', 'start_line': 5, 'end_line': 15, 'content': 'content of file 2'}
SAMPLE_CONTEXT_DOC_DUPLICATE = {'file_path': 'path/to/file1.py', 'start_line': 10, 'end_line': 20, 'content': 'duplicate content'}
SAMPLE_CONTEXT_DOC_3 = {'file_path': 'another/file.py', 'start_line': 1, 'end_line': 5, 'content': 'content of file 3'}

def make_context_retriever(mock_code_indexer=None):
    if mock_code_indexer is None:
        mock_code_indexer = MagicMock(spec=CodeIndexer)
    return ContextRetriever(mock_code_indexer)

def test_get_symbol_context(sample_git_diff_simple):
    mock_code_indexer = MagicMock(spec=CodeIndexer)
    mock_code_indexer.retrieve_symbol_context.side_effect = [
        [{'file_path': 'file1.py', 'start_line': 1, 'end_line': 5, 'content': 'def new_function(): ...'}] 
    ]
    retriever = make_context_retriever(mock_code_indexer)
    result = retriever.get_symbol_context(sample_git_diff_simple)
    assert mock_code_indexer.retrieve_symbol_context.call_count == 1 
    mock_code_indexer.retrieve_symbol_context.assert_called_once_with('new_function')
    assert len(result) == 1
    assert result[0] == [{'file_path': 'file1.py', 'start_line': 1, 'end_line': 5, 'content': 'def new_function(): ...'}]

def test_get_modified_files_context(sample_git_diff_simple):
    mock_code_indexer = MagicMock(spec=CodeIndexer)
    mock_code_indexer.retrieve_file_context.side_effect = [
        [{'file_path': 'file1.py', 'start_line': 1, 'end_line': 10, 'content': 'content file1.py'}], 
        [{'file_path': 'file2.txt', 'start_line': 1, 'end_line': 5, 'content': 'content file2.txt'}]  
    ]
    retriever = make_context_retriever(mock_code_indexer)
    file_count = 2
    result = retriever.get_modified_files_context(sample_git_diff_simple, file_count)
    assert mock_code_indexer.retrieve_file_context.call_count == 2 
    mock_code_indexer.retrieve_file_context.assert_has_calls([
        call('file1.py'),
        call('file2.txt'),
    ], any_order=True)
    assert len(result) == 2
    assert any(d == {'file_path': 'file1.py', 'start_line': 1, 'end_line': 10, 'content': 'content file1.py'} for d in result)
    assert any(d == {'file_path': 'file2.txt', 'start_line': 1, 'end_line': 5, 'content': 'content file2.txt'} for d in result)

def test_get_symbol_context_complex(complex_diff):
    mock_code_indexer = MagicMock(spec=CodeIndexer)
    # Simulate four symbols in the diff (function add, function remove, class rename, variable change)
    side_effect_values = [
        [{'file_path': 'file1.py', 'start_line': 1, 'end_line': 5, 'content': 'def new_function(): ...'}],
        [{'file_path': 'file1.py', 'start_line': 6, 'end_line': 10, 'content': 'def helper(): ...'}],
        [{'file_path': 'file1.py', 'start_line': 11, 'end_line': 15, 'content': 'class RenamedClass: ...'}],
    ]
    def side_effect(*args, **kwargs):
        if side_effect.values:
            return side_effect.values.pop(0)
        return [{'file_path': 'file1.py', 'start_line': 100, 'end_line': 101, 'content': 'default'}]
    side_effect.values = side_effect_values.copy()
    mock_code_indexer.retrieve_symbol_context.side_effect = side_effect
    retriever = make_context_retriever(mock_code_indexer)
    result = retriever.get_symbol_context(complex_diff)
    assert mock_code_indexer.retrieve_symbol_context.call_count == 4
    assert any(doc[0]['file_path'] == 'file1.py' for doc in result)


def test_get_modified_files_context_complex(modified_files_complex_diff):
    mock_code_indexer = MagicMock(spec=CodeIndexer)
    mock_code_indexer.retrieve_file_context.side_effect = [
        [{'file_path': 'file1.py', 'start_line': 1, 'end_line': 10, 'content': 'content file1.py'}],
        [{'file_path': 'file2_renamed.py', 'start_line': 11, 'end_line': 20, 'content': 'content file2_renamed.py'}],
        [{'file_path': 'file3.py', 'start_line': 21, 'end_line': 30, 'content': 'content file3.py'}]
    ]
    retriever = make_context_retriever(mock_code_indexer)
    file_count = 3
    result = retriever.get_modified_files_context(modified_files_complex_diff, file_count)
    assert mock_code_indexer.retrieve_file_context.call_count == 3
    assert any(d['file_path'] == 'file1.py' for d in result)
    assert any(d['file_path'] == 'file2_renamed.py' for d in result)
    assert any(d['file_path'] == 'file3.py' for d in result)

def test_dedup_context():
    retriever = make_context_retriever()
    context_docs = [SAMPLE_CONTEXT_DOC_1, SAMPLE_CONTEXT_DOC_2, SAMPLE_CONTEXT_DOC_DUPLICATE, SAMPLE_CONTEXT_DOC_3]
    result = retriever.dedup_context(context_docs)
    assert len(result) == 3
    assert SAMPLE_CONTEXT_DOC_1 in result 
    assert SAMPLE_CONTEXT_DOC_2 in result
    assert SAMPLE_CONTEXT_DOC_3 in result
    paths_lines = {(d['file_path'], d['start_line'], d['end_line']) for d in result}
    assert ('path/to/file1.py', 10, 20) in paths_lines
    assert ('path/to/file2.js', 5, 15) in paths_lines
    assert ('another/file.py', 1, 5) in paths_lines

def test_dedup_context_empty():
    retriever = make_context_retriever()
    assert retriever.dedup_context([]) == []

def test_dedup_context_no_duplicates():
    retriever = make_context_retriever()
    context_docs = [SAMPLE_CONTEXT_DOC_1, SAMPLE_CONTEXT_DOC_2, SAMPLE_CONTEXT_DOC_3]
    result = retriever.dedup_context(context_docs)
    assert result == context_docs 
