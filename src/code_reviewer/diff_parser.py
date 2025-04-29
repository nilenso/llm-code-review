import os
import re
from typing import Dict, Set, Generator, Tuple
from unidiff import PatchSet


def extract_diff_lines(git_diff: str) -> Generator[Tuple[str, int, str, str], None, None]:
    """
    Parse git diff to extract file names and both added and removed lines.
    Returns tuples of (file_name, line_number, code_line, line_type)
    line_type is 'added' or 'removed'
    Line number refers to the line number in the new file for added lines,
    and the line number in the old file for removed lines.
    """
    patch_set = PatchSet.from_string(git_diff)

    for patched_file in patch_set:
        file_path = patched_file.path  # Or patched_file.target_file for consistency
        if file_path.startswith('b/'): # Handle git's 'b/' prefix
            file_path = file_path[2:]

        for hunk in patched_file:
            for line in hunk:
                if line.is_added:
                    # Use target_line_no for added lines
                    yield file_path, line.target_line_no, line.value, 'added'
                elif line.is_removed:
                    # Use source_line_no for removed lines
                    yield file_path, line.source_line_no, line.value, 'removed'


def extract_added_lines(git_diff: str) -> Generator[Tuple[str, int, str], None, None]:
    """
    Parse git diff to extract file names and added lines.
    Returns tuples of (file_name, line_number, code_line)
    Line number refers to the line number in the new file.
    """
    patch_set = PatchSet.from_string(git_diff)

    for patched_file in patch_set:
        file_path = patched_file.path
        if file_path.startswith('b/'):
            file_path = file_path[2:]

        for hunk in patched_file:
            for line in hunk:
                if line.is_added:
                    yield file_path, line.target_line_no, line.value


def extract_modified_files(git_diff: str) -> Set[str]:
    """
    Extract the list of modified files from the git diff
    """
    modified_files: Set[str] = set()
    patch_set = PatchSet.from_string(git_diff)

    for patched_file in patch_set:
        file_path = patched_file.path
        if file_path.startswith('b/'):
            file_path = file_path[2:]
        # Extract only the filename, not the full path
        file_name = os.path.basename(file_path)
        modified_files.add(file_name)

    return modified_files


def extract_modified_symbols(git_diff: str) -> Dict[str, Set[str]]:
    """
    Extract names of functions, classes, and variables that are being modified in the diff
    """
    modified_symbols: Dict[str, Set[str]] = {}

    # Patterns to identify symbols in various languages (keep existing patterns)
    patterns = {
        'function': r'(?:function|def|public|private|protected)\s+(\w+)\s*\(',
        'class': r'(?:class)\s+(\w+)',
        'variable': r'(?:var|let|const|int|long|double|float|boolean|string|private\s+val)\s+(\w+)\s*[=;:]'
    }

    patch_set = PatchSet.from_string(git_diff)

    for patched_file in patch_set:
        file_path = patched_file.path
        if file_path.startswith('b/'):
            file_path = file_path[2:]
        # Extract only the filename, not the full path
        current_file_basename = os.path.basename(file_path)
        if current_file_basename not in modified_symbols:
            modified_symbols[current_file_basename] = set()

        for hunk in patched_file:
            for line in hunk:
                # Only process added lines
                if line.is_added:
                    code_line = line.value

                    # Check for symbols using existing patterns
                    for symbol_type, pattern in patterns.items():
                        matches = re.finditer(pattern, code_line)
                        for match in matches:
                            symbol_name = match.group(1)
                            if symbol_name:
                                modified_symbols[current_file_basename].add(symbol_name)

    return modified_symbols
