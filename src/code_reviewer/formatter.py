import re
import os
import jinja2
from typing import Dict, List, Optional, Tuple
from .models import CodeReviewResponse, CodeReviewComment, CodeReviewCategory

class CodeReviewFormatter:
    """Base class for formatting code reviews"""
    def __init__(self, git_diff: str, response: CodeReviewResponse):
        self.git_diff = git_diff
        self.response = response
        self.comments_by_file = self._organize_comments_by_file()
    
    def _organize_comments_by_file(self) -> Dict[str, List[Tuple[int, CodeReviewComment]]]:
        """Group comments by file and line number for easier processing"""
        comments_by_file = {}
        
        for category, comments in self.response.categories.items():
            for comment in comments:
                # Skip "No X issues detected" comments
                if "no issues detected" in comment.comment.lower():
                    continue
                
                file_name = comment.file_name
                line_number = comment.line_number
                
                if file_name:
                    if file_name not in comments_by_file:
                        comments_by_file[file_name] = []
                    
                    if line_number:
                        comments_by_file[file_name].append((line_number, comment))
                    else:
                        # For file-level comments (no specific line), use line 0
                        comments_by_file[file_name].append((0, comment))
        
        return comments_by_file
    
    def format(self) -> str:
        """Format the code review (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement this method")


class MarkdownFormatter(CodeReviewFormatter):
    """Format code reviews as Markdown"""
    
    def format(self) -> str:
        md_output = [f"# Code Review"]
        md_output.append("\n## Changelog\n")
        md_output.append(f"{self.response.summary}\n")

        md_output.append("## Review Comments\n")
        md_output.append("| File | Line | Category | Comment |")
        md_output.append("|------|------|----------|---------|")
        
        for category, comments in self.response.categories.items():
            for comment in comments:
                if "no issues detected" in comment.comment.lower():
                    continue
                
                file_name = comment.file_name if comment.file_name else "-"
                line_no = str(comment.line_number) if comment.line_number else "-"
                md_output.append(f"| {file_name} | {line_no} | {category.value} | {comment.comment} |")
        
        # Add detailed sections for each file with comments
        md_output.append("\n## Details by File\n")
        
        for file_name, comments in self.comments_by_file.items():
            md_output.append(f"### {file_name}\n")
            
            # Sort comments by line number
            sorted_comments = sorted(comments, key=lambda x: x[0])
            
            for line_number, comment in sorted_comments:
                line_info = f"Line {line_number}" if line_number > 0 else "File-level comment"
                md_output.append(f"- **{line_info}** ({comment.category.value}): {comment.comment}")
            
            md_output.append("")  # Empty line between files
        
        return "\n".join(md_output)


class HTMLFormatter(CodeReviewFormatter):
    """Format code reviews as HTML with inline comments using Jinja2 templates"""
    
    def __init__(self, git_diff: str, response: CodeReviewResponse):
        super().__init__(git_diff, response)
        # Initialize Jinja2 environment
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def _parse_diff(self) -> Dict[str, List[Tuple[int, str, str]]]:
        """
        Parse git diff to extract file names and lines.
        Returns a dict mapping file names to lists of (line_number, line_type, content) tuples.
        line_type can be 'header', 'added', 'removed', 'context', or 'info'
        """
        result = {}
        current_file = None
        line_numbers_map = {}  # Map to keep track of displayed line numbers to actual line numbers
        absolute_line_number = 0  # Absolute line number counter
        
        for line in self.git_diff.split('\n'):
            # File header
            if line.startswith('+++'):
                current_file = line[4:].strip()
                if current_file.startswith('b/'):
                    current_file = current_file[2:]
                
                if current_file not in result:
                    result[current_file] = []
                    line_numbers_map[current_file] = {}
                
                absolute_line_number = 0  # Reset for new file
                result[current_file].append((0, 'header', line))
                continue
            
            # Skip other headers
            if line.startswith('---') or line.startswith('diff ') or line.startswith('index '):
                continue
            
            # Hunk header
            if line.startswith('@@'):
                hunk_match = re.search(r'@@ -\d+,\d+ \+(\d+),\d+ @@', line)
                if hunk_match:
                    absolute_line_number = int(hunk_match.group(1))
                
                if current_file:
                    result[current_file].append((0, 'info', line))
                continue
            
            # We need a current file to process lines
            if not current_file:
                continue
            
            # Added line
            if line.startswith('+') and not line.startswith('+++'):
                # Store the mapping from absolute line number to display line number
                line_numbers_map[current_file][absolute_line_number] = absolute_line_number
                result[current_file].append((absolute_line_number, 'added', line[1:]))
                absolute_line_number += 1
                continue
            
            # Removed line
            if line.startswith('-') and not line.startswith('---'):
                result[current_file].append((0, 'removed', line[1:]))
                continue
            
            # Context line
            if not line.startswith('+') and not line.startswith('-'):
                # Store the mapping from absolute line number to display line number
                line_numbers_map[current_file][absolute_line_number] = absolute_line_number
                result[current_file].append((absolute_line_number, 'context', line))
                absolute_line_number += 1
        
        return result, line_numbers_map
    
    def format(self) -> str:
        """Format the code review as HTML with inline comments using Jinja2 template"""
        parsed_diff, line_numbers_map = self._parse_diff()
        
        # Prepare the template data
        files_data = {}
        
        for file_name, lines in parsed_diff.items():
            file_data = {"lines": [], "file_comments": []}
            
            # Add file-level comments
            if file_name in self.comments_by_file:
                file_comments = [
                    {"category": comment.category.value, "text": comment.comment}
                    for line_no, comment in self.comments_by_file.get(file_name, [])
                    if line_no == 0
                ]
                file_data["file_comments"] = file_comments
            
            # Process line-by-line diff with inline comments
            for line_no, line_type, content in lines:
                line_data = {
                    "number": line_no,
                    "type": line_type,
                    "content": content,
                    "comments": []
                }

                # Only add comments to non-removed lines and if comments exist for the file
                if line_type != 'removed' and line_no > 0 and file_name in self.comments_by_file:
                    # Find comments for the exact line number
                    comments_for_this_line = []
                    added_comment_texts = set()
                    for comment_line, comment in self.comments_by_file.get(file_name, []):
                        # Check if the comment line number exactly matches the diff line number
                        if comment_line == line_no:
                            # Normalize text for deduplication
                            comment_text_normalized = comment.comment.strip().lower()
                            if comment_text_normalized not in added_comment_texts:
                                comments_for_this_line.append(
                                    {"category": comment.category.value, "text": comment.comment}
                                )
                                added_comment_texts.add(comment_text_normalized)

                    line_data["comments"] = comments_for_this_line

                file_data["lines"].append(line_data)
            
            files_data[file_name] = file_data
        
        # Render the template
        template = self.jinja_env.get_template("code_review.html.j2")
        return template.render(
            summary=self.response.summary,
            files=files_data
        )
    
class ComprehensiveHTMLFormatter(CodeReviewFormatter):
    """
    HTML formatter that shows the entire file content with the diff applied,
    and comments aligned to the correct lines.
    """
    
    def __init__(self, git_diff: str, response: CodeReviewResponse, repo_path: Optional[str] = None):
        super().__init__(git_diff, response)
        self.repo_path = repo_path
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

    def _extract_modified_files(self) -> Dict[str, Dict]:
        """
        Extract the filenames and changes from the git diff.
        Returns a dict mapping filenames to their changes.
        """
        modified_files = {}
        current_file = None
        hunk_info = []
        
        for line in self.git_diff.split('\n'):
            # File header
            if line.startswith('+++'):
                if current_file and hunk_info:
                    modified_files[current_file]['hunks'] = hunk_info
                
                current_file = line[4:].strip()
                if current_file.startswith('b/'):
                    current_file = current_file[2:]
                
                hunk_info = []
                if current_file not in modified_files:
                    modified_files[current_file] = {'hunks': []}
                continue
            
            # Hunk header
            if line.startswith('@@'):
                hunk_match = re.search(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', line)
                if hunk_match:
                    old_start = int(hunk_match.group(1))
                    old_count = int(hunk_match.group(2))
                    new_start = int(hunk_match.group(3))
                    new_count = int(hunk_match.group(4))
                    
                    hunk_info.append({
                        'header': line,
                        'old_start': old_start,
                        'old_count': old_count,
                        'new_start': new_start,
                        'new_count': new_count,
                        'lines': []
                    })
                continue
            
            # Process lines in current hunk
            if current_file and hunk_info and line and line[0] in ['+', '-', ' ']:
                if hunk_info[-1]['lines']:
                    hunk_info[-1]['lines'].append(line)
                else:
                    hunk_info[-1]['lines'] = [line]
        
        # Add the last hunk if any
        if current_file and hunk_info:
            modified_files[current_file]['hunks'] = hunk_info
        
        return modified_files
    
    def _read_original_file(self, file_path: str) -> List[str]:
        """Read the original file content from the repository."""
        if not self.repo_path:
            return []
        
        try:
            full_path = os.path.join(self.repo_path, file_path)
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
    
    def _apply_diff_to_file(self, file_path: str, original_lines: List[str], hunks: List[Dict]) -> Tuple[List[Dict], Dict[int, int]]:
        """
        Apply the diff hunks to the original file content.
        Returns a list of line objects with content, line number, and change type,
        along with a mapping from original line numbers to new line numbers.
        """
        if not original_lines:
            return self._reconstruct_from_diff(hunks)
        
        result = []
        line_mapping = {}  # Maps original line numbers to new line numbers
        current_line = 1
        new_line_counter = 1
        
        # Convert original_lines list to include line numbers and change type
        numbered_lines = [{'number': i+1, 'content': line.rstrip('\n'), 'type': 'original'}
                         for i, line in enumerate(original_lines)]
        
        # Apply each hunk
        for hunk in hunks:
            old_start = hunk['old_start']
            new_start = hunk['new_start']
            
            # Add unchanged lines before the hunk
            while current_line < old_start:
                if current_line - 1 < len(numbered_lines):
                    line_mapping[current_line] = new_line_counter
                    result.append({
                        'number': new_line_counter, 
                        'content': numbered_lines[current_line - 1]['content'], 
                        'type': 'context',
                        'old_number': current_line
                    })
                    new_line_counter += 1
                current_line += 1
            
            # Process the hunk lines
            old_line = old_start
            
            for line in hunk['lines']:
                if line.startswith(' '):  # Context line
                    if old_line - 1 < len(numbered_lines):
                        line_mapping[old_line] = new_line_counter
                        context_line = {
                            'number': new_line_counter, 
                            'content': line[1:], 
                            'type': 'context',
                            'old_number': old_line
                        }
                        result.append(context_line)
                        new_line_counter += 1
                    old_line += 1
                    
                elif line.startswith('-'):  # Removed line
                    if old_line - 1 < len(numbered_lines):
                        # Important: Store original line number for removed lines
                        removed_line = {
                            'number': 0,  # No new line number for removed lines
                            'content': line[1:],
                            'type': 'removed',
                            'old_number': old_line
                        }
                        result.append(removed_line)
                    old_line += 1
                    
                elif line.startswith('+'):  # Added line
                    # Added lines get a new line number but no old number
                    added_line = {
                        'number': new_line_counter,
                        'content': line[1:],
                        'type': 'added',
                        'old_number': None  # No original line for added content
                    }
                    result.append(added_line)
                    new_line_counter += 1
            
            current_line = old_line
        
        # Add any remaining unchanged lines
        while current_line - 1 < len(numbered_lines):
            line_mapping[current_line] = new_line_counter
            result.append({
                'number': new_line_counter,
                'content': numbered_lines[current_line - 1]['content'],
                'type': 'context',
                'old_number': current_line
            })
            new_line_counter += 1
            current_line += 1
        
        return result, line_mapping
    
    def _reconstruct_from_diff(self, hunks: List[Dict]) -> Tuple[List[Dict], Dict[int, int]]:
        """
        Reconstruct file content from diff hunks when original file isn't available.
        Returns both the reconstructed lines and a mapping of original line numbers.
        """
        result = []
        line_mapping = {}  # Maps original line numbers to new line numbers
        
        new_line = 1  # Track new line numbers
        
        for hunk in hunks:
            old_line = hunk.get('old_start', 0)  # Get old start if available
            
            for line in hunk['lines']:
                if line.startswith(' '):  # Context line
                    line_mapping[old_line] = new_line
                    result.append({
                        'number': new_line,
                        'content': line[1:],
                        'type': 'context',
                        'old_number': old_line
                    })
                    new_line += 1
                    old_line += 1
                    
                elif line.startswith('+'):  # Added line
                    result.append({
                        'number': new_line,
                        'content': line[1:],
                        'type': 'added',
                        'old_number': None  # No original line for added content
                    })
                    new_line += 1
                    
                elif line.startswith('-'):  # Removed line
                    result.append({
                        'number': 0,  # Removed lines don't have new line numbers
                        'content': line[1:],
                        'type': 'removed',
                        'old_number': old_line
                    })
                    old_line += 1
            
        return result, line_mapping

    def _is_similar_comment(self, comment1, comment2, similarity_threshold=0.6):
        """
        Check if two comments are semantically similar.
        Uses a simple word overlap heuristic to determine similarity.
        """
        # Check for null-related comments
        null_keywords = ['null', 'npe', 'nullpointer', 'nullpointerexception', 'not null']
        
        text1 = comment1['text'].lower()
        text2 = comment2['text'].lower()

        # Check for specific patterns
        is_null_comment1 = any(keyword in text1 for keyword in null_keywords)
        is_null_comment2 = any(keyword in text2 for keyword in null_keywords)
        
        # If both are about null checks, consider them similar
        if is_null_comment1 and is_null_comment2:
            return True
            
        # Otherwise use word overlap for similarity
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        
        return similarity > similarity_threshold

    def _merge_similar_comments(self, comments_list):
        """
        Merge similar comments into a single comment.
        Returns a list of deduplicated comments.
        """
        if not comments_list:
            return []
            
        result = []
        
        for new_comment in comments_list:
            # Check if we already have a similar comment
            found_similar = False
            for existing_comment in result:
                if self._is_similar_comment(new_comment, existing_comment):
                    # Don't add this comment as it's similar to an existing one
                    found_similar = True
                    break
            
            if not found_similar:
                result.append(new_comment)
        
        return result

    def _validate_and_group_comment(self, ln: int, comment: CodeReviewComment, file_path: str, 
                                    line_by_number: Dict[int, Dict], 
                                    comments_to_attach: Dict[int, List[Dict]]):
        """
        Validates the line number provided by the LLM for a comment and groups
        valid comments by their target line number. If invalid, fallback to file-level comment.
        """
        if ln is None or ln <= 0:
            print(f"[Fallback] Attaching comment to file-level for {file_path}: {comment.comment}")
            # Attach as file-level comment if not already present
            if file_path in self.comments_by_file:
                self.comments_by_file[file_path].append((0, comment))
            else:
                self.comments_by_file[file_path] = [(0, comment)]
            return
        target_line_number = ln
        if target_line_number not in line_by_number:
            print(f"[LLM Validation Warning] Fallback: Attaching comment to file-level for {file_path}: LLM provided non-existent line number {target_line_number}. Comment: '{comment.comment}'")
            if file_path in self.comments_by_file:
                self.comments_by_file[file_path].append((0, comment))
            else:
                self.comments_by_file[file_path] = [(0, comment)]
            return
        target_line_obj = line_by_number[target_line_number]
        if target_line_obj['type'] != 'added':
            print(f"[LLM Validation Warning] Fallback: Attaching comment to file-level for {file_path}:{target_line_number}: LLM provided line number for a non-added line (type: {target_line_obj['type']}). Comment: '{comment.comment}'")
            if file_path in self.comments_by_file:
                self.comments_by_file[file_path].append((0, comment))
            else:
                self.comments_by_file[file_path] = [(0, comment)]
            return
        comment_obj = {
            'category': comment.category.value,
            'text': comment.comment
        }
        if target_line_number not in comments_to_attach:
            comments_to_attach[target_line_number] = []

        comments_to_attach[target_line_number].append(comment_obj)

    def format(self) -> str:
        """
        Format the code review as HTML showing complete files with the diff applied
        and comments at the correct lines.
        """
        modified_files = self._extract_modified_files()
        files_data = {}
        
        for file_path, file_info in modified_files.items():
            # Read the original file
            original_lines = self._read_original_file(file_path)
            
            # Apply the diff to get the modified file with change information
            if original_lines:
                file_lines, line_mapping = self._apply_diff_to_file(file_path, original_lines, file_info['hunks'])
            else:
                file_lines, line_mapping = self._reconstruct_from_diff(file_info['hunks'])
            
            # Create a reverse mapping from new line numbers to line objects
            line_by_number = {}
            for i, line in enumerate(file_lines):
                if line['number'] > 0:  # Skip removed lines
                    line_by_number[line['number']] = line
                # Initialize comments array for each line
                line['comments'] = []
            
            # Add file-level comments
            file_comments = []
            if file_path in self.comments_by_file:
                file_level_comments = [
                    {"category": comment.category.value, "text": comment.comment}
                    for ln, comment in self.comments_by_file.get(file_path, [])
                    if ln == 0
                ]
                # Deduplicate file-level comments
                file_comments = self._merge_similar_comments(file_level_comments)
            
            # Process comments and attach them to the right lines
            if file_path in self.comments_by_file:
                # Use a dictionary to group comments by the target NEW line number
                comments_to_attach = {}

                for ln, comment in self.comments_by_file.get(file_path, []):
                    if ln == 0:  # Skip file-level comments (already handled)
                        continue
                    
                    # Call the validation and grouping function
                    self._validate_and_group_comment(ln, comment, file_path, line_by_number, comments_to_attach)

                # Now attach the grouped and deduplicated comments to the lines
                for target_line_num, comments in comments_to_attach.items():
                    # Validation is done in the helper, just need to find the line object again
                    if target_line_num in line_by_number:
                        target_line_obj = line_by_number[target_line_num]
                        # Attach comments (assuming type check was passed in helper)
                        deduplicated_comments = self._merge_similar_comments(comments)
                        target_line_obj['comments'] = deduplicated_comments
                    else:
                        print(f"Error: Target line {target_line_num} not found during attachment in {file_path}")

            files_data[file_path] = {
                "lines": file_lines,
                "file_comments": file_comments
            }
        
        # Render the HTML template
        return self._render_html(files_data)
    
    def _render_html(self, files_data):
        """Render the HTML template with the data."""
        template = self.jinja_env.get_template("comprehensive_review.html.j2")
        return template.render(
            summary=self.response.summary,
            files=files_data
        )

def format_review(git_diff: str, response: CodeReviewResponse, format_type: str = "markdown", repo="./") -> str:
    """
    Format a code review response in the specified format.
    
    Args:
        git_diff: The original git diff
        response: The code review response
        format_type: The output format ("markdown", "html", "comprehensive_html")
        repo_path: Optional path to the repository root for the comprehensive formatter.

    Returns:
        Formatted review as a string
    """
    if format_type.lower() == "html":
        formatter = HTMLFormatter(git_diff, response)
    elif format_type.lower() == "comprehensive_html":
        formatter = ComprehensiveHTMLFormatter(git_diff, response, repo_path=repo)
    else:
        formatter = MarkdownFormatter(git_diff, response)

    return formatter.format()
