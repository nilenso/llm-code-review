from .models import CodeReviewCategory


CODE_REVIEW_PROMPTS = {
    CodeReviewCategory.DESIGN: """
    Find design flaws and improvements:
    - Identify classes/functions violating single responsibility
    - Flag improper separation of concerns
    - Spot excessive complexity that should be simplified
    - Identify tight coupling that should be loosened
    - Point out missing abstractions or poor component organization
    """,    

    CodeReviewCategory.FUNCTIONALITY: """
    Identify bugs and functional problems:
    - Flag incorrect conditionals or logic errors
    - Find unhandled edge cases
    - Identify unchecked error conditions
    - Spot null/undefined handling issues
    - Point out potential security vulnerabilities
    - Flag performance bottlenecks
    """,

    CodeReviewCategory.NAMING: """
    Find naming problems:
    - Flag non-descriptive or misleading names
    - Identify inconsistencies with codebase conventions
    - Point out missing prefixes for booleans (is/has)
    - Flag confusing abbreviations
    - Identify names that don't reflect purpose
    """,

    CodeReviewCategory.CONSISTENCY: """
    Identify consistency issues:
    - Flag style inconsistencies with codebase
    - Point out inconsistent error handling
    - Identify mixed patterns for similar operations
    - Flag mixed conventions (indentation, bracing)
    - Note inconsistent API design
    """,

    CodeReviewCategory.CODING_STYLE: """
    Find style problems:
    - Flag improper indentation or formatting
    - Identify missing comments for complex logic
    - Point out excessive line lengths
    - Flag commented-out code without explanation
    - Identify unnecessarily complex expressions
    """,

    CodeReviewCategory.TESTS: """
    Identify testing gaps:
    - Flag missing tests for functionality
    - Point out uncovered edge cases
    - Identify weak or unclear assertions
    - Flag interdependent tests
    - Note brittle test implementations
    """,

    CodeReviewCategory.ROBUSTNESS: """
    Find error handling weaknesses:
    - Identify missing exception handling
    - Flag absent input validation
    - Point out resource leaks
    - Identify inadequate logging
    - Flag potential concurrency issues
    """,

    CodeReviewCategory.READABILITY: """
    Find readability issues:
    - Flag complex code without explanatory comments
    - Identify deeply nested control structures
    - Point out excessively long functions
    - Flag overly complex boolean expressions
    - Identify cryptic algorithms
    """,

    CodeReviewCategory.ABSTRACTIONS: """
    Find abstraction problems:
    - Identify repeated code needing abstraction
    - Flag poor encapsulation exposing internals
    - Point out overly complex interfaces
    - Identify primitive obsession
    - Flag mixed abstraction levels
    """
}

TARGETED_REVIEW_PROMPTS = {
    ### DESIGN SUBCATEGORIES ###
    "DESIGN_SRP": """
    Find single responsibility principle violations:
    - Identify functions doing multiple unrelated operations
    - Flag classes with multiple responsibilities
    - Point out modules handling too many concerns
    """,
    
    "DESIGN_COUPLING": """
    Find coupling problems:
    - Identify tight coupling between components
    - Flag excessive dependencies between modules
    - Point out inappropriate inheritance relationships
    - Identify components that should communicate through interfaces
    """,
    
    "DESIGN_COMPLEXITY": """
    Find complexity issues:
    - Identify overly complex algorithms or workflows
    - Flag methods with too many parameters
    - Point out deeply nested code that could be simplified
    - Identify convoluted business logic
    """,
    
    ### FUNCTIONALITY SUBCATEGORIES ###
    "FUNCTIONALITY_BUGS": """
    Find logic bugs and errors:
    - Identify incorrect conditionals or logic errors
    - Flag off-by-one errors in loops or calculations
    - Point out incorrect operator usage (e.g., = vs ==)
    - Identify incorrect return values
    """,
    
    "FUNCTIONALITY_EDGE_CASES": """
    Find unhandled edge cases:
    - Identify missing null/undefined checks
    - Flag potential division by zero
    - Point out unchecked array bounds
    - Identify unhandled empty collections
    """,
    
    "FUNCTIONALITY_SECURITY": """
    Find security issues:
    - Identify missing input validation
    - Flag potential injection vulnerabilities
    - Point out insecure data handling
    - Identify authentication/authorization weaknesses
    """,
    
    ### ROBUSTNESS SUBCATEGORIES ###
    "ROBUSTNESS_ERROR_HANDLING": """
    Find error handling problems:
    - Identify missing try-catch blocks
    - Flag empty catch blocks
    - Point out swallowed exceptions
    - Identify incorrect error propagation
    """,
    
    "ROBUSTNESS_RESOURCE_MANAGEMENT": """
    Find resource management issues:
    - Identify unclosed resources (files, connections)
    - Flag potential memory leaks
    - Point out missing cleanup code
    - Identify unmanaged external resources
    """,
    
    "ROBUSTNESS_CONCURRENCY": """
    Find concurrency issues:
    - Identify potential race conditions
    - Flag unsynchronized shared state
    - Point out deadlock possibilities
    - Identify misuse of asynchronous operations
    """,
    
    ### CODE QUALITY SUBCATEGORIES ###
    "QUALITY_NAMING": """
    Find naming problems:
    - Identify unclear or misleading variable names
    - Flag inconsistent naming conventions
    - Point out poor function/method names
    - Identify cryptic abbreviations
    """,
    
    "QUALITY_READABILITY": """
    Find readability issues:
    - Identify complex code without comments
    - Flag excessively long functions
    - Point out convoluted expressions
    - Identify poor code organization
    """,
    
    "QUALITY_DUPLICATION": """
    Find code duplication:
    - Identify repeated logic that should be abstracted
    - Flag copy-pasted code with minor variations
    - Point out redundant calculations
    - Identify duplicate validation logic
    """,
}

# Map from standard categories to targeted subcategories
CATEGORY_TO_SUBCATEGORIES = {
    CodeReviewCategory.DESIGN: ["DESIGN_SRP", "DESIGN_COUPLING", "DESIGN_COMPLEXITY"],
    CodeReviewCategory.FUNCTIONALITY: ["FUNCTIONALITY_BUGS", "FUNCTIONALITY_EDGE_CASES", "FUNCTIONALITY_SECURITY"],
    CodeReviewCategory.NAMING: ["QUALITY_NAMING"],
    CodeReviewCategory.CONSISTENCY: ["QUALITY_NAMING", "DESIGN_COUPLING"],
    CodeReviewCategory.CODING_STYLE: ["QUALITY_READABILITY", "QUALITY_DUPLICATION"],
    CodeReviewCategory.TESTS: ["FUNCTIONALITY_EDGE_CASES"],
    CodeReviewCategory.ROBUSTNESS: ["ROBUSTNESS_ERROR_HANDLING", "ROBUSTNESS_RESOURCE_MANAGEMENT", "ROBUSTNESS_CONCURRENCY"],
    CodeReviewCategory.READABILITY: ["QUALITY_READABILITY"],
    CodeReviewCategory.ABSTRACTIONS: ["DESIGN_SRP", "QUALITY_DUPLICATION"],
}

def worker_system_prompt(category, subcategories):
    return f"""
    You are a critical code reviewer focusing on {category} issues in code.
    
    You will be given:
    1. The code diff to review (only comment on these specific changes)
    2. Additional context from the repository to help you understand the codebase better
    
    IMPORTANT:
    - Do NOT mention positive aspects or praise the code
    - Focus on problems and improvements ONLY in the ADDED lines in the diff(lines starting with '+').
    - Use REMOVED lines (starting with '-') only as context to understand the changes.
    - Be direct and specific in your criticism
    
    Find critical issues related to:
    {subcategories}
    
    Respond ONLY with a valid JSON array of comments. Each comment MUST have a "comment" field:
    [
      {{
         "file_name": "example.py", // optional
         "line_number": 42, // optional
         "comment": "Your critical feedback goes here" // REQUIRED
       }}
    ]              
    
    Only include file_name and line_number if your comment applies to a specific line.
    If you have no comments for this category within the diff, return an empty array [].
    """

def worker_user_prompt(category, code_to_review, context):
    return f"""                
    --- GIT DIFF (review this) ---
    {code_to_review}
    --- END DIFF ---

    --- CONTEXT (for reference only, DO NOT REVIEW) ---
    {context}
    --- END CONTEXT ---
        
    Review ONLY the ADDED lines (starting with '+') in the git diff for {category} problems ONLY, focusing 
    on the specific subcategories mentioned. Use REMOVED lines (starting with '-') as context to understand 
    what changed. Be critical - focus exclusively on issues, not strengths.
    
    Reply with JSON array of comments with the format:
    [
      {{
         "file_name": "Example.kt", // optional
         "line_number": 123, // optional
         "comment": "Critical feedback with improvement suggestion" // REQUIRED
      }}
    ]
    
    Only include file_name and line_number if your comment applies to a specific line.
    If you have no comments for this category within the diff, return an empty array [].
    """

def changelog_system_prompt():
    return """
    You are a helpful assistant that generates a changelog from a git diff.
    **Goal:** Create a simple, human-readable list summarizing the actual code changes.
    
    **Instructions:**
    1.  **Analyze Diff:** Look at the added (+) and removed (-) lines in the provided git diff.
    2.  **Describe Changes:** For each distinct code change (like fixing a bug, adding a feature, refactoring code, removing unused code), write one bullet point (`-`) describing what was done.
    3.  **Be Concrete:** Describe the action taken (e.g., "Fixed calculation in...", "Added method X to class Y...", "Removed function Z...", "Refactored function A for clarity..."). Focus on WHAT changed in the code.
    4.  **Exclude Non-Code:** **DO NOT** include changes to comments, documentation, or whitespace/formatting.
    5.  **Keep it Simple:** Use clear language. Be concise. Do not include file names or line numbers.
    6.  **Output Format:** Respond ONLY with a flat list of bullet points (`-`). **DO NOT** use any headings or other formatting.
    
    **Example Output (Flat list ONLY):**
    - Fixed calculation error in the `add` function.
    - Added a `double` method to the `Bar` class.
    - Added an initialization message to `Bar` class.
    - Refactored the `foo` function for clarity and updated its logic.
    - Renamed `greet_old` function to `greet` and updated message.
    - Removed the unused function `unused_func`.
    """

def changelog_user_prompt(git_diff: str) -> str:
    return f"""
    Analyze the following git diff and generate a concise, human-readable changelog summarizing the key changes. Group similar items.

    Git Diff:
    ```diff
    {git_diff}
    ```
    
    Generate the changelog based ONLY on the provided diff:
    """
