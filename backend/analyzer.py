"""
Code analysis using TreeSitter for AST parsing and Hugging Face models for summarization.
Uses LangChain to orchestrate the LLM calls.
"""

try:
    from tree_sitter import Language, Parser
except ImportError:
    Language = None
    Parser = None
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional
import torch
import torch
import re
import os
import datetime
import google.generativeai as genai

# Initialize TreeSitter for Python
# Note: For simplicity and reliability, we primarily use regex parsing
# TreeSitter can be enabled if properly configured, but regex works out-of-the-box
parser = None
try:
    # Try to use tree-sitter-python if properly installed
    import tree_sitter_python as tspython
    # The tree-sitter-python package structure varies by version
    # We'll try common patterns but fall back to regex if it fails
    if hasattr(tspython, 'language') and callable(getattr(tspython, 'language')):
        lang_obj = tspython.language()
        if lang_obj:
            PY_LANGUAGE = Language(lang_obj)
            parser = Parser(PY_LANGUAGE)
except Exception:
    # TreeSitter not available or not properly configured
    # This is fine - we'll use regex parsing which works reliably
    parser = None


def parse_files_with_treesitter(files: List[Dict]) -> List[Dict]:
    """
    Parse code files using TreeSitter to extract important functions and classes.
    
    Args:
        files: List of dicts with "path" and "code" keys
    
    Returns:
        List of dicts with "path", "code", and "functions" keys
        where "functions" is a list of function/class names found
    """
    parsed_data = []
    
    for file_info in files:
        path = file_info["path"]
        code = file_info["code"]
        
        # Extract functions and classes using TreeSitter or regex fallback
        functions = []
        
        if path.endswith(".py") and parser is not None:
            # Use TreeSitter for Python files if available
            try:
                tree = parser.parse(bytes(code, "utf8"))
                root_node = tree.root_node
                
                def traverse(node):
                    if node.type == "function_definition":
                        # Get function name
                        for child in node.children:
                            if child.type == "identifier":
                                functions.append(child.text.decode("utf8"))
                    elif node.type == "class_definition":
                        # Get class name
                        for child in node.children:
                            if child.type == "identifier":
                                functions.append(f"class {child.text.decode('utf8')}")
                    
                    for child in node.children:
                        traverse(child)
                
                traverse(root_node)
            except Exception as e:
                print(f"Warning: TreeSitter parsing failed for {path}: {e}")
                # Fall through to regex parsing
        
        # Fallback: Use simple regex to find functions/classes if TreeSitter not available
        if not functions and path.endswith(".py"):
            # Find function definitions: def function_name(
            func_matches = re.findall(r'^\s*def\s+(\w+)\s*\(', code, re.MULTILINE)
            functions.extend(func_matches)
            # Find class definitions: class ClassName
            class_matches = re.findall(r'^\s*class\s+(\w+)', code, re.MULTILINE)
            functions.extend([f"class {name}" for name in class_matches])
        
        parsed_data.append({
            "path": path,
            "code": code,
            "functions": functions
        })
    
    return parsed_data


class SimpleCodeLLM:
    """
    Simple wrapper around Hugging Face CodeT5 model for code summarization.
    Uses CodeT5-base-multi-sum which is fine-tuned for code summarization.
    """
    
    def __init__(self):
        # Switch to the multi-sum model which is fine-tuned for summarization
        model_name = "Salesforce/codet5-base-multi-sum"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # Use CPU by default
        self.device = "cpu"
        self.model.to(self.device)
    
    def generate_summary(self, code_text: str, max_length: int = 150) -> str:
        """Generate a summary from the code content."""
        # CodeT5-sum models are trained to take code and output summary.
        # We prepend 'summarize: ' as per T5 conventions.
        input_text = "summarize: " + code_text[:1500]  # Allow more context (1500 chars)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=2048, truncation=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,        # Configurable length
                min_length=50,                # Ensure substantial output
                num_beams=4,                  # Better quality search
                early_stopping=True,
                length_penalty=1.0,    
                no_repeat_ngram_size=3, 
                repetition_penalty=1.2  
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

    def generate_line_by_line_explanation(self, code_text: str, file_path: str, max_length: int = 1024) -> str:
        """
        Generate a detailed line-by-line explanation of the code.
        Uses a more detailed prompt and longer output for comprehensive explanations.
        """
        # Create a detailed prompt that explicitly requests line-by-line explanation
        # Split code into lines for better context
        code_lines = code_text.split('\n')
        num_lines = len(code_lines)
        
        # For very long files, process in chunks
        if num_lines > 200:
            # Process first 200 lines and last 50 lines
            code_preview = '\n'.join(code_lines[:200]) + '\n\n... [middle section omitted] ...\n\n' + '\n'.join(code_lines[-50:])
            code_text = code_preview
        
        # Enhanced prompt for line-by-line explanation
        prompt = (
            f"Explain the following code from file '{file_path}' in detail, line by line.\n"
            f"For each important line or code block, explain:\n"
            f"1. What the line does\n"
            f"2. Why it's needed\n"
            f"3. How it connects to other parts\n"
            f"4. Any important variables, functions, or concepts\n\n"
            f"Code:\n{code_text}\n\n"
            f"Provide a comprehensive, beginner-friendly explanation:"
        )
        
        # Use 'explain:' prefix for better model understanding
        input_text = "explain: " + prompt[:3000]  # Limit input to avoid token limits
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=2048, truncation=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,        # Much longer for detailed explanations
                min_length=200,               # Ensure substantial output
                num_beams=5,                  # Better quality search
                early_stopping=False,         # Don't stop early for detailed output
                length_penalty=0.6,           # Prefer longer explanations
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,       # Reduce repetition
                temperature=0.7               # Slightly more creative
            )
        
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation


class GeminiCodeLLM:
    """
    Wrapper around Google's Gemini Pro model for superior code summarization and explanation.
    """
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Use gemini-flash-latest as verified working model
        self.model = genai.GenerativeModel('gemini-flash-latest')
    
    def _call_with_retry(self, func, *args, **kwargs):
        """
        Helper to call a function with exponential backoff retry for 429 errors.
        Respects 'retry_delay' if present in the error message.
        """
        import time
        import random
        import re
        
        max_retries = 5  # Increased retries for better resilience
        base_delay = 5.0
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e)
                # Check for 429 error code or "quota" in message
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt == max_retries - 1:
                        # Last attempt failed, raise the error
                        raise e
                    
                    # Try to parse strict retry delay from error message
                    # Look for: retry_delay { seconds: 46 }   OR   retry in 46.8s
                    delay_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_str)
                    if not delay_match:
                         delay_match = re.search(r'retry in\s*([\d\.]+)', error_str)
                    
                    if delay_match:
                        # Add a small buffer (1s) to be safe
                        wait_time = float(delay_match.group(1)) + 1.0
                        print(f"Gemini Rate Limit: Server requested wait of {wait_time:.2f}s. Sleeping...")
                        time.sleep(wait_time)
                    else:
                        # Fallback to exponential backoff
                        delay = (base_delay * (2 ** attempt)) + (random.random() * 1.0)
                        print(f"Gemini Rate Limit hit. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                else:
                    # Not a rate limit error, raise immediately
                    raise e

    def generate_summary(self, code_text: str, max_length: int = 150) -> str:
        """Generate a summary from the code content using Gemini."""
        prompt = (
            f"Summarize the following code. keeping it concise (under {max_length} characters if possible):\n\n"
            f"{code_text[:3000]}"
        )
        try:
            response = self._call_with_retry(self.model.generate_content, prompt)
            if response.candidates and response.candidates[0].content.parts:
                return response.text
            else:
                return "Summary unavailable: Model returned no content."
        except Exception as e:
            import traceback
            error_msg = f"Gemini summary generation failed: {e}\n{traceback.format_exc()}"
            print(error_msg)
            with open("gemini_debug_error.log", "a") as f:
                f.write(f"[{datetime.datetime.now()}] ERROR in generate_summary: {error_msg}\n")
            return f"Summary unavailable due to API error: {str(e)}"

    def generate_line_by_line_explanation(self, code_text: str, file_path: str, max_length: int = 2048) -> str:
        """
        Generate a detailed line-by-line explanation using the AutoReasoner persona.
        """
        prompt = f"""
You are AutoReasoner, an expert code comprehension assistant integrated into a GitHub repository analysis system.

The user has provided a GitHub repository URL.
The repository has already been fetched using the GitHub API.
The source code for the current file has already been extracted and prepared by the system.

Your task is to analyze the given source code content for ONE file at a time and generate clear, accurate, and human-readable explanations.
You are responsible for understanding and explaining the code 

────────────────────────────────────────
STEP 1: Code Structure Analysis
────────────────────────────────────────
- Analyze the provided source code for this file.
- Detect whether the code contains:
  a) Functions or methods
  b) Logical blocks (if-else, loops, try-except / try-catch, switch-case, etc.)

Rules:
- If functions are present:
  - Treat EACH function independently
  - Inside each function, further split the code into meaningful logical blocks
- If NO functions are present:
  - Explain the code LINE BY LINE in execution order

────────────────────────────────────────
STEP 2: Splitting Rules (IMPORTANT)
────────────────────────────────────────
- Do NOT merge unrelated logic
- Each logical block must represent ONE meaningful operation
- A logical block can be:
  - Initialization
  - Conditional logic
  - Loop
  - Function or API call
  - Error handling
  - Return statement
- Preserve execution order and intent

────────────────────────────────────────
STEP 3: Explanation Requirements
────────────────────────────────────────
For EACH function or logical block, explain:

- HOW the function or block works internally
- WHY this block exists
- The flow of execution step-by-step
- Input → processing → output
- How this function or block contributes to the overall file and repository

Guidelines:
- Use simple, natural, human-readable language
- Assume the reader is a student or junior developer
- Do NOT repeat or rewrite the code
- Do NOT hallucinate functionality
- Do NOT assume missing behavior
- Be precise, clear, and instructional

────────────────────────────────────────
STEP 4: Output Format (STRICT — FOLLOW EXACTLY)
────────────────────────────────────────

--- FILE OVERVIEW ---
Briefly explain what this file does and how it fits into the overall repository (e.g., backend API, analysis logic, database layer, GitHub integration).

--- FUNCTION: <function_name> ---
Purpose:
Explain the role of this function in simple terms.

Logic Breakdown:
Block 1: <Block Name>
- Step-by-step explanation of how this block works

Block 2: <Block Name>
- Step-by-step explanation

(Repeat for all blocks)

Return Value:
Explain what the function returns and under what conditions.

--- NO FUNCTIONS PRESENT ---
Line-by-Line Explanation:

Line 1:
- Explanation

Line 2:
- Explanation

(Continue sequentially in execution order)

--- END OF EXPLANATION ---

────────────────────────────────────────
STEP 5: Quality Constraints
────────────────────────────────────────
- Explanations must be:
  ✔ Clear
  ✔ Correct
  ✔ Concise
  ✔ Beginner-friendly
- Focus on reasoning and execution flow
- The explanation must be accurate enough for:
  - Learning
  - Code review
  - Developer onboarding

────────────────────────────────────────
SOURCE CODE FOR CURRENT FILE:
{code_text}
"""
        try:
            # Generate content safely with retry
            response = self._call_with_retry(self.model.generate_content, prompt)
            if response.candidates and response.candidates[0].content.parts:
                return response.text
            else:
                return "Explanation unavailable: Model returned no content."
        except Exception as e:
            error_msg = f"Gemini explanation generation failed: {e}"
            print(error_msg)
            return f"Failed to generate explanation. Error: {str(e)}"



    def generate_dependency_analysis(self, code_text: str, file_path: str) -> Dict:
        """
        Generate strict JSON dependency analysis for the file.
        """
        prompt = f"""
You are AutoReasoner, an expert code analysis assistant.

Keeping the model 'gemini-3-flash-preview'. This task is to identify dependencies and call relationships.

The source code has already been fetched from a GitHub repository.
Analyze the given file in isolation, but infer dependencies when visible.

────────────────────────────────────────
TASK 1: File-Level Dependencies
────────────────────────────────────────
Identify:
- Which external files/modules are imported or referenced
- Whether this file depends on:
  - Database layer
  - API layer
  - Utility/helper modules

Output each dependency as a directional relationship:
<Current File> → <Dependent File or Module>

────────────────────────────────────────
TASK 2: Function-Level Call Flow
────────────────────────────────────────
For each function in the file:
- Identify which functions it:
  - Calls internally
  - Calls from other files
- Ignore standard library calls unless important

Represent relationships as:
<Caller Function> → <Callee Function>

────────────────────────────────────────
RULES (IMPORTANT)
────────────────────────────────────────
- Do NOT explain logic
- Do NOT summarize code
- Do NOT hallucinate dependencies
- Only include relationships clearly visible in code
- If no dependencies exist, explicitly say: "No dependencies found"

────────────────────────────────────────
OUTPUT FORMAT (STRICT – JSON ONLY)
────────────────────────────────────────
{{
  "file_dependencies": [
    {{
      "from": "<current_file>",
      "to": "<dependent_file_or_module>",
      "type": "import | usage"
    }}
  ],
  "function_dependencies": [
    {{
      "from": "<caller_function>",
      "to": "<callee_function>",
      "type": "internal | external"
    }}
  ]
}}

Now analyze the following code from file: {file_path}

{code_text}
"""
        try:
            # We want strict JSON
            response = self._call_with_retry(self.model.generate_content, prompt, generation_config={"response_mime_type": "application/json"})
            import json
            if response.candidates and response.candidates[0].content.parts:
                return json.loads(response.text)
            else:
                print("Gemini returned no content for dependency analysis.")
                return {"file_dependencies": [], "function_dependencies": []}
        except Exception as e:
            print(f"Dependency analysis failed: {e}")
            return {"file_dependencies": [], "function_dependencies": []}


# Global LLM instance to avoid reloading logic
_global_llm = None

def get_llm():
    global _global_llm
    if _global_llm is None:
        # Check for Gemini API Key first
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                print("Initializing GeminiCodeLLM...")
                _global_llm = GeminiCodeLLM(api_key)
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}. Falling back to SimpleCodeLLM.")
                _global_llm = SimpleCodeLLM()
        else:
            print("No GEMINI_API_KEY found. Using SimpleCodeLLM (Hugging Face).")
            _global_llm = SimpleCodeLLM()
    return _global_llm

def summarize_repository(parsed_data: List[Dict]) -> Dict:
    """
    Use LLM to create summaries for the repository and each file.
    
    Args:
        parsed_data: List of parsed file data from parse_files_with_treesitter
    
    Returns:
        Dict with "repo_summary" (string) and "file_summaries" (list of dicts)
    """
    # Initialize the LLM
    llm = get_llm()
    
    # Step 3: Create repository summary
    # Combine important parts of files for repo context
    all_code_snippets = []
    for item in parsed_data[:5]:  # Limit to first 5 important files
        # Include filename and a bit of code
        code_preview = item["code"][:200]
        all_code_snippets.append(f"# File: {item['path']}\n{code_preview}")
    
    # Join nicely
    repo_code_context = "\n".join(all_code_snippets)
    repo_summary = llm.generate_summary(repo_code_context)
    
    # Step 3: Create file summaries
    file_summaries = []
    for item in parsed_data:
        path = item["path"]
        code = item["code"]
        functions = item["functions"]
        
        # Instead of a long LLM summary, build a short, human-friendly description
        # that simply states what the file represents. Keep it deterministic and
        # easy to read for the UI table.
        file_summary = _simple_file_description(path, functions)
        
        file_summaries.append({
            "path": path,
            "summary": file_summary
        })
    
    return {
        "repo_summary": repo_summary,
        "file_summaries": file_summaries
    }


def _simple_file_description(path: str, functions: List[str]) -> str:
    """
    Build a short, easy-to-read description of what the file represents.
    Keeps wording simple and avoids long summaries.
    """
    lower = path.lower()
    
    if lower.endswith(".html"):
        return "HTML page structure and layout for this project section."
    if lower.endswith(".css"):
        return "Stylesheet that defines the look, colors, and spacing."
    if lower.endswith((".js", ".jsx", ".ts", ".tsx")):
        if functions:
            # Mention the first couple of functions/classes if available
            top_items = ", ".join(functions[:2])
            return f"Script handling UI or logic; key items: {top_items}."
        return "Script file that implements client-side logic or helpers."
    if lower.endswith(".py"):
        if functions:
            top_items = ", ".join(functions[:2])
            return f"Python module for backend logic; key items: {top_items}."
        return "Python module used in the backend services."
    if lower.endswith(".md"):
        return "Documentation or README content for the project."
    if lower.endswith((".json", ".yml", ".yaml")):
        return "Configuration or metadata file for the project."
    
    # Default fallback
    return "Project source file containing code or resources."


def generate_single_file_explanation(code: str, path: str) -> Dict:
    """
    Generate a detailed line-by-line explanation for a single file using LLM.
    Returns both the code and explanation in a structured format.
    
    Returns:
        Dict with "code", "code_lines", and "explanation" keys
    """
    llm = get_llm()
    
    # Split code into lines for structured display
    code_lines = code.split('\n')
    
    # Use the dedicated line-by-line explanation method
    explanation = llm.generate_line_by_line_explanation(code, path, max_length=1024)
    
    # If the explanation is too short or seems incomplete, try with even more length
    if len(explanation) < 200:
        explanation = llm.generate_line_by_line_explanation(code, path, max_length=1536)
    
    # Generate dependencies
    dependencies = {"file_dependencies": [], "function_dependencies": []}
    if isinstance(llm, GeminiCodeLLM):
        dependencies = llm.generate_dependency_analysis(code, path)
    
    # Generate explanations for code blocks/chunks
    # Only do this for smaller files to avoid too many API calls
    block_explanations = []
    if len(code_lines) <= 100:  # Only generate block explanations for files with 100 or fewer lines
        block_explanations = _generate_block_explanations(code, path, llm)
    
    return {
        "code": code,
        "code_lines": code_lines,
        "explanation": explanation,
        "dependencies": dependencies,
        "block_explanations": block_explanations
    }


def _generate_block_explanations(code: str, path: str, llm) -> List[Dict]:
    """
    Break code into logical blocks (functions, classes, or code sections) 
    and generate explanations for each block.
    This helps provide more granular line-by-line explanations.
    
    Returns:
        List of dicts with "start_line", "end_line", "code_block", and "explanation"
    """
    code_lines = code.split('\n')
    blocks = []
    
    # Simple approach: split by empty lines or function/class definitions
    current_block_lines = []
    current_start = 0
    
    for i, line in enumerate(code_lines):
        stripped = line.strip()
        
        # Detect new block: function/class definition (except first one)
        is_new_block = (
            (stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("export ") or stripped.startswith("function "))
            and current_block_lines
        )
        
        # If we hit a new block and have content, save previous block
        if is_new_block and current_block_lines:
            block_code = '\n'.join(current_block_lines)
            if block_code.strip():
                try:
                    block_explanation = llm.generate_line_by_line_explanation(
                        block_code[:1000],  # Limit block size to avoid token limits
                        f"{path} (lines {current_start+1}-{i})",
                        max_length=300
                    )
                    blocks.append({
                        "start_line": current_start + 1,
                        "end_line": i,
                        "code_block": block_code,
                        "explanation": block_explanation
                    })
                except Exception as e:
                    print(f"Warning: Failed to generate block explanation: {e}")
            
            # Start new block
            current_block_lines = [line]
            current_start = i
        else:
            current_block_lines.append(line)
    
    # Handle last block
    if current_block_lines:
        block_code = '\n'.join(current_block_lines)
        if block_code.strip():
            try:
                block_explanation = llm.generate_line_by_line_explanation(
                    block_code[:1000],
                    f"{path} (lines {current_start+1}-{len(code_lines)})",
                    max_length=300
                )
                blocks.append({
                    "start_line": current_start + 1,
                    "end_line": len(code_lines),
                    "code_block": block_code,
                    "explanation": block_explanation
                })
            except Exception as e:
                print(f"Warning: Failed to generate block explanation: {e}")
    
    # If file is too simple (no blocks detected), return empty list
    # The overall explanation will be sufficient
    return blocks

