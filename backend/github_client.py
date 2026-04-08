"""
Simple GitHub API client to fetch repository code.
Uses the GitHub REST API with authentication token.
"""

import requests
import os
from typing import List, Dict, Optional

import json

# Get GitHub token from environment variable
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def get_default_branch(owner: str, repo: str) -> str:
    """
    Get the default branch name for a repository.
    
    Args:
        owner: Repository owner (e.g., "facebook")
        repo: Repository name (e.g., "react")
    
    Returns:
        Default branch name (e.g., "main" or "master")
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    return data["default_branch"]


def list_files(owner: str, repo: str, branch: str, path: str = "") -> List[Dict]:
    """
    List all files in a repository directory.
    Recursively gets all files, not just the top level.
    
    Args:
        owner: Repository owner
        repo: Repository name
        branch: Branch name (e.g., "main")
        path: Directory path (empty string for root)
    
    Returns:
        List of file info dicts with "path", "type", "download_url" keys
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    items = response.json()
    files = []
    
    # Define excluded directories/files
    EXCLUDED_PREFIXES = [".ipynb_checkpoints", "__pycache__", ".git", ".idea", ".vscode", "node_modules", "venv", "env"]
    
    for item in items:
        # Skip excluded items
        if any(item["name"].startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
            
        if item["type"] == "file":
            # Only include code files (simple extension check)
            code_extensions = [
                ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", 
                ".go", ".rs", ".rb", ".php", ".swift", ".m", ".mm", ".kt", ".cs",
                ".html", ".css", ".sh", ".bat", ".scala", ".lua", ".dart", ".ipynb"
            ]
            if any(item["name"].endswith(ext) for ext in code_extensions):
                files.append({
                    "path": item["path"],
                    "download_url": item["download_url"]
                })
        elif item["type"] == "dir":
            # Recursively get files from subdirectories
            subfiles = list_files(owner, repo, branch, item["path"])
            files.extend(subfiles)
    
    return files


def download_file_content(download_url: str) -> str:
    """
    Download the content of a file from GitHub.
    
    Args:
        download_url: The download URL from the GitHub API
    
    Returns:
        File content as a string
    """
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    
    response = requests.get(download_url, headers=headers)
    response.raise_for_status()
    
    return response.text


def fetch_repository_code(repo_url: str) -> List[Dict]:
    """
    Main function to fetch all code files from a GitHub repository.
    
    Args:
        repo_url: Full GitHub URL (e.g., "https://github.com/owner/repo")
    
    Returns:
        List of dicts with "path" and "code" keys
    """
    # Parse URL to get owner and repo
    # Example: "https://github.com/facebook/react" -> owner="facebook", repo="react"
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {repo_url}")
    
    repo = parts[-1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    
    owner = parts[-2]
    
    # Step 2: Get default branch
    branch = get_default_branch(owner, repo)
    
    # Step 2: List all files
    file_list = list_files(owner, repo, branch)
    
    # Step 2: Download content of each file
    files_with_content = []
    for file_info in file_list:
        try:
            content = download_file_content(file_info["download_url"])
            
            # Special handling for Jupyter Notebooks
            if file_info["path"].endswith(".ipynb"):
                try:
                    nb = json.loads(content)
                    cells = []
                    for cell in nb.get("cells", []):
                        if cell.get("cell_type") == "code":
                            source = "".join(cell.get("source", []))
                            cells.append(source)
                    # Join code cells for analysis
                    content = "\n\n# --- Jupyter Cell ---\n".join(cells)
                except Exception:
                    # If parsing fails, just use raw content (it's JSON)
                    pass
            
            files_with_content.append({
                "path": file_info["path"],
                "code": content
            })
        except Exception as e:
            # Skip files that can't be downloaded
            print(f"Warning: Could not download {file_info['path']}: {e}")
            continue
    
    return files_with_content




def fetch_single_file(repo_url: str, file_path: str) -> Optional[str]:
    """
    Fetch content of a single file from GitHub.
    
    Args:
        repo_url: Full GitHub URL
        file_path: Path to the file in the repo
        
    Returns:
        File content string or None if failed
    """
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        return None
    
    repo = parts[-1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    owner = parts[-2]
    
    try:
        # Get metadata for the file
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        headers = {}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if data["type"] == "file":
             # Use download_url to get raw content
             return download_file_content(data["download_url"])
        return None
        
    except Exception as e:
        print(f"Error fetching file {file_path}: {e}")
        return None
