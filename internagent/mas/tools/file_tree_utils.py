"""
Utility functions for generating file trees from GitHub repositories.
"""

import tempfile
import subprocess
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# File extensions for code files when generating repository trees
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.kt', '.scala', '.groovy',
    '.cpp', '.c', '.h', '.hpp', '.cc', '.cxx', '.go', '.rs', '.rb', '.php',
    '.html', '.htm', '.css', '.scss', '.less', '.json', '.yaml', '.yml', '.toml',
    '.md', '.rst', '.txt', '.sh', '.bash', '.ps1', '.bat', '.cmd',
    '.sql', '.proto', '.gradle', '.xml', '.csv', '.ipynb', '.r', '.jl',
    '.swift', '.m', '.mm', '.cs', '.fs', '.vb', '.lua', '.pl', '.pm',
    '.hs', '.lhs', '.clj', '.cljs', '.cljc', '.edn', '.ex', '.exs',
    '.erl', '.hrl', '.dockerfile', '.makefile', '.cmake', '.toml', '.ini',
    '.cfg', '.conf', '.properties', '.env', '.gitignore', '.gitattributes',
    '.yml', '.yaml', '.lock', '.tf', '.tfvars', '.hcl', '.sol'
}


def generate_tree(dir_path: Path, prefix: str = '', level: int = -1) -> str:
    """
    Generate a filtered directory tree as a string.
    
    Args:
        dir_path: Path to the directory
        prefix: Prefix for tree visualization
        level: Maximum depth to traverse (-1 for unlimited)
        
    Returns:
        Formatted tree string
    """
    if level == 0:
        return ''

    # Get contents
    contents = []
    for path in dir_path.iterdir():
        if path.is_dir():
            if path.name == '.git':
                continue
            sub_tree = generate_tree(path, '', level - 1 if level > 0 else -1)
            if sub_tree:  # Only include if subtree is not empty
                contents.append((path, sub_tree))
        else:
            if path.suffix.lower() in CODE_EXTENSIONS:
                contents.append((path, None))

    if not contents:
        return ''  # Prune empty directory

    tree_str = ''
    last_index = len(contents) - 1
    for i, (path, sub_tree) in enumerate(contents):
        pointer = '├── ' if i < last_index else '└── '
        tree_str += prefix + pointer + path.name + '\n'
        if sub_tree is not None:
            extension = '│   ' if i < last_index else '    '
            if sub_tree:
                lines = sub_tree.split('\n')
                if lines and lines[-1] == '':
                    lines = lines[:-1]
                # Add prefix extension to all subtree lines
                indented = [prefix + extension + line for line in lines]
                tree_str += '\n'.join(indented) + '\n'
    return tree_str


def get_repo_tree(url: str, max_level: int = -1) -> str:
    """
    Clone a GitHub repo to a temp dir and return its directory tree as a string.
    
    Args:
        url: GitHub repository URL
        max_level: Maximum depth to traverse (-1 for unlimited)
        
    Returns:
        Formatted tree string or error message
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"[Git Clone] Starting clone: {url}")
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                result = subprocess.run(
                    ['git', 'clone', '--depth', '1', '--progress', url, temp_dir],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60  # 60 second timeout
                )
                logger.info(f"[Git Clone] Clone completed: {url}")
                cloned_path = Path(temp_dir)
                tree = generate_tree(cloned_path, level=max_level)
                return tree
            except subprocess.CalledProcessError as e:
                err_msg = e.stderr if e.stderr else str(e)
                if attempt < max_retries:
                    logger.warning(f"[Git Clone] Clone failed (attempt {attempt}), retrying: {err_msg}")
                    # Clean up temp directory
                    for item in Path(temp_dir).iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    continue
                else:
                    logger.error(f"[Git Clone] Failed after {max_retries} attempts: {err_msg}")
                    return f"Error cloning repository after {max_retries} attempts: {err_msg}"
            except subprocess.TimeoutExpired:
                logger.error(f"[Git Clone] Timeout cloning {url}")
                return f"Error: Git clone timeout after 60 seconds"
            except Exception as e:
                logger.error(f"[Git Clone] Unexpected error: {str(e)}")
                return f"Unexpected error: {str(e)}"
    
    return "Error: Unknown error occurred"

