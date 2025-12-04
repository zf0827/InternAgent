"""Setup configuration for repo_analysis_toolkit."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="repo_analysis_toolkit",
    version="0.1.0",
    author="RepoMaster Team",
    author_email="",
    description="A toolkit for analyzing code repositories and extracting contextual information",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx>=2.6.0",
        "tiktoken>=0.5.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "full": [
            "grep-ast>=0.3.0",
            "tree-sitter>=0.20.0",
            "tree-sitter-language-pack>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "repo-analysis=repo_analysis.cli:main",
        ],
    },
)

