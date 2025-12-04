from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="deep_research",
    version="0.1.0",
    author="RepoMaster Team",
    description="Deep web research agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pyautogen",
        "tiktoken",
        "aiohttp",
        "requests",
        "beautifulsoup4",
        "typing_extensions",
    ],
)

