"""
Data Models for Search Agent

This module defines the core data structures used by the Search Agent:
- Idea: Input structure representing a research idea
- SearchQuery: Structured queries for different platforms
- Source: Individual search result (paper, repo, webpage)
- SearchResults: Aggregated results from all platforms
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid
from enum import Enum


class SourceType(Enum):
    """Type of source"""
    PAPER = "paper"
    CODE = "code"
    WEBPAGE = "webpage"
    SCHOLAR = "scholar"


class Platform(Enum):
    """Search platform"""
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    GITHUB = "github"
    GOOGLE_SEARCH = "google_search"
    GOOGLE_SCHOLAR = "google_scholar"
    KAGGLE = "kaggle"


@dataclass
class Idea:
    """
    Represents a research idea with all its components.

    Attributes:
        motivation: The motivation behind the research
        research_question: The main research question
        method: Proposed methodology
        experimental_setting: Experimental setup description
        expected_results: Expected outcomes (optional)
        raw_text: Original full text of the idea (optional)
        part: List of fields to include when generating full text. 
              Valid values: 'motivation', 'research_question', 'method', 'experimental_setting'.
              If None or empty, includes all fields.
    """
    motivation: str
    research_question: str
    method: str
    experimental_setting: str
    expected_results: Optional[str] = None
    raw_text: Optional[str] = None
    part: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "motivation": self.motivation,
            "research_question": self.research_question,
            "method": self.method,
            "experimental_setting": self.experimental_setting,
            "expected_results": self.expected_results,
            "raw_text": self.raw_text,
            "part": self.part
        }

    def get_full_text(self) -> str:
        """Get full text representation of the idea"""
        # If part is specified, use partial selection logic even if raw_text exists
        if self.part:
            valid_parts = {'motivation', 'research_question', 'method', 'experimental_setting'}
            parts = []
            field_map = {
                'motivation': ('Motivation', self.motivation),
                'research_question': ('Research Question', self.research_question),
                'method': ('Method', self.method),
                'experimental_setting': ('Experimental Setting', self.experimental_setting),
            }
            for p in self.part:
                if p in valid_parts and p in field_map:
                    label, content = field_map[p]
                    if content:  # Only add if content is not empty
                        parts.append(f"{label}: {content}")
            
            if self.expected_results and 'expected_results' in self.part:
                parts.append(f"Expected Results: {self.expected_results}")
            
            # If we have parts from structured fields, return them
            if parts:
                return "\n\n".join(parts)
            # If part is specified but no structured fields, fall back to raw_text if available
            if self.raw_text:
                return self.raw_text
            return ""
        
        # If no part is specified, use raw_text if available, otherwise build from structured fields
        if self.raw_text:
            return self.raw_text

        # Default: include all fields
        parts = []
        if self.motivation:
            parts.append(f"Motivation: {self.motivation}")
        if self.research_question:
            parts.append(f"Research Question: {self.research_question}")
        if self.method:
            parts.append(f"Method: {self.method}")
        if self.experimental_setting:
            parts.append(f"Experimental Setting: {self.experimental_setting}")

        if self.expected_results:
            parts.append(f"Expected Results: {self.expected_results}")

        return "\n\n".join(parts) if parts else ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Idea":
        return cls(
            motivation=data.get("motivation", ""),
            research_question=data.get("research_question", ""),
            method=data.get("method", ""),
            experimental_setting=data.get("experimental_setting", ""),
            expected_results=data.get("expected_results"),
            raw_text=data.get("raw_text"),
            part=data.get("part"),
        )


@dataclass
class SearchQuery:
    """
    Represents structured queries for different platforms.

    Attributes:
        paper_queries: Queries for academic paper search (arXiv, Semantic Scholar, PubMed)
        code_queries: Queries for code repository search (GitHub)
        web_queries: Queries for web search (Google)
        scholar_queries: Queries for Google Scholar
    """
    paper_queries: List[str] = field(default_factory=list)
    github_queries: List[str] = field(default_factory=list)
    kaggle_queries: List[str] = field(default_factory=list)
    web_queries: List[str] = field(default_factory=list)
    scholar_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary"""
        return {
            "paper_queries": self.paper_queries,
            "github_queries": self.github_queries,
            "kaggle_queries": self.kaggle_queries,
            "web_queries": self.web_queries,
            "scholar_queries": self.scholar_queries
        }

    def get_all_queries(self) -> List[str]:
        """Get all queries as a flat list"""
        return (self.paper_queries + self.github_queries + self.kaggle_queries +
                self.web_queries + self.scholar_queries)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchQuery":
        return cls(
            paper_queries=data.get("paper_queries", []),
            github_queries=data.get("github_queries", []),
            kaggle_queries=data.get("kaggle_queries", []),
            web_queries=data.get("web_queries", []),
            scholar_queries=data.get("scholar_queries", []),
        )


@dataclass
class Source:
    """
    Represents a single search result source.

    Attributes:
        title: Title of the source
        url: URL or link to the source
        source_type: Type of source (paper, code, webpage)
        platform: Platform where the source was found
        description: Brief description or abstract
        authors: List of authors (for papers)
        year: Publication year (for papers)
        citations: Citation count (for papers)
        metadata: Additional metadata
    """
    title: str
    url: str
    source_type: SourceType
    platform: Platform
    description: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    citations: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    normalized_title: Optional[str] = None
    doi: Optional[str] = None
    pdf_url: Optional[str] = None
    timestamp: Optional[str] = None
    kaggle_ref: Optional[str] = None
    kaggle_item_type: Optional[str] = None
    kaggle_subtitle: Optional[str] = None
    arxiv_id: Optional[str] = None
    s2_tldr: Optional[str] = None
    s2_open_access_pdf: Optional[str] = None
    web_source_name: Optional[str] = None
    web_rank: Optional[int] = None
    scholar_publication_info: Optional[str] = None
    scholar_rank: Optional[int] = None
    scholar_cited_by: Optional[int] = None
    page_title: Optional[str] = None
    page_headings: Optional[List[str]] = None
    page_links: Optional[List[str]] = None
    page_raw_text: Optional[str] = None
    page_structured_summary: Optional[str] = None
    page_structured_evidence: Optional[str] = None
    page_structured_rational: Optional[str] = None
    page_extractor_model: Optional[str] = None
    page_extractor_goal: Optional[str] = None
    page_read_source: Optional[str] = None
    file_tree: Optional[str] = None
    raw_api: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.title and not self.normalized_title:
            self.normalized_title = ''.join(self.title.lower().split())
        if not self.id:
            base = f"{self.platform.value}:{self.url or ''}:{self.normalized_title or self.title}"
            self.id = str(uuid.uuid5(uuid.NAMESPACE_URL, base))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type.value,
            "platform": self.platform.value,
            "description": self.description,
            "authors": self.authors,
            "year": self.year,
            "citations": self.citations,
            "metadata": self.metadata,
            "id": self.id,
            "normalized_title": self.normalized_title,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "timestamp": self.timestamp,
            "kaggle_ref": self.kaggle_ref,
            "kaggle_item_type": self.kaggle_item_type,
            "kaggle_subtitle": self.kaggle_subtitle,
            "arxiv_id": self.arxiv_id,
            "s2_tldr": self.s2_tldr,
            "s2_open_access_pdf": self.s2_open_access_pdf,
            "web_source_name": self.web_source_name,
            "web_rank": self.web_rank,
            "scholar_publication_info": self.scholar_publication_info,
            "scholar_rank": self.scholar_rank,
            "scholar_cited_by": self.scholar_cited_by,
            "page_title": self.page_title,
            "page_headings": self.page_headings,
            "page_links": self.page_links,
            "page_raw_text": self.page_raw_text,
            "page_structured_summary": self.page_structured_summary,
            "page_structured_evidence": self.page_structured_evidence,
            "page_structured_rational": self.page_structured_rational,
            "page_extractor_model": self.page_extractor_model,
            "page_extractor_goal": self.page_extractor_goal,
            "page_read_source": self.page_read_source,
            "file_tree": self.file_tree,
            "raw_api": self.raw_api,
            "extra": self.extra
        }

    def __str__(self) -> str:
        """String representation"""
        parts = [f"[{self.source_type.value.upper()}] {self.title}"]

        if self.authors:
            authors_str = ", ".join(self.authors[:3])
            if len(self.authors) > 3:
                authors_str += " et al."
            parts.append(f"Authors: {authors_str}")

        if self.year:
            parts.append(f"Year: {self.year}")

        if self.citations:
            parts.append(f"Citations: {self.citations}")

        parts.append(f"Platform: {self.platform.value}")
        parts.append(f"URL: {self.url}")

        if self.description:
            desc = self.description[:200] + "..." if len(self.description) > 200 else self.description
            parts.append(f"Description: {desc}")

        return "\n".join(parts)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            source_type=SourceType(data.get("source_type", SourceType.WEBPAGE.value)),
            platform=Platform(data.get("platform", Platform.GOOGLE_SEARCH.value)),
            description=data.get("description"),
            authors=data.get("authors"),
            year=data.get("year"),
            citations=data.get("citations"),
            metadata=data.get("metadata", {}),
            id=data.get("id"),
            normalized_title=data.get("normalized_title"),
            doi=data.get("doi"),
            pdf_url=data.get("pdf_url"),
            timestamp=data.get("timestamp"),
            kaggle_ref=data.get("kaggle_ref"),
            kaggle_item_type=data.get("kaggle_item_type"),
            kaggle_subtitle=data.get("kaggle_subtitle"),
            arxiv_id=data.get("arxiv_id"),
            s2_tldr=data.get("s2_tldr"),
            s2_open_access_pdf=data.get("s2_open_access_pdf"),
            web_source_name=data.get("web_source_name"),
            web_rank=data.get("web_rank"),
            scholar_publication_info=data.get("scholar_publication_info"),
            scholar_rank=data.get("scholar_rank"),
            scholar_cited_by=data.get("scholar_cited_by"),
            page_title=data.get("page_title"),
            page_headings=data.get("page_headings"),
            page_links=data.get("page_links"),
            page_raw_text=data.get("page_raw_text"),
            page_structured_summary=data.get("page_structured_summary"),
            page_structured_evidence=data.get("page_structured_evidence"),
            page_structured_rational=data.get("page_structured_rational"),
            page_extractor_model=data.get("page_extractor_model"),
            page_extractor_goal=data.get("page_extractor_goal"),
            page_read_source=data.get("page_read_source"),
            file_tree=data.get("file_tree"),
            raw_api=data.get("raw_api", {}),
            extra=data.get("extra", {}),
        )


@dataclass
class ResourceNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    depth: int = 0
    papers: List[Source] = field(default_factory=list)
    repos: List[Source] = field(default_factory=list)
    kaggle: List[Source] = field(default_factory=list)
    web: List[Source] = field(default_factory=list)
    scholar: List[Source] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    queries_used: Optional[SearchQuery] = None
    goals_by_type: Dict[str, str] = field(default_factory=dict)
    children_ids: List[str] = field(default_factory=list)

    def top_k_by_type(self, k: int = 3) -> Dict[str, List[Source]]:
        return {
            "papers": self.papers[:k],
            "repos": self.repos[:k],
            "kaggle": self.kaggle[:k],
            "web": self.web[:k],
            "scholar": self.scholar[:k],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "papers": [s.to_dict() for s in self.papers],
            "repos": [s.to_dict() for s in self.repos],
            "kaggle": [s.to_dict() for s in self.kaggle],
            "web": [s.to_dict() for s in self.web],
            "scholar": [s.to_dict() for s in self.scholar],
            "source_ids": self.source_ids,
            "queries_used": self.queries_used.to_dict() if self.queries_used else None,
            "goals_by_type": self.goals_by_type,
            "children_ids": self.children_ids,
        }


@dataclass
class ResourceTree:
    root_id: Optional[str] = None
    nodes: Dict[str, ResourceNode] = field(default_factory=dict)

    def add_node(self, node: ResourceNode) -> None:
        self.nodes[node.id] = node
        if self.root_id is None:
            self.root_id = node.id

    def link_child(self, parent_id: str, child_id: str) -> None:
        parent = self.nodes.get(parent_id)
        child = self.nodes.get(child_id)
        if parent and child:
            if child_id not in parent.children_ids:
                parent.children_ids.append(child_id)
            child.parent_id = parent_id

    def get_node(self, node_id: str) -> Optional[ResourceNode]:
        return self.nodes.get(node_id)

    def all_sources(self) -> List[Source]:
        out: List[Source] = []
        for n in self.nodes.values():
            out.extend(n.papers)
            out.extend(n.repos)
            out.extend(n.kaggle)
            out.extend(n.web)
            out.extend(n.scholar)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }


@dataclass
class SearchResults:
    """
    Aggregated search results from all platforms.

    Attributes:
        idea: The original idea that was searched
        queries: The queries that were generated
        papers: List of paper sources
        github_repos: List of GitHub repository sources
        kaggle_results: List of Kaggle datasets/notebooks
        web_pages: List of web page sources
        scholar_results: List of Google Scholar results
        total_count: Total number of sources found
    """
    idea: Idea
    queries: SearchQuery
    papers: List[Source] = field(default_factory=list)
    github_repos: List[Source] = field(default_factory=list)
    kaggle_results: List[Source] = field(default_factory=list)
    web_pages: List[Source] = field(default_factory=list)
    scholar_results: List[Source] = field(default_factory=list)
    resource_tree: Optional[ResourceTree] = None

    @property
    def total_count(self) -> int:
        """Get total number of sources"""
        return (len(self.papers) + len(self.github_repos) + len(self.kaggle_results) +
                len(self.web_pages) + len(self.scholar_results))

    def get_all_sources(self) -> List[Source]:
        """Get all sources as a flat list"""
        return (self.papers + self.github_repos + self.kaggle_results +
                self.web_pages + self.scholar_results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "idea": self.idea.to_dict(),
            "queries": self.queries.to_dict(),
            "papers": [s.to_dict() for s in self.papers],
            "github_repos": [s.to_dict() for s in self.github_repos],
            "kaggle_results": [s.to_dict() for s in self.kaggle_results],
            "web_pages": [s.to_dict() for s in self.web_pages],
            "scholar_results": [s.to_dict() for s in self.scholar_results],
            "total_count": self.total_count,
            "resource_tree": self.resource_tree.to_dict() if self.resource_tree else None,
        }

    def summary(self) -> str:
        """Get a summary of the search results"""
        lines = [
            "=" * 80,
            "SEARCH RESULTS SUMMARY",
            "=" * 80,
            f"Total Sources Found: {self.total_count}",
            f"  - Papers: {len(self.papers)}",
            f"  - GitHub Repositories: {len(self.github_repos)}",
            f"  - Kaggle Results: {len(self.kaggle_results)}",
            f"  - Web Pages: {len(self.web_pages)}",
            f"  - Scholar Results: {len(self.scholar_results)}",
            "=" * 80
        ]
        return "\n".join(lines)

    def detailed_report(self, max_sources_per_type: int = 5) -> str:
        """Generate a detailed report of the search results"""
        lines = [self.summary(), ""]

        # Papers
        if self.papers:
            lines.append("\n" + "=" * 80)
            lines.append("PAPERS")
            lines.append("=" * 80)
            for i, paper in enumerate(self.papers[:max_sources_per_type], 1):
                lines.append(f"\n{i}. {paper}")
                lines.append("-" * 80)

        # GitHub Repositories
        if self.github_repos:
            lines.append("\n" + "=" * 80)
            lines.append("GITHUB REPOSITORIES")
            lines.append("=" * 80)
            for i, repo in enumerate(self.github_repos[:max_sources_per_type], 1):
                lines.append(f"\n{i}. {repo}")
                lines.append("-" * 80)

        # Kaggle Results
        if self.kaggle_results:
            lines.append("\n" + "=" * 80)
            lines.append("KAGGLE RESULTS")
            lines.append("=" * 80)
            for i, item in enumerate(self.kaggle_results[:max_sources_per_type], 1):
                lines.append(f"\n{i}. {item}")
                lines.append("-" * 80)

        # Web Pages
        if self.web_pages:
            lines.append("\n" + "=" * 80)
            lines.append("WEB PAGES")
            lines.append("=" * 80)
            for i, page in enumerate(self.web_pages[:max_sources_per_type], 1):
                lines.append(f"\n{i}. {page}")
                lines.append("-" * 80)

        # Scholar Results
        if self.scholar_results:
            lines.append("\n" + "=" * 80)
            lines.append("SCHOLAR RESULTS")
            lines.append("=" * 80)
            for i, result in enumerate(self.scholar_results[:max_sources_per_type], 1):
                lines.append(f"\n{i}. {result}")
                lines.append("-" * 80)

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResults":
        return cls(
            idea=Idea.from_dict(data.get("idea", {})),
            queries=SearchQuery.from_dict(data.get("queries", {})),
            papers=[Source.from_dict(s) for s in data.get("papers", [])],
            github_repos=[Source.from_dict(s) for s in data.get("github_repos", [])],
            kaggle_results=[Source.from_dict(s) for s in data.get("kaggle_results", [])],
            web_pages=[Source.from_dict(s) for s in data.get("web_pages", [])],
            scholar_results=[Source.from_dict(s) for s in data.get("scholar_results", [])],
            resource_tree=None,
        )

