"""
通用富化工具：paper 抽取、web/code 报告生成。
保持与 agent 解耦，仅暴露实际使用的异步富化函数。
"""

from .enricher import (
    enrich_papers_with_extraction,
    enrich_web_with_reports,
    enrich_code_with_rawtext,
    enrich_code_with_repo,
)

__all__ = [
    "enrich_papers_with_extraction",
    "enrich_web_with_reports",
    "enrich_code_with_rawtext",
    "enrich_code_with_repo",
]


