"""
Filtering utilities for search results using dspy.
"""

import json
import random
import os
import logging
from typing import List, Dict, Any

import dspy

from .searchers.models import SearchResults, Source

logger = logging.getLogger(__name__)


class FilterSignature(dspy.Signature):
    """Signature for filtering search results by relevance."""
    idea = dspy.InputField(
        desc="Structured research idea text with sections labeled exactly: Motivation:, Research Question:, "
             "Method:, Experimental Setting:, and optionally Expected Results:. Content may be multi-paragraph. "
             "Treat this as the authoritative description of the idea's motivation and methodology."
    )
    source_context = dspy.InputField(
        desc="Up to six sources from one platform chunk. Each line uses: index. title | summary. "
             "The summary is an extract of the source page or description. Use only these lines to judge relevance; "
             "do not invent content."
    )
    score_jsonl = dspy.OutputField(
        desc='Return a pure JSON array aligned to input order. Each element is {"score": <integer 0–10>, '
             '"reason": <1–2 sentences explaining the score based on direct relevance>. '
             'Array length must match the number of input lines. No extra text before or after the JSON.'
    )


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Extract JSON array from text."""
    try:
        s = text.strip()
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end+1])
        return []
    except Exception:
        return []


def _load_lm_from_env():
    """Load dspy LM from environment variables."""
    ds_api_key = os.getenv("DS_API_KEY")
    if ds_api_key:
        return dspy.LM(
            model="openai/deepseek-v3",
            api_key=ds_api_key,
            api_base=os.getenv("DS_API_BASE_URL")
        )
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return dspy.LM(
            model="openai/gpt-4o-mini",
            api_key=openai_api_key,
            api_base=os.getenv("OPENAI_API_BASE_URL")
        )
    
    raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY.")


def filter_search_results(sr: SearchResults, 
                          top_k_papers: int = 10,
                          top_k_code: int = 6,
                          top_k_web: int = 10,
                          batch_size: int = 10) -> SearchResults:
    """
    Filter search results by relevance to the research idea.
    
    Args:
        sr: SearchResults object to filter
        top_k_papers: Number of top papers/scholar results to keep
        top_k_code: Number of top code results (GitHub/Kaggle) to keep
        top_k_web: Number of top web results to keep
        batch_size: Batch size for processing
        
    Returns:
        Filtered SearchResults object
    """
    if sr is None:
        raise ValueError("SearchResults cannot be None")
    
    idea_text = sr.idea.get_full_text()
    
    # Load LM
    lm = _load_lm_from_env()
    
    # Create filter generators with different system prompts
    gen_paper = dspy.ChainOfThought(FilterSignature)
    gen_paper.system_prompt = (
        "Task: For each input line, assign a relevance score 0–10 to the idea and give a brief reason. "
        "High (8–10): strong alignment in motivation, methodology, problem framing, and experimental setting; "
        "provides evidence or directly supports the proposed method. "
        "Medium (4–7): partial alignment or similar domain with some methodological overlap. "
        "Low (0–3): superficial topical overlap or different goals/methods. "
        "Use only the title and summary. Output a JSON array aligned to input order; each element is {score, reason}."
    )
    
    gen_web = dspy.ChainOfThought(FilterSignature)
    gen_web.system_prompt = (
        "Task: For each input line, assign a relevance score 0–10 to the idea and give a brief reason. "
        "High (8–10): arguments/analysis/engineering notes clearly align with the idea's motivation and "
        "methodological approach, offering actionable insight or evidence. "
        "Medium (4–7): partial thematic alignment with limited methodological fit. "
        "Low (0–3): loose topical relation or opinion with no methodological relevance. "
        "Use only the title and summary. Output a JSON array aligned to input order; each element is {score, reason}."
    )
    
    gen_code = dspy.ChainOfThought(FilterSignature)
    gen_code.system_prompt = (
        "Task: For each input line, assign an implementation utility score 0–10 to the idea and give a brief reason. "
        "High (8–10): directly enables the methodology (solver implementation, training pipeline, evaluation harness, "
        "dataset loaders, reproducible scripts). "
        "Medium (4–7): partially useful components requiring significant adaptation. "
        "Low (0–3): paper-only repos, tutorials/discussions without implementable tooling, or superficial relation. "
        "Use only the title and summary. Output a JSON array aligned to input order; each element is {score, reason}."
    )
    
    def _accumulate_scores(items: List[Source], generator: dspy.Module) -> Dict[str, Dict[str, float]]:
        """Accumulate relevance scores for sources."""
        scores: Dict[str, Dict[str, float]] = {}
        if not items:
            return scores
        
        for i in range(0, len(items), batch_size):
            chunk = items[i:i+batch_size]
            if not chunk:
                continue
            
            lines = []
            for idx, s in enumerate(chunk):
                title = (s.title or "").strip()
                summary = (s.page_structured_summary or s.description or s.page_raw_text or "").strip()
                summary = summary[:240]
                lines.append(f"{idx}. {title} | {summary}")
            
            ctx = "\n".join(lines)
            
            try:
                with dspy.settings.context(lm=lm):
                    out = generator(idea=idea_text, source_context=ctx)
                arr = _extract_json_array(getattr(out, "score_jsonl", ""))
                
                for j, s in enumerate(chunk):
                    obj = arr[j] if j < len(arr) else {}
                    try:
                        sc = float(obj.get("score", 0))
                        sc = max(0.0, min(10.0, sc))  # Clamp to [0, 10]
                    except Exception:
                        sc = 0.0
                    
                    sid = s.id
                    if sid not in scores:
                        scores[sid] = {"sum": 0.0, "cnt": 0.0}
                    scores[sid]["sum"] += sc
                    scores[sid]["cnt"] += 1.0
            except Exception as e:
                logger.warning(f"Error scoring batch: {e}")
                continue
        
        return scores
    
    # Combine and shuffle sources
    papers_scholar = list(sr.papers or []) + list(sr.scholar_results or [])
    github_kaggle = list(sr.github_repos or []) + list(sr.kaggle_results or [])
    webpages = list(sr.web_pages or [])
    random.shuffle(papers_scholar)
    random.shuffle(github_kaggle)
    random.shuffle(webpages)
    
    # Score all sources
    logger.info(f"Filtering {len(papers_scholar)} papers/scholar, {len(github_kaggle)} code, {len(webpages)} web")
    ps_scores = _accumulate_scores(papers_scholar, gen_paper)
    gk_scores = _accumulate_scores(github_kaggle, gen_code)
    web_scores = _accumulate_scores(webpages, gen_web)
    
    def _top_ids(scores: Dict[str, Dict[str, float]], k: int) -> List[str]:
        """Get top k source IDs by average score."""
        if not scores:
            return []
        rows = []
        for sid, meta in scores.items():
            cnt = meta.get("cnt", 0.0)
            if cnt > 0:
                avg = meta.get("sum", 0.0) / cnt
                rows.append((avg, cnt, sid))
        rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [r[2] for r in rows[:k]]
    
    # Get top IDs
    ps_ids = set(_top_ids(ps_scores, top_k_papers))
    gk_ids = set(_top_ids(gk_scores, top_k_code))
    web_ids = set(_top_ids(web_scores, top_k_web))
    
    # Filter sources
    fp = [s for s in sr.papers if s.id in ps_ids]
    fs = [s for s in sr.scholar_results if s.id in ps_ids]
    fr = [s for s in sr.github_repos if s.id in gk_ids]
    fk = [s for s in sr.kaggle_results if s.id in gk_ids]
    fw = [s for s in sr.web_pages if s.id in web_ids]
    
    logger.info(f"Filtered to {len(fp)} papers, {len(fs)} scholar, {len(fr)} github, "
               f"{len(fk)} kaggle, {len(fw)} web")
    
    return SearchResults(
        idea=sr.idea,
        queries=sr.queries,
        papers=fp,
        github_repos=fr,
        kaggle_results=fk,
        web_pages=fw,
        scholar_results=fs,
        resource_tree=sr.resource_tree,
    )

