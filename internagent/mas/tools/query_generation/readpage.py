import os
import re
import json
import time
import logging
from typing import Dict, List, Optional

import requests
import tiktoken
import dspy

logger = logging.getLogger(__name__)


def _truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def read_with_jina(url: str, timeout: int = 50, retries: int = 3) -> str:
    api_key = os.getenv("JINA_API_KEY") or os.getenv("JINA_API_KEYS", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
            if resp.status_code == 200:
                text = resp.text or ""
                if text.strip():
                    return text
                return "[readpage] Empty content."
            last_err = f"status={resp.status_code} body={resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.5)
    return f"[readpage] Failed to read page: {last_err}"


class ExtractSignature(dspy.Signature):
    webpage_content = dspy.InputField(desc="Full webpage text or markdown contents")
    goal = dspy.InputField(desc="User's reading goal for extraction")
    rational = dspy.OutputField(desc="Locate goal-relevant sections/data from the content")
    evidence = dspy.OutputField(desc="Most relevant information with original context, multi-paragraph allowed")
    summary = dspy.OutputField(desc="Concise, logical summary with contribution to the goal")


def extract_structured(content: str, goal: str, max_retries: int = 2) -> Dict[str, str]:
    # Load LLM configuration from environment variables
    ds_api_key = os.getenv("DS_API_KEY")
    if ds_api_key:
        lm = dspy.LM(
            model="openai/deepseek-v3",
            api_key=ds_api_key,
            api_base=os.getenv("DS_API_BASE_URL")
        )
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY.")
        lm = dspy.LM(
            model="openai/gpt-4o-mini",
            api_key=openai_api_key,
            api_base=os.getenv("OPENAI_API_BASE_URL")
        )
    
    extractor = dspy.ChainOfThought(ExtractSignature)
    truncated = _truncate_to_tokens(content, max_tokens=95000)
    last = None
    for attempt in range(max_retries + 1):
        try:
            with dspy.settings.context(lm=lm):
                res = extractor(webpage_content=truncated, goal=goal)
            out = {
                "rational": getattr(res, "rational", "") or "",
                "evidence": getattr(res, "evidence", "") or "",
                "summary": getattr(res, "summary", "") or "",
            }
            if len(out["summary"].strip()) > 0 or len(out["evidence"].strip()) > 0:
                return out
        except Exception as e:
            last = str(e)
    return {"rational": "", "evidence": "", "summary": f"[readpage] structured extraction failed: {last}"}


def parse_markdown_metadata(text: str) -> Dict[str, Optional[str]]:
    title = None
    headings: List[str] = []
    links: List[str] = []

    for line in text.splitlines():
        if not title:
            m = re.match(r"^#\s+(.*)$", line.strip())
            if m:
                title = m.group(1).strip()
        hm = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if hm:
            headings.append(hm.group(2).strip())
        for url in re.findall(r"https?://[^\s)]+", line):
            links.append(url)
        for mlink in re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", line):
            links.append(mlink)

    if not title:
        mt = re.search(r"^\s*title\s*:\s*(.*)$", text, re.IGNORECASE | re.MULTILINE)
        if mt:
            title = mt.group(1).strip()

    return {
        "title": title,
        "headings": headings,
        "links": links,
    }


def read_page(url: str, goal: Optional[str] = None, mode: str = "raw+structured") -> Dict[str, object]:
    logger.info(f"read_page: Starting to read page - URL: {url}, goal: '{goal[:50] if goal else None}...', mode: {mode}")
    
    raw = read_with_jina(url)
    if raw.startswith("[readpage] Failed"):
        logger.error(f"read_page: Failed to read page from Jina - URL: {url}, error: {raw}")
        return {"url": url, "error": raw, "metadata": {"source": "jina"}}
    
    logger.info(f"read_page: Successfully retrieved raw content, length: {len(raw)} characters for URL: {url}")

    md = parse_markdown_metadata(raw)
    logger.info(f"read_page: Parsed metadata - title: '{md.get('title')}', headings: {len(md.get('headings', []))}, links: {len(md.get('links', []))} for URL: {url}")

    result: Dict[str, object] = {
        "url": url,
        "raw_text": raw,
        "metadata": md,
    }

    if goal and ("structured" in mode):
        structured = extract_structured(raw, goal)
        result["structured"] = structured
        logger.info(f"read_page: Completed structured extraction for URL: {url} - summary length: {len(structured.get('summary', ''))}, evidence length: {len(structured.get('evidence', ''))}")
    else:
        logger.info(f"read_page: Skipping structured extraction - goal present: {bool(goal)}, structured in mode: {'structured' in mode} for URL: {url}")

    logger.info(f"read_page: Successfully completed reading page for URL: {url}")
    return result

