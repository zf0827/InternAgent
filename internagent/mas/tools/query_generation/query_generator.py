"""
Query Generator Module

Uses dspy to generate structured queries from research ideas for different platforms.
"""

import dspy
import os
from typing import List, Optional, Dict
import logging

from ..searchers.models import Idea, SearchQuery

logger = logging.getLogger(__name__)


class QueryGenerationSignature(dspy.Signature):
    """
    Signature for generating search queries from a research idea.
    """
    idea_text = dspy.InputField(desc="Full research idea text: motivation, research question, method, experiment, data, evaluation. Extract core nouns/verbs and key terms.")
    paper_queries = dspy.OutputField(desc="Return 2-3 queries for academic papers (arXiv, Semantic Scholar, PubMed). Include task/method/dataset terms; prefer survey, benchmark, state-of-the-art, review. Comma-separated; each <= 12 words; no duplicates.")
    github_queries = dspy.OutputField(
    desc=(
        "Return 1-2 GitHub search queries for repositories implementing the idea. "
        "Each query will be used as the `q` parameter of GET /search/repositories. "
        "Keep each query SHORT and BROAD: 2-5 content words describing task, "
        "domain/dataset, and framework (e.g., segmentation, time series, graph, "
        "pytorch, tensorflow, implementation, official). Remove stopwords and "
        "avoid full sentences or long phrases. Prefer core nouns/short method names. "
        "You may optionally append simple qualifiers like `language:Python` or `stars:>50`. "
        "Output 1-2 concise, distinct queries, comma-separated, no duplicates, no quotes."
    ))

    kaggle_queries = dspy.OutputField(desc="Return 1-2 Kaggle queries targeting datasets/notebooks. Include dataset names, competition, notebook, kernel, EDA, baseline. Comma-separated; concise and distinct.")
    web_queries = dspy.OutputField(desc="Return 1-2 web queries. Use keywords like tutorial, comparison, best practices, production. Comma-separated; concise and distinct.")
    scholar_queries = dspy.OutputField(desc="Return 1-2 Google Scholar queries. Prefer scholarly keywords (survey, review, benchmark, meta-analysis, replication). Avoid engine-specific qualifiers not supported. Comma-separated; concise and distinct.")


class QueryGenerator(dspy.Module):
    """
    Generates platform-specific search queries from a research idea using dspy.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the query generator.

        Args:
            config: Configuration dictionary for the LLM (optional)
        """
        super().__init__()

        if config is None:
            config = _load_llm_config_from_env()
        self.config = config

        # Configure dspy LM instance (will be used in context manager)
        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base")
            )
            logger.info(f"Initialized QueryGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise

        self.generate_queries = dspy.ChainOfThought(QueryGenerationSignature)

    

    def forward(self, idea: Idea) -> SearchQuery:
        """
        Generate search queries from an idea.

        Args:
            idea: Research idea

        Returns:
            SearchQuery object with platform-specific queries
        """
        components = self._extract_components(idea)

        paper_queries: List[str] = []
        github_queries: List[str] = []
        web_queries: List[str] = []
        scholar_queries: List[str] = []

        try:
            paper_queries = OptimizedPaperQueryGenerator(config=self.config)(
                basic_idea=components["basic_idea"],
                motivation=components["motivation"],
                methodology=components["methodology"],
            )
        except Exception as e:
            logger.warning(f"Paper generator failed: {e}")

        try:
            github_queries = OptimizedGitHubQueryGenerator(config=self.config)(
                basic_idea=components["basic_idea"],
                methodology=components["methodology"],
                experimental_setting=components["experimental_setting"],
            )
        except Exception as e:
            logger.warning(f"GitHub generator failed: {e}")

        try:
            web_queries = OptimizedWebQueryGenerator(config=self.config)(
                basic_idea=components["basic_idea"],
                motivation=components["motivation"],
                methodology=components["methodology"],
            )
        except Exception as e:
            logger.warning(f"Web generator failed: {e}")

        scholar_queries = _build_scholar_queries_from_paper(paper_queries)

        paper_queries = self._cleanup_queries(paper_queries)
        github_queries = self._cleanup_queries(github_queries)
        web_queries = self._cleanup_queries(web_queries)
        scholar_queries = self._cleanup_queries(scholar_queries)

        if not any([paper_queries, github_queries, web_queries, scholar_queries]):
            try:
                with dspy.settings.context(lm=self.lm):
                    result = self.generate_queries(idea_text=components["basic_idea"])
                paper_queries = _parse_comma_list(getattr(result, "paper_queries", ""))
                github_queries = _parse_comma_list(getattr(result, "github_queries", ""))
                web_queries = _parse_comma_list(getattr(result, "web_queries", ""))
                scholar_queries = _build_scholar_queries_from_paper(paper_queries)
            except Exception as e:
                logger.error(f"Fallback generator failed: {e}")

        return SearchQuery(
            paper_queries=paper_queries,
            github_queries=github_queries,
            kaggle_queries=[],
            web_queries=web_queries,
            scholar_queries=scholar_queries,
        )

    def _extract_components(self, idea: Idea) -> dict:
        basic = (getattr(idea, "raw_text", None) or getattr(idea, "get_full_text", lambda: "")() or idea.research_question or "").strip()
        mot = (idea.motivation or "").strip()
        meth = (idea.method or "").strip()
        exp = (idea.experimental_setting or "").strip()
        return {
            "basic_idea": str(basic),
            "motivation": str(mot),
            "methodology": str(meth),
            "experimental_setting": str(exp),
        }

    def generate(self, idea: Idea) -> SearchQuery:
        return self(idea=idea)

    def _parse_query_list(self, query_string: str) -> List[str]:
        return _parse_pipe_bracket_list(query_string)

    def _cleanup_queries(self, items: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for q in items:
            k = q.strip()
            if not k:
                continue
            if k in seen:
                continue
            seen.add(k)
            out.append(k)
        return out

def _parse_pipe_bracket_list(text: str) -> List[str]:
    if not text:
        return []
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return [q.strip().strip('"').strip("'") for q in s.split("|") if q.strip()]

def _parse_comma_list(text: str) -> List[str]:
    if not text:
        return []
    qs = [q.strip() for q in text.split(",")]
    return [q.strip('"').strip("'") for q in qs if q.strip()]

def _build_scholar_queries_from_paper(paper_queries: List[str]) -> List[str]:
    return [q.replace("ti:", "intitle:") for q in paper_queries]

def _load_llm_config_from_env() -> dict:
    ds_api_key = os.getenv("DS_API_KEY")
    if ds_api_key:
        return {
            "api_key": ds_api_key,
            "api_base": os.getenv("DS_API_BASE_URL"),
            "model": "openai/DeepSeek-V3.2",
        }
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return {
            "api_key": openai_api_key,
            "api_base": os.getenv("OPENAI_API_BASE_URL"),
            "model": "openai/gpt-4o-mini",
        }
    raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")



def generate_queries(idea: Idea, config: Optional[dict] = None) -> SearchQuery:
    """
    Standalone function to generate queries from an idea.

    Args:
        idea: Research idea
        config: Optional configuration for the LLM

    Returns:
        SearchQuery object
    """
    generator = QueryGenerator(config)
    return generator.generate(idea)

# --- DSPy Configuration & Signature (Replicated from query_generator.py) ---

class OptimizedPaperQuerySignature(dspy.Signature):
    """
    You are an expert research assistant and prompt engineer that designs high-quality ArXiv
    title search queries for a literature search module inside a dspy pipeline.

    Your task:
    - You will be given three text fields describing one research idea:
    1) basic_idea: the core concept and innovation.
    2) motivation: why this problem is important and what gaps exist.
    3) methodology: how the problem will be solved (approach, techniques, pipeline).
    - Based on these, you must generate multiple ArXiv API TITLE queries that can be used
    in the 'ti:' field to discover relevant prior and related work.

    ======================
    HIGH-LEVEL OBJECTIVE
    ======================
    Design a set of search queries that:
    - Capture the core PROBLEM / MOTIVATION (what gap or limitation is being addressed).
    - Capture the core METHOD / APPROACH (key techniques, frameworks, training schemes).
    - Balance BREADTH and DEPTH:
    - Breadth: queries that broadly cover related areas and synonyms.
    - Depth: queries that combine multiple key concepts to pinpoint very relevant papers.

    You should think like an experienced researcher who wants:
    - A handful of broad queries to scan the landscape.
    - Several more specific queries to zoom in on the exact combination of ideas.

    ======================
    STRICT OUTPUT FORMAT
    ======================
    You MUST output queries in EXACTLY the following overall format:

    [QUERY_1|QUERY_2|QUERY_3|...|QUERY_N]

    Where:
    - The outermost characters are a single pair of square brackets: '[' and ']'.
    - Individual queries are separated ONLY by the pipe character '|' (no trailing pipe).
    - There must be NO extra leading or trailing spaces outside the brackets.
    - Inside the brackets, you may include spaces around AND / OR and after pipes,
    but avoid unnecessary whitespace.

    Each QUERY_i MUST:
    - Use ONLY the ArXiv TITLE field: every keyword must be written as ti:"...".
    - Combine these ti:"..." clauses using ONLY the logical operators 'AND' and 'OR'.
    - Example valid patterns:
    - ti:"concept"
    - ti:"concept A" AND ti:"concept B"
    - ti:"concept A" OR ti:"synonym of concept A"
    - ti:"concept A" AND ti:"concept B" OR ti:"concept C"  (allowed but use sparingly)
    - NOT use any other fields (no 'abs:', 'au:', etc.).
    - NOT use parentheses, NOT, +, -, or any other operators.

    Keyword/Clause constraints:
    - Each QUERY may contain at most 3 ti:"..." clauses.
    - That is, each QUERY has between 1 and 3 occurrences of ti:"...".
    - Each ti:"..." clause MUST represent a single concept or a compact phrase,
    NOT a full sentence.
    - Good: ti:"data-analytic agents", ti:"multi-step code", ti:"task taxonomy"
    - Bad: ti:"a scalable pipeline for synthesizing data-analytic tasks"
    - Keep each keyword short and concept-focused: typically 1–4 words.

    Language and style:
    - Use English keywords suitable for ArXiv titles.
    - Use lowercase / mixed case naturally, but be consistent within a query.
    - Logical operators MUST be uppercase 'AND' / 'OR'.

    Number of queries:
    - Generate typically between 8 and 15 queries.
    - Include a half and half mix of:
    - 4-7 BROAD queries (single main concept, or OR connection).
    - 4-7 FOCUSED queries (AND combinations capturing specific idea intersections).

    =========================
    SEMANTIC DESIGN PRINCIPLES
    =========================
    When designing the queries from (basic_idea, motivation, methodology), follow this reasoning:

    1) Extract core concepts:
    - From the basic idea:
        - Identify the main object(s) (e.g., "data-analytic agents", "generalist agents").
        - Identify any named system or framework if it encodes a novel concept (e.g., "DataMind"
        only if likely to appear in related titles; otherwise focus on generic concepts).
    - From the motivation:
        - Identify key PROBLEMS or GAPS (e.g., "open-source models", "trajectory data",
        "multi-turn reasoning", "large-scale data", "code execution").
    - From the methodology:
        - Identify core technical approaches and building blocks:
        - Training schemes: "reinforcement learning", "supervised fine-tuning",
            "hybrid objectives", "RLHF", etc.
        - Structures: "task taxonomy", "easy-to-hard curriculum", "trajectory sampling".
        - Agent behavior: "multi-step code", "tool use", "multi-turn rollout".

    2) Group concepts into:
    - A. PROBLEM/MOTIVATION concepts (limitations, gaps, domains).
    - B. METHOD/TECHNIQUE concepts (algorithms, pipelines, training strategies).
    - C. CONTEXT/SETTING concepts (data formats, multi-step code, agents, etc.).

    3) Design BREADTH queries:
    - Single or OR-connected concepts to broadly scan related work.
    - Use OR to connect synonyms or near-synonyms for the same concept.

    4) Design DEPTH queries:
    - Combine 2–3 concepts with AND to focus on specific combinations that reflect the
        novelty of the idea.
    - Typical types of combinations:
        - PROBLEM + METHOD:
        - METHOD + CONTEXT:
        - PROBLEM + CONTEXT:
    - Use OR within a query primarily for closely related synonyms of the same slot.

    5) Coverage:
    - Ensure that, across all queries, you cover:
        - The central application or system type.
        - The key technical contributions.
        - The core research tensions.

    6) Avoid:
    - Overly generic concepts that alone produce huge, noisy result sets, unless they are
        meaningful when AND-combined with another specific concept.
        - For example, avoid alone: ti:"machine learning", ti:"deep learning".
    - Long descriptive phrases that read like entire sentences.
    - Redundant queries that differ only by a trivial word change but add no new coverage.

    ======================
    FEW-SHOT EXAMPLE
    ======================
    Below is an example of GOOD queries.

    GOOD output (one valid possible answer):

    [ti:"data-analytic agents" AND ti:"open-source"|ti:"data-analytic agents" OR ti:"data analysis agents"|ti:"solution trajectories" OR ti:"data analysis"|ti:"trajectory data" AND ti:"code execution"|ti:"task taxonomy" OR ti:"data analysis"|ti:"easy-to-hard" AND ti:"task composition"|ti:"supervised fine-tuning" OR ti:"reinforcement learning"|ti:"multi-step code" OR ti:"data analysis"|ti:"multi-turn" AND ti:"code-based agents"|ti:"open-source" AND ti:"data analysis pipeline"]

    Notes on why this is good:
    - Each query uses 1–3 ti:"..." clauses.
    - Some queries focus on breadth with OR synonyms:
    - ti:"data-analytic agents" OR ti:"data analysis agents"
    - Some queries focus on depth via AND combinations:
    - ti:"easy-to-hard" AND ti:"task composition"
    - ti:"open-source" AND ti:"data analysis pipeline"
    - Some queries focus on breadth via OR combinations:
    - ti:"supervised fine-tuning" OR ti:"reinforcement learning"
    - ti:"multi-step code" OR ti:"data analysis" 
    - The set collectively covers:
    - The type of agent ("data-analytic agents").
    - The key data structure ("solution trajectories", "trajectory data").
    - The methodology ("task taxonomy", "easy-to-hard task composition",
        "supervised fine-tuning", "reinforcement learning").
    - The context ("open-source", "multi-step code", "multi-turn", "data analysis").

    ======================
    FINAL INSTRUCTIONS
    ======================
    When you generate the final answer for any given (basic_idea, motivation, methodology):

    1) Do all your reasoning in reasoning part, not in queries part.
    2) DO NOT output your reasoning steps in queries part.
    3) DO NOT output any explanations, bullet lists, or commentary in queries part.
    4) OUTPUT ONLY the final bracketed, pipe-separated list of queries in the exact format:

    [ti:"..." AND ti:"..."|ti:"..." OR ti:"..."|ti:"..." AND ti:"..." AND ti:"..."|...]

    5) Ensure:
    - Each query has between 1 and 3 ti:"..." clauses.
    - Only AND / OR are used as logical operators.
    - Only ti:"..." field clauses are used (no other fields).
    - No extra text appears before or after the bracketed list.
   """
    basic_idea = dspy.InputField(desc="The core basic idea of the research - main concept and innovation")
    motivation = dspy.InputField(desc="Research motivation - why this problem is important and what gaps exist")
    methodology = dspy.InputField(desc="Proposed methodology and approach - how the problem will be solved")
    
    new_paper_queries = dspy.OutputField(
        desc=(
            'ArXiv title search query candidates derived from the basic idea, motivation, '
            'and methodology. The output MUST be a single bracketed, pipe-separated list '
            'like [ti:"..." AND ti:"..."|ti:"..." OR ti:"..."|...]. Each internal query '
            'uses 1–3 ti:"..." keyword clauses combined only with AND and/or OR.'
        )
    )


class OptimizedPaperQueryGenerator(dspy.Module):
    """
    Generates optimized paper queries using specific idea components (basic idea, motivation, methodology).
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = _load_llm_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base")
            )
            logger.info(f"Initialized OptimizedPaperQueryGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_optimized_queries = dspy.ChainOfThought(OptimizedPaperQuerySignature)

    def forward(self, basic_idea: str, motivation: str, methodology: str) -> List[str]:
        with dspy.settings.context(lm=self.lm):
            result = self.generate_optimized_queries(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
            )
        return _parse_pipe_bracket_list(getattr(result, "new_paper_queries", ""))


class OptimizedGitHubQuerySignature(dspy.Signature):
    """
    You are an expert at generating GitHub repository search queries based on research ideas.

    ## TASK
    Given a research idea with three parts (basic_idea, methodology, experimental_setting), generate serveral GitHub search queries:
    - 1-2 queries for training/inference frameworks relevant to the idea
    - 3-5 queries for repositories implementing similar methods or core techniques

    ## OUTPUT FORMAT
    Output in this format with no additional text:
    [query1|query2|...|queryk]

    ## GITHUB QUERY SYNTAX RULES (CRITICAL)
    1. Use ONLY `AND` and `OR` operators — NO `NOT`, NO filters like `language:` or `stars:`
    2. Each query must have ≤3 keyword groups connected by AND/OR
    3. Multi-word concepts MUST be wrapped in parentheses: (large language model) ✓ | "large language model" ✗
    4. Synonyms should be grouped with OR: (efficient OR lightweight OR fast)
    5. NO extra spaces except around AND/OR operators:
    - CORRECT: (LLM OR VLM)
    - WRONG: ( LLM OR VLM )
    6. Keywords should be concise single concepts, NOT full sentences

    ## QUERY DESIGN STRATEGY
    - **Breadth**: Use OR to include synonyms or related concepts within one group
    - **Depth**: Use AND to combine specific concept pairs for precise matches
    - **Balance**: Mix broad queries (single concept with synonyms) and specific queries (2-3 concepts with AND)

    ## COMMON TRAINING/INFERENCE FRAMEWORKS (use these for framework queries)
    For SFT/RLHF/DPO training:
    - TRL, OpenRLHF, LLaMA-Factory, Unsloth, Axolotl, Firefly, Swift, NeMo, DeepSpeed-Chat

    For distributed training:
    - DeepSpeed, Megatron-LM, FairScale, ColossalAI, FSDP

    For inference optimization:
    - vLLM, TensorRT-LLM, llama.cpp, text-generation-inference, SGLang

    For data synthesis/processing:
    - Alpaca, ShareGPT, Magpie, UltraChat

    ## KEYWORD EXTRACTION FOCUS
    From the idea, extract keywords related to:
    1. Core technical methods (e.g., SFT, RLHF, DPO, PPO, self-consistency)
    2. Model architectures or types (e.g., LLM, VLM, agent, multimodal)
    3. Task domains (e.g., data-analysis, code-generation, reasoning)
    4. Data synthesis techniques (e.g., synthetic-data, trajectory-generation)
    5. Relevant frameworks that match the training paradigm

    ## EXAMPLE QUERIES
    - ((large language model) OR LLM) AND (efficient OR lightweight)
    - (SFT OR (supervised fine-tuning) OR RLHF OR DPO)
    - (TRL OR OpenRLHF OR LLaMA-Factory OR Unsloth)
    - (vLLM OR TensorRT-LLM OR SGLang) AND inference

    ## EXECUTION CHECKLIST
    Before outputting, verify each query:
    ☐ Uses only AND/OR operators
    ☐ Has ≤3 keyword groups
    ☐ Multi-word concepts are in parentheses (not quotes)
    ☐ No extra spaces inside parentheses
    ☐ Keywords are concise concepts, not sentences
    ☐ Output format is exactly [q1|q2|...|qk]

    Generate queries that would help find: (1) similar research implementations, (2) relevant training frameworks, (3) useful toolkits for the proposed methodology.
    """
    basic_idea = dspy.InputField(desc="The core basic idea of the research - main concept and innovation")
    methodology = dspy.InputField(desc="Proposed methodology and approach - how the problem will be solved")
    experimental_setting = dspy.InputField(desc="Experimental setting and evaluation approach")
    
    new_github_queries = dspy.OutputField(
        desc=(
            """Serveral GitHub search queries in format: [query1|query2|...|queryk]
    Each query uses only AND/OR operators with ≤3 keyword groups. Multi-word concepts in parentheses like (large language model). No extra spaces except around AND/OR. No quotes, NOT operator, or filters."""
        )
    )


class OptimizedGitHubQueryGenerator(dspy.Module):
    """
    Generates optimized GitHub queries using specific idea components (basic idea, methodology, experimental_setting).
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = _load_llm_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base")
            )
            logger.info(f"Initialized OptimizedGitHubQueryGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_github_queries = dspy.ChainOfThought(OptimizedGitHubQuerySignature)

    def forward(self, basic_idea: str, methodology: str, experimental_setting: str) -> List[str]:
        with dspy.settings.context(lm=self.lm):
            result = self.generate_github_queries(
                basic_idea=basic_idea,
                methodology=methodology,
                experimental_setting=experimental_setting,
            )
        return _parse_pipe_bracket_list(getattr(result, "new_github_queries", ""))


class OptimizedWebQuerySignature(dspy.Signature):
    """
    You are an expert in generating precise search queries for academic and research-oriented web searches, specifically tailored to uncover related works, evidence, criticisms, and diverse viewpoints on innovative research ideas. Your task is to analyze the provided basic_idea, motivation, and methodology sections, then synthesize 3-5 targeted queries that can be directly inserted into a Google Search API restricted to sites like x.com, medium.com, towardsdatascience.com, substack.com, and reddit.com/r/MachineLearning.

    Key guidelines:
    - Extract core keywords, phrases, and concepts from the three sections, emphasizing the motivation (gaps and importance) and methodology (key approaches and innovations). Prioritize elements that highlight novelty, challenges, or proposed solutions to guide searches toward discussions of similar methods, empirical evidence, critiques, or extensions in related literature.
    - Each query must use only AND and OR operators, with no other Boolean operators (e.g., no NOT), filters (e.g., no site:), or extraneous elements. Limit each query to 1-3 keywords or phrases.
    - For multi-word concepts, enclose in double quotes (e.g., "supervised fine-tuning").
    - Use OR within parentheses for synonyms or alternative terms to broaden recall and improve precision on a single concept (e.g., (efficient OR lightweight OR fast)).
    - Use AND between distinct concepts to probe specific combinations for depth (e.g., ("large language model" OR LLM) AND (efficient OR lightweight)) But don't use too much AND to limit the scope of the search.
    - Use OR across major concepts for breadth when exploring related works (e.g., (SFT OR RLHF) OR ("data synthesis" OR "trajectory generation")).
    - Avoid terms implying tutorials, best practices, implementations, benchmarks, or guides (e.g., no 'tutorial', 'how-to', 'implementation', 'best practice', 'benchmark'). Focus exclusively on analytical discussions: related works, evidence from studies, criticisms of approaches, or viewpoints on gaps/solutions.
    - Ensure no extra spaces around operators (e.g., (LLM OR VLM), not ( LLM OR VLM )).
    - Output exactly in the format [query1|query2|...], with 3-5 queries separated by pipes (|). No introductions, explanations, or additional text.

    Few-shot examples:
    Input idea: Basic idea involves efficient training of large language models. Motivation: High computational costs limit accessibility. Methodology: Use lightweight fine-tuning with RL.
    Output: [("large language model" OR LLM) AND (efficient OR lightweight OR fast)|("supervised fine-tuning" OR SFT) OR (RL OR "reinforcement learning")|(RLHF OR DPO)]

    Input idea: Reinforcement learning for aligning language models. Motivation: Safety and bias issues in outputs. Methodology: Direct preference optimization.
    Output: [(RLHF OR "reinforcement learning from human feedback") AND (alignment OR safety)|(DPO OR "direct preference optimization") OR PPO|("language model" OR LLM) AND (bias OR criticism OR evidence)]

    Generate queries that facilitate deeper evaluation of the idea by surfacing comparable research trajectories, not user-facing resources.
    """
    basic_idea = dspy.InputField(desc="The core basic idea of the research - main concept and innovation")
    motivation = dspy.InputField(desc="Research motivation - why this problem is important and what gaps exist")
    methodology = dspy.InputField(desc="Proposed methodology and approach - how the problem will be solved")
    
    new_web_queries = dspy.OutputField(
        desc=(
            "A list of search queries in the format [query1|query2|...], where each query consists of 1-3 keywords or phrases connected solely by AND or OR operators, enclosed in parentheses where appropriate for grouping. Use double quotes for multi-word phrases (e.g., 'large language model') and OR for synonyms or alternatives. Focus on core concepts from the inputs to identify related works, evidence, criticisms, or views, balancing depth (via AND) and breadth (via OR). Output only the bracketed list, with no additional text."
        )
    )


class OptimizedWebQueryGenerator(dspy.Module):
    """
    Generates optimized web queries using specific idea components (basic idea, motivation, methodology).
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = _load_llm_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base")
            )
            logger.info(f"Initialized OptimizedWebQueryGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_web_queries = dspy.ChainOfThought(OptimizedWebQuerySignature)

    def forward(self, basic_idea: str, motivation: str, methodology: str) -> List[str]:
        with dspy.settings.context(lm=self.lm):
            result = self.generate_web_queries(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
            )
        return _parse_pipe_bracket_list(getattr(result, "new_web_queries", ""))
