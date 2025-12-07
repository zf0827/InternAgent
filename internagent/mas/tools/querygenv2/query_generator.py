"""
Query Generator Module for V2

Generates optimized search queries for arXiv and web platforms.
"""

import dspy
import os
import json
import logging
from typing import List, Optional, Dict, Any

from ..searchersv2.models import Idea, SearchQuery, Source

logger = logging.getLogger(__name__)


def _load_llm_config_from_env() -> dict:
    """Load LLM configuration from environment variables."""
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


def _parse_pipe_bracket_list(text: str) -> List[str]:
    """Parse a pipe-separated list from bracket format [a|b|c]."""
    if not text:
        return []
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return [q.strip().strip('"').strip("'") for q in s.split("|") if q.strip()]


class OptimizedCoreSignature(dspy.Signature):
    """
    You are an expert academic keyword extraction specialist. Your task is to distill a research idea into search-optimized keywords at the atomic concept level for precise matching on OpenAlex.

    ====================== TASK OVERVIEW ======================
    Given a research idea (basic_idea, motivation, methodology), extract:
    1. ONE core essence — the central concept of this work
    2. ONE main motivation — the key problem/gap being addressed
    3. Core technologies — the decisive technical components (multiple allowed, no more than 5 techs)
    4. Baselines — the baseline methods or models that are referenced in the experimental setting part

    ====================== KEYWORD QUALITY REQUIREMENTS ======================
    1. **Conciseness**: Each keyword MUST be <= 3 words.
    2. **Specificity**: Prefer precise terms over vague ones.
    3. **Essentialism**: Extract the most essential concepts. Do NOT add unnecessary words.
    4. **Academic style**: Use formal academic vocabulary.
    5. **Atomic correctness**: Each keyword represents ONE semantic concept.

    ====================== OUTPUT FORMAT (STRICT) ======================
    - core_essence: A single string (<= 3 words).
    - main_motivation: A single string (<= 3 words).
    - tech: A JSON-style list of <= 5 strings (each <= 3 words). Example: ["keyword1", "keyword2", ...]

    ====================== EXAMPLES ======================
    For an idea about "using LLMs to automatically fix code bugs":
    
    core_essence: "program repair"
    main_motivation: "software bug"
    tech: ["large language model", "code generation"]
    baselines: ["directly prompt",...](need to be extracted from the real experimental setting part)

    For an idea about "efficient inference of diffusion models via distillation":
    
    core_essence: "diffusion acceleration"
    main_motivation: "inference efficiency"
    tech: ["knowledge distillation", "model compression"]
    baselines: ["Easy-Diffusion", "Diffusers", ...](need to be extracted from the real experimental setting part)

    ====================== NOW PROCESS THE INPUT ======================
    Analyze the given idea deeply, identify the ONE core essence, ONE main motivation, key technologies, and baselines.
    """

    basic_idea = dspy.InputField(
        desc="The core basic idea of the research - main concept and claimed innovation"
    )
    motivation = dspy.InputField(
        desc="Research motivation - why this problem matters and what gaps exist"
    )
    methodology = dspy.InputField(
        desc="Proposed methodology and technical approach - how the problem will be solved"
    )
    experimental_setting = dspy.InputField(
        desc="Major And Analysis Experiments - Including datasets, baselines, metrics, and hardware"
    )

    core_essence = dspy.OutputField(
        desc="A single string (<= 3 words) representing the central concept."
    )
    main_motivation = dspy.OutputField(
        desc="A single string (<= 3 words) representing the key problem/gap."
    )
    tech = dspy.OutputField(
        desc='JSON-style list of keyword strings (each <= 3 words) representing core technical components. Format: ["tech1", "tech2", ...]'
    )
    baselines = dspy.OutputField(
        desc='JSON-style list of keyword strings (each <= 3 words) representing baseline methods or models. Format: ["baseline1", "baseline2", ...]'
    )


class OptimizedCoreGenerator(dspy.Module):
    """
    Generates core concept, motivation, and techs from idea text.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = self._load_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base")
            )
            logger.info(f"Initialized OptimizedCoreGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_core = dspy.ChainOfThought(OptimizedCoreSignature)

    def _load_config_from_env(self) -> dict:
        ds_api_key = os.getenv("DS_API_KEY")
        if ds_api_key:
            logger.info(f"Using DeepSeek API")
            return {
                "api_key": ds_api_key,
                "api_base": os.getenv("DS_API_BASE_URL"),
                "model": "openai/DeepSeek-V3.2"
            }
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            return {
                "api_key": openai_api_key,
                "api_base": os.getenv("OPENAI_API_BASE_URL"),
                "model": "openai/gpt-4o-mini"
            }
        
        raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")

    def forward(self, basic_idea: str, motivation: str, methodology: str, experimental_setting: str) -> Dict[str, Any]:
        """
        Generate core essence, motivation and techs from idea text.
        """
        logger.info("Generating core essence, motivation, techs, and baselines for idea...")
        
        logger.info(f"Components - Basic idea: {len(basic_idea)} chars, "
                   f"Motivation: {len(motivation)} chars, "
                   f"Methodology: {len(methodology)} chars, "
                   f"Experimental setting: {len(experimental_setting)} chars")
        
        with dspy.settings.context(lm=self.lm):
            result = self.generate_core(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
                experimental_setting=experimental_setting,
            )
        
        techs = []
        baselines = []
        if result.tech:
             # Try to parse the list string
            try:
                tech_str = result.tech.strip()
                if tech_str.startswith('[') and tech_str.endswith(']'):
                    techs = json.loads(tech_str)
                else:
                    # Fallback splitting if not valid JSON
                    techs = [t.strip().strip('"').strip("'") for t in tech_str.split(',')]
            except Exception as e:
                logger.error(f"Failed to parse tech list: {e}")
                techs = [result.tech]
        if result.baselines:
            try:
                baseline_str = result.baselines.strip()
                if baseline_str.startswith('[') and baseline_str.endswith(']'):
                    baselines = json.loads(baseline_str)
                else:
                    # Fallback splitting if not valid JSON
                    baselines = [t.strip().strip('"').strip("'") for t in baseline_str.split(',')]
            except Exception as e:
                logger.error(f"Failed to parse baseline list: {e}")
                baselines = [result.baselines]
                
        return {
            "core_essence": result.core_essence.strip('"'),
            "main_motivation": result.main_motivation.strip('"'),
            "tech": techs,
            "baselines": baselines,
        }


class OptimizedSynonymsSignature(dspy.Signature):
    """You are an expert academic search strategist on arXiv. Your goal is to generate 3-5 EFFECTIVE lexical variants 
    of a given research concept that will maximize recall of RELATED PRIOR WORK in the same research area.
    
    INPUTS:
    basic_idea: The core basic idea of the research - main concept and claimed innovation
    motivation & methodology: Research motivation and proposed methodology - only to better understand the basic_idea, NOT important
    core_essence: The core concept in the basic idea, we FOCUS ON generating related entities for this concept

    CRITICAL UNDERSTANDING:
    These are NOT strict synonyms — they are alternative terms/phrasings that researchers would use when working on 
    the SAME PROBLEM SPACE or SAME INNOVATION. Your job is to think: "What other terms would appear in 
    titles of papers that a researcher should read if they're working on this idea?"
    
    CORE OBJECTIVE:
    Generate variants that capture:
    1. Direct lexical alternatives (e.g., "vision-language models" ↔ "multimodal LLMs" ↔ "VLMs")
    2. Related technical approaches in the same problem space (e.g., "in-context learning" → "few-shot prompting", "prompt-based learning")
    3. Broader/narrower terms that cover similar work (e.g., "test-time adaptation" → "test-time training", "online adaptation")
    4. Common domain-specific phrasings used interchangeably in literature (even if semantically different)
    
    WHAT TO INCLUDE:
    ✓ Terms that solve the same core problem with different naming
    ✓ Alternative technical formulations of the same concept
    ✓ Established acronyms and their expansions (e.g., RAG ↔ retrieval-augmented generation)
    ✓ Related methods that would appear in a "related work" section
    ✓ Domain-specific jargon variants (e.g., "uncertainty quantification" vs "uncertainty estimation")
    
    WHAT TO AVOID:
    ✗ Mere syntactic transformations ("multi-step reasoning" → "reasoning in multiple steps")
    ✗ Word reorderings that don't change search results ("data-analytic agents" → "agents for data analytics")
    ✗ Over-generic terms that would return thousands of irrelevant papers
    ✗ Highly specialized sub-concepts that narrow the search too much
    ✗ Avoid add additional concept from the content of motivation & methodology parts, these part are provided to better understand the core essence
    
    REASONING PROCESS (internal, before output):
    In your reasoning, think of these following questions step and step.
    1. Given the core_essence, ask: "What papers should a researcher read if working on this?"
    2. Consider: What related methods exist? What alternative terminologies are used? What's the broader problem class?
    3. Use basic_idea, motivation, methodology as context to identify the true research area
    4. Generate 3-5 terms that would each retrieve a useful subset of related work when used in an OR query
    
    FINAL INSTRUCTION:
    Think like you're building an OR query: ti:"variant1" OR ti:"variant2" OR ti:"variant3" to catch all relevant papers.
    Prioritize RECALL of related work over strict semantic equivalence.
    
    OUTPUT FORMAT:
    Strictly a clean JSON list of 3-5 strings. Each entity should be no more than 3 words. No explanations, no markdown, no extra text.
    """
    
    core_essence = dspy.InputField(desc="The exact central concept phrase")
    basic_idea = dspy.InputField(desc="Brief context of the research topic to help identify related work")
    motivation = dspy.InputField(desc="The main motivation/pain point this idea is directly solving")
    methodology = dspy.InputField(desc="The main detailed methodology in this idea")
    
    related_entities = dspy.OutputField(desc='JSON list of 3-5 lexically distinct terms for maximizing related work recall')


class OptimizedSynonymsGenerator(dspy.Module):
    """
    Generates synonyms for a given core essence using dspy.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = self._load_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base"),
                temperature=1.0
            )
            logger.info(f"Initialized OptimizedSynonymsGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_synonyms = dspy.ChainOfThought(OptimizedSynonymsSignature)

    def _load_config_from_env(self) -> dict:
        ds_api_key = os.getenv("DS_API_KEY")
        if ds_api_key:
            return {
                "api_key": ds_api_key,
                "api_base": os.getenv("DS_API_BASE_URL"),
                "model": "openai/DeepSeek-V3.2"
            }
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            print(openai_api_key)
            print(os.getenv("OPENAI_API_BASE_URL"))
            return {
                "api_key": openai_api_key,
                "api_base": os.getenv("OPENAI_API_BASE_URL"),
                "model": "openai/gpt-4o-mini"
            }
        
        raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")

    def forward(self, core_essence: str, basic_idea: str, motivation: str, methodology: str) -> List[str]:
        """
        Generate synonyms for the core essence.
        Returns a list of strings where the first element is the core_essence itself.
        """
        logger.info(f"Generating synonyms for: {core_essence}")
        
        with dspy.settings.context(lm=self.lm):
            result = self.generate_synonyms(
                core_essence=core_essence,
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
            )
            
        # Parse output
        synonyms = []
        raw_output = result.related_entities
        try:
            # Attempt JSON parse
            cleaned = raw_output.strip()
            if cleaned.startswith("```json"):
                 cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                 cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            if cleaned.startswith('[') and cleaned.endswith(']'):
                synonyms = json.loads(cleaned)
            else:
                # Fallback: split by comma
                synonyms = [s.strip().strip('"').strip("'") for s in cleaned.split(',')]
        except Exception as e:
            logger.warning(f"Failed to parse synonyms as JSON: {e}. Raw: {raw_output}")
            # Fallback split
            synonyms = [s.strip() for s in raw_output.split(',')]

        # Clean up
        final_list = []
        seen = set()
        
        # Add core_essence first
        if core_essence:
            final_list.append(core_essence)
            seen.add(core_essence.lower())

        # Add synonyms
        for s in synonyms:
            if isinstance(s, str):
                s_clean = s.strip()
                if s_clean and s_clean.lower() not in seen:
                    final_list.append(s_clean)
                    seen.add(s_clean.lower())

        # 2-gram distance sorting
        def get_ngrams(text, n=2):
            text = text.lower()
            return [text[i:i+n] for i in range(len(text)-n+1)]

        def calculate_ngram_distance(s1, s2, n=2):
            if not s1 or not s2:
                return 1.0
            
            ngrams1 = set(get_ngrams(s1, n))
            ngrams2 = set(get_ngrams(s2, n))
            
            if not ngrams1 and not ngrams2:
                return 0.0
            if not ngrams1 or not ngrams2:
                return 1.0
                
            intersection = len(ngrams1.intersection(ngrams2))
            union = len(ngrams1.union(ngrams2))
            
            return 1.0 - (intersection / union)

        # Calculate distances and sort
        # Skip the first element (core_essence) for sorting, but include it in result
        if len(final_list) > 1:
            synonyms_to_sort = final_list[1:]
            # Calculate distance to core_essence for each synonym
            synonyms_with_scores = []
            for syn in synonyms_to_sort:
                dist = calculate_ngram_distance(core_essence, syn)
                synonyms_with_scores.append((syn, dist))
            
            # Sort by distance (descending: far to near? User said "from far to near" -> far first?)
            # "从远到近排序输出" usually means from far distance (least similar) to near distance (most similar)?
            # Or does it mean "from far (conceptually) to near (conceptually)"?
            # Usually "rank from far to near" implies distance descending.
            # However, if we want synonyms, we usually want the most similar ones. 
            # But let's follow instruction "从远到近" (from far to near). 
            # Far distance = high value (dissimilar). Near distance = low value (similar).
            # So sort by distance descending.
            
            synonyms_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            sorted_synonyms = [item[0] for item in synonyms_with_scores]
            final_list = [core_essence] + sorted_synonyms
            print(final_list)
                    
        return final_list


class OptimizedPaperQuerySignature(dspy.Signature):
    """
    You are an expert academic search strategist. Your only goal is to generate 6–10 extremely precise ArXiv TITLE queries 
    that can directly retrieve the most relevant prior work to a given research idea — nothing more, nothing less.

    Input:
      - basic_idea: core concept and claimed innovation of the idea
      - motivation: why this problem matters and the key existing gap
      - methodology: concrete technical approach that solves the problem

    ======================
    CORE OBJECTIVE
    ======================
    1. First, in your reasoning (not in output), deeply understand and condense the idea into:
       - One single core essence (usually 3–8 words that capture what this work is truly about)
       - One single most important motivation/pain point this idea is directly solving
       - One or (rarely) two truly decisive technical components that define the method

    2. All generated queries MUST revolve only around these 2–4 ultra-core concepts identified above.
       No secondary or peripheral concepts are allowed.

    3. For each core concept, expand 2–5 academic synonyms or alternative phrasings that commonly appear 
       in real paper titles (e.g., "vision-language models" ↔ "multimodal large language models" ↔ "VLMs").

    4. Generate queries using primarily OR within the same concept slot to maximize recall of different expressions, 
       and use AND extremely sparingly — only when combining two truly indispensable core concepts 
       (core problem + core method, or core method + core context). 
       Most queries should be single-concept with rich OR chains or at most one AND.

    5. Final goal: every returned paper from these 6–10 queries should feel "this is almost exactly our idea" 
       to a human researcher. Precision > breadth.

    ======================
    STRICT OUTPUT FORMAT (UNCHANGED)
    ======================
    Output ONLY:

    [QUERY_1|QUERY_2|...|QUERY_N]

    - 6 ≤ N ≤ 10
    - Each QUERY contains 1 to 3 ti:"..." clauses
    - Only ti:"..." clauses + uppercase AND / OR are allowed
    - No parentheses, no NOT, no other fields, no extra text

    ======================
    REASONING REQUIREMENTS (MUST DO BEFORE OUTPUT)
    ======================
    In your internal reasoning (never visible in final answer), you MUST explicitly write:
    1. Core essence (one phrase):      "The true core of this idea is: X"
    2. Most direct motivation/gap:     "The single most important pain point being solved is: Y"
    3. Decisive technical component(s): "The truly novel/enabling technique(s) are: Z (and W if any)"
    4. For each of X, Y, Z, list 3–5 title-level synonyms/alternative phrasings

    Only after this analysis do you design the 6–10 queries.

    ======================
    WHAT IS NOW FORBIDDEN
    ======================
    - Using AND to combine two non-essential or loosely related concepts
    - Queries that would return >200–300 results on arXiv (too noisy)
    - Including minor technical details, datasets, benchmarks, or secondary contributions
    - More than 10 queries or fewer than 6

    ======================
    FINAL OUTPUT RULE
    ======================
    Only output the bracketed list. No reasoning, no explanation, no bullet points, no extra words.
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
            config = self._load_config_from_env()

        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base"),
                temperature=1.0
            )
            logger.info(f"Initialized OptimizedPaperQueryGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_optimized_queries = dspy.ChainOfThought(OptimizedPaperQuerySignature)

    def _load_config_from_env(self) -> dict:
        ds_api_key = os.getenv("DS_API_KEY")
        if ds_api_key:
            logger.info(f"Using DeepSeek API")
            return {
                "api_key": ds_api_key,
                "api_base": os.getenv("DS_API_BASE_URL"),
                "model": "openai/DeepSeek-V3.2"
            }
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            return {
                "api_key": openai_api_key,
                "api_base": os.getenv("OPENAI_API_BASE_URL"),
                "model": "openai/gpt-4o-mini"
            }
        
        raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")

    def forward(self, basic_idea: str, motivation: str, methodology: str) -> List[str]:
        """
        Generate optimized paper queries from idea text.
        Returns a list of new paper queries.
        """
        logger.info("Generating optimized paper queries for idea...")
        
        logger.info(f"Components - Basic idea: {len(basic_idea)} chars, "
                   f"Motivation: {len(motivation)} chars, "
                   f"Methodology: {len(methodology)} chars")
        
        with dspy.settings.context(lm=self.lm):
            result = self.generate_optimized_queries(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology
            )

        new_paper_queries = self._parse_query_list(result.new_paper_queries)
        return new_paper_queries

    def _parse_query_list(self, query_string: str) -> List[str]:
        if not query_string:
            return []
        # 去掉外层的中括号
        query_string = query_string.strip()
        if query_string.startswith('[') and query_string.endswith(']'):
            query_string = query_string[1:-1].strip()
        # 按 '|' 分割并去掉多余空格
        queries = [q.strip() for q in query_string.split("|")]
        return queries


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
    """Generates optimized web queries using specific idea components."""
    
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


class RefineQuerySignature(dspy.Signature):
    """
    You are an expert academic search strategist helping to refine and extend an existing ArXiv title search.

    GOAL:
    Given (1) the original research idea, (2) the top-ranked papers found so far (including the queries that
    retrieved them), and (3) the full set of original queries, you will reflect on what has worked well and
    what has not, then propose improved follow-up title queries that complement the current results.

    INPUTS:
    - basic_idea: The original research idea.
    - top_papers_info: JSON string with top papers, including for each paper its title, similarity_score,
      and the specific query that retrieved it:
        [{"title": "...", "similarity_score": 0.95, "query": "..."}, ...]
    - original_queries: JSON array of all queries used in the first search round, including both effective
      and ineffective ones.

    INTERPRETATION OF QUERIES:
    - Treat the queries that successfully retrieved the papers in top_papers_info as “good” queries:
      they are reasonably well-aligned with the basic_idea and the actual literature.
    - Treat the remaining queries in original_queries (that did not retrieve top papers) as “weak” or
      “less useful” queries, because they are likely:
        - too specific (overly detailed constraints that hurt recall),
        - too broad (introducing a lot of noise), or
        - partially off-topic relative to the basic_idea.

    ANALYSIS PROCESS:
    1. Analyze good queries and top paper titles:
       - Extract recurring, high-signal keywords/phrases and phrasings that characterize the core topic,
         tasks, methods, or domains.
       - Notice terminology and synonyms that appear to be widely used and well-matched to the idea.

    2. Analyze weak queries:
       - Identify over-specific fragments (very detailed or niche conditions) that likely prevent finding
         additional relevant papers; consider how they could be generalized or removed.
       - Identify low-relevance or noisy keywords and avoid reusing them in new queries.

    3. Reflect on coverage and gaps:
       - Determine which aspects of the basic_idea are already well-covered by the current top papers
         (e.g., particular methods, datasets, problem settings).
       - Identify missing or under-explored perspectives, such as:
         alternative methods, related tasks, adjacent application domains, different terminology,
         or broader/narrower variants of the problem.

    4. Design refined queries:
       - Reuse and recombine high-signal keywords from good queries and from top paper titles.
       - Generalize over-specific fragments from weak queries (e.g., shorten overly detailed phrases,
         drop unnecessary constraints, or replace them with slightly broader terms).
       - Avoid low-relevance or noisy keywords observed in weak queries.
       - Introduce alternative but clearly related terminology that might surface complementary or
         previously missed papers, while remaining focused on the basic_idea.
       - Aim for queries that extend the current search (new angles, related subproblems,
         complementary approaches) without drifting off-topic.

    OUTPUT REQUIREMENTS:
    - Generate 6–10 new ArXiv title queries.
    - Each query must use only ti:"..." clauses combined with uppercase AND / OR.
    - Each query must contain 1–3 ti:"..." clauses.
    - Do NOT duplicate any of the original_queries verbatim; new queries should be refinements,
      recombinations, or generalizations.
    - Focus on discovering papers that complement or extend the current top results, improving recall
      while maintaining good precision.

    ======================
    STRICT OUTPUT FORMAT (UNCHANGED)
    ======================
    Output ONLY:

    [QUERY_1|QUERY_2|...|QUERY_N]

    - 6 ≤ N ≤ 10
    - Each QUERY contains 1 to 3 ti:"..." clauses
    - Only ti:"..." clauses + uppercase AND / OR are allowed
    - No parentheses, no NOT, no other fields, no extra text
    """

    basic_idea = dspy.InputField(
        desc="The original research idea that the search should stay focused on."
    )
    top_papers_info = dspy.InputField(
        desc='JSON string with top papers: [{"title": "...", "similarity_score": 0.95, "query": "..."}, ...]'
    )
    original_queries = dspy.InputField(
        desc="JSON array of all original title queries used in the first search round."
    )
    refined_queries = dspy.OutputField(
        desc=(
            'Refined ArXiv title search queries derived from the basic idea and from analysis of which '
            'initial queries and retrieved papers worked well or poorly. The output MUST be a single '
            'bracketed, pipe-separated list like [ti:"..." AND ti:"..."|ti:"..." OR ti:"..."|...]. '
            'Each internal query uses 1–3 ti:"..." clauses combined only with AND and/or OR, and should '
            'extend or complement the current set of top-ranked papers while avoiding ineffective patterns '
            'from the original queries.'
        )
    )


class RefineGenerator(dspy.Module):
    """
    Generates refined queries based on top-ranked search results.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        if config is None:
            config = _load_llm_config_from_env()
        
        try:
            self.lm = dspy.LM(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config["api_key"],
                api_base=config.get("api_base"),
                temperature=1.0
            )
            logger.info(f"Initialized RefineGenerator with model: {config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy with provided config: {e}")
            raise
        
        self.generate_refined_queries = dspy.ChainOfThought(RefineQuerySignature)
    
    def forward(
        self,
        basic_idea: str,
        top_sources: List[Source],
        similarity_scores: List[float],
        source_queries: List[str],
        original_queries: List[str]
    ) -> List[str]:
        """
        Generate refined queries based on top-ranked sources.
        
        Args:
            basic_idea: The original research idea
            top_sources: List of top-ranked Source objects
            similarity_scores: List of similarity scores corresponding to top_sources
            source_queries: List of queries that found each source (by query index)
            original_queries: List of original queries used in first search
        
        Returns:
            List of refined query strings
        """
        logger.info(f"Generating refined queries based on {len(top_sources)} top sources...")
        
        # Build top_papers_info JSON
        papers_info = []
        for i, source in enumerate(top_sources):
            papers_info.append({
                "title": source.title,
                "similarity_score": similarity_scores[i] if i < len(similarity_scores) else 0.0,
                "query": source_queries[i] if i < len(source_queries) else ""
            })
        
        top_papers_info_str = json.dumps(papers_info, ensure_ascii=False)
        original_queries_str = json.dumps(original_queries, ensure_ascii=False)
        
        with dspy.settings.context(lm=self.lm):
            result = self.generate_refined_queries(
                basic_idea=basic_idea,
                top_papers_info=top_papers_info_str,
                original_queries=original_queries_str
            )
        
        # Parse refined queries
        refined_queries = self._parse_query_list(result.refined_queries)
        refined_queries_abs = [q.replace("ti:", "abs:") for q in refined_queries]
        refined_queries_abs = [f'(({q}) NOT (ti:"survey" OR ti:"benchmark" OR ti:"overview"))' for q in refined_queries_abs]
        logger.info(f"Generated {len(refined_queries)} refined queries")
        return refined_queries_abs
    
    def _parse_query_list(self, query_string: str) -> List[str]:
        """Parse query list from string format."""
        if not query_string:
            return []
        query_string = query_string.strip()
        if query_string.startswith('[') and query_string.endswith(']'):
            query_string = query_string[1:-1].strip()
        queries = [q.strip() for q in query_string.split("|")]
        return queries


class QueryGenerator:
    """
    Main query generator that generates queries for different platforms.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize the query generator."""
        if config is None:
            config = _load_llm_config_from_env()
        self.config = config
        
        self.core_generator = OptimizedCoreGenerator(config=config)
        self.synonym_generator = OptimizedSynonymsGenerator(config=config)
        self.web_query_generator = OptimizedWebQueryGenerator(config=config)
        self.paper_query_generator = OptimizedPaperQueryGenerator(config=config)
    
    def generate(self, idea: Idea) -> SearchQuery:
        """
        Generate search queries from an idea.
        
        Args:
            idea: Research idea
            
        Returns:
            SearchQuery object with platform-specific queries
        """
        # Extract components from idea
        basic_idea = (idea.basic_idea or "").strip()
        motivation = (idea.motivation or "").strip()
        methodology = (idea.method or "").strip()
        experimental_setting = (idea.experimental_setting or "").strip()
        
        paper_queries: List[str] = []
        web_queries: List[str] = []
        
        # Generate paper queries using new logic: core -> synonyms -> queries
        try:
            # Step 1: Generate core essence, motivation and techs
            core_info = self.core_generator(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
                experimental_setting=experimental_setting,
            )
            
            # Step 2: Generate synonyms for core essence
            core_essence = core_info.get("core_essence")
            logger.info(f"core_info: {core_info}")
            if core_essence:
                synonyms = self.synonym_generator(
                    core_essence=core_essence,
                    basic_idea=basic_idea,
                    motivation=motivation,
                    methodology=methodology,
                )
                
                # Step 3: Build paper queries from synonyms
                if synonyms:
                    paper_queries = [
                        f'(ti:"{s}" NOT (ti:"survey" OR ti:"benchmark" OR ti:"overview"))'
                        for s in synonyms
                    ]
            else:
                logger.warning("No core_essence generated, skipping paper query generation")
                
            # Step 4: Generate baselines from experimental setting
            baselines = core_info.get("baselines")
            if baselines:
                for baseline in baselines:
                    paper_queries.append(f'(abs:"{baseline}" NOT (ti:"survey" OR ti:"benchmark" OR ti:"overview"))')
            else:
                logger.warning("No baselines generated, skipping paper query generation")

        except Exception as e:
            logger.warning(f"Paper query generator (synonyms) failed: {e}")
        
        # Generate paper queries from direct generation using OptimizedPaperQueryGenerator
        try:
            direct_queries = self.paper_query_generator(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology
            )
            # Replace ti: with abs: as in test_paperv4.py
            direct_queries_abs = [q.replace("ti:", "abs:") for q in direct_queries]
            direct_queries_abs = [f'(({q}) NOT (ti:"survey" OR ti:"benchmark" OR ti:"overview"))' for q in direct_queries_abs]
            paper_queries.extend(direct_queries_abs)
            logger.info(f"Generated {len(direct_queries_abs)} direct paper queries")
        except Exception as e:
            logger.warning(f"Direct paper query generator failed: {e}")
        
        try:
            web_queries = self.web_query_generator(
                basic_idea=basic_idea,
                motivation=motivation,
                methodology=methodology,
            )
        except Exception as e:
            logger.warning(f"Web query generator failed: {e}")
        
        # Clean up queries
        paper_queries = self._cleanup_queries(paper_queries)
        web_queries = self._cleanup_queries(web_queries)
        
        return SearchQuery(
            paper_queries=paper_queries,
            github_queries=[],  # Placeholder
            kaggle_queries=[],  # Placeholder
            web_queries=web_queries,
            scholar_queries=[],  # Placeholder
        )
    
    def _cleanup_queries(self, items: List[str]) -> List[str]:
        """Remove duplicates and empty queries."""
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

