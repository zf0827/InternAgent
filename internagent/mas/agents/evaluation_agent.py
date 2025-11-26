"""
Evaluation Agent for InternAgent

This module provides the EvaluationAgent that evaluates research ideas from multiple
aspects: clarity (faithfulness), novelty, and feasibility. It contains three sub-agents
that perform specialized evaluations and then synthesizes their results.
"""

import logging
import asyncio
import random
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class EvaluationAgent(BaseAgent):
    """
    Agent that evaluates research ideas from multiple aspects.
    
    Contains three sub-agents:
    - ClarityAgent: Evaluates logical consistency, factual correctness, and structure
    - NoveltyAgent: Evaluates originality and uniqueness compared to prior work
    - FeasibilityAgent: Evaluates implementation feasibility based on code repositories
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "EvaluationAgent"
        self.temperature = config.get("temperature", 0.7)
        
        # Initialize sub-agents (as internal classes)
        self.clarity_agent = self._ClarityAgent(model, config)
        self.novelty_agent = self._NoveltyAgent(model, config)
        self.feasibility_agent = self._FeasibilityAgent(model, config)
        
        logger.info(f"Initialized EvaluationAgent with sub-agents")
    
    def _extract_idea_text(self, idea: Any) -> str:
        """
        Extract idea text from various input formats.
        
        Args:
            idea: Can be a string, dict with idea fields, or Idea object
            
        Returns:
            Formatted idea text string
        """
        if isinstance(idea, str):
            return idea
        
        if isinstance(idea, dict):
            raw = idea.get("raw_text")
            if raw:
                return raw
            
            # Build from structured fields
            parts = []
            if idea.get('motivation'):
                parts.append(f"Motivation: {idea.get('motivation', '')}")
            if idea.get('research_question'):
                parts.append(f"Research Question: {idea.get('research_question', '')}")
            if idea.get('method'):
                parts.append(f"Method: {idea.get('method', '')}")
            if idea.get('experimental_setting'):
                parts.append(f"Experimental Setting: {idea.get('experimental_setting', '')}")
            if idea.get('expected_results'):
                parts.append(f"Expected Results: {idea.get('expected_results', '')}")
            
            return "\n\n".join(parts) if parts else ""
        
        # If it's an Idea object with get_full_text method
        if hasattr(idea, 'get_full_text'):
            return idea.get_full_text()
        
        return str(idea)
    
    def _extract_idea_parts(self, idea: Any) -> Dict[str, str]:
        """
        Extract specific parts of idea (motivation, research_question, method).
        
        Args:
            idea: Can be a string, dict with idea fields, or Idea object
            
        Returns:
            Dictionary with motivation, research_question, and method
        """
        result = {
            "motivation": "",
            "research_question": "",
            "method": ""
        }
        
        if isinstance(idea, dict):
            result["motivation"] = idea.get("motivation", "")
            result["research_question"] = idea.get("research_question", "")
            result["method"] = idea.get("method", "")
        elif hasattr(idea, 'motivation'):
            result["motivation"] = getattr(idea, 'motivation', '')
            result["research_question"] = getattr(idea, 'research_question', '')
            result["method"] = getattr(idea, 'method', '')
        
        return result
    
    def _extract_github_file_trees(self, search_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract file_tree from github_repos in search_results.
        
        Args:
            search_results: Dictionary from SearchResults.to_dict()
            
        Returns:
            List of dictionaries with title, url, and file_tree
        """
        github_repos = search_results.get("github_repos", [])
        file_trees = []
        
        for repo in github_repos:
            if isinstance(repo, dict):
                file_tree = repo.get("file_tree")
                if file_tree:
                    file_trees.append({
                        "title": repo.get("title", ""),
                        "url": repo.get("url", ""),
                        "file_tree": file_tree
                    })
        
        return file_trees
    
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute evaluation of research idea from multiple aspects.
        
        Args:
            context: Must contain:
                - idea: The research idea to evaluate
                - search_results: SearchResults dictionary
                - persona: Reviewer persona dictionary (optional)
                - web_report: Web research report (optional)
                - code_report: Code research report (optional)
                - paper_report: Paper research report (optional)
            params: Additional parameters (temperature override, etc.)
            
        Returns:
            Dictionary containing:
                - clarity: {score, reason}
                - novelty: {score, reason}
                - feasibility: {score, reason, pseudocode}
                - overall: {summary, overall_rating}
        """
        # Extract inputs
        idea = context.get("idea")
        if not idea:
            raise AgentExecutionError("context must contain 'idea'")
        
        search_results = context.get("search_results") or context.get("search_result")
        if not search_results:
            raise AgentExecutionError("context must contain 'search_results'")
        
        persona = context.get("persona", {})
        web_report = context.get("web_report", "")
        code_report = context.get("code_report", "")
        paper_report = context.get("paper_report", "")
        
        # Extract idea text
        idea_text = self._extract_idea_text(idea)
        idea_parts = self._extract_idea_parts(idea)
        
        # Extract github file trees
        github_file_trees = self._extract_github_file_trees(search_results)
        
        # Execute three sub-agents in parallel
        clarity_task = self.clarity_agent.evaluate(
            idea_text=idea_text,
            web_report=web_report,
            code_report=code_report,
            paper_report=paper_report,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        )
        
        novelty_task = self.novelty_agent.evaluate(
            idea_parts=idea_parts,
            paper_report=paper_report,
            web_report=web_report,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        )
        
        feasibility_task = self.feasibility_agent.evaluate(
            idea_text=idea_text,
            code_report=code_report,
            github_file_trees=github_file_trees,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        )
        
        # Wait for all evaluations to complete
        try:
            clarity_result, novelty_result, feasibility_result = await asyncio.gather(
                clarity_task,
                novelty_task,
                feasibility_task,
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error during parallel evaluation: {e}")
            raise AgentExecutionError(f"Evaluation failed: {str(e)}")
        
        # Handle exceptions from sub-agents
        if isinstance(clarity_result, Exception):
            logger.error(f"ClarityAgent failed: {clarity_result}")
            clarity_result = {"score": 1, "reason": f"Evaluation failed: {str(clarity_result)}"}
        
        if isinstance(novelty_result, Exception):
            logger.error(f"NoveltyAgent failed: {novelty_result}")
            novelty_result = {"score": 1, "reason": f"Evaluation failed: {str(novelty_result)}"}
        
        if isinstance(feasibility_result, Exception):
            logger.error(f"FeasibilityAgent failed: {feasibility_result}")
            feasibility_result = {"score": 1, "reason": f"Evaluation failed: {str(feasibility_result)}", "pseudocode": ""}
        
        # Generate overall summary using LLM
        overall_summary = await self._generate_overall_summary(
            clarity_result,
            novelty_result,
            feasibility_result,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        )
        
        return {
            "clarity": clarity_result,
            "novelty": novelty_result,
            "feasibility": feasibility_result,
            "overall": overall_summary
        }
    
    async def _generate_overall_summary(self, clarity: Dict, novelty: Dict, feasibility: Dict, 
                                        persona: Dict[str, Any] = None, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate overall evaluation summary and overall rating using LLM.
        
        Args:
            clarity: Clarity evaluation result with score and reason
            novelty: Novelty evaluation result with score and reason
            feasibility: Feasibility evaluation result with score, reason, and pseudocode
            persona: Reviewer persona dictionary (optional)
            temperature: Temperature for LLM generation
            
        Returns:
            Dictionary with summary and overall_rating
        """
        clarity_score = clarity.get("score", 0.0)
        novelty_score = novelty.get("score", 0.0)
        feasibility_score = feasibility.get("score", 0.0)
        clarity_reason = clarity.get("reason", "")
        novelty_reason = novelty.get("reason", "")
        feasibility_reason = feasibility.get("reason", "")
        
        # Build prompt for overall summary generation
        persona_section = self._build_persona_section(persona) if persona else ""
        
        prompt = f"""{persona_section}You are a senior research reviewer synthesizing evaluation results from three aspects: clarity, novelty, and feasibility.

=== Evaluation Results ===

**Clarity Evaluation:**
Score: {clarity_score}/10
Reason: {clarity_reason}

**Novelty Evaluation:**
Score: {novelty_score}/10
Reason: {novelty_reason}

**Feasibility Evaluation:**
Score: {feasibility_score}/10
Reason: {feasibility_reason}

=== Task ===
Based on the three evaluation results above, write a comprehensive review summary that:
1. Describes how this reviewer views the research idea overall
2. Synthesizes the evaluations from all three aspects (clarity, novelty, feasibility)
3. Provides an overall assessment

Then, assign an overall rating using the ICLR 6-level scale (only allowed values: 1, 3, 5, 6, 8, 10).

=== ICLR Overall Rating Scale ===
- **1 (Strong Reject)**: The submission is fundamentally flawed — e.g., the problem is unimportant or trivial, the method is unsound, experiments are meaningless or missing, writing is incoherent, or novelty is negligible.

- **3 (Reject)**: The submission has substantial flaws. Perhaps some interesting ideas, but major weaknesses (soundness, clarity, evaluation) preclude acceptance in current form.

- **5 (Weak Reject / Borderline Reject)**: The submission shows some promise (some novelty or interesting ideas) but also significant weaknesses. It merits serious improvements before it could be considered for acceptance.

- **6 (Weak Accept / Borderline Accept)**: The submission has acceptable quality overall. It has reasonable novelty, soundness, and adequate evaluation, but may lack strong impact, novelty, or polish. Acceptable but not outstanding.

- **8 (Accept)**: The submission is solid: contributions are clear, novelty/impact is decent, evaluation is thorough, writing is good. This is a paper worthy of acceptance.

- **10 (Strong Accept)**: The submission is excellent — high novelty or impact, sound methodology, thorough and convincing experiments, clear writing, and likely to lead to significant interest and follow-up.

=== Output Requirements ===
Provide a comprehensive summary that synthesizes all three evaluation aspects and an overall rating based on the ICLR scale above."""
        
        schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Comprehensive review summary describing how the reviewer views the research idea, synthesizing evaluations from clarity, novelty, and feasibility aspects"
                },
                "overall_rating": {
                    "type": "number",
                    "description": "Overall rating using ICLR 6-level scale. Only allowed values: 1, 3, 5, 6, 8, 10",
                    "enum": [1, 3, 5, 6, 8, 10]
                }
            },
            "required": ["summary", "overall_rating"]
        }
        
        system_prompt = (
            "You are an expert research reviewer specializing in synthesizing evaluation results "
            "and providing comprehensive assessments of research ideas. Your task is to integrate "
            "evaluations from multiple aspects and provide a balanced overall judgment."
        )
        
        try:
            result = await self.model.generate_json(
                prompt=prompt,
                schema=schema,
                system_prompt=system_prompt,
                temperature=temperature
            )
            return result
        except Exception as e:
            logger.error(f"Overall summary generation failed: {e}")
            # Fallback to simple summary if LLM generation fails
            avg_score = (clarity_score + novelty_score + feasibility_score) / 3.0
            fallback_rating = 1 if avg_score < 2 else (3 if avg_score < 4 else (5 if avg_score < 5.5 else (6 if avg_score < 7 else (8 if avg_score < 9 else 10))))
            return {
                "summary": f"Clarity: {clarity_score}/10 - {clarity_reason[:200]}...\nNovelty: {novelty_score}/10 - {novelty_reason[:200]}...\nFeasibility: {feasibility_score}/10 - {feasibility_reason[:200]}...",
                "overall_rating": fallback_rating
            }
    
    @staticmethod
    def _mask_report_by_ratio(text: str, ratio: float) -> str:
        """
        Randomly mask report text by ratio (fine-grained, split by sentences/phrases).
        
        Split the text into shorter segments by commas, periods, line breaks, etc., then randomly
        select some segments whose total length is approximately original_length * ratio, and mask the rest.
        
        Args:
            text: Text to be masked
            ratio: Retention ratio (0.0-1.0), e.g., 0.7 means retaining 70% of the content
                
        Returns:
            Masked text
        """
        import re

        if not text or ratio >= 1.0:
            return text
        if ratio <= 0.0:
            return "[Content masked based on reviewer's background knowledge level]"

        # Fine-grained segmentation: split by ". ", "。", "，", ",", "、" and newlines, preserving separators
        # Use regex to capture separators, maintaining their distinction when masked
        segs = []
        last_end = 0
        for m in re.finditer(r"([,，。.\n])", text):
            seg = text[last_end:m.end()]
            segs.append(seg)
            last_end = m.end()
        if last_end < len(text):
            segs.append(text[last_end:])

        total_len = sum(len(seg) for seg in segs)
        target_len = max(1, int(total_len * ratio))

        # Random sampling, aiming for total length after sampling not exceeding target_len
        indices = list(range(len(segs)))
        random.shuffle(indices)

        chosen = set()
        chosen_len = 0
        for idx in indices:
            seg_len = len(segs[idx])
            if chosen_len + seg_len <= target_len or len(chosen) == 0:
                chosen.add(idx)
                chosen_len += seg_len
            if chosen_len >= target_len:
                break

        # Construct masked text
        masked = []
        for i, seg in enumerate(segs):
            if i in chosen:
                masked.append(seg)
            else:
                masked.append("[...]")
        res = "".join(masked)
        # Prevent all content from being masked
        if len(res.replace("[...]", "").strip()) == 0:
            # At least keep the first segment
            res = segs[0] + "[...]" * (len(segs) - 1)
        return res
    
    @staticmethod
    def _filter_file_trees_by_ratio(file_trees: List[Dict[str, str]], ratio: float) -> List[Dict[str, str]]:
        """
        Filter file_trees by ratio, keeping only a certain proportion of repositories.
        
        Args:
            file_trees: List of file trees
            ratio: Retention ratio (0.0-1.0), e.g., 0.7 means retaining 70% of repositories
            
        Returns:
            Filtered list of file trees
        """
        if not file_trees or ratio >= 1.0:
            return file_trees
        if ratio <= 0.0:
            return []
        
        # Calculate number of repositories to keep
        num_keep = max(1, int(len(file_trees) * ratio))
        
        # Randomly select repositories to keep
        return random.sample(file_trees, num_keep)
    
    @staticmethod
    def _build_persona_section(persona: Dict[str, Any]) -> str:
        """
        Build persona section for prompt (shared by all sub-agents).
        
        Args:
            persona: Reviewer persona dictionary containing background, background_knowledge, goal, constraints
            
        Returns:
            Formatted persona section string for prompt
        """
        if not persona:
            return ""
        
        background = persona.get("background", "")
        background_knowledge = persona.get("background_knowledge", {})
        goal = persona.get("goal", "")
        constraints = persona.get("constraints", "")
        
        persona_section = "\n=== Reviewer Persona ===\n"
        if background:
            persona_section += f"Background: {background}\n\n"
        if background_knowledge:
            lit = background_knowledge.get("literature_familiarity", "N/A")
            meth = background_knowledge.get("methodology_depth", "N/A")
            app = background_knowledge.get("application_experience", "N/A")
            persona_section += f"Background Knowledge:\n"
            persona_section += f"  - Literature Familiarity: {lit}/10\n"
            persona_section += f"  - Methodology Depth: {meth}/10\n"
            persona_section += f"  - Application Experience: {app}/10\n\n"
            persona_section += "Note: Based on the background knowledge scores above, the research reports and code repositories provided below have been randomly masked to reflect the reviewer's knowledge level. Lower scores result in more content being masked.\n\n"
        if goal:
            persona_section += f"Goal: {goal}\n\n"
        if constraints:
            persona_section += f"Constraints: {constraints}\n"
        persona_section += "\nPlease evaluate the research idea from the perspective of this reviewer persona.\n"
        
        return persona_section
    
    # ==================== Sub-Agent Classes ====================
    
    class _ClarityAgent:
        """Internal agent for evaluating idea clarity, faithfulness, and logical consistency."""
        
        def __init__(self, model, config: Dict[str, Any]):
            self.model = model
            self.config = config
            self.system_prompt = config.get("clarity_system_prompt", self._default_system_prompt())
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the clarity, "
                "logical consistency, and factual correctness of research ideas. "
                "Your task is to evaluate whether an idea is reasonable, well-structured, "
                "and consistent with established knowledge and evidence."
            )
        
        def _build_clarity_schema(self) -> Dict[str, Any]:
            """Build JSON schema for clarity evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Clarity score using ICLR 6-level scale. Only allowed values: 1, 3, 5, 6, 8, 10",
                        "enum": [1, 3, 5, 6, 8, 10]
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the clarity evaluation, including assessment of logical consistency, factual correctness, and structural quality"
                    }
                },
                "required": ["score", "reason"]
            }
        
        def _build_clarity_prompt(self, idea_text: str, web_report: str, code_report: str, 
                                  paper_report: str, persona: Dict[str, Any] = None) -> str:
            """Build prompt for clarity evaluation."""
            persona_section = EvaluationAgent._build_persona_section(persona) if persona else ""
            
            # Extract background_knowledge scores and apply masking
            lit_ratio = 1.0
            meth_ratio = 1.0
            if persona and persona.get("background_knowledge"):
                bg_knowledge = persona.get("background_knowledge", {})
                lit_score = bg_knowledge.get("literature_familiarity", 10)
                meth_score = bg_knowledge.get("methodology_depth", 10)
                lit_ratio = max(0.0, min(1.0, lit_score / 10.0))
                meth_ratio = max(0.0, min(1.0, meth_score / 10.0))
            
            # Apply masking: lit controls paper_report and web_report, meth controls code_report
            masked_web_report = EvaluationAgent._mask_report_by_ratio(web_report, lit_ratio) if web_report else ""
            masked_paper_report = EvaluationAgent._mask_report_by_ratio(paper_report, lit_ratio) if paper_report else ""
            masked_code_report = EvaluationAgent._mask_report_by_ratio(code_report, meth_ratio) if code_report else ""
            
            reports_section = ""
            if masked_web_report:
                reports_section += f"\n\n=== Web Research Report ===\n{masked_web_report}\n"
            if masked_code_report:
                reports_section += f"\n\n=== Code Research Report ===\n{masked_code_report}\n"
            if masked_paper_report:
                reports_section += f"\n\n=== Paper Research Report ===\n{masked_paper_report}\n"
            
            if not reports_section:
                reports_section = "\n\nNo research reports are available."

            # reports_section = "No research reports are available."
            
            return f"""{persona_section}You are evaluating the clarity, logical consistency, and factual correctness of a research idea.

=== Research Idea ===
{idea_text}
{reports_section}

=== Evaluation Task ===
Based on your knowledge and the provided research reports, evaluate the idea from the following perspectives:

1. **Logical Consistency**: Is the idea internally consistent? Do the motivation, research question, method, and expected results align logically?

2. **Factual Correctness**: Are the claims and assumptions in the idea consistent with established scientific knowledge and the evidence presented in the reports? Are there any obvious factual errors or contradictions?

3. **Structural Quality**: Is the idea well-structured and clearly presented? Are all components (motivation, research question, method, experimental setting) clearly defined and coherent?

4. **Reasonableness**: Based on the evidence and your knowledge, is the idea reasonable and plausible? Are there any obvious flaws or contradictions?

=== Output Requirements ===
Provide a score using the ICLR 6-level scale (only allowed values: 1, 3, 5, 6, 8, 10) and a detailed reason explaining your evaluation.

=== ICLR Scoring Scale for Clarity ===
- **1 (Strong Reject / Very Poor)**: The idea is fundamentally flawed in clarity — writing is incoherent, logical structure is severely broken, factual errors are pervasive, or the idea is incomprehensible.

- **3 (Reject)**: The idea has substantial clarity issues — significant logical inconsistencies, multiple factual errors, poor structure, or unclear presentation that precludes understanding.

- **5 (Borderline / Weak Reject)**: The idea shows some clarity but has notable weaknesses — some logical inconsistencies, occasional factual issues, or structural problems that need improvement.

- **6 (Weak Accept / Borderline Accept)**: The idea has acceptable clarity overall — reasonable logical consistency, mostly factually correct, adequate structure, but may lack polish or precision.

- **8 (Accept)**: The idea is clear and well-presented — logically consistent, factually sound, well-structured, and easy to understand. This is a solid idea in terms of clarity.

- **10 (Strong Accept / Excellent)**: The idea demonstrates excellent clarity — highly logical, factually accurate, exceptionally well-structured, and clearly presented. The clarity significantly enhances the idea's value.

Focus on evidence-based evaluation using both your knowledge and the provided reports."""
        
        async def evaluate(self, idea_text: str, web_report: str, code_report: str, 
                          paper_report: str, persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea clarity."""
            prompt = self._build_clarity_prompt(idea_text, web_report, code_report, paper_report, persona)
            schema = self._build_clarity_schema()
            
            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"ClarityAgent evaluation failed: {e}")
                raise
    
    class _NoveltyAgent:
        """Internal agent for evaluating idea novelty and originality."""
        
        def __init__(self, model, config: Dict[str, Any]):
            self.model = model
            self.config = config
            self.system_prompt = config.get("novelty_system_prompt", self._default_system_prompt())
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the novelty "
                "and originality of research ideas. Your task is to evaluate whether an idea "
                "is truly novel compared to existing work, or if it has significant overlap "
                "with prior research."
            )
        
        def _build_novelty_schema(self) -> Dict[str, Any]:
            """Build JSON schema for novelty evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Novelty score using ICLR 6-level scale. Only allowed values: 1, 3, 5, 6, 8, 10",
                        "enum": [1, 3, 5, 6, 8, 10]
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the novelty evaluation, including comparison with related work and identification of unique contributions"
                    }
                },
                "required": ["score", "reason"]
            }
        
        def _build_novelty_prompt(self, idea_parts: Dict[str, str], paper_report: str, 
                                  web_report: str, persona: Dict[str, Any] = None) -> str:
            """Build prompt for novelty evaluation."""
            persona_section = EvaluationAgent._build_persona_section(persona) if persona else ""
            
            # Extract background_knowledge scores and apply masking
            lit_ratio = 1.0
            if persona and persona.get("background_knowledge"):
                bg_knowledge = persona.get("background_knowledge", {})
                lit_score = bg_knowledge.get("literature_familiarity", 10)
                lit_ratio = max(0.0, min(1.0, lit_score / 10.0))
            
            # Apply masking: lit controls paper_report and web_report
            masked_paper_report = EvaluationAgent._mask_report_by_ratio(paper_report, lit_ratio) if paper_report else ""
            masked_web_report = EvaluationAgent._mask_report_by_ratio(web_report, lit_ratio) if web_report else ""
            
            idea_section = ""
            if idea_parts.get("motivation"):
                idea_section += f"Motivation: {idea_parts['motivation']}\n\n"
            if idea_parts.get("research_question"):
                idea_section += f"Research Question: {idea_parts['research_question']}\n\n"
            if idea_parts.get("method"):
                idea_section += f"Method: {idea_parts['method']}\n\n"
            
            reports_section = ""
            if masked_paper_report:
                reports_section += f"\n\n=== Paper Research Report (Related Work) ===\n{masked_paper_report}\n"
            if masked_web_report:
                reports_section += f"\n\n=== Web Research Report (Related Discussions) ===\n{masked_web_report}\n"
            
            if not reports_section:
                reports_section = "\n\nNo research reports are available."
            
            # reports_section = "No research reports are available."

            return f"""{persona_section}You are evaluating the novelty and originality of a research idea.

=== Research Idea (Key Components) ===
{idea_section}
{reports_section}

=== Evaluation Task ===
Based on the provided research reports describing related work, evaluate the novelty of the idea from the following perspectives:

1. **Methodological Novelty**: Does the proposed method differ significantly from existing approaches? Are there similar methods that achieve the same goals?

2. **Conceptual Novelty**: Does the idea introduce new concepts, perspectives, or theoretical frameworks? Or does it primarily apply existing concepts?

3. **Conclusion/Result Novelty**: Are the expected conclusions or results similar to what has been found in prior work? Or do they represent new insights?

4. **Analysis Approach**: Does the analysis approach differ from existing work, or is it similar to prior studies?

5. **Overall Originality**: Considering all aspects, how original is this idea compared to the related work described in the reports?

=== Output Requirements ===
Provide a score using the ICLR 6-level scale (only allowed values: 1, 3, 5, 6, 8, 10) and a detailed reason explaining your evaluation.

=== ICLR Scoring Scale for Novelty ===
- **1 (Strong Reject / Very Poor)**: The idea has negligible novelty — essentially identical to existing work in methods, concepts, conclusions, and analysis approach. No meaningful contribution.

- **3 (Reject)**: The idea has minimal novelty — significant overlap with existing work. Minor variations but no substantial differences in methods, concepts, or conclusions.

- **5 (Borderline / Weak Reject)**: The idea shows some promise in novelty but also significant overlap — some novel aspects but many similarities with existing work. Needs more differentiation.

- **6 (Weak Accept / Borderline Accept)**: The idea has acceptable novelty — reasonable differences from existing work in some aspects (methods, concepts, or conclusions), but may lack strong impact or originality.

- **8 (Accept)**: The idea demonstrates decent novelty — clear differences from existing work in methods, concepts, or conclusions. The novelty is meaningful and contributes to the field.

- **10 (Strong Accept / Excellent)**: The idea is highly novel and original — significant differences from existing work in multiple aspects. High novelty or impact, likely to lead to significant interest and follow-up.

Focus on identifying specific similarities and differences with the related work described in the reports."""

        async def evaluate(self, idea_parts: Dict[str, str], paper_report: str, 
                          web_report: str, persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea novelty."""
            prompt = self._build_novelty_prompt(idea_parts, paper_report, web_report, persona)
            schema = self._build_novelty_schema()
            
            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"NoveltyAgent evaluation failed: {e}")
                raise
    
    class _FeasibilityAgent:
        """Internal agent for evaluating idea implementation feasibility."""
        
        def __init__(self, model, config: Dict[str, Any]):
            self.model = model
            self.config = config
            self.system_prompt = config.get("feasibility_system_prompt", self._default_system_prompt())
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert software engineer and research implementer specializing in "
                "assessing the feasibility of implementing research ideas. Your task is to "
                "evaluate whether an idea can be practically implemented using existing code "
                "repositories and libraries."
            )
        
        def _build_feasibility_schema(self) -> Dict[str, Any]:
            """Build JSON schema for feasibility evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Feasibility score using ICLR 6-level scale. Only allowed values: 1, 3, 5, 6, 8, 10",
                        "enum": [1, 3, 5, 6, 8, 10]
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the feasibility evaluation, including assessment of implementation challenges and opportunities"
                    },
                    "pseudocode": {
                        "type": "string",
                        "description": "Pseudocode or implementation plan showing how to integrate existing code repositories to implement the idea's methodology"
                    }
                },
                "required": ["score", "reason", "pseudocode"]
            }
        
        def _build_feasibility_prompt(self, idea_text: str, code_report: str, 
                                     github_file_trees: List[Dict[str, str]], 
                                     persona: Dict[str, Any] = None) -> str:
            """Build prompt for feasibility evaluation."""
            persona_section = EvaluationAgent._build_persona_section(persona) if persona else ""
            
            # Extract background_knowledge scores and apply masking
            meth_ratio = 1.0
            app_ratio = 1.0
            if persona and persona.get("background_knowledge"):
                bg_knowledge = persona.get("background_knowledge", {})
                meth_score = bg_knowledge.get("methodology_depth", 10)
                app_score = bg_knowledge.get("application_experience", 10)
                meth_ratio = max(0.0, min(1.0, meth_score / 10.0))
                app_ratio = max(0.0, min(1.0, app_score / 10.0))
            
            # Apply masking: meth controls code_report, app controls file_trees
            masked_code_report = EvaluationAgent._mask_report_by_ratio(code_report, meth_ratio) if code_report else ""
            filtered_file_trees = EvaluationAgent._filter_file_trees_by_ratio(github_file_trees, app_ratio) if github_file_trees else []
            
            file_trees_section = ""
            if filtered_file_trees:
                file_trees_section = "\n\n=== Available Code Repositories (File Trees) ===\n"
                for i, repo in enumerate(filtered_file_trees, 1):
                    file_trees_section += f"\n--- Repository {i}: {repo.get('title', 'Unknown')} ---\n"
                    file_trees_section += f"URL: {repo.get('url', '')}\n"
                    file_trees_section += f"File Tree:\n{repo.get('file_tree', '')}\n"
            else:
                file_trees_section = "\n\nNo GitHub repository file trees are available."
            
            code_report_section = ""
            if masked_code_report:
                code_report_section = f"\n\n=== Code Research Report ===\n{masked_code_report}\n"
            else:
                code_report_section = "\n\nNo code research report is available."
            
            # code_report_section = "No code research report is available."
            # file_trees_section = "No GitHub repository file trees are available."
            
            return f"""{persona_section}You are evaluating the implementation feasibility of a research idea.

=== Research Idea ===
{idea_text}
{code_report_section}
{file_trees_section}

=== Evaluation Task ===
Based on the provided code repositories (file trees) and code research report, evaluate the feasibility of implementing this idea:

1. **Code Availability**: Can the idea's methodology be implemented using the available code repositories? Are there relevant modules, functions, or components that can be leveraged?

2. **Integration Feasibility**: How easy would it be to integrate and combine existing code from different repositories? Are there compatible interfaces and dependencies?

3. **Implementation Complexity**: How complex would the implementation be? Are there missing components that would need to be implemented from scratch?

4. **Library Support**: Are there standard Python libraries or frameworks that can support the implementation?

5. **Overall Feasibility**: Considering all factors, how feasible is it to implement this idea using existing code resources?

=== Output Requirements ===
Provide:
1. A score using the ICLR 6-level scale (only allowed values: 1, 3, 5, 6, 8, 10) indicating feasibility
2. A detailed reason explaining your assessment
3. Pseudocode or an implementation plan showing how to integrate the available code repositories to implement the idea's methodology

=== ICLR Scoring Scale for Feasibility ===
- **1 (Strong Reject / Very Poor)**: The idea is fundamentally infeasible — missing critical code components, incompatible dependencies, or requires implementation from scratch with no available resources. Implementation is essentially impossible.

- **3 (Reject)**: The idea has substantial feasibility challenges — major missing components, significant integration difficulties, or requires substantial new code development that makes implementation impractical.

- **5 (Borderline / Weak Reject)**: The idea shows some feasibility but also significant challenges — some available code but requires considerable integration work, missing components, or complex dependencies that need resolution.

- **6 (Weak Accept / Borderline Accept)**: The idea has acceptable feasibility — reasonable code availability and integration possibilities, but may require some additional work or components. Implementation is possible but not straightforward.

- **8 (Accept)**: The idea is feasible — good code availability, reasonable integration possibilities, and can be implemented using existing resources with moderate effort. This is a solid idea in terms of feasibility.

- **10 (Strong Accept / Excellent)**: The idea is highly feasible — excellent code availability, easy integration, minimal new development needed. Implementation can be done easily using available code repositories and libraries.

The pseudocode should be clear and show how existing code repositories can be integrated to achieve the idea's goals."""
        
        async def evaluate(self, idea_text: str, code_report: str, 
                          github_file_trees: List[Dict[str, str]], 
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea feasibility."""
            prompt = self._build_feasibility_prompt(idea_text, code_report, github_file_trees, persona)
            schema = self._build_feasibility_schema()
            
            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"FeasibilityAgent evaluation failed: {e}")
                raise

