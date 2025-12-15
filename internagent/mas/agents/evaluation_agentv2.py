"""
Evaluation Agent V2 for InternAgent

This module provides the EvaluationAgentV2 that evaluates research ideas from multiple
aspects using grounded reports from GroundingAgentV2. It processes idea parts and their
associated report summaries to provide comprehensive evaluations.
"""

import logging
import asyncio
import random
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.searchersv2.models import Idea

logger = logging.getLogger(__name__)


class EvaluationAgentV2(BaseAgent):
    """
    Agent that evaluates research ideas from multiple aspects using grounded reports.
    
    Contains five sub-agents:
    - ClarityAgent: Evaluates how well the title and abstract summarize the paper, clarity and structure
    - NoveltyAgent: Evaluates whether it introduces new problems, perspectives, or techniques
    - ValidityAgent: Evaluates theoretical foundations, robust algorithms, and detailed methodologies
    - FeasibilityAgent: Evaluates research design, methodology robustness, and result analysis
    - SignificanceAgent: Evaluates potential contribution and impact on the research community
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "EvaluationAgentV2"
        self.temperature = config.get("temperature", 0.7)
        
        # Initialize sub-agents (as internal classes)
        self.clarity_agent = self._ClarityAgent(model, config)
        self.novelty_agent = self._NoveltyAgent(model, config)
        self.validity_agent = self._ValidityAgent(model, config)
        self.feasibility_agent = self._FeasibilityAgent(model, config)
        self.significance_agent = self._SignificanceAgent(model, config)
        
        logger.info(f"Initialized EvaluationAgentV2 with five sub-agents")
    
    def _extract_idea_text(self, idea: Any) -> str:
        """
        Extract idea text from various input formats.
        
        Args:
            idea: Can be a string, dict with idea fields, or Idea object
            
        Returns:
            Formatted idea text string
        """
        if isinstance(idea, Idea):
            return idea.get_full_text()
        
        if isinstance(idea, str):
            return idea
        
        if isinstance(idea, dict):
            idea_obj = Idea.from_dict(idea)
            return idea_obj.get_full_text()
        
        return str(idea)
    
    def _extract_idea_parts(self, idea: Any) -> Dict[str, str]:
        """
        Extract specific parts of idea as dictionary.
        
        Args:
            idea: Can be a string, dict with idea fields, or Idea object
            
        Returns:
            Dictionary with part_name -> part_content
        """
        if isinstance(idea, Idea):
            idea_dict = idea.to_dict()
        elif isinstance(idea, dict):
            idea_obj = Idea.from_dict(idea)
            idea_dict = idea_obj.to_dict()
        else:
            return {}
        
        # Extract only the part fields (not the _list fields)
        result = {}
        for key in ['basic_idea', 'motivation', 'research_question', 
                   'method', 'experimental_setting', 'expected_results']:
            value = idea_dict.get(key)
            if value:
                result[key] = value
        
        return result
    
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute evaluation of research idea from multiple aspects.
        
        Args:
            context: Must contain:
                - idea: The research idea to evaluate
                - grounding_results: GroundingAgentV2 output dictionary
                    Format: {part_name: {report_type: [{summary, score, report_id}, ...]}}
                - persona: Reviewer persona dictionary (optional)
            params: Additional parameters (temperature override, etc.)
            
        Returns:
            Dictionary containing:
                - clarity: {score, reason}
                - novelty: {score, reason}
                - validity: {score, reason}
                - feasibility: {score, reason, pseudocode}
                - significance: {score, reason}
                - overall: {summary, recommendation}
        """
        # Extract inputs
        idea = context.get("idea")
        if not idea:
            raise AgentExecutionError("context must contain 'idea'")
        
        grounding_results = context.get("grounding_results")
        if not grounding_results:
            raise AgentExecutionError("context must contain 'grounding_results'")
        
        persona = context.get("persona", {})
        
        # Extract idea text and parts
        idea_text = self._extract_idea_text(idea)
        idea_parts = self._extract_idea_parts(idea)
        
        # Execute five sub-agents in parallel
        clarity_task = self.clarity_agent.evaluate(
            idea_text=idea_text,
            idea_parts=idea_parts,
            grounding_results=grounding_results,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        )
        
        novelty_task = self.novelty_agent.evaluate(
            idea_parts=idea_parts,
            grounding_results=grounding_results,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        )
        
        validity_task = self.validity_agent.evaluate(
            idea_text=idea_text,
            idea_parts=idea_parts,
            grounding_results=grounding_results,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        )
        
        feasibility_task = self.feasibility_agent.evaluate(
            idea_text=idea_text,
            idea_parts=idea_parts,
            grounding_results=grounding_results,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        )
        
        significance_task = self.significance_agent.evaluate(
            idea_parts=idea_parts,
            grounding_results=grounding_results,
            persona=persona,
            temperature=params.get("temperature", self.temperature)
        )
        
        # Wait for all evaluations to complete
        try:
            clarity_result, novelty_result, validity_result, feasibility_result, significance_result = await asyncio.gather(
                clarity_task,
                novelty_task,
                validity_task,
                feasibility_task,
                significance_task,
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error during parallel evaluation: {e}")
            raise AgentExecutionError(f"Evaluation failed: {str(e)}")
        
        # Handle exceptions from sub-agents
        if isinstance(clarity_result, Exception):
            logger.error(f"ClarityAgent failed: {clarity_result}")
            clarity_result = {"score": 0.0, "reason": f"Evaluation failed: {str(clarity_result)}"}
        
        if isinstance(novelty_result, Exception):
            logger.error(f"NoveltyAgent failed: {novelty_result}")
            novelty_result = {"score": 0.0, "reason": f"Evaluation failed: {str(novelty_result)}"}
        
        if isinstance(validity_result, Exception):
            logger.error(f"ValidityAgent failed: {validity_result}")
            validity_result = {"score": 0.0, "reason": f"Evaluation failed: {str(validity_result)}"}
        
        if isinstance(feasibility_result, Exception):
            logger.error(f"FeasibilityAgent failed: {feasibility_result}")
            feasibility_result = {"score": 0.0, "reason": f"Evaluation failed: {str(feasibility_result)}", "pseudocode": ""}
        
        if isinstance(significance_result, Exception):
            logger.error(f"SignificanceAgent failed: {significance_result}")
            significance_result = {"score": 0.0, "reason": f"Evaluation failed: {str(significance_result)}"}
        
        # Generate overall summary
        overall_summary = self._generate_overall_summary(
            clarity_result,
            novelty_result,
            validity_result,
            feasibility_result,
            significance_result
        )
        
        return {
            "clarity": clarity_result,
            "novelty": novelty_result,
            "validity": validity_result,
            "feasibility": feasibility_result,
            "significance": significance_result,
            "overall": overall_summary
        }
    
    def _generate_overall_summary(self, clarity: Dict, novelty: Dict, validity: Dict, 
                                  feasibility: Dict, significance: Dict) -> Dict[str, str]:
        """
        Generate overall evaluation summary and recommendation.
        
        Args:
            clarity: Clarity evaluation result
            novelty: Novelty evaluation result
            validity: Validity evaluation result
            feasibility: Feasibility evaluation result
            significance: Significance evaluation result
            
        Returns:
            Dictionary with summary and recommendation
        """
        clarity_score = clarity.get("score", 0.0)
        novelty_score = novelty.get("score", 0.0)
        validity_score = validity.get("score", 0.0)
        feasibility_score = feasibility.get("score", 0.0)
        significance_score = significance.get("score", 0.0)
        
        avg_score = (clarity_score + novelty_score + validity_score + feasibility_score + significance_score) / 5.0
        
        # Generate summary
        summary_parts = [
            f"Clarity Score: {clarity_score:.2f}/10 - {clarity.get('reason', '')[:200]}...",
            f"Novelty Score: {novelty_score:.2f}/10 - {novelty.get('reason', '')[:200]}...",
            f"Validity Score: {validity_score:.2f}/10 - {validity.get('reason', '')[:200]}...",
            f"Feasibility Score: {feasibility_score:.2f}/10 - {feasibility.get('reason', '')[:200]}...",
            f"Significance Score: {significance_score:.2f}/10 - {significance.get('reason', '')[:200]}...",
            f"\nOverall Average Score: {avg_score:.2f}/10"
        ]
        summary = "\n".join(summary_parts)
        
        # Generate recommendation
        if avg_score >= 7.0:
            recommendation = "This idea shows strong potential. It demonstrates good clarity, novelty, validity, feasibility, and significance. Consider proceeding with implementation."
        elif avg_score >= 5.0:
            recommendation = "This idea has moderate potential but may need refinement in some areas. Review the specific concerns raised in each evaluation dimension."
        else:
            recommendation = "This idea faces significant challenges. Consider substantial revisions or exploring alternative approaches before proceeding."
        
        return {
            "summary": summary,
            "recommendation": recommendation
        }
    
    @staticmethod
    def _filter_reports_by_ratio(reports: List[Dict[str, Any]], ratio: float) -> List[Dict[str, Any]]:
        """
        Filter reports by ratio - select integer number of reports to show.
        
        Args:
            reports: List of report dictionaries with {summary, score, report_id}
            ratio: Retention ratio (0.0-1.0), e.g., 0.7 means retaining 70% of reports
            
        Returns:
            Filtered list of reports
        """
        if not reports:
            return []
        
        if ratio >= 1.0:
            return reports
        
        if ratio <= 0.0:
            return []
        
        # Calculate number of reports to keep (integer)
        num_reports = len(reports)
        num_keep = max(1, int(num_reports * ratio))
        
        # Randomly select reports to keep
        selected_indices = random.sample(range(num_reports), num_keep)
        return [reports[idx] for idx in sorted(selected_indices)]
    
    @staticmethod
    def _format_grounded_reports(reports: List[Dict[str, Any]], report_type: str) -> str:
        """
        Format grounded reports (from GroundingAgentV2) for display in prompt.
        
        Args:
            reports: List of report dictionaries with {summary, score, report_id}
            report_type: Type of report - "paper_report", "web_report", or "code_report"
            
        Returns:
            Formatted string containing report summaries
        """
        if not reports:
            return ""
        
        formatted_reports = []
        for i, report in enumerate(reports, 1):
            if not isinstance(report, dict):
                continue
            
            summary = report.get("summary", "")
            score = report.get("score", 0)
            report_id = report.get("report_id", f"{report_type}_{i}")
            
            if summary:
                formatted_reports.append(f"--- {report_id} (Relevance Score: {score}/10) ---\n{summary}")
        
        if formatted_reports:
            return "\n\n".join(formatted_reports)
        else:
            return ""
    
    @staticmethod
    def _build_context_from_grounding(grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                     idea_parts: Dict[str, str],
                                     part_list: List[str],
                                     type_list: List[str],
                                     persona: Dict[str, Any]) -> str:
        """
        Build context string from grounding results, filtered by part_list and type_list.
        
        Args:
            grounding_results: Output from GroundingAgentV2
                Format: {part_name: {report_type: [{summary, score, report_id}, ...]}}
            idea_parts: Dictionary of idea parts (part_name -> content)
            part_list: List of part names to include
            type_list: List of report types to include (e.g., ["paper_report", "web_report"])
            persona: Reviewer persona dictionary
            
        Returns:
            Formatted context string
        """
        # Extract background_knowledge ratios
        lit_ratio = 1.0
        meth_ratio = 1.0
        frontier_ratio = 1.0
        
        if persona and persona.get("background_knowledge"):
            bg_knowledge = persona.get("background_knowledge", {})
            lit_score = bg_knowledge.get("literature_familiarity", 10)
            meth_score = bg_knowledge.get("methodology_depth", 10)
            frontier_score = bg_knowledge.get("frontier_sensitivity", 10)
            lit_ratio = max(0.0, min(1.0, lit_score / 10.0))
            meth_ratio = max(0.0, min(1.0, meth_score / 10.0))
            frontier_ratio = max(0.0, min(1.0, frontier_score / 10.0))
        
        context_parts = []
        
        # Process each part in part_list
        for part_name in part_list:
            # Get idea part content
            part_content = idea_parts.get(part_name, "")
            if not part_content:
                # Skip if no content for this part
                continue
            
            # Build part section
            part_section = f"\n=== {part_name.upper().replace('_', ' ')} ===\n{part_content}\n"
            
            # Process each report type in type_list
            has_reports = False
            part_reports = grounding_results.get(part_name, {})
            
            for report_type in type_list:
                if report_type not in part_reports:
                    continue
                
                reports = part_reports[report_type]
                if not reports:
                    continue
                
                # Apply filtering based on report type
                if report_type == "paper_report":
                    # Use literature_familiarity for paper reports
                    filtered_reports = EvaluationAgentV2._filter_reports_by_ratio(reports, lit_ratio)
                elif report_type == "code_report":
                    # Use methodology_depth for code reports
                    filtered_reports = EvaluationAgentV2._filter_reports_by_ratio(reports, meth_ratio)
                elif report_type == "web_report":
                    # Use frontier_sensitivity for web reports
                    filtered_reports = EvaluationAgentV2._filter_reports_by_ratio(reports, frontier_ratio)
                else:
                    filtered_reports = reports
                
                if filtered_reports:
                    has_reports = True
                    formatted = EvaluationAgentV2._format_grounded_reports(filtered_reports, report_type)
                    if formatted:
                        type_label = report_type.replace("_", " ").title()
                        part_section += f"\n--- {type_label} ---\n{formatted}\n"
            
            # Always add part section if it has content, even if no reports
            context_parts.append(part_section)
        
        if context_parts:
            return "\n".join(context_parts)
        else:
            return "\n[No relevant reports available for the selected parts and types.]"
    
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
            frontier = background_knowledge.get("frontier_sensitivity", "N/A")
            persona_section += f"Background Knowledge:\n"
            persona_section += f"  - Literature Familiarity: {lit}/10 (controls paper reports)\n"
            persona_section += f"  - Methodology Depth: {meth}/10 (controls code reports)\n"
            persona_section += f"  - Frontier Sensitivity: {frontier}/10 (controls web reports)\n\n"
            persona_section += "Note: Based on the background knowledge scores above, the research reports provided below have been randomly filtered to reflect the reviewer's knowledge level. Lower scores result in fewer reports being shown. Paper reports are filtered by literature familiarity, code reports by methodology depth, and web reports by frontier sensitivity.\n\n"
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
            # Define which parts and report types this agent needs
            self.part_list = config.get("clarity_part_list", ["basic_idea", "motivation", "research_question", "method", "experimental_setting"])
            self.type_list = config.get("clarity_type_list", ["paper_report", "web_report", "code_report"])
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the clarity "
                "of research ideas. Your task is to evaluate how well the title and abstract "
                "summarize the paper, whether the content is clear, concise, and informative, "
                "and whether the idea is well-structured and easy to follow."
            )
        
        def _build_clarity_schema(self) -> Dict[str, Any]:
            """Build JSON schema for clarity evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Clarity score from 0 to 10, where 10 indicates excellent clarity, well-summarized title and abstract, clear and concise presentation, and well-structured content",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the clarity evaluation, including assessment of title/abstract quality, clarity and conciseness, and structural quality"
                    }
                },
                "required": ["score", "reason"]
            }
        
        def _build_clarity_prompt(self, idea_text: str, idea_parts: Dict[str, str],
                                  grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                  persona: Dict[str, Any] = None) -> str:
            """Build prompt for clarity evaluation."""
            persona_section = EvaluationAgentV2._build_persona_section(persona) if persona else ""
            
            # Build context from grounding results
            context_section = EvaluationAgentV2._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )
            
            return f"""{persona_section}You are evaluating the clarity of a research idea.

=== Research Idea ===
{idea_text}
{context_section}

=== Evaluation Task ===
Based on your knowledge and the provided research reports, evaluate the idea from the following perspectives:

1. **Title and Abstract Quality**: How well do the title and abstract summarize the paper? Are they clear, concise, and informative? Do they effectively convey the significance and main contributions of the research?

2. **Alignment**: Are the title and abstract well-aligned with each other? Do they accurately represent the core idea and content of the paper?

3. **Clarity and Conciseness**: Is the content clear, concise, and informative? Does the paper effectively convey its significance and main contributions?

4. **Structure and Organization**: Is the content well-structured and easy to follow? Are all components (motivation, research question, method, experimental setting, expected results) clearly defined and coherent?

=== Scoring Guidelines ===
As a peer reviewer, you must maintain strict and fair evaluation standards:

1. **Critical Perspective**: Approach this evaluation with a critical eye. Scrutinize the idea thoroughly and identify any weaknesses, gaps, or areas that need improvement. Do not be overly lenient.

2. **Review Standards**: This is a peer review process. Maintaining fairness and rigor is essential. Being overly generous undermines the integrity of the review process and does not serve the research community.

3. **Score Distribution**: Based on statistical constraints, approximately only 30% of submissions should receive a passing score (>=7.0). When assigning scores, consider this distribution. Reserve scores of 7.0 or higher for ideas that truly demonstrate excellence. Most ideas should receive scores below 7.0.

**Clarity Scoring Scale** (evaluate title/abstract quality, clarity, conciseness, and structure):
   - Score 8-10: Exceptional clarity - title and abstract perfectly summarize the paper, content is exceptionally clear and concise, structure is exemplary with all components well-defined and coherent (exceedingly rare, reserve for truly outstanding cases)
   - Score 6-7: Good clarity - title and abstract effectively summarize the paper, content is generally clear and well-structured, most components are well-defined (use conservatively, only when clarity is genuinely strong)
   - Score 4-5: Moderate clarity - title and abstract provide basic summary but lack precision, content has some clarity issues or structural weaknesses, some components need improvement (more common - use when clarity is adequate but not strong)
   - Score 2-3: Poor clarity - title and abstract are vague or misaligned, content is unclear or poorly organized, significant structural issues or missing components (common - use when clarity problems are evident)
   - Score 0-1: Very poor clarity - title and abstract fail to convey the idea, content is confusing or incoherent, severe structural problems or major missing components

=== Output Requirements ===
Provide a score from 0 to 10 and a detailed reason explaining your evaluation.

Focus on evidence-based evaluation using both your knowledge and the provided reports. Apply strict standards and justify your score accordingly."""
        
        async def evaluate(self, idea_text: str, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea clarity."""
            prompt = self._build_clarity_prompt(idea_text, idea_parts, grounding_results, persona)
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
            # Define which parts and report types this agent needs
            self.part_list = config.get("novelty_part_list", ["motivation", "basic_idea", "research_question", "method"])
            self.type_list = config.get("novelty_type_list", ["paper_report", "web_report"])
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the novelty "
                "and originality of research ideas. Your task is to evaluate whether an idea "
                "introduces new problems or perspectives, new techniques, or represents a "
                "significant advancement compared to existing methods, and how it aligns with "
                "or diverges from current research trends."
            )
        
        def _build_novelty_schema(self) -> Dict[str, Any]:
            """Build JSON schema for novelty evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Novelty score from 0 to 10, where 10 indicates the idea introduces new problems/perspectives or new techniques with significant advancement, and 0 indicates significant overlap with existing work",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the novelty evaluation, including assessment of new problems/perspectives, new techniques, advancement compared to existing methods, and alignment with research trends"
                    }
                },
                "required": ["score", "reason"]
            }
        
        def _build_novelty_prompt(self, idea_parts: Dict[str, str],
                                  grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                  persona: Dict[str, Any] = None) -> str:
            """Build prompt for novelty evaluation."""
            persona_section = EvaluationAgentV2._build_persona_section(persona) if persona else ""
            
            # Build context from grounding results
            context_section = EvaluationAgentV2._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )
            
            return f"""{persona_section}You are evaluating the novelty and originality of a research idea.

=== Research Idea (Key Components) ===
{context_section}

=== Evaluation Task ===
Based on the provided research reports describing related work, evaluate the novelty of the idea from the following perspectives:

1. **New Problem or Perspective**: Does it introduce a new problem or perspective that has not been explored before? Does it address questions or angles that existing work has not covered?

2. **New Techniques**: Does it introduce new techniques or represent a significant advancement compared to existing methods? Are the proposed techniques substantially different from prior approaches?

3. **Advancement Compared to Existing Methods**: How does the proposed approach compare to existing methods? Does it represent a meaningful improvement or advancement?

4. **Alignment with Research Trends**: How does it align with or diverge from current research trends? Does it follow existing trends or chart new directions?

=== Scoring Guidelines ===
As a peer reviewer, you must maintain strict and fair evaluation standards:

1. **Critical Perspective**: Approach this evaluation with a critical eye. Scrutinize the idea thoroughly and identify any weaknesses, gaps, or areas that need improvement. Do not be overly lenient.

2. **Review Standards**: This is a peer review process. Maintaining fairness and rigor is essential. Being overly generous undermines the integrity of the review process and does not serve the research community.

3. **Score Distribution**: Based on statistical constraints, approximately only 30% of submissions should receive a passing score (>=7.0). When assigning scores, consider this distribution. Reserve scores of 7.0 or higher for ideas that truly demonstrate excellence. Most ideas should receive scores below 7.0.

**Novelty Scoring Scale** (evaluate new problems/perspectives, new techniques, advancement over existing methods, and alignment with trends):
   - Score 8-10: Exceptional novelty - introduces genuinely new problems or perspectives not explored before, proposes substantially new techniques with significant advancement, represents major departure from existing methods (exceedingly rare, reserve for truly groundbreaking cases)
   - Score 6-7: Good novelty - introduces new angles or perspectives, proposes meaningful improvements or new techniques, shows clear advancement over existing methods (use conservatively, only when novelty is genuinely strong)
   - Score 4-5: Moderate novelty - introduces some new aspects but with significant overlap with existing work, proposes incremental improvements, shows limited advancement over existing methods (more common - use when novelty exists but is limited)
   - Score 2-3: Low novelty - largely follows existing problems/perspectives, minor variations of existing techniques, minimal advancement over existing methods (common - use when novelty is weak)
   - Score 0-1: No novelty - completely overlaps with existing work, no new problems/perspectives or techniques, essentially replicates existing methods

=== Output Requirements ===
Provide a score from 0 to 10 and a detailed reason explaining your evaluation.
Focus on identifying specific similarities and differences with the related work described in the reports, and assess whether the idea introduces genuinely new problems, perspectives, or techniques. Apply strict standards and justify your score accordingly."""
        
        async def evaluate(self, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea novelty."""
            prompt = self._build_novelty_prompt(idea_parts, grounding_results, persona)
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
            # Define which parts and report types this agent needs
            self.part_list = config.get("feasibility_part_list", ["method", "experimental_setting"])
            self.type_list = config.get("feasibility_type_list", ["code_report"])
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the feasibility "
                "of research ideas. Your task is to evaluate whether the research design and "
                "methods are clearly described and justified, whether the methodology is robust "
                "and appropriate for addressing the research questions, and whether the results "
                "are well-analyzed and interpreted."
            )
        
        def _build_feasibility_schema(self) -> Dict[str, Any]:
            """Build JSON schema for feasibility evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Feasibility score from 0 to 10, where 10 indicates clear research design, robust methodology, well-analyzed results, and findings that support claims, and 0 indicates significant issues in these areas",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the feasibility evaluation, including assessment of research design clarity, methodology robustness, result analysis quality, and whether findings support claims"
                    },
                    "pseudocode": {
                        "type": "string",
                        "description": "Pseudocode or implementation plan showing how to integrate existing code repositories to implement the idea's methodology (if applicable)"
                    }
                },
                "required": ["score", "reason", "pseudocode"]
            }
        
        def _build_feasibility_prompt(self, idea_text: str, idea_parts: Dict[str, str],
                                     grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                     persona: Dict[str, Any] = None) -> str:
            """Build prompt for feasibility evaluation."""
            persona_section = EvaluationAgentV2._build_persona_section(persona) if persona else ""
            
            # Build context from grounding results
            context_section = EvaluationAgentV2._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )
            
            return f"""{persona_section}You are evaluating the feasibility of a research idea.

=== Research Idea ===
{idea_text}
{context_section}

=== Evaluation Task ===
Based on the provided research reports and code repositories, evaluate the feasibility of this idea from the following perspectives:

1. **Research Design and Methods**: Are the research design and methods clearly described and justified? Is the experimental setup well-defined and appropriate?

2. **Methodology Robustness**: Is the methodology robust and appropriate for addressing the research questions? Are the methods sound and well-suited to the problem?

3. **Result Analysis and Interpretation**: Are the results well-analyzed and interpreted? Is the analysis thorough and appropriate for the type of results expected?

4. **Findings Support Claims**: Do the findings support the claims made in the idea? Is there a logical connection between the methodology, results, and conclusions?

5. **Implementation Feasibility** (if applicable): Can the idea's methodology be implemented using available code repositories? Are there relevant modules, functions, or components that can be leveraged?

=== Scoring Guidelines ===
As a peer reviewer, you must maintain strict and fair evaluation standards:

1. **Critical Perspective**: Approach this evaluation with a critical eye. Scrutinize the idea thoroughly and identify any weaknesses, gaps, or areas that need improvement. Do not be overly lenient.

2. **Review Standards**: This is a peer review process. Maintaining fairness and rigor is essential. Being overly generous undermines the integrity of the review process and does not serve the research community.

3. **Score Distribution**: Based on statistical constraints, approximately only 30% of submissions should receive a passing score (>=7.0). When assigning scores, consider this distribution. Reserve scores of 7.0 or higher for ideas that truly demonstrate excellence. Most ideas should receive scores below 7.0.

**Feasibility Scoring Scale** (evaluate research design clarity, methodology robustness, result analysis quality, and implementation feasibility):
   - Score 8-10: Exceptional feasibility - research design is exceptionally clear and well-justified, methodology is highly robust and perfectly suited to research questions, result analysis plan is thorough and appropriate, implementation is clearly feasible with available resources (exceedingly rare, reserve for truly outstanding cases)
   - Score 6-7: Good feasibility - research design is clear and justified, methodology is robust and appropriate, result analysis plan is adequate, implementation appears feasible (use conservatively, only when feasibility is genuinely strong)
   - Score 4-5: Moderate feasibility - research design has some clarity issues or gaps, methodology has some weaknesses or may not fully address research questions, result analysis plan needs improvement, implementation has some concerns (more common - use when feasibility is adequate but not strong)
   - Score 2-3: Poor feasibility - research design is unclear or poorly justified, methodology has significant weaknesses or is inappropriate, result analysis plan is inadequate, implementation faces major challenges (common - use when feasibility problems are evident)
   - Score 0-1: Very poor feasibility - research design is severely flawed or missing, methodology is fundamentally unsound or inappropriate, result analysis plan is missing or inadequate, implementation is not feasible

=== Output Requirements ===
Provide:
1. A score from 0 to 10 indicating feasibility
2. A detailed reason explaining your assessment
3. Pseudocode or an implementation plan (if applicable) showing how to integrate the available code repositories to implement the idea's methodology

Focus on evaluating the research design, methodology quality, and result analysis rather than just implementation feasibility. Apply strict standards and justify your score accordingly."""
        
        async def evaluate(self, idea_text: str, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea feasibility."""
            prompt = self._build_feasibility_prompt(idea_text, idea_parts, grounding_results, persona)
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
    
    class _ValidityAgent:
        """Internal agent for evaluating idea validity, theoretical foundations, and methodological rigor."""
        
        def __init__(self, model, config: Dict[str, Any]):
            self.model = model
            self.config = config
            self.system_prompt = config.get("validity_system_prompt", self._default_system_prompt())
            # Define which parts and report types this agent needs
            self.part_list = config.get("validity_part_list", ["motivation", "basic_idea", "research_question", "method", "experimental_setting"])
            self.type_list = config.get("validity_type_list", ["paper_report", "web_report", "code_report"])
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the validity "
                "of research ideas. Your task is to evaluate whether an idea includes solid "
                "theoretical foundations, robust algorithms, and detailed methodologies, and "
                "whether the underlying principles are well-defined and logically consistent."
            )
        
        def _build_validity_schema(self) -> Dict[str, Any]:
            """Build JSON schema for validity evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Validity score from 0 to 10, where 10 indicates solid theoretical foundations, robust algorithms, detailed methodologies, and well-defined logically consistent principles, and 0 indicates significant gaps in these areas",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the validity evaluation, including assessment of theoretical foundations, algorithm robustness, methodology detail, and logical consistency"
                    }
                },
                "required": ["score", "reason"]
            }
        
        def _build_validity_prompt(self, idea_text: str, idea_parts: Dict[str, str],
                                   grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                   persona: Dict[str, Any] = None) -> str:
            """Build prompt for validity evaluation."""
            persona_section = EvaluationAgentV2._build_persona_section(persona) if persona else ""
            
            # Build context from grounding results
            context_section = EvaluationAgentV2._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )
            
            return f"""{persona_section}You are evaluating the validity of a research idea.

=== Research Idea ===
{idea_text}
{context_section}

=== Evaluation Task ===
Based on your knowledge and the provided research reports, evaluate the validity of the idea from the following perspectives:

1. **Theoretical Foundations**: Does it include solid theoretical foundations? Are the underlying principles well-defined and logically consistent? Is there a sound theoretical basis for the approach?

2. **Robust Algorithms**: Are the algorithms robust? Are they well-designed, theoretically sound, and appropriate for addressing the research problem? Do they have clear algorithmic foundations?

3. **Detailed Methodologies**: Are the methodologies detailed and well-specified? Is there sufficient detail to understand and potentially replicate the approach? Are all necessary steps clearly described?

4. **Logical Consistency**: Are the underlying principles well-defined and logically consistent? Do the theoretical foundations, algorithms, and methodologies align coherently?

5. **Addressing Research Problem**: Do the theoretical foundations, algorithms, and methodologies effectively address the research problem? Is there a clear connection between the problem and the proposed solution?

=== Scoring Guidelines ===
As a peer reviewer, you must maintain strict and fair evaluation standards:

1. **Critical Perspective**: Approach this evaluation with a critical eye. Scrutinize the idea thoroughly and identify any weaknesses, gaps, or areas that need improvement. Do not be overly lenient.

2. **Review Standards**: This is a peer review process. Maintaining fairness and rigor is essential. Being overly generous undermines the integrity of the review process and does not serve the research community.

3. **Score Distribution**: Based on statistical constraints, approximately only 30% of submissions should receive a passing score (>=7.0). When assigning scores, consider this distribution. Reserve scores of 7.0 or higher for ideas that truly demonstrate excellence. Most ideas should receive scores below 7.0.

**Validity Scoring Scale** (evaluate theoretical foundations, algorithm robustness, methodology detail, and logical consistency):
   - Score 8-10: Exceptional validity - solid and well-established theoretical foundations, highly robust and theoretically sound algorithms, detailed and comprehensive methodologies, perfect logical consistency throughout (exceedingly rare, reserve for truly outstanding cases)
   - Score 6-7: Good validity - sound theoretical foundations, robust and well-designed algorithms, adequate methodology detail, generally logically consistent (use conservatively, only when validity is genuinely strong)
   - Score 4-5: Moderate validity - theoretical foundations have some gaps or weaknesses, algorithms have some robustness concerns, methodology lacks sufficient detail, some logical inconsistencies (more common - use when validity is adequate but not strong)
   - Score 2-3: Poor validity - weak or missing theoretical foundations, algorithms have significant robustness issues, methodology lacks critical details, notable logical inconsistencies (common - use when validity problems are evident)
   - Score 0-1: Very poor validity - severely flawed or missing theoretical foundations, algorithms are fundamentally unsound, methodology is severely lacking, major logical inconsistencies or contradictions

=== Output Requirements ===
Provide a score from 0 to 10 and a detailed reason explaining your evaluation.

Focus on evidence-based evaluation using both your knowledge and the provided reports. Assess the rigor and validity of the theoretical and methodological foundations. Apply strict standards and justify your score accordingly."""
        
        async def evaluate(self, idea_text: str, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea validity."""
            prompt = self._build_validity_prompt(idea_text, idea_parts, grounding_results, persona)
            schema = self._build_validity_schema()
            
            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"ValidityAgent evaluation failed: {e}")
                raise
    
    class _SignificanceAgent:
        """Internal agent for evaluating idea significance and potential impact."""
        
        def __init__(self, model, config: Dict[str, Any]):
            self.model = model
            self.config = config
            self.system_prompt = config.get("significance_system_prompt", self._default_system_prompt())
            # Define which parts and report types this agent needs
            self.part_list = config.get("significance_part_list", ["motivation", "basic_idea", "research_question", "method"])
            self.type_list = config.get("significance_type_list", ["paper_report", "web_report"])
        
        def _default_system_prompt(self) -> str:
            return (
                "You are an expert research evaluator specializing in assessing the significance "
                "and potential impact of research ideas. Your task is to evaluate the potential "
                "contribution and impact on the research community in its specific domain and "
                "beyond, and how it compares to existing works in terms of impact."
            )
        
        def _build_significance_schema(self) -> Dict[str, Any]:
            """Build JSON schema for significance evaluation output."""
            return {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Significance score from 0 to 10, where 10 indicates high potential contribution and impact on the research community in its domain and beyond, and 0 indicates minimal impact",
                        "minimum": 0,
                        "maximum": 10
                    },
                    "reason": {
                        "type": "string",
                        "description": "Detailed explanation of the significance evaluation, including assessment of potential contribution, impact on research community, and comparison with existing works"
                    }
                },
                "required": ["score", "reason"]
            }
        
        def _build_significance_prompt(self, idea_parts: Dict[str, str],
                                       grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                                       persona: Dict[str, Any] = None) -> str:
            """Build prompt for significance evaluation."""
            persona_section = EvaluationAgentV2._build_persona_section(persona) if persona else ""
            
            # Build context from grounding results
            context_section = EvaluationAgentV2._build_context_from_grounding(
                grounding_results=grounding_results,
                idea_parts=idea_parts,
                part_list=self.part_list,
                type_list=self.type_list,
                persona=persona
            )
            
            return f"""{persona_section}You are evaluating the significance and potential impact of a research idea.

=== Research Idea (Key Components) ===
{context_section}

=== Evaluation Task ===
Based on the provided research reports describing related work, evaluate the significance of the idea from the following perspectives:

1. **Potential Contribution**: What is the potential contribution of this idea? What new insights, solutions, or understanding does it offer?

2. **Impact on Specific Domain**: How significant is the potential impact on the research community in its specific domain? Does it address important problems or advance the field meaningfully?

3. **Impact Beyond Domain**: Does the idea have potential impact beyond its specific domain? Could it influence other research areas or have broader applications?

4. **Comparison with Existing Works**: How does it compare to existing works in terms of impact? Does it represent a meaningful advancement or improvement over prior research?

5. **Long-term Significance**: What is the long-term significance? Could this idea lead to further research directions or practical applications?

=== Scoring Guidelines ===
As a peer reviewer, you must maintain strict and fair evaluation standards:

1. **Critical Perspective**: Approach this evaluation with a critical eye. Scrutinize the idea thoroughly and identify any weaknesses, gaps, or areas that need improvement. Do not be overly lenient.

2. **Review Standards**: This is a peer review process. Maintaining fairness and rigor is essential. Being overly generous undermines the integrity of the review process and does not serve the research community.

3. **Score Distribution**: Based on statistical constraints, approximately only 30% of submissions should receive a passing score (>=7.0). When assigning scores, consider this distribution. Reserve scores of 7.0 or higher for ideas that truly demonstrate excellence. Most ideas should receive scores below 7.0.

**Significance Scoring Scale** (evaluate potential contribution, impact on domain and beyond, comparison with existing works, and long-term significance):
   - Score 8-10: Exceptional significance - offers transformative insights or solutions, has profound impact potential in domain and beyond, represents major advancement over existing works, has strong potential for long-term influence and further research directions (exceedingly rare, reserve for truly groundbreaking cases)
   - Score 6-7: Good significance - offers meaningful contributions and insights, has substantial impact potential in domain with some broader implications, represents clear advancement over existing works, has potential for further research (use conservatively, only when significance is genuinely strong)
   - Score 4-5: Moderate significance - offers some contributions but limited in scope, has moderate impact potential primarily within domain, represents incremental improvement over existing works, limited long-term potential (more common - use when significance exists but is limited)
   - Score 2-3: Low significance - offers minimal contributions, has limited impact potential even within domain, represents minor improvement or comparable to existing works, little long-term potential (common - use when significance is weak)
   - Score 0-1: No significance - offers no meaningful contributions, has negligible impact potential, does not advance beyond existing works, no long-term potential

=== Output Requirements ===
Provide a score from 0 to 10 and a detailed reason explaining your evaluation.

Focus on assessing the potential contribution and impact, considering both the immediate domain and broader research community. Compare with existing works to contextualize the significance. Apply strict standards and justify your score accordingly."""
        
        async def evaluate(self, idea_parts: Dict[str, str],
                          grounding_results: Dict[str, Dict[str, List[Dict[str, Any]]]],
                          persona: Dict[str, Any] = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
            """Evaluate idea significance."""
            prompt = self._build_significance_prompt(idea_parts, grounding_results, persona)
            schema = self._build_significance_schema()
            
            try:
                result = await self.model.generate_json(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=self.system_prompt,
                    temperature=temperature
                )
                return result
            except Exception as e:
                logger.error(f"SignificanceAgent evaluation failed: {e}")
                raise

