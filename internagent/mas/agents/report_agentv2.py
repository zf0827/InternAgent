import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.searchersv2.models import Idea, SearchResults, Source, SourceType

logger = logging.getLogger(__name__)


class ReportAgentV2(BaseAgent):
    """
    根据多位 persona 的评审结果生成最终报告（final_report）。
    - 输入：idea、sources（SearchResults 或包含 papers/web_pages/github_repos 的 dict）、
      evaluation_results（来自 EvaluationAgentV2 的列表）、future_papers（已 enrich，metadata.paper_extract）
    - 输出：包含 final_report（字符串）、final_decision（dict）、revision_advice（字符串）
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "ReportAgentV2"
        self.temperature = config.get("temperature", 0.4)

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        idea = context.get("idea")
        evaluation_results = context.get("evaluation_results")
        sources = context.get("sources") or context.get("search_results") or context.get("search_result")
        future_papers = context.get("future_papers", [])

        if not idea:
            raise AgentExecutionError("context 必须包含 'idea'")
        if not evaluation_results or not isinstance(evaluation_results, list):
            raise AgentExecutionError("context 必须包含非空的 'evaluation_results' 列表")

        idea_text = self._extract_idea_text(idea)
        paper_sources, web_sources, code_sources = self._extract_sources(sources)

        paper_block = self._format_paper_resources(paper_sources)
        web_block = self._format_web_resources(web_sources)
        code_block = self._format_code_resources(code_sources)
        evaluation_block = self._format_evaluations(evaluation_results)

        final_decision = await self._generate_final_decision(
            evaluation_results=evaluation_results,
            idea_text=idea_text,
            params=params,
        )
        revision_advice = await self._generate_revision_advice(
            idea_text=idea_text,
            future_papers=future_papers,
            params=params,
        )

        final_report = self._assemble_final_report(
            idea_text=idea_text,
            paper_block=paper_block,
            web_block=web_block,
            code_block=code_block,
            evaluation_block=evaluation_block,
            final_decision=final_decision,
            revision_advice=revision_advice,
        )

        return {
            "final_report": final_report,
            "final_decision": final_decision,
            "revision_advice": revision_advice,
            "params": params,
        }

    # ------------------------------------------------------------------ #
    # 数据提取与格式化
    # ------------------------------------------------------------------ #
    def _extract_idea_text(self, idea: Any) -> str:
        if isinstance(idea, Idea):
            return idea.get_full_text()
        if isinstance(idea, str):
            return idea
        if isinstance(idea, dict):
            try:
                return Idea.from_dict(idea).get_full_text()
            except Exception:
                pass
            parts = []
            for key in [
                "basic_idea",
                "motivation",
                "research_question",
                "method",
                "experimental_setting",
                "expected_results",
            ]:
                if idea.get(key):
                    parts.append(f"{key.replace('_', ' ').title()}: {idea[key]}")
            return "\n".join(parts)
        return str(idea)

    def _extract_sources(self, sources: Any) -> Tuple[List[Source], List[Source], List[Source]]:
        papers: List[Source] = []
        webs: List[Source] = []
        codes: List[Source] = []

        if not sources:
            return papers, webs, codes

        def _to_source(item: Union[Source, Dict[str, Any]]) -> Optional[Source]:
            if isinstance(item, Source):
                return item
            if isinstance(item, dict):
                try:
                    return Source.from_dict(item)
                except Exception:
                    return None
            return None

        if isinstance(sources, SearchResults):
            papers = [s for s in sources.papers if isinstance(s, Source)]
            webs = [s for s in sources.web_pages if isinstance(s, Source)]
            codes = [s for s in sources.github_repos if isinstance(s, Source)]
        elif isinstance(sources, dict):
            for item in sources.get("papers", []) + sources.get("scholar_results", []):
                src = _to_source(item)
                if src:
                    papers.append(src)
            for item in sources.get("web_pages", []):
                src = _to_source(item)
                if src:
                    webs.append(src)
            for item in sources.get("github_repos", []) + sources.get("kaggle_results", []):
                src = _to_source(item)
                if src:
                    codes.append(src)

        return papers, webs, codes

    def _format_paper_resources(self, papers: List[Source]) -> str:
        if not papers:
            return "No paper resources."

        blocks = []
        for idx, paper in enumerate(papers, 1):
            meta = paper.metadata or {}
            extract = meta.get("paper_extract") or {}
            desc_parts = []
            for key in ["basic_idea", "motivation", "method", "research_question", "expected_results"]:
                val = extract.get(key)
                if not val:
                    continue
                if isinstance(val, list):
                    val = " ".join([str(v) for v in val])
                desc_parts.append(f"{key.replace('_', ' ').title()}: {val}")
            desc = "\n".join(desc_parts) if desc_parts else (paper.description or "")
            blocks.append(f"- Paper {idx}: {paper.title or 'Unknown'}\n{desc}".strip())
        return "\n".join(blocks)

    def _format_web_resources(self, webs: List[Source]) -> str:
        if not webs:
            return "No web resources."
        blocks = []
        for idx, web in enumerate(webs, 1):
            meta = web.metadata or {}
            content = ""
            if meta.get("web_report") and isinstance(meta["web_report"], dict):
                content = meta["web_report"].get("summary") or meta["web_report"].get("report_content", "")
            if not content:
                content = web.description or (web.page_raw_text or "")[:400]
            blocks.append(f"- Web {idx}: {web.title or web.url or 'Unknown'}\n{content}".strip())
        return "\n".join(blocks)

    def _format_code_resources(self, codes: List[Source]) -> str:
        if not codes:
            return "No code resources."
        blocks = []
        for idx, code in enumerate(codes, 1):
            meta = code.metadata or {}
            content = ""
            if meta.get("code_report") and isinstance(meta["code_report"], dict):
                content = meta["code_report"].get("summary") or meta["code_report"].get("report_content", "")
            if not content:
                content = code.description or code.repo_context or ""
            blocks.append(f"- Code {idx}: {code.title or code.url or 'Unknown'}\n{content}".strip())
        return "\n".join(blocks)

    def _format_evaluations(self, evaluation_results: List[Dict[str, Any]]) -> str:
        sections = []
        for idx, item in enumerate(evaluation_results, 1):
            evaluation = item.get("evaluation", item)
            persona = item.get("persona", {})
            reviewer_header = f"## Reviewer {idx}"
            persona_line = persona.get("background") or ""

            def _section(name: str) -> str:
                data = evaluation.get(name, {}) or {}
                reason = data.get("reason", "No reason provided.")
                score = data.get("score", "N/A")
                return f"### {name.capitalize()}\n{reason}\nScore: {score}"

            clarity = _section("clarity")
            novelty = _section("novelty")
            validity = _section("validity")
            feasibility = _section("feasibility")
            significance = _section("significance")

            overall = evaluation.get("overall", {})
            overall_reason = overall.get("summary", overall.get("reason", "No overall summary."))
            overall_score = overall.get("score") or overall.get("average_score") or ""
            overall_block = f"### Summary\n{overall_reason}\nOverall Score: {overall_score}"

            parts = [
                reviewer_header,
                persona_line,
                clarity,
                novelty,
                validity,
                feasibility,
                significance,
                overall_block,
            ]
            sections.append("\n\n".join([p for p in parts if p]))

        return "\n\n".join(sections)

    # ------------------------------------------------------------------ #
    # 生成 Final Decision / Revision Advice
    # ------------------------------------------------------------------ #
    def _build_final_decision_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Overall rationale synthesizing all reviewers' evaluations"},
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Final score on a continuous scale from 0 to 10 (decimals allowed). The score should align with the decision category based on ICLR2025 statistics.",
                },
                "decision": {
                    "type": "string",
                    "enum": ["reject", "poster", "spotlight", "oral"],
                    "description": "Final acceptance decision type",
                },
            },
            "required": ["reason", "score", "decision"],
        }

    def _build_final_decision_prompt(self, evaluation_results: List[Dict[str, Any]], idea_text: str) -> str:
        eval_summaries = []
        for idx, item in enumerate(evaluation_results, 1):
            evaluation = item.get("evaluation", item)
            persona = item.get("persona", {})
            persona_tag = persona.get("background") or persona.get("goal") or f"Reviewer {idx}"
            parts = []
            for key in ["clarity", "novelty", "validity", "feasibility", "significance"]:
                data = evaluation.get(key, {}) or {}
                parts.append(f"{key.title()}: {data.get('score', 'N/A')}/10 – {data.get('reason', '')}")
            overall = evaluation.get("overall", {})
            overall_txt = overall.get("summary", "")
            parts.append(f"Overall: {overall_txt}")
            eval_summaries.append(f"Reviewer {idx} ({persona_tag}):\n" + "\n".join(parts))

        iclr_scale = """
1 (Strong Reject): Fatal flaws (unsound, trivial, missing evaluation, incoherent, or negligible novelty)
3 (Reject): Major weaknesses preclude acceptance; some interesting ideas but substantial issues
5 (Weak Reject): Some promise but clear significant weaknesses; needs major improvements
6 (Weak Accept): Acceptable overall quality; adequate novelty/rigor but not outstanding
8 (Accept): Solid contribution, clear writing, decent novelty/impact, thorough evaluation
10 (Strong Accept): Excellent novelty/impact, sound methodology, convincing experiments, very clear
""".strip()

        decision_stats = """
ICLR2025 Review Statistics (for reference):
- reject: 42.17% of submissions, average score 4.77, standard deviation 0.92
- poster: 25.65% of submissions, average score 6.23, standard deviation 0.50
- spotlight: 3.26% of submissions, average score 7.35, standard deviation 0.26
- oral: 1.82% of submissions, average score 7.79, standard deviation 0.55

These statistics reflect the actual distribution and scoring patterns from ICLR2025. Use them as guidance to ensure your scoring aligns with realistic review standards.
""".strip()

        return f"""
You are an ICLR-style meta-reviewer conducting a rigorous peer review. Your goal is to provide an objective evaluation that aligns with human reviewer decisions and ICLR2025 review patterns.

=== Review Guidelines ===
1. **Align with human reviewer decisions**: The ICLR2025 statistics below represent actual human reviewer behavior and decision patterns. Your evaluation should closely match these patterns. This is the PRIMARY guideline for your evaluation.
2. **Objective and balanced assessment**: Conduct a thorough, objective analysis that fairly weighs both strengths and weaknesses. Avoid systematic bias toward either extreme leniency or excessive strictness.
3. **Adopt a critical perspective**: While maintaining objectivity, approach the review with analytical rigor. Identify both strengths and weaknesses, limitations and contributions, to ensure a balanced evaluation.
4. **ICLR2025 statistics alignment**: The following distribution and scoring patterns from ICLR2025 represent how human reviewers actually evaluated submissions. Your decision and score should align with these patterns:
{decision_stats}
5. **Score alignment with decision category**: Your final score (0-10, decimals allowed) must logically align with your decision category based on the ICLR2025 statistics:
   - reject decisions typically correspond to scores around 4.77 ± 0.92
   - poster decisions typically correspond to scores around 6.23 ± 0.50
   - spotlight decisions typically correspond to scores around 7.35 ± 0.26
   - oral decisions typically correspond to scores around 7.79 ± 0.55
   Ensure your score distribution matches the human reviewer patterns shown in the statistics above.

=== Research Idea ===
{idea_text}

=== Reviewer Evaluations (verbatim) ===
{chr(10).join(eval_summaries)}

=== ICLR Overall Rating Scale (Reference) ===
{iclr_scale}

=== Decision Types ===
- reject: not accepted
- poster: accepted (regular)
- spotlight: accepted (notable)
- oral: accepted (great)

=== Requirements ===
- Use ONLY the provided reviewer evidence; do not invent new facts.
- Aggregate consensus and highlight divergences; call out any outlier scores.
- Weigh all five dimensions (clarity, novelty, validity, feasibility, significance); justify trade-offs explicitly.
- **CRITICAL: Align with human reviewer patterns**: Your decision distribution and scoring should match the ICLR2025 statistics. For example, approximately 25.65% of submissions should receive "poster" decisions with scores around 6.23, not systematically lower. Ensure your evaluation reflects the actual human reviewer behavior patterns.
- Score on a continuous 0-10 scale (decimals allowed). The score must align with your decision category based on the ICLR2025 statistics provided above.
- Be objective and balanced: neither overly generous nor overly strict. Let the ICLR2025 statistics guide your calibration.
- Tone: concise, professional, decision-focused, and objective.

=== Output Format ===
Return pure JSON (no code fences) conforming to the schema: reason, score (0-10, decimals allowed), decision.
"""

    async def _generate_final_decision(
        self, evaluation_results: List[Dict[str, Any]], idea_text: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = self._build_final_decision_prompt(evaluation_results, idea_text)
        schema = self._build_final_decision_schema()
        try:
            resp = await self._call_model(
                prompt=prompt,
                system_prompt="You are an experienced ICLR meta-reviewer. Output must be pure JSON only.",
                schema=schema,
                temperature=params.get("temperature", self.temperature),
            )
            return {
                "reason": resp.get("reason", ""),
                "score": resp.get("score", ""),
                "decision": resp.get("decision", ""),
            }
        except Exception as e:
            logger.warning(f"Final decision generation failed: {e}")
            return {
                "reason": "Meta-review generation failed.",
                "score": "",
                "decision": "reject",
            }

    def _build_revision_prompt(self, idea_text: str, future_papers: List[Any]) -> str:
        paper_sections = []
        for idx, paper in enumerate(future_papers, 1):
            meta = {}
            title = f"Future Paper {idx}"
            if isinstance(paper, Source):
                meta = paper.metadata or {}
                title = paper.title or title
            elif isinstance(paper, dict):
                meta = paper.get("metadata", {})
                title = paper.get("title", title)

            extract = meta.get("paper_extract") or {}
            lines = []
            for key in ["basic_idea", "method", "research_question", "motivation", "expected_results"]:
                val = extract.get(key)
                if not val:
                    continue
                if isinstance(val, list):
                    val = " ".join([str(v) for v in val])
                lines.append(f"{key.replace('_', ' ').title()}: {val}")
            if lines:
                paper_sections.append(f"--- {title} ---\n" + "\n".join(lines))

        future_block = "\n\n".join(paper_sections) if paper_sections else "No future papers with extracted info."

        return f"""
You are a senior researcher. Using the current idea and the extracted future papers (already enriched), produce precise revision advice (future-work style) grounded ONLY in the provided content.

=== Current Idea (Idea fields: basic_idea, motivation, research_question, method, experimental_setting, expected_results) ===
{idea_text}

=== Future Papers (extracted) ===
{future_block}

=== Requirements ===
- Derive suggestions strictly from the supplied idea and future papers; no external knowledge.
- Cover: methodology/model improvements; experiment & evaluation enhancements; data/task extensions; risks/feasibility flags; measurable next steps.
- Be specific, actionable, and succinct; tie each suggestion to a concrete gap or inspiration point from the future papers or current idea.
- Prioritize high-impact, feasible actions; avoid generic advice.
- Output as Markdown text (no JSON, no code fences).
"""

    async def _generate_revision_advice(
        self, idea_text: str, future_papers: List[Any], params: Dict[str, Any]
    ) -> str:
        prompt = self._build_revision_prompt(idea_text, future_papers)
        try:
            resp = await self._call_model(
                prompt=prompt,
                system_prompt="You are a helpful senior researcher.",
                schema=None,
                temperature=params.get("temperature", self.temperature),
            )
            if isinstance(resp, str):
                return resp
            return str(resp)
        except Exception as e:
            logger.warning(f"Revision advice generation failed: {e}")
            return "Failed to generate revision advice."

    # ------------------------------------------------------------------ #
    # 最终报告组装
    # ------------------------------------------------------------------ #
    def _assemble_final_report(
        self,
        idea_text: str,
        paper_block: str,
        web_block: str,
        code_block: str,
        evaluation_block: str,
        final_decision: Dict[str, Any],
        revision_advice: str,
    ) -> str:
        reason = final_decision.get("reason", "")
        score = final_decision.get("score", "")
        decision = final_decision.get("decision", "")

        return f"""{idea_text}

# Searched Resources

## Paper Resources
{paper_block}

## Web Resources
{web_block}

## Code Resources
{code_block}

# Evaluation Results
{evaluation_block}

# Final Decision
Reason: {reason}
Score: {score}
Decision: {decision}

# Revision Advice
{revision_advice}
"""

