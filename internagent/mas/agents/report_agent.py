import asyncio
import random
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentExecutionError
from ..tools.searchersv2.models import Idea, SearchResults, Source, Platform, SourceType
from .extraction_agent import ExtractionAgent

class ReportAgent(BaseAgent):
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "ReportAgent"
        self.temperature = config.get("temperature")   
        extraction_config = config.get("extraction_config")
        
        # 如果extraction_config为None，从主配置中派生一个默认配置
        if extraction_config is None:
            extraction_config = {
                "name": "ExtractionAgent",
                "model_provider": config.get("model_provider", "default"),
                "extract_temperature": config.get("extract_temperature", 0.3),
                "_global_config": config.get("_global_config", {})
            }
        
        self.extraction_agent = ExtractionAgent(model, extraction_config)

    async def extract_papers(self, papers: List[Source]) -> List[Dict[str, Any]]:
        """
        Extract PDF URLs from papers and call ExtractionAgent to get structured information
        
        """
        extracted_results = []
        
        # 1. collect PDF URL
        paper_pdf_pairs = []
        for paper in papers:
            pdf_url = None
            if paper.pdf_url:
                pdf_url = paper.pdf_url
            elif paper.url and "pdf" in paper.url.lower():
                pdf_url = paper.url
            elif paper.url:  
                if "arxiv.org" in paper.url:
                    arxiv_id = paper.url.split("/")[-1].replace("v", "").replace(".html", "")
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                elif "openreview.net" in paper.url:
                    pdf_url = paper.url.replace("https://", "https://openreview.net/pdf?id=")
            
            if pdf_url:
                paper_pdf_pairs.append({
                    "paper": paper,
                    "pdf_url": pdf_url
                })
        
        print(f"📚 Find {len(paper_pdf_pairs)} extractable PDF links")
        
        for idx, item in enumerate(paper_pdf_pairs, 1):
            paper = item["paper"]
            pdf_url = item["pdf_url"]
            
            print(f"\n[{idx}/{len(paper_pdf_pairs)}] Extracting paper:")
            print(f"   Title: {paper.title[:60] if paper.title else 'Unknown'}...")
            print(f"   PDF:   {pdf_url[:80]}...")

            result = await self._extract_single_paper(paper, pdf_url, idx)

            if result:
                extracted_results.append(result)
                print(f"   ✅ Extraction successful")
            else:
                print(f"   ❌ Extraction failed")
        
        print(f"\n Sequential extraction completed! Successfully extracted {len(extracted_results)}/{len(paper_pdf_pairs)} papers")
        return extracted_results

    async def _extract_single_paper(self, paper: Source, pdf_url: str, idx: int) -> Dict[str, Any]:
        try:
            context = {"url": pdf_url}           
            extraction_result = await self.extraction_agent.execute(context, {})
            
            # add metadata
            enhanced_result = {
                "paper_metadata": {
                    "title": paper.title or "Unknown",
                    "url": paper.url or "",
                    "pdf_url": pdf_url,
                    "authors": paper.authors or [],
                    "year": paper.year or "",
                    "platform": paper.platform.value if paper.platform else ""
                },
                **extraction_result
            }
            
            return enhanced_result
            
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}...")
            return None

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. Mode 1: Generate research reports (web, code, paper) from search_result
        2. Mode 2: Generate review reports (evaluation_reports) from evaluation_results
        """
        evaluation_results = context.get("evaluation_results")
        if evaluation_results and isinstance(evaluation_results, list):
            # Mode 2
            print("\n" + "="*60)
            print("REVIEW REPORT GENERATION MODE")
            print("="*60)
            return await self._generate_review_reports(evaluation_results, params)
        else:
            # Mode 1
            print("\n" + "="*60)
            print("RESEARCH REPORT GENERATION MODE")
            print("="*60)
            return await self._generate_research_reports(context, params)

    async def _generate_review_reports(self, evaluation_results: List[Dict[str, Any]], 
                                    params: Dict[str, Any]) -> Dict[str, Any]:
        """Mode 2: Generate a review report from evaluation_results"""
        evaluation_reports = []
        
        print(f"Generating review reports for {len(evaluation_results)} reviewers...")
        
        for idx, eval_result in enumerate(evaluation_results, 1):
            print(f"\n[{idx}/{len(evaluation_results)}] Generating report for Reviewer {idx}...")
            
            persona = eval_result.get("persona", {})
            evaluation = eval_result.get("evaluation", {})
            
            review_prompt = self._build_review_prompt(persona, evaluation, idx)
            review_schema = self._build_review_schema()
            
            # Generate review report
            review_report = await self._call_model(
                prompt=review_prompt,
                system_prompt=self._get_review_system_prompt(),
                schema=review_schema,
                temperature=params.get("temperature", self.temperature),
            )
            
            full_report = review_report.get("full_review_report", "")
            evaluation_reports.append({
                "reviewer_index": idx,
                "persona": persona,
                "evaluation": evaluation,
                "full_review_report": full_report 
            })
            
            print(f"  ✅ Review report generated for Reviewer {idx}")
        
        print(f"\n{'='*60}")
        print(f"REVIEW REPORTS COMPLETED: {len(evaluation_reports)} reports generated")
        print(f"{'='*60}")
        
        return {
            "evaluation_reports": evaluation_reports,
            "params": params
        }

    async def _generate_research_reports(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """模式1：从search_result生成research reports"""
        sr = context.get("search_result") or context.get("search_results")
        if not isinstance(sr, dict):
            raise AgentExecutionError("search_result must be a dict containing idea and search results")
        sr = SearchResults.from_dict(sr)

        idea = sr.idea
        idea_text = self._get_idea_text(sr)

        papers = sr.papers + sr.scholar_results
        github_repos = sr.github_repos
        kaggle_results = sr.kaggle_results
        web_pages = sr.web_pages
        
        print(f"total: Papers={len(papers)}, Code={len(github_repos)}, Web={len(web_pages)}")
        
        print("\n" + "="*60)
        print("STARTING PARALLEL PROCESSING (Three Main Tasks)")
        print("="*60)
        
        # Create functions for three main tasks
        async def paper_task():
            """Paper extraction"""
            print("\n[PAPER TASK] Starting sequential paper extraction...")
            result = await self.extract_papers(papers)
            print(f"[PAPER TASK] Completed: {len(result) if result else 0} papers extracted")
            return result
        
        async def web_task():
            """Web reports"""
            print("\n[WEB TASK] Starting sequential web report generation...")
            result = await self._generate_web_reports_sequential(idea_text, web_pages, params)
            print(f"[WEB TASK] Completed: {len(result) if result else 0} web reports generated")
            return result
        
        async def code_task():
            """Code reports"""
            print("\n[CODE TASK] Starting sequential code report generation...")
            result = await self._generate_code_reports_sequential(idea_text, github_repos + kaggle_results, params)
            print(f"[CODE TASK] Completed: {len(result) if result else 0} code reports generated")
            return result
        
        paper_extraction_results, web_reports, code_reports = await asyncio.gather(
            paper_task(),
            web_task(),
            code_task(),
            return_exceptions=False
        )
        
        print("\n" + "="*60)
        print("PARALLEL PROCESSING COMPLETED")
        print("="*60)
        print(f"  Papers: {len(paper_extraction_results) if paper_extraction_results else 0} extracted")
        print(f"  Web: {len(web_reports) if web_reports else 0} reports generated")
        print(f"  Code: {len(code_reports) if code_reports else 0} reports generated")
        print("="*60)
        
        return {
            "web_reports": web_reports, 
            "code_reports": code_reports,
            "paper_reports": paper_extraction_results,
            "params": params,
        }

    async def _generate_web_reports_sequential(self, idea_text: str, web_pages: List[Source], 
                                          params: Dict[str, Any]) -> List[Dict[str, Any]]:
        web_descriptions = self.build_descriptions(web_pages)
        web_reports = []
        
        if not web_descriptions:
            print("  No web descriptions to process")
            return web_reports
        
        print(f"  Processing {len(web_descriptions)} web sources sequentially...")
        
        for idx, single_desc in enumerate(web_descriptions, 1):
            print(f"  [{idx}/{len(web_descriptions)}] Generating web report...")
            
            web_prompt = self._build_web_prompt(idea_text, single_desc, idx)
            web_schema = self._build_output_schema(analysis_type="web")
            
            single_web_report = await self._call_model(
                prompt=web_prompt,
                system_prompt=self.system_prompt,
                schema=web_schema,
                temperature=params.get("temperature", self.temperature),
            )
            
            web_reports.append({
                "report_id": f"web_report_{idx:02d}",
                "source_description": single_desc,
                "content": single_web_report
            })
            
            print(f"    ✅ Web report {idx} generated")
        
        return web_reports
    
    async def _generate_code_reports_sequential(self, idea_text: str, code_items: List[Source], 
                                           params: Dict[str, Any]) -> List[Dict[str, Any]]:
        code_desc = self.build_descriptions(code_items)
        code_reports = []
        
        if not code_desc:
            print("  No code descriptions to process")
            return code_reports
        
        print(f"  Processing {len(code_desc)} code sources sequentially...")
        
        for idx, single_desc in enumerate(code_desc, 1):
            print(f"  [{idx}/{len(code_desc)}] Generating code report...")
            
            code_prompt = self._build_code_prompt(idea_text, single_desc, idx)
            code_schema = self._build_output_schema(analysis_type="code")
            
            single_code_report = await self._call_model(
                prompt=code_prompt,
                system_prompt=self.system_prompt,
                schema=code_schema,
                temperature=params.get("temperature", self.temperature),
            )
            
            code_reports.append({
                "report_id": f"code_report_{idx:02d}",
                "source_description": single_desc,
                "content": single_code_report
            })
            
            print(f"    ✅ Code report {idx} generated")
        
        return code_reports
        
        
    def build_descriptions(self, items: List[Source]) -> List[str]:
        """
        Build descriptions from Source objects.
        For web pages, use page_raw_text.
        For GitHub repos, use repo_context.
        """
        descs = []
        for index, source in enumerate(items, 1):
            if not isinstance(source, Source):
                continue
            
            block = f"Source{index}:\n"
            has_content = False
            
            # For web pages, use page_raw_text
            if source.source_type == SourceType.WEBPAGE and source.page_raw_text:
                block += f"Webpage raw text:\n{source.page_raw_text}\n"
                has_content = True
            
            # For GitHub repos, use repo_context
            elif source.platform == Platform.GITHUB and source.repo_context:
                block += f"Repo Summary:\n{source.repo_context}\n"
                has_content = True
            
            # For Kaggle, also use repo_context if available
            elif source.platform == Platform.KAGGLE and source.repo_context:
                block += f"Repo Summary:\n{source.repo_context}\n"
                has_content = True
            
            # Only add if there's actual content
            if has_content:
                descs.append(block)
        
        return descs

    def _get_idea_text(self, sr: SearchResults) -> str:
        """Extract idea text from SearchResults using Idea.get_full_text() method"""
        idea = sr.idea
        # Use the get_full_text() method which handles all formatting
        return idea.get_full_text()

    def _build_review_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "full_review_report": {
                    "type": "string",
                    "description": "Complete, well-formatted review report including all sections: executive summary, detailed analysis, critical feedback, implementation roadmap, final recommendation, and reviewer perspective"
                }
            },
            "required": ["full_review_report"],
            "additionalProperties": False
        }

    def _get_review_system_prompt(self) -> str:
        return """You are an expert research reviewer and report writer. 
Your task is to synthesize evaluation results into comprehensive, professional review reports 
that provide valuable feedback to researchers while maintaining academic rigor and objectivity.

Focus on:
1. Evidence-based analysis using the provided evaluation results
2. Clear, actionable recommendations for improvement
3. Professional tone suitable for academic review
4. Comprehensive coverage of all evaluation dimensions
5. Consideration of the reviewer's background and perspective
Ensure your reports are well-structured, detailed, and provide genuine value to the research team.
RESTRICTION：
- Output MUST be pure JSON only, without any code block markers like ```json or ```, strictly matching the provided schema — no extra commentary, no markdown, no text outside JSON.

"""

    def _build_review_prompt(self, persona: Dict[str, Any], evaluation: Dict[str, Any], 
                       reviewer_idx: int) -> str:
        # Extract reviewer information
        background = persona.get("background", "Unknown background")
        goal = persona.get("goal", "Evaluate research idea")
        constraints = persona.get("constraints", "")
        
        # Extract Evaluation Results
        clarity = evaluation.get("clarity", {})
        novelty = evaluation.get("novelty", {})
        feasibility = evaluation.get("feasibility", {})
        overall = evaluation.get("overall", {})
        
        clarity_score = clarity.get("score", 0.0)
        clarity_reason = clarity.get("reason", "")
        
        novelty_score = novelty.get("score", 0.0)
        novelty_reason = novelty.get("reason", "")
        
        feasibility_score = feasibility.get("score", 0.0)
        feasibility_reason = feasibility.get("reason", "")
        feasibility_pseudocode = feasibility.get("pseudocode", "")
        
        overall_summary = overall.get("summary", "")
        overall_recommendation = overall.get("recommendation", "")
        
        return f"""
    You are generating a comprehensive review report for a research idea based on evaluation results.

    === REVIEWER INFORMATION ===
    Reviewer #{reviewer_idx}
    Background: {background}
    Goal: {goal}
    {('Constraints: ' + constraints) if constraints else ''}

    === EVALUATION RESULTS ===

    1. CLARITY EVALUATION:
    Score: {clarity_score}/10
    Reason: {clarity_reason}

    2. NOVELTY EVALUATION:
    Score: {novelty_score}/10
    Reason: {novelty_reason}

    3. FEASIBILITY EVALUATION:
    Score: {feasibility_score}/10
    Reason: {feasibility_reason}

    Implementation Pseudocode/Plan:
    {feasibility_pseudocode}

    4. OVERALL ASSESSMENT:
    {overall_summary}

    Recommendation:
    {overall_recommendation}

    === REVIEW REPORT REQUIREMENTS ===

    Your task is to synthesize these evaluation results into a comprehensive review report.
    The report should be structured as follows:

    1. EXECUTIVE SUMMARY
    - Brief overview of the research idea
    - Overall assessment and key findings
    - Main strengths and weaknesses

    2. DETAILED EVALUATION ANALYSIS
    a. Clarity Assessment
        - Logical consistency and structural quality
        - Factual correctness and reasoning soundness
        - Specific observations and concerns

    b. Novelty Assessment  
        - Originality compared to existing work
        - Methodological and conceptual contributions
        - Areas of overlap and differentiation

    c. Feasibility Assessment
        - Implementation challenges and opportunities
        - Technical viability and resource requirements
        - Implementation plan analysis

    3. CRITICAL FEEDBACK
    - Major concerns and potential issues
    - Specific improvements and suggestions
    - Areas needing clarification or refinement

    4. IMPLEMENTATION ROADMAP
    - Step-by-step implementation recommendations
    - Technical dependencies and considerations
    - Milestones and deliverables

    5. FINAL RECOMMENDATION
    - Clear recommendation (Accept/Revise/Reject)
    - Priority level and timeline suggestions
    - Next steps for the researchers

    6. APPENDIX: REVIEWER PERSPECTIVE
    - How the reviewer's background influenced the evaluation
    - Any biases or limitations in the assessment
    - Additional context or insights

    === CRITICAL REQUIREMENTS ===

    1. Maintain objectivity and evidence-based reasoning
    2. Connect all feedback to specific evaluation results
    3. Provide actionable recommendations for improvement
    4. Consider the reviewer's background and perspective
    5. Balance strengths and weaknesses fairly
    6. Include specific examples and references where relevant
    7. Ensure the report is comprehensive yet concise

    === OUTPUT FORMAT ===

    You must output a JSON object with the following structure:
    {{
    "full_review_report": "string"
    }}
    Output MUST be pure JSON only, without any code block markers like ```json or ```!

    The "full_review_report" should contain the complete, well-formatted review report including all required sections:
    1. Executive Summary
    2. Detailed Analysis (clarity, novelty, feasibility)
    3. Critical Feedback  
    4. Implementation Roadmap
    5. Final Recommendation
    6. Reviewer Perspective

    Ensure the report is professional, detailed, and suitable for academic/research evaluation.
    """

    def _build_output_schema(self, analysis_type: str = "generic") -> Dict[str, Any]:
        if analysis_type == "code":
            return {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Comprehensive summary of the code resource content"
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Specific evidence and code implementation details extracted from the raw text"
                    },
                    "rational": {
                        "type": "string", 
                        "description": "Logical reasoning on how the evidence supports or relates to the research idea"
                    },
                    "report_content": {
                        "type": "string",
                        "description": "Final code report"
                    }
                },
                "required": ["summary", "evidence", "rational", "report_content"],
                "additionalProperties": False
            }
        elif analysis_type == "web":
            return {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Comprehensive summary of the web page content"
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Specific evidence and viewpoints extracted from the raw text"
                    },
                    "rational": {
                        "type": "string",
                        "description": "Logical reasoning on how the evidence supports, challenges, or relates to the research idea"
                    },
                    "report_content": {
                        "type": "string",
                        "description": "Final web report"
                    }
                },
                "required": ["summary", "evidence", "rational", "report_content"],
                "additionalProperties": False
            }
        
    def _build_web_prompt(self, idea_text: str, description: str, source_idx: int) -> str:
        return (
        f"You are generating a comprehensive analysis for a research idea based on a SINGLE web resource.\n\n"
        f"=== Input Specification (Web Source #{source_idx}) ===\n"
        f"Research Idea:\n{idea_text}\n\n"
        f"Web Resource Raw Text (Content to analyze):\n{description}\n\n"
        "=== Analysis Requirements ===\n"
        "Your task is to analyze this web content and produce a structured output with FOUR components:\n\n"
        "1. SUMMARY: A concise overview of the web content's main themes, arguments, and relevance to the research idea.\n"
        "2. EVIDENCE: Specific textual evidence extracted from the raw text that either SUPPORTS, CONTRADICTS, or COMPLICATES the research idea. "
        "Include both positive and negative evidence. Provide direct quotes or paraphrased content with context.\n"
        "3. RATIONAL: Logical reasoning explaining HOW the evidence relates to the research idea - whether it validates, challenges, qualifies, or provides context.\n"
        "4. REPORT_CONTENT: A complete analytical report with the following sections:\n"
        "   a. Idea Overview - Brief restatement of the research idea\n"
        "   b. Extracted Viewpoints - All relevant claims, arguments, evidence (both supportive and contradictory)\n"
        "   c. Key Insights - Most significant findings from this source\n"
        "   d. Implications - What this suggests for feasibility, novelty, risks\n"
        "   e. Source-Specific Notes - Unique aspects of this source\n\n"
        "   f. SUMMARY - from 1. Copy all the content over\n\n"
        "   g. EVIDENCE - from 2. Copy all the content over\n\n"
        "   h. RATIONAL - from 3. Copy all the content over\n\n"
        "=== Critical Requirements ===\n"
        "1. EVIDENCE MUST include BOTH supporting and contradictory evidence if present in the text.\n"
        "2. Do NOT invent or assume information not present in the raw text.\n"
        "3. Maintain strict objectivity - do not bias toward supporting the idea.\n"
        "4. Focus on relevance: Only include content that directly relates to evaluating the research idea.\n"
        "5. For EVIDENCE, clearly indicate whether each piece supports or contradicts the idea.\n"
        "6. For RATIONAL, explain the logical connection between evidence and idea evaluation.\n"
        "7. REPORT_CONTENT should synthesize the analysis into a coherent, well-structured report.\n\n"
        "=== Output Format ===\n"
        "You must output a JSON object with exactly these four fields:\n"
        "- summary: (string) Comprehensive summary\n"
        "- evidence: (string) Specific evidence with clear labels [SUPPORTING] or [CONTRADICTING]\n"
        "- rational: (string) Logical reasoning connecting evidence to idea\n"
        "- report_content: (string) Complete analytical report as described above\n\n"
        "Ensure all four components are distinct, detailed, and based SOLELY on the provided raw text."
        "- Output MUST be pure JSON only, without any code block markers like ```json or ```, strictly matching the provided schema — no extra commentary, no markdown, no text outside JSON."
    )

    def _build_code_prompt(self, idea_text: str, description: str, source_idx: int) -> str:
        return (
        f"You are generating a technical implementation analysis for a research idea based on a SINGLE code resource.\n\n"
        f"=== Input Specification (Code Source #{source_idx}) ===\n"
        f"Research Idea:\n{idea_text}\n\n"
        f"Code Resource Raw Text (Content to analyze):\n{description}\n\n"
        "=== Analysis Requirements ===\n"
        "Your task is to analyze this code/documentation content and produce a structured output with FOUR components:\n\n"
        "1. SUMMARY: A concise overview of the code resource - what it offers, its architecture, and relevance to implementing the research idea.\n"
        "2. EVIDENCE: Specific technical details extracted from the raw text that could either ENABLE or COMPLICATE implementation of the research idea. "
        "Include both useful features/practices and limitations/challenges. Pay special attention to file structures, dependencies, and architecture.\n"
        "3. RATIONAL: Logical reasoning explaining HOW the evidence informs the implementation strategy - whether it provides solutions, reveals gaps, or suggests modifications.\n"
        "4. REPORT_CONTENT: A complete technical report with the following sections:\n"
        "   a. Idea Overview - Brief restatement of the research idea\n"
        "   b. Useful Components - Tools, modules, models applicable to the idea\n"
        "   c. Repository Structure Analysis - File tree examination and architectural assessment\n"
        "   d. Typical Pipelines - Common workflows that can inform implementation\n"
        "   e. Integration Strategy & Considerations - How to use this resource, including obstacles\n"
        "   f. Limitations & Risks - Constraints, missing components, compatibility issues\n\n"
        "   g. SUMMARY - from 1.  Copy all the content over\n\n"
        "   h. EVIDENCE - from 2. Copy all the content over\n\n"
        "   i. RATIONAL - from 3. Copy all the content over\n\n"
        "=== Critical Requirements ===\n"
        "1. EVIDENCE MUST include both enabling features and implementation challenges if present.\n"
        "2. Analyze file structures (file_tree) to understand code organization and reusability.\n"
        "3. Identify technical constraints, dependencies, and compatibility issues.\n"
        "4. Focus on practical implementation: How can this code actually be used or adapted?\n"
        "5. For EVIDENCE, clearly label whether each item [ENABLES] implementation or [COMPLICATES] it.\n"
        "6. For RATIONAL, explain the technical logic behind implementation feasibility.\n"
        "7. REPORT_CONTENT should provide actionable technical guidance.\n\n"
        "=== Output Format ===\n"
        "You must output a JSON object with exactly these four fields:\n"
        "- summary: (string) Comprehensive summary\n"
        "- evidence: (string) Specific technical evidence with clear labels [ENABLING] or [COMPLICATING]\n"
        "- rational: (string) Logical reasoning connecting evidence to implementation feasibility\n"
        "- report_content: (string) Complete technical report as described above\n\n"
        "Ensure all four components are technically precise and based SOLELY on the provided raw text."
        "- Output MUST be pure JSON only, without any code block markers like ```json or ```, strictly matching the provided schema — no extra commentary, no markdown, no text outside JSON."
    )