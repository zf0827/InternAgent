import asyncio
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentExecutionError
from ..tools.searchersv2.models import SearchResults, Source, Platform, SourceType
from .extraction_agent import ExtractionAgent

class ReportAgent(BaseAgent):
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "ReportAgent"
        self.temperature = config.get("temperature", 0.7)
        
        extraction_config = config.get("extraction_config", {
            "name": "ExtractionAgent",
            "model_provider": config.get("model_provider", "default"),
            "extract_temperature": config.get("extract_temperature", 0.3),
            "_global_config": config.get("_global_config", {})
        })

        self.extraction_agent = ExtractionAgent(model, extraction_config)

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Two primary operational modes:
        1. Only evaluation_results -> Generate review reports
        2. evaluation_results + future_papers -> Generate review reports first, then final report
        3. Only search_result -> Generate research reports
        """
        
        evaluation_results = context.get("evaluation_results")
        
        if evaluation_results and isinstance(evaluation_results, list):
            print("\n" + "="*60)
            print("REVIEW REPORT GENERATION MODE")
            print("="*60)
            
            # Generate review reports first
            review_reports_result = await self._generate_review_reports(evaluation_results, params)
            
            # Extract generated review reports
            generated_review_reports = review_reports_result.get("evaluation_reports", [])
            
            print(f"\n✅ REVIEW REPORTS completed: {len(generated_review_reports)} reports generated")
            
            # Check if final report generation is required
            if context.get("future_papers"):
                print("\n" + "="*60)
                print("FINAL REPORT GENERATION")
                print("="*60)
                print("Detected future_papers, initiating final report generation...")
                
                final_report = await self._generate_final_report(
                    review_reports=generated_review_reports,  # Use recently generated review reports
                    research_reports=context.get("research_reports", {}),
                    idea=context.get("idea", {}),
                    future_papers=context["future_papers"],
                    params=params
                )
                
                print(f"✅ FINAL REPORT completed: {len(final_report)} characters")
                
                # Return results including both review reports and final report
                return {
                    "evaluation_reports": generated_review_reports,
                    "final_report": final_report,
                    "params": params
                }
            else:
                # Return only review reports
                print("\n⚠️ No future_papers detected, returning review reports only")
                return {
                    "evaluation_reports": generated_review_reports,
                    "params": params
                }
        
        elif context.get("search_result") or context.get("search_results"):
            print("\n" + "="*60)
            print("RESEARCH REPORT GENERATION MODE")
            print("="*60)
            return await self._generate_research_reports(context, params)
        
        else:
            raise AgentExecutionError(
                "No valid input provided. Required input options:\n"
                "1. evaluation_results (for review reports)\n"
                "2. evaluation_results + future_papers (for review reports + final report)\n"
                "3. search_result (for research reports)"
            )
    
    async def extract_papers(self, papers: List[Source]) -> List[Dict[str, Any]]:
        """
        Extract PDF URLs from papers and call ExtractionAgent to obtain structured information
        """
        extracted_results = []
        
        # 1. Collect PDF URLs
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
        
        print(f"📚 Identified {len(paper_pdf_pairs)} extractable PDF links")
        
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
            
            # Add metadata
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

    async def _generate_review_reports(self, evaluation_results: List[Dict[str, Any]], 
                                    params: Dict[str, Any]) -> Dict[str, Any]:
        """Mode 2: Generate review reports from evaluation_results"""
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
        """Mode 1: Generate research reports from search_result"""
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
        
        print(f"Total sources: Papers={len(papers)}, Code={len(github_repos)}, Web={len(web_pages)}")
        
        print("\n" + "="*60)
        print("INITIATING PARALLEL PROCESSING (Three Primary Tasks)")
        print("="*60)
        
        # Create functions for three main tasks
        async def paper_task():
            """Paper extraction task"""
            print("\n[PAPER TASK] Initiating sequential paper extraction...")
            result = await self.extract_papers(papers)
            print(f"[PAPER TASK] Completed: {len(result) if result else 0} papers extracted")
            return result
        
        async def web_task():
            """Web reports generation task"""
            print("\n[WEB TASK] Initiating sequential web report generation...")
            result = await self._generate_web_reports_sequential(idea_text, web_pages, params)
            print(f"[WEB TASK] Completed: {len(result) if result else 0} web reports generated")
            return result
        
        async def code_task():
            """Code reports generation task"""
            print("\n[CODE TASK] Initiating sequential code report generation...")
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
    
    # ---------------------------------------------------------------
    # Web reports generation
    # ---------------------------------------------------------------
    async def _generate_web_reports_sequential(self, idea_text: str, web_pages: List[Source], params: Dict[str, Any]):
        web_descriptions = self.build_descriptions(web_pages)
        web_reports = []
        
        if not web_descriptions:
            print("  No web descriptions available for processing")
            return web_reports
        
        print(f"  Processing {len(web_descriptions)} web sources sequentially...")
        
        for idx, single_desc in enumerate(web_descriptions, 1):
            print(f"  [{idx}/{len(web_descriptions)}] Generating web report...")
            
            web_prompt = self._build_web_prompt(idea_text, single_desc, idx)
            web_schema = self._build_output_schema()
            
            single_web_report = await self._call_model(
                prompt=web_prompt,
                system_prompt=self.system_prompt,
                schema=web_schema,
                temperature=params.get("temperature", self.temperature),
            )
            
            web_reports.append({
                "report_id": f"web_report_{idx:02d}",
                "source_description": single_desc,
                "summary": single_web_report.get("summary", ""),
                "report_content": single_web_report.get("report_content", "")
            })
            
            print(f"    ✅ Web report {idx} generated")
        
        return web_reports

    # ---------------------------------------------------------------
    # Code reports generation
    # ---------------------------------------------------------------
    async def _generate_code_reports_sequential(self, idea_text: str, code_items: List[Source], params: Dict[str, Any]):
        code_desc = self.build_descriptions(code_items)
        code_reports = []
        
        if not code_desc:
            print("  No code descriptions available for processing")
            return code_reports
        
        print(f"  Processing {len(code_desc)} code sources sequentially...")
        
        for idx, single_desc in enumerate(code_desc, 1):
            print(f"  [{idx}/{len(code_desc)}] Generating code report...")
            
            code_prompt = self._build_code_prompt(idea_text, single_desc, idx)
            code_schema = self._build_output_schema()
            
            single_code_report = await self._call_model(
                prompt=code_prompt,
                system_prompt=self.system_prompt,
                schema=code_schema,
                temperature=params.get("temperature", self.temperature),
            )
            
            code_reports.append({
                "report_id": f"code_report_{idx:02d}",
                "source_description": single_desc,
                "summary": single_code_report.get("summary", ""),
                "report_content": single_code_report.get("report_content", "")
            })
            
            print(f"    ✅ Code report {idx} generated")
        
        return code_reports

    def _build_output_schema(self) -> Dict[str, Any]:
        """Return schema containing summary and report_content"""
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary, 2-3 sentences capturing the core content and relevance to the idea"
                },
                "report_content": {
                    "type": "string",
                    "description": "Complete report content with detailed Markdown-formatted analysis"
                }
            },
            "required": ["summary", "report_content"],
            "additionalProperties": False
        }

    def build_descriptions(self, items: List[Source]) -> List[str]:
        """
        Construct descriptions from Source objects.
        For web pages, utilize page_raw_text.
        For GitHub repositories, utilize repo_context.
        """
        descs = []
        for index, source in enumerate(items, 1):
            if not isinstance(source, Source):
                continue
            
            block = f"Source{index}:\n"
            has_content = False
            
            # For web pages, utilize page_raw_text
            if source.source_type == SourceType.WEBPAGE and source.page_raw_text:
                block += f"Webpage raw text:\n{source.page_raw_text}\n"
                has_content = True
            
            # For GitHub repositories, utilize repo_context
            elif source.platform == Platform.GITHUB and source.repo_context:
                block += f"Repository Summary:\n{source.repo_context}\n"
                has_content = True
            
            # For Kaggle, also utilize repo_context if available
            elif source.platform == Platform.KAGGLE and source.repo_context:
                block += f"Repository Summary:\n{source.repo_context}\n"
                has_content = True
            
            # Only append if substantial content exists
            if has_content:
                descs.append(block)
        
        return descs

    def _get_idea_text(self, sr: SearchResults) -> str:
        """Extract idea text from SearchResults using Idea.get_full_text() method"""
        idea = sr.idea
        # Utilize the get_full_text() method which manages all formatting
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
1. Evidence-based analysis utilizing the provided evaluation results
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
    - Principal strengths and weaknesses

    2. DETAILED EVALUATION ANALYSIS
    a. Clarity Assessment
        - Logical consistency and structural quality
        - Factual accuracy and reasoning soundness
        - Specific observations and concerns

    b. Novelty Assessment  
        - Originality relative to existing work
        - Methodological and conceptual contributions
        - Areas of overlap and differentiation

    c. Feasibility Assessment
        - Implementation challenges and opportunities
        - Technical viability and resource requirements
        - Implementation plan analysis

    3. CRITICAL FEEDBACK
    - Major concerns and potential issues
    - Specific improvements and suggestions
    - Areas requiring clarification or refinement

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

    def _build_output_schema(self) -> Dict[str, Any]:
        """Return schema containing summary and report_content"""
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary, 2-3 sentences capturing the core content and relevance to the idea"
                },
                "report_content": {
                    "type": "string",
                    "description": "Complete report content with detailed Markdown-formatted analysis"
                }
            },
            "required": ["summary", "report_content"],
            "additionalProperties": False
        }
        
    def _build_web_prompt(self, idea_text: str, description: str, idx: int) -> str:
        return f"""
    You are an expert research analyst. Your task is to analyze a SINGLE web resource and generate:
1. A concise summary
2. A detailed report_content that comprehensively summarizes the raw resource itself

Research Context:
{idea_text}

Web Resource Content:
{description}

=== SUMMARY REQUIREMENTS ===
- Provide a concise 5-6 sentence summary
- Focus on the web content itself - its principal themes, arguments, and factual content

=== REPORT_CONTENT REQUIREMENTS ===
Provide a detailed Markdown-formatted analysis that comprehensively summarizes ONLY the raw resource content. Structure your analysis as follows:

## Key Insights
- Most significant findings, conclusions, or claims made in the source
- Important data points or statistics presented
- Novel perspectives or noteworthy arguments

## Implementation Implications
- Methodologies, techniques, or approaches described
- Tools, frameworks, or technologies referenced
- Practical considerations, requirements, or constraints mentioned
- Case studies, applications, or examples provided

## Source-Specific Notes
- Unique characteristics or limitations of this source
- Contextual factors like publication date, authorship, or purpose
- Methodological approach or data quality aspects
- Format, structure, or presentation details

=== FORMATTING REQUIREMENTS ===
- Utilize clear Markdown headers (##, ###, ####) for sections
- Utilize bullet points for all lists and details
- Utilize bold text for sub-section headings within bullet points
- Include italic text for emphasis on important terms
- Maintain academic/professional tone throughout
- Ensure comprehensive coverage of ALL content in the raw resource

=== RESTRICTIONS ===
- DO NOT provide recommendations or suggestions
- DO NOT incorporate external knowledge or information
- Focus EXCLUSIVELY on summarizing what is present in the raw resource

=== OUTPUT FORMAT ===
Output MUST be pure JSON only, without any code block markers like ```json or ```, strictly matching this schema:
{{
  "summary": "A concise 5-6 sentence overview of the web content itself, focusing only on the raw resource",
  "report_content": "Detailed Markdown-formatted analysis comprehensively summarizing the raw resource content with all required sections"
}}
"""

    def _build_code_prompt(self, idea_text: str, description: str, idx: int) -> str:
        return f"""
    You are an expert research analyst. Your task is to analyze a SINGLE code resource and generate:
    1. A concise summary (5-6 sentences)
    2. A detailed report_content that comprehensively describes the raw code resource itself

    Research Context:
    {idea_text}

    Code Resource Content:
    {description}

    === SUMMARY REQUIREMENTS ===
    - Provide a concise 5-6 sentence summary
    - Focus on the code resource itself - what it offers, its architecture, and technical specifications

    === REPORT_CONTENT REQUIREMENTS ===
    Provide a detailed Markdown-formatted analysis that comprehensively describes ONLY the raw code resource. Structure your analysis with these EXACT sections:

    ## Useful Components
    - Tools, modules, models applicable to the idea
    - List available components with brief descriptions
    - Note pre-trained models or datasets if present
    - Mention utility scripts or helper functions

    ## Repository Structure Analysis
    - File tree examination and architectural assessment
    - Overview of directory organization
    - Key files and their purposes
    - Architecture patterns and design principles

    ## Typical Pipelines
    - Common workflows that can inform implementation
    - Data processing sequences
    - Training and evaluation procedures
    - Deployment or serving workflows

    ## Integration Strategy & Considerations
    - How to utilize this resource, including obstacles
    - Setup and installation requirements
    - API interfaces and usage patterns
    - Configuration options and parameters
    - Potential integration challenges

    ## Limitations & Risks
    - Constraints, missing components, compatibility issues
    - Technical limitations or performance boundaries
    - Missing features or incomplete implementations
    - Compatibility requirements and dependencies
    - Maintenance status and documentation quality

    === FORMATTING REQUIREMENTS ===
    - Utilize EXACTLY the section headers shown above (## Useful Components, ## Repository Structure Analysis, etc.)
    - Utilize bullet points for all details within each section
    - Utilize consistent Markdown formatting throughout
    - Include technical specifics and concrete details
    - Maintain professional, technical tone

    === RESTRICTIONS ===
    - DO NOT analyze relevance to the research idea
    - DO NOT compare with other repositories
    - DO NOT suggest improvements or modifications
    - DO NOT incorporate external technical knowledge
    - Focus EXCLUSIVELY on describing what is present in the raw code resource

    === OUTPUT FORMAT ===
    Output MUST be pure JSON only, without any code block markers like ```json or ```, strictly matching this schema:
    {{
    "summary": "A concise 5-6 sentence overview of the code resource itself, focusing on the raw technical details",
    "report_content": "Detailed Markdown-formatted analysis with EXACT sections: Useful Components, Repository Structure Analysis, Typical Pipelines, Integration Strategy & Considerations, Limitations & Risks"
    }}
    """
    
    # ==================== Final Report Generation ====================
    
    async def _generate_final_report(self, review_reports: List[Dict[str, Any]], 
                                   research_reports: Dict[str, Any],
                                   idea: Dict[str, Any], 
                                   future_papers: List[Dict[str, Any]], 
                                   params: Dict[str, Any]) -> str:
        """Generate final comprehensive report"""
        print("Initiating final report generation...")
        
        # 1. Extract information from future_papers
        print("  Extracting information from future papers...")
        extracted_future_papers = await self._extract_future_papers(future_papers, params)
        
        # 2. Generate evaluation summary from review_reports (utilizing full_review_report)
        print("  Generating evaluation summary...")
        evaluation_summary = await self._generate_evaluation_summary(review_reports, params)
        
        # 3. Generate research activities summary (extracting summary fields from web/code reports)
        print("  Generating research activities summary...")
        search_summary = await self._generate_search_summary(research_reports, params)
        
        # 4. Generate improvement recommendations
        print("  Generating improvement recommendations...")
        improvements = await self._generate_improvements(idea, extracted_future_papers, params)
        
        # 5. Assemble final report
        final_report = self._assemble_final_report(
            idea=idea,
            search_summary=search_summary,
            evaluation_summary=evaluation_summary,
            improvements=improvements
        )
        
        print(f"✅ Final report generated ({len(final_report)} characters)")
        return final_report
    
    async def _extract_future_papers(self, future_papers: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract information from future papers - no quantity limitations"""
        extracted_papers = []
        
        for idx, paper in enumerate(future_papers, 1):
            print(f"  [{idx}/{len(future_papers)}] Extracting paper information...")
            
            pdf_url = paper.get("pdf_url")
            if not pdf_url:
                print(f"    Skipping: No PDF URL available")
                continue
            
            try:
                # Utilize extraction agent for information extraction
                extraction_result = await self.extraction_agent.execute({"url": pdf_url}, {})
                
                extracted_papers.append({
                    "title": paper.get("title", f"Paper {idx}"),
                    "pdf_url": pdf_url,
                    "extraction": extraction_result
                })
                print(f"    ✅ Extraction successful")
                
            except Exception as e:
                print(f"    ❌ Extraction failed: {str(e)}")
                continue
        
        return extracted_papers
    
    async def _generate_evaluation_summary(self, review_reports: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
        """Generate evaluation summary from review_reports - directly utilizing complete full_review_report"""
        if not review_reports:
            return "No evaluation data available."
        
        review_texts = []
        for report in review_reports:
            review_text = report.get("full_review_report", "")
            if review_text:
                review_texts.append(review_text)
        
        if not review_texts:
            return "No review reports available for summarization."
        
        review_sections = []
        for i, text in enumerate(review_texts, 1):
            review_sections.append(f"\nReviewer {i} Report:\n{text}")
        
        prompt = f"""
    Review Reports from {len(review_texts)} reviewers:
    {"---".join(review_sections)}

    Based on these complete review reports, provide a comprehensive summary of the evaluation results including:
    1. Overall assessment and consensus points
    2. Principal strengths and contributions identified by reviewers
    3. Key concerns, limitations, and weaknesses raised
    4. Major recommendations and suggestions for improvement
    5. Diverging opinions or contradictory feedback
    6. Final evaluation conclusions

    Provide a detailed, thorough summary that captures all important aspects from the review reports.

    Output JSON: {{"summary": "text"}}
    """
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt="You are an expert at synthesizing review feedback. Provide comprehensive summaries of evaluation results from complete review reports.",
                schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
                temperature=params.get("temperature", 0.5),
            )
            return response.get("summary", "")
        except Exception as e:
            print(f"Error generating evaluation summary: {e}")
            return f"## Evaluation Summary\n\nBased on {len(review_texts)} review reports. Summary generation failed."
    
    async def _generate_search_summary(self, research_reports: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Generate research activities summary - extracting summary fields from web/code reports, no quantity limitations"""
        paper_reports = research_reports.get("paper_reports", [])
        web_reports = research_reports.get("web_reports", [])
        code_reports = research_reports.get("code_reports", [])
        
        # Statistical overview
        stats = {
            "papers": len(paper_reports),
            "web_sources": len(web_reports),
            "code_repos": len(code_reports)
        }
        
        # Extract paper information - all papers
        paper_info = []
        for paper in paper_reports:
            title = paper.get("paper_metadata", {}).get("title", "Unknown Paper")
            basic_idea = paper.get("basic_idea", [])
            if basic_idea and isinstance(basic_idea, list) and basic_idea:
                # Utilize complete basic_idea
                paper_info.append(f"- **{title}**: {' '.join([str(item) for item in basic_idea])}")
            elif basic_idea:
                paper_info.append(f"- **{title}**: {basic_idea}")
        
        # Extract web summaries - all sources
        web_summaries = []
        for report in web_reports:
            # Directly extract summary field from report object
            if "summary" in report:
                summary = report["summary"]
                web_summaries.append(f"- {summary}")
            # Or extract from content if available
            elif "content" in report and isinstance(report["content"], dict) and "summary" in report["content"]:
                summary = report["content"]["summary"]
                web_summaries.append(f"- {summary}")
        
        # Extract code summaries - all repositories
        code_summaries = []
        for report in code_reports:
            if "summary" in report:
                summary = report["summary"]
                code_summaries.append(f"- {summary}")
            elif "content" in report and isinstance(report["content"], dict) and "summary" in report["content"]:
                summary = report["content"]["summary"]
                code_summaries.append(f"- {summary}")
        
        # Generate summary
        prompt = f"""
Comprehensive Research Analysis Summary:

Papers Analyzed ({stats['papers']} papers):
{chr(10).join(paper_info) if paper_info else "No papers analyzed."}

Web Sources Analyzed ({stats['web_sources']} sources):
{chr(10).join(web_summaries) if web_summaries else "No web sources analyzed."}

Code Repositories Analyzed ({stats['code_repos']} repositories):
{chr(10).join(code_summaries) if code_summaries else "No code repositories analyzed."}

Provide a comprehensive overview of research activities including:
1. Research scope and methodology
2. Key sources analyzed in each category (papers, web, code)
3. Principal findings and discoveries from each source type
4. Relationships and connections between different sources
5. Gaps or limitations in the research coverage
6. Overall research quality and completeness

Provide a detailed, thorough summary that captures the full breadth of research activities.

Output JSON: {{"summary": "text"}}
"""
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt="You are an expert research analyst. Provide comprehensive summaries of research activities and findings.",
                schema={"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
                temperature=params.get("temperature", 0.5),
            )
            return response.get("summary", "")
        except Exception as e:
            print(f"Error generating search summary: {e}")
            return f"## Research Activities\n\nAnalyzed {stats['papers']} papers, {stats['web_sources']} web sources, {stats['code_repos']} code repositories."
    
    async def _generate_improvements(self, idea: Dict[str, Any], extracted_papers: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
        """Generate improvement recommendations - no quantity limitations"""
        if not extracted_papers:
            return "No future papers analyzed for improvement suggestions."
        
        # Extract key paper information - all papers
        papers_info = []
        for paper in extracted_papers:
            title = paper.get("title", "Unknown Paper")
            extraction = paper.get("extraction", {})
            
            # Construct complete information
            info_parts = []
            
            # basic_idea - complete
            basic_idea = extraction.get("basic_idea", [])
            if basic_idea and isinstance(basic_idea, list):
                info_parts.append(f"Basic Idea: {' '.join([str(item) for item in basic_idea])}")
            elif basic_idea:
                info_parts.append(f"Basic Idea: {basic_idea}")
            
            # method - complete
            method = extraction.get("method", [])
            if method and isinstance(method, list):
                info_parts.append(f"Method: {' '.join([str(item) for item in method])}")
            elif method:
                info_parts.append(f"Method: {method}")
            
            # research_question - complete
            research_question = extraction.get("research_question", [])
            if research_question and isinstance(research_question, list):
                info_parts.append(f"Research Question: {' '.join([str(item) for item in research_question])}")
            elif research_question:
                info_parts.append(f"Research Question: {research_question}")
            
            # motivation - complete
            motivation = extraction.get("motivation", [])
            if motivation and isinstance(motivation, list):
                info_parts.append(f"Motivation: {' '.join([str(item) for item in motivation])}")
            elif motivation:
                info_parts.append(f"Motivation: {motivation}")
            
            if info_parts:
                papers_info.append(f"### {title}\n{chr(10).join(info_parts)}\n")
        
        if not papers_info:
            return "Future papers analyzed but insufficient information extracted."
        
        # Format idea - complete
        idea_text = ""
        if isinstance(idea, dict):
            idea_parts = []
            if idea.get('research_question'):
                idea_parts.append(f"**Research Question**: {idea['research_question']}")
            if idea.get('method'):
                idea_parts.append(f"**Method**: {idea['method']}")
            if idea.get('motivation'):
                idea_parts.append(f"**Motivation**: {idea['motivation']}")
            if idea.get('experimental_setting'):
                idea_parts.append(f"**Experimental Setting**: {idea['experimental_setting']}")
            if idea.get('expected_results'):
                idea_parts.append(f"**Expected Results**: {idea['expected_results']}")
            
            idea_text = "\n\n".join(idea_parts)
        
        prompt = f"""
Original Research Idea:
{idea_text}

Future Papers Analyzed ({len(extracted_papers)} papers):
{chr(10).join(papers_info)}

Based on these future papers and their complete extracted information, provide comprehensive improvement suggestions for the original research idea including:

1. **Methodological Improvements**: Recommendations for enhancing the research methodology based on approaches in future papers
2. **Conceptual Enhancements**: Approaches to strengthen the theoretical foundations or conceptual framework
3. **Technical Refinements**: Specific technical improvements or implementations inspired by future work
4. **Scope Expansion**: Opportunities to broaden the research scope or address additional aspects
5. **Risk Mitigation**: Strategies to address potential limitations or risks identified in future papers
6. **Implementation Roadmap**: Concrete steps for implementing these improvements

Provide detailed, actionable suggestions that are specifically grounded in the analysis of future papers.

Output JSON: {{"improvements": "text"}}
"""
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt="You are a research strategy expert. Provide comprehensive improvement suggestions based on detailed analysis of future research papers.",
                schema={"type": "object", "properties": {"improvements": {"type": "string"}}, "required": ["improvements"]},
                temperature=params.get("temperature", 0.5),
            )
            return response.get("improvements", "")
        except Exception as e:
            print(f"Error generating improvement suggestions: {e}")
            return "## Improvement Suggestions\n\nFailed to generate suggestions from future papers."
    
    def _assemble_final_report(self, idea: Dict[str, Any], search_summary: str, 
                             evaluation_summary: str, improvements: str) -> str:
        """Assemble final report - comprehensive format"""
        # Format idea - complete
        idea_sections = []
        if isinstance(idea, dict):
            if idea.get('motivation'):
                idea_sections.append(f"**Motivation**:\n{idea['motivation']}")
            if idea.get('research_question'):
                idea_sections.append(f"**Research Question**:\n{idea['research_question']}")
            if idea.get('method'):
                idea_sections.append(f"**Method**:\n{idea['method']}")
            if idea.get('experimental_setting'):
                idea_sections.append(f"**Experimental Setting**:\n{idea['experimental_setting']}")
            if idea.get('expected_results'):
                idea_sections.append(f"**Expected Results**:\n{idea['expected_results']}")
        
        idea_display = "\n\n".join(idea_sections) if idea_sections else "No idea details available."
        
        return f"""# Final Research Analysis Report

## 1. Research Idea Overview
{idea_display}

## 2. Research Activities Summary  
{search_summary}

## 3. Evaluation Results Summary
{evaluation_summary}

## 4. Improvement Recommendations
{improvements}

---

### Report Metadata
- **Report Type**: Comprehensive Research Analysis
- **Components Included**: Research Activities, Evaluation Summary, Improvement Recommendations
- **Data Sources**: Research reports, Reviewer evaluations, Future papers analysis

*This report was generated automatically based on comprehensive analysis of all available data sources.*"""