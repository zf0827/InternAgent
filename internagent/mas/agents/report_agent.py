import random
from typing import  Dict, Any, List
from .base_agent import BaseAgent, AgentExecutionError
from ..tools.searchers.models import SearchResults
from .extraction_agent import ExtractionAgent

class ReportAgent(BaseAgent):
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "ReportAgent"
        self.temperature = config.get("temperature")   
        extraction_config = config.get("extraction_config")
        self.extraction_agent = ExtractionAgent(model, extraction_config)

    async def extract_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract PDF URLs from papers and call ExtractionAgent to get structured information
        
        """
        extracted_results = []
        
        # 1. collect PDF URL
        pdf_urls = []
        for paper in papers:
            if paper.get("pdf_url"):
                pdf_url = paper.get("pdf_url")
            elif paper.get("url") and "pdf" in paper["url"].lower():
                pdf_url = paper["url"]
            elif paper.get("url"):  
                if "arxiv.org" in paper["url"]:
                    arxiv_id = paper["url"].split("/")[-1].replace("v", "").replace(".html", "")
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                elif "openreview.net" in paper["url"]:
                    pdf_url = paper["url"].replace("https://", "https://openreview.net/pdf?id=")
            pdf_urls.append({
                "paper": paper,
                "pdf_url": pdf_url
                })
        
        print(f"📚 Find {len(pdf_urls)} extractable PDF links")
        extracted_results = []

        # 2. Randomly select up to 10 papers
        max_papers = 10
        if len(pdf_urls) > max_papers:
            selected_papers = random.sample(pdf_urls, max_papers)
            print(f"Randomly select {max_papers} papers for extraction")
        else:
            selected_papers = pdf_urls
            print(f"extract {len(selected_papers)} papers")

        for idx, item in enumerate(selected_papers,1):
            paper = item["paper"]
            pdf_url = item["pdf_url"]
            
            print(f"\n[{idx}/{len(selected_papers)}] Extracting paper:")
            print(f"   Title: {paper.get('title', 'Unknown')[:60]}...")
            print(f"   PDF:   {pdf_url[:80]}...")

            result = await self._extract_single_paper(paper, pdf_url, idx)

            if result:
                extracted_results.append(result)
                print(f"Extraction successful")
            else:
                print(f"Extraction failed")
            
            print(f"\n Sequential extraction completed! Successfully extracted {len(extracted_results)}/{len(selected_papers)} papers")
        return extracted_results

    async def _extract_single_paper(self, paper: Dict[str, Any], pdf_url: str, idx: int) -> Dict[str, Any]:
        try:
            context = {"url": pdf_url}           
            extraction_result = await self.extraction_agent.execute(context, {})
            
            # add metadata
            enhanced_result = {
                "paper_metadata": {
                    "title": paper.get("title", "Unknown"),
                    "url": paper.get("url", ""),
                    "pdf_url": pdf_url,
                    "authors": paper.get("authors", []),
                    "year": paper.get("year", ""),
                    "platform": paper.get("platform", "")
                },
                **extraction_result
            }
            
            return enhanced_result
            
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}...")
            return None

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        sr = context.get("search_result") or context.get("search_results")
        if not isinstance(sr, dict):
            raise AgentExecutionError("search_result must be a dict containing idea and search results")

        papers = sr.get("papers", []) + sr.get("scholar_results", [])
        github_repos = sr.get("github_repos", [])
        kaggle_results = sr.get("kaggle_results", [])
        web_pages = sr.get("web_pages", [])
        code_desc = self._pick_descriptions((github_repos +kaggle_results), 10)
        print(f"total: Papers={len(papers)}, Code={len(github_repos)}, Web={len(web_pages)}")
        idea_text = self._get_idea_text(sr)
        print("\n Start structured extraction of the paper...")
        paper_extraction_results = await self.extract_papers(papers)

  
        print("Generate Web Report...")
        web_descriptions = self._pick_descriptions(web_pages, 10) 
        web_reports = []
        
        for idx, single_desc in enumerate(web_descriptions, 1):
            web_prompt = self._build_web_prompt(idea_text, [single_desc], idx)
            
            print(f"  Generate the {idx}th Web Report...")
            web_schema=self._build_output_schema(analysis_type="web")
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

        
        print("\n Generate Code report...")
        code_reports = []
        code_schema=self._build_output_schema(analysis_type="code")
        for idx, single_desc in enumerate(code_desc, 1):
            code_prompt = self._build_code_prompt(idea_text, [single_desc], idx)
            
            print(f"  Generate the {idx}th Code Report......")
            
            single_code_report = await self._call_model(
                prompt=code_prompt,
                system_prompt=self.system_prompt,
                schema=code_schema,
                temperature=params.get("temperature", self.temperature),
            )
            

            code_reports.append({
                "report_id": f"web_report_{idx:02d}",
                "source_description": single_desc,
                "content": single_code_report
            })
   

        return {
            "web_reports": web_reports, 
            "code_reports": code_reports,
            "paper_reports": paper_extraction_results,
            "params": params,
        }
        
        
    def _pick_descriptions(self, items: List[Dict[str, Any]], k: int) -> List[str]:
        descs = []
        for index, i in enumerate(items):
            if not isinstance(i, dict):
                continue
            text = i.get("page_raw_text","")
            platform = i.get("platform", "")
            file_tree = i.get("file_tree", "")
            block = f"Source{index}: \n"
            if text!="":
                block+=f"page raw text: \n{text}\n"
            if platform == "github" and file_tree:
                block += f"file_tree: \n{file_tree}\n"
            descs.append(block)
        if not descs:
            return []
        if len(descs) <= k:
            return descs
        return random.sample(descs, k)
    
 

    def _get_idea_text(self, sr: Dict[str, Any]) -> str:
        idea = sr.get("idea") or {}
        raw = idea.get("raw_text")
        
        # Check if part is specified
        part_list = idea.get("part")
        
        # If part is specified, use partial selection logic even if raw_text exists
        if part_list:
            # If part is specified, only include those fields
            valid_parts = {'motivation', 'research_question', 'method', 'experimental_setting'}
            parts = []
            field_map = {
                'motivation': ('Motivation', idea.get('motivation', '')),
                'research_question': ('Research Question', idea.get('research_question', '')),
                'method': ('Method', idea.get('method', '')),
                'experimental_setting': ('Experimental Setting', idea.get('experimental_setting', '')),
            }
            for p in part_list:
                if p in valid_parts and p in field_map:
                    label, content = field_map[p]
                    if content:  # Only add if content is not empty
                        parts.append(f"{label}: {content}")
            
            exp = idea.get("expected_results")
            if exp and 'expected_results' in part_list:
                parts.append(f"Expected Results: {exp}")
            
            # If we have parts from structured fields, return them
            if parts:
                return "\n\n".join(parts)
            # If part is specified but no structured fields, fall back to raw_text if available
            if raw:
                return raw
            return ""
        
        # If no part is specified, use raw_text if available
        if raw:
            return raw
        
        # Default: include all fields
        parts = []
        if idea.get('motivation'):
            parts.append(f"Motivation: {idea.get('motivation', '')}")
        if idea.get('research_question'):
            parts.append(f"Research Question: {idea.get('research_question', '')}")
        if idea.get('method'):
            parts.append(f"Method: {idea.get('method', '')}")
        if idea.get('experimental_setting'):
            parts.append(f"Experimental Setting: {idea.get('experimental_setting', '')}")
        
        exp = idea.get("expected_results")
        if exp:
            parts.append(f"Expected Results: {exp}")
        
        return "\n\n".join(parts) if parts else ""

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
        
    def _build_web_prompt(self, idea_text: str, description: List[str], source_idx: int) -> str:
        raw_text = description[0] if description else ""
        return (
        f"You are generating a comprehensive analysis for a research idea based on a SINGLE web resource.\n\n"
        f"=== Input Specification (Web Source #{source_idx}) ===\n"
        f"Research Idea:\n{idea_text}\n\n"
        f"Web Resource Raw Text (Content to analyze):\n{raw_text}\n\n"
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
    )

    def _build_code_prompt(self, idea_text: str, description: List[str],source_idx: int) -> str:
        raw_text = description[0] if description else ""
        return (
        f"You are generating a technical implementation analysis for a research idea based on a SINGLE code resource.\n\n"
        f"=== Input Specification (Code Source #{source_idx}) ===\n"
        f"Research Idea:\n{idea_text}\n\n"
        f"Code Resource Raw Text (Content to analyze):\n{raw_text}\n\n"
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
    )