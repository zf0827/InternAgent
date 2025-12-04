
"""
GroundingAgent
- Deal with reports separately: paper_reports, web_reports, code_reports
- Extract evidence from each report and select the best supporting and contradicting evidence
- Accumulate all results into the same results list
- Sort and filter by score
"""

import json
import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)

class GroundingAgent(BaseAgent):
    """
    Revised Unified Grounding Agent - Processes each report separately.
    
    Input format (compatible with ReportAgent output):
    {
      "claims": {"part_name": ["claim1", "claim2", ...]},
      "reports": {
        "web_reports": [  # List of web report dicts
          {
            "report_id": "...",
            "source_description": "...",
            "content": {
              "summary": "...",
              "evidence": "...",
              "rational": "...",
              "report_content": "The actual web report text here"
            }
          },
          ...
        ],
        "code_reports": [  # List of code report dicts (same structure as web)
          {
            "report_id": "...",
            "source_description": "...",
            "content": {
              "summary": "...",
              "evidence": "...",
              "rational": "...",
              "report_content": "The actual code report text here"
            }
          },
          ...
        ],
        "paper_reports": [  # List of paper reports (from paper extraction)
          {
            "paper_metadata": {"title": "...", ...},
            "basic_idea": [...],
            "motivation": [...],
            "research_question": [...],
            "method": [...],
            "experimental_setting": [...],
            "expected_results": [...]
          },
          ...
        ]
      }
    }
    
    Output format:
    {
      "grounding_results": [
        {
          "claim": "...",
          "part": "motivation",
          "report_type": "web_report|code_report|paper_report",
          "report_id": "web_report_01" or "code_01" or "paper:Title",
          "support_evidence": "...",
          "support_score": 8,
          "support_source": "...",
          "contradiction": "...",
          "contradiction_score": -3,
          "contradiction_source": "..."
        },
        ...  # results from all reports accumulated
      ]
    }
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.model = model
        self.extract_temperature = float(config.get("extract_temperature", 0.3))
        self.ground_temperature = float(config.get("ground_temperature", 0.3))
        self.top_k = int(config.get("top_k_evidence", 20))
    
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        # Validate claims input
        claims_input = context.get("claims")
        if not claims_input or not isinstance(claims_input, dict):
            raise AgentExecutionError("UnifiedGroundingAgent requires 'claims' as a dict with one part.")
        
        if len(claims_input) != 1:
            raise AgentExecutionError("UnifiedGroundingAgent expects exactly one part in 'claims'.")
        
        part_name, claims = next(iter(claims_input.items()))
        if not isinstance(claims, list) or len(claims) == 0:
            raise AgentExecutionError("The part must contain a non-empty list of claims.")
        
        # Get reports
        reports_input = context.get("reports", {})
        if not isinstance(reports_input, dict):
            raise AgentExecutionError("'reports' must be a dictionary containing web_reports, code_reports, paper_reports")
        
        web_reports = reports_input.get("web_reports", [])
        code_reports = reports_input.get("code_reports", [])
        paper_reports = reports_input.get("paper_reports", [])
        
        # Validate reports format
        if not isinstance(web_reports, list):
            web_reports = [web_reports]
        if not isinstance(code_reports, list):
            code_reports = [code_reports]
        if not isinstance(paper_reports, list):
            paper_reports = [paper_reports]
        
        # Process all reports and accumulate results
        all_results = []
        
        # 1. Process paper reports
        paper_results = await self._process_paper_reports(part_name, claims, paper_reports)
        all_results.extend(paper_results)
        
        # 2. Process web reports
        web_results = await self._process_web_reports(part_name, claims, web_reports)
        all_results.extend(web_results)
        
        # 3. Process code reports
        code_results = await self._process_code_reports(part_name, claims, code_reports)
        all_results.extend(code_results)
        final_results = self._select_best_evidence(part_name, claims, all_results)

        print(json.dumps(all_results, indent=2, ensure_ascii=False))
        print("After ranking...")
        return {"grounding_results": final_results}
    
    def _select_best_evidence(self, part_name: str, claims: List[str], 
                             all_results: List[Dict]) -> List[Dict]:
        """Select best supporting and contradicting evidence for each claim from all reports"""
        final_results = []
        
        # Group results by claim
        claim_results = {}
        for result in all_results:
            claim = result.get("claim")
            if claim not in claim_results:
                claim_results[claim] = []
            claim_results[claim].append(result)
        
        # For each claim, select best evidence
        for claim in claims:
            claim_specific_results = claim_results.get(claim, [])
            
            # Initialize with default values
            best_support = {
                "evidence": "",
                "score": 0,
                "source": "",
                "report_type": "",
                "report_id": ""
            }
            
            best_contradiction = {
                "evidence": "",
                "score": 0,  
                "source": "",
                "report_type": "",
                "report_id": ""
            }
            
            
            # Find best supporting and contradicting evidence
            for result in claim_specific_results:
                report_type = result.get("report_type", "")
                report_id = result.get("report_id", "")
                
                # Check supporting evidence
                support_score = result.get("support_score", 0)
                if isinstance(support_score, (int, float)) and support_score > best_support["score"]:
                    best_support = {
                        "evidence": result.get("support_evidence", ""),
                        "score": support_score,
                        "source": result.get("support_source", ""),
                        "report_type": report_type,
                        "report_id": report_id
                    }
                
                # Check contradicting evidence
                contradiction_score = result.get("contradiction_score", 0)
                if isinstance(contradiction_score, (int, float)) and contradiction_score < best_contradiction["score"]:
                    best_contradiction = {
                        "evidence": result.get("contradiction", ""),
                        "score": contradiction_score,
                        "source": result.get("contradiction_source", ""),
                        "report_type": report_type,
                        "report_id": report_id
                    }
            
            
            # Create final result for this claim
            final_result = {
                "claim": claim,
                "part": part_name,
                "support_evidence": best_support["evidence"] if best_support["evidence"] else None,
                "support_score": best_support["score"],
                "support_source": best_support["source"] if best_support["source"] else None,
                "contradiction": best_contradiction["evidence"] if best_contradiction["evidence"] else None,
                "contradiction_score": best_contradiction["score"],
                "contradiction_source": best_contradiction["source"] if best_contradiction["source"] else None
            }
            
            final_results.append(final_result)
        
        return final_results
    async def _process_paper_reports(self, part_name: str, claims: List[str], paper_reports: List[Dict]) -> List[Dict]:
        """Process each paper report separately"""
        results = []
        
        for idx, paper in enumerate(paper_reports):
            if not paper:
                continue
            
            # Extract paper metadata
            metadata = paper.get("paper_metadata", {})
            title = metadata.get("title", f"Paper_{idx}").replace("\n", " ").strip()[:100]
            report_id = f"paper:{title}"
            
            # Build paper content from all sections
            paper_content = self._build_paper_content(paper)
            
            # Create labeled paper block
            paper_block = f"""[PAPER_REPORT:{title}]
            {paper_content}
            [/PAPER_REPORT:{title}]"""
            
            # Process this paper report
            report_content = {
                "paper_report": paper_block,
                "web_report": "[WEB_REPORT:none] No web report provided [/WEB_REPORT:none]",
                "code_report": "[CODE_REPORT:none] No code report provided [/CODE_REPORT:none]"
            }
            
            try:
                paper_result = await self._process_single_report(
                    part_name, claims, report_content, "paper_report", report_id
                )
                results.extend(paper_result)
            except Exception as e:
                logger.error(f"Error processing paper report {report_id}: {e}")
                # Create empty results for this paper
                results.extend(self._create_empty_results(part_name, claims, "paper_report", report_id))
        
        return results
    
    async def _process_web_reports(self, part_name: str, claims: List[str], web_reports: List[Dict]) -> List[Dict]:
        """Process each web report separately"""
        results = []
        
        for idx, web_report in enumerate(web_reports):
            if not web_report:
                continue
            
            # Extract web report content
            if isinstance(web_report, dict):
                report_id = web_report.get("report_id", f"web_report_{idx:02d}")
                source_desc = web_report.get("source_description", f"Web Report {idx}")
                content = web_report.get("content", {})
                
                # Use report_content from content dict
                report_text = content.get("report_content", "")
                if not report_text:
                    # Fallback to other fields
                    report_text = content.get("summary", "") + "\n" + \
                                 content.get("evidence", "") + "\n" + \
                                 content.get("rational", "")
            else:
                report_id = f"web_report_{idx:02d}"
                source_desc = f"Web Report {idx}"
                report_text = str(web_report)
            
            if not report_text.strip():
                continue
            
            # Create labeled web report block
            web_block = f"""[WEB_REPORT:{report_id}]
Source: {source_desc}
{report_text.strip()}
[/WEB_REPORT:{report_id}]"""
            
            # Process this web report
            report_content = {
                "web_report": web_block,
                "paper_report": "[PAPER_REPORT:none] No paper report provided [/PAPER_REPORT:none]",
                "code_report": "[CODE_REPORT:none] No code report provided [/CODE_REPORT:none]"
            }
            
            try:
                web_result = await self._process_single_report(
                    part_name, claims, report_content, "web_report", report_id
                )
                results.extend(web_result)
            except Exception as e:
                logger.error(f"Error processing web report {report_id}: {e}")
                # Create empty results for this web report
                results.extend(self._create_empty_results(part_name, claims, "web_report", report_id))
        
        return results
    
    async def _process_code_reports(self, part_name: str, claims: List[str], code_reports: List[Dict]) -> List[Dict]:
        """Process each code report separately"""
        results = []
        
        for idx, code_report in enumerate(code_reports):
            if not code_report:
                continue
            
            # Extract code report content
            if isinstance(code_report, dict):
                report_id = code_report.get("report_id", f"code_{idx:02d}")
                source_desc = code_report.get("source_description", f"Code Report {idx}")
                content = code_report.get("content", {})
                
                # Use report_content from content dict
                report_text = content.get("report_content", "")
                if not report_text:
                    # Fallback to other fields
                    report_text = content.get("summary", "") + "\n" + \
                                 content.get("evidence", "") + "\n" + \
                                 content.get("rational", "")
            else:
                report_id = f"code_{idx:02d}"
                source_desc = f"Code Report {idx}"
                report_text = str(code_report)
            
            if not report_text.strip():
                continue
            
            # Create labeled code report block
            code_block = f"""[CODE_REPORT:{report_id}]
Source: {source_desc}
{report_text.strip()}
[/CODE_REPORT:{report_id}]"""
            
            # Process this code report
            report_content = {
                "code_report": code_block,
                "web_report": "[WEB_REPORT:none] No web report provided [/WEB_REPORT:none]",
                "paper_report": "[PAPER_REPORT:none] No paper report provided [/PAPER_REPORT:none]"
            }
            
            try:
                code_result = await self._process_single_report(
                    part_name, claims, report_content, "code_report", report_id
                )
                results.extend(code_result)
            except Exception as e:
                logger.error(f"Error processing code report {report_id}: {e}")
                # Create empty results for this code report
                results.extend(self._create_empty_results(part_name, claims, "code_report", report_id))
        
        return results
    
    def _build_paper_content(self, paper: Dict) -> str:
        """Build paper content from all sections"""
        sections = []
        
        # Basic idea
        basic_idea = paper.get("basic_idea", [])
        if basic_idea:
            sections.append("BASIC IDEA:")
            sections.extend([f"- {item}" for item in basic_idea])
            sections.append("")
        
        # Motivation
        motivation = paper.get("motivation", [])
        if motivation:
            sections.append("MOTIVATION:")
            sections.extend([f"- {item}" for item in motivation])
            sections.append("")
        
        # Research question
        research_question = paper.get("research_question", [])
        if research_question:
            sections.append("RESEARCH QUESTION:")
            sections.extend([f"- {item}" for item in research_question])
            sections.append("")
        
        # Method
        method = paper.get("method", [])
        if method:
            sections.append("METHOD:")
            sections.extend([f"- {item}" for item in method])
            sections.append("")
        
        # Experimental setting
        experimental_setting = paper.get("experimental_setting", [])
        if experimental_setting:
            sections.append("EXPERIMENTAL SETTING:")
            sections.extend([f"- {item}" for item in experimental_setting])
            sections.append("")
        
        # Expected results
        expected_results = paper.get("expected_results", [])
        if expected_results:
            sections.append("EXPECTED RESULTS:")
            sections.extend([f"- {item}" for item in expected_results])
            sections.append("")
        
        return "\n".join(sections)
    
    def _create_empty_results(self, part_name: str, claims: List[str], 
                            report_type: str, report_id: str) -> List[Dict]:
        """Create empty grounding results for failed reports"""
        results = []
        for claim in claims:
            results.append({
                "claim": claim,
                "part": part_name,
                "report_type": report_type,
                "report_id": report_id,
                "support_evidence": "",
                "support_score": 0,
                "support_source": "",
                "contradiction": "",
                "contradiction_score": 0,
                "contradiction_source": ""
            })
        return results
    
    async def _process_single_report(self, part_name: str, claims: List[str], 
                                   report_content: Dict[str, str], report_type: str, 
                                   report_id: str) -> List[Dict]:
        """Process a single report and return grounding results"""
        # Stage 1: Evidence extraction
        extract_prompt = self._build_extract_prompt(
            part_name, claims, report_content, report_type, report_id
        )
        extract_schema = self._extract_schema()
        
        try:
            extract_response = await self._call_model(
                prompt=extract_prompt,
                system_prompt=self._get_extract_system_prompt(report_type),
                schema=extract_schema,
                temperature=self.extract_temperature,
            )
        except Exception as e:
            raise AgentExecutionError(f"Evidence extraction failed for {report_type} {report_id}: {e}")
        
        evidences_obj = extract_response.get("evidences", {})
        if not isinstance(evidences_obj, dict):
            evidences_obj = {}
        
        # Trim and normalize evidences
        for claim in list(evidences_obj.keys()):
            if isinstance(evidences_obj[claim], list):
                evidences_obj[claim] = [
                    str(x).strip() for x in evidences_obj[claim] 
                    if x and str(x).strip()
                ][:self.top_k]
            else:
                evidences_obj[claim] = []
        
        # Stage 2: Grounding / scoring
        ground_prompt = self._build_ground_prompt(
            part_name, claims, evidences_obj, report_content, report_type, report_id
        )
        ground_schema = self._ground_schema()
        
        try:
            ground_response = await self._call_model(
                prompt=ground_prompt,
                system_prompt=self._get_ground_system_prompt(report_type),
                schema=ground_schema,
                temperature=self.ground_temperature,
            )
        except Exception as e:
            raise AgentExecutionError(f"Grounding failed for {report_type} {report_id}: {e}")
        
        results = ground_response.get("grounding_results", [])
        if not isinstance(results, list):
            results = []
        
        # Add report_type and report_id to each result
        for result in results:
            result["report_type"] = report_type
            result["report_id"] = report_id
        
        return results
    
    # ---------------- PROMPT BUILDERS ----------------
    def _get_extract_system_prompt(self, report_type: str) -> str:
        """Get system prompt for evidence extraction based on report type"""
        if report_type == "paper_report":
            return """You are an expert evidence extractor for PAPER reports.
            
CRITICAL RULES FOR PAPER REPORTS:
1. You are analyzing a SINGLE academic paper report.
2. You MUST extract evidence from ALL sections of the paper:
   - BASIC IDEA: The paper's core concepts
   - MOTIVATION: The paper's research motivations
   - RESEARCH QUESTION: The paper's research questions
   - METHOD: The paper's methodologies and approaches
   - EXPERIMENTAL SETTING: The paper's experimental setups
   - EXPECTED RESULTS: The paper's expected outcomes
3. These are ALL VALID evidence from an external paper!
4. Each evidence string MUST end with the exact source identifier in parentheses: (PAPER_REPORT:paper:Title)
5. Example: "The paper demonstrates CoT improves accuracy by 20% on complex tasks (PAPER_REPORT:paper:Faithful CoT)"
6. Extract ALL relevant evidence that either SUPPORTS or CONTRADICTS each claim.
7. If no evidence is found for a claim, return an empty list for that claim."""
        
        elif report_type == "web_report":
            return """You are an expert evidence extractor for WEB reports.
            
CRITICAL RULES FOR WEB REPORTS:
1. You are analyzing a SINGLE web analysis report.
2. DO NOT extract evidence from any section that is summarizing/restating THE IDEA BEING EVALUATED.
3. Specifically avoid: "## 1. Idea Overview", "### 1. Idea Restatement", or similar summary sections.
4. ONLY extract from analytical sections like:
   - Extracted Viewpoints
   - Supportive Evidence
   - Contradictory Evidence
   - Limitations and Considerations
   - Implementation Details
   - Key Insights
   - Experimental Results
   - Critical Analysis sections
5. Each evidence string MUST end with the exact source identifier in parentheses: (WEB_REPORT:report_id)
6. Example: "Web analysis confirms CoT enhances reasoning capabilities (WEB_REPORT:web_report_02)"
7. Extract ALL relevant evidence that either SUPPORTS or CONTRADICTS each claim.
8. If no evidence is found for a claim, return an empty list for that claim."""
        
        else:  # code_report
            return """You are an expert evidence extractor for CODE reports.
            
CRITICAL RULES FOR CODE REPORTS:
1. You are analyzing a SINGLE code analysis report.
2. DO NOT extract evidence from sections summarizing THE IDEA BEING EVALUATED.
3. Specifically avoid: "### 1. Idea Restatement" or similar summary sections.
4. Extract from technical analysis sections only, such as:
   - Critical Missing Components
   - Architectural Mismatches & Inflexibility
   - Dependency & Environment Issues
   - Performance & Scalability Failures
   - Maintenance & Sustainability Red Flags
   - Historical Implementation Failures
5. Each evidence string MUST end with the exact source identifier in parentheses: (CODE_REPORT:report_id)
6. Example: "Existing frameworks lack executable reasoning modules (CODE_REPORT:code_01)"
7. Extract ALL relevant evidence that either SUPPORTS or CONTRADICTS each claim.
8. If no evidence is found for a claim, return an empty list for that claim."""
    
    def _build_extract_prompt(self, part_name: str, claims: List[str], 
                            report_content: Dict[str, str], report_type: str, 
                            report_id: str) -> str:
        """Build evidence extraction prompt for a single report"""
        # Build claims block
        claims_block = "".join([f"{i}. {claim}\n" for i, claim in enumerate(claims, 1)])
        
        # Get the relevant report content based on type
        if report_type == "paper_report":
            report_text = report_content.get("paper_report", "")
            report_label = "PAPER REPORT"
        elif report_type == "web_report":
            report_text = report_content.get("web_report", "")
            report_label = "WEB REPORT"
        else:  # code_report
            report_text = report_content.get("code_report", "")
            report_label = "CODE REPORT"
        
        template = f"""
You are extracting evidence from a SINGLE {report_label} to ground specific claims.

IMPORTANT INSTRUCTIONS:
1. You are analyzing claims for the part: {part_name}
2. These claims come from ONE SPECIFIC RESEARCH IDEA that is being evaluated.
3. You are working with ONLY ONE {report_label}: {report_id}

FOR THIS {report_label} ONLY:
- Extract evidence from the content below
- Each evidence MUST end with the exact source identifier in parentheses
- Use the format: "FUll Evidence text here ({report_type.upper()}:{report_id})"
- Look for evidence that either SUPPORTS or CONTRADICTS each claim

Return a JSON object with a single top-level key "evidences" mapping each claim to a list of evidence strings.
You MUST output valid JSON ONLY.
Do NOT wrap the output in triple backticks.
Do NOT use code blocks.
Do NOT include markup such as ```json, ``` or any other fences.
Do NOT add explanations, comments, or text outside the JSON.
Your entire response MUST be a single valid JSON object and nothing else.

Example output format:
{{
"evidences": {{
    "Standard chain-of-thought reasoning improves model performance on complex tasks.": [
    "The paper demonstrates CoT improves accuracy by 20% on complex tasks (PAPER_REPORT:paper:Faithful CoT)"
    ],
    "Another claim here.": []
}}
}}

--- CLAIMS TO GROUND (from the idea being evaluated) ---
PART: {part_name}
{claims_block}

--- SINGLE {report_label}: {report_id} ---
{report_text}
"""
        return template
    
    def _extract_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "evidences": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "required": ["evidences"]
        }
    
    def _get_ground_system_prompt(self, report_type: str) -> str:
        """Get system prompt for grounding/scoring based on report type"""
        report_type_name = report_type.replace("_", " ").title()
        
        return f"""You are an expert scientific adjudicator evaluating evidence from a SINGLE {report_type_name}.

IMPORTANT RULES:
1. You are evaluating evidence from ONLY ONE {report_type_name}.
2. For each claim, select the SINGLE BEST supporting evidence from this report.
3. For each claim, select the SINGLE BEST contradicting evidence from this report.
4. All evidence must come from this same {report_type_name}.
5. Scores must be assigned using the standard scales:
   - Support scores: 0-10 (0=no evidence, 10=strongest direct validation)
   - Contradiction scores: -10 to 0 (0=no evidence, -10=strongest direct refutation)
6. Source identifiers MUST be extracted from the evidence text (the part in parentheses).

Output only valid JSON with the exact required structure."""
    
    def _build_ground_prompt(self, part_name: str, claims: List[str], 
                           evidences_dict: Dict[str, List[str]], 
                           report_content: Dict[str, str], report_type: str,
                           report_id: str) -> str:
        """Build grounding/scoring prompt for a single report"""
        # Build claims with evidences
        blocks = []
        for idx, claim in enumerate(claims, start=1):
            evs = evidences_dict.get(claim) or []
            blocks.append(f"ITEM {idx} - CLAIM: {claim}")
            if not evs:
                blocks.append("  EVIDENCES: (no evidences found in this report)")
            else:
                blocks.append(f"  EVIDENCES FROM {report_type.upper()}:{report_id}:")
                for j, ev in enumerate(evs, start=1):
                    blocks.append(f"    [{j}] {ev}")
            blocks.append("")
        
        block_text = "\n".join(blocks)
        
        # Get report text for reference
        if report_type == "paper_report":
            report_text = report_content.get("paper_report", "")
            report_label = "PAPER REPORT"
        elif report_type == "web_report":
            report_text = report_content.get("web_report", "")
            report_label = "WEB REPORT"
        else:  # code_report
            report_text = report_content.get("code_report", "")
            report_label = "CODE REPORT"
        
        template = f"""
You are evaluating evidence from a SINGLE {report_label} for claims in the part: {part_name}

CRITICAL: You are working with ONLY ONE {report_label} ({report_id}). 
All evidence selections MUST come from this same report.

FOR EACH CLAIM:
1. From the available evidence for this claim (all from {report_id}), select:
   a. The SINGLE BEST supporting evidence (if any)
   b. The SINGLE BEST contradicting evidence (if any)

2. For each selected evidence:
   - Copy the FULL evidence text (including the source identifier in parentheses)
   - Extract the source identifier (the part in parentheses without the parentheses)
   - Assign appropriate scores:
     * Support: 0-10 (10=strongest support)
     * Contradiction: -10 to 0 (-10=strongest contradiction)
     * 0 means no evidence for that category

3. IMPORTANT SCORING GUIDELINES:
   - 8-10: Direct experimental validation, quantitative results, explicit confirmation
   - 5-7: Strong theoretical support, detailed implementation, clear conceptual alignment
   - 1-4: Weak or indirect support, tangential relevance
   - -1 to -4: Weak contradiction or counter-argument
   - -5 to -7: Clear conceptual misalignment or reasonable counter-evidence
   - -8 to -10: Direct experimental contradiction or explicit refutation

4. REMEMBER: You are evaluating ONE report only. If multiple evidences exist, choose the BEST one for each category.

Output strict JSON with this exact structure:
{{
  "grounding_results": [
    {{
      "claim": "...",
      "part": "{part_name}",
      "support_evidence": "...",  // FULL evidence text including source
      "support_score": 0,
      "support_source": "SOURCE_IDENTIFIER",  // e.g., "PAPER_REPORT:paper:Title"
      "contradiction": "...",  // FULL evidence text including source
      "contradiction_score": 0,
      "contradiction_source": "SOURCE_IDENTIFIER"  // e.g., "CODE_REPORT:code_01"
    }}
  ]
}}

--- ITEMS TO EVALUATE (from {report_label}:{report_id}) ---
{block_text}

--- REFERENCE: THE {report_label} BEING EVALUATED ---
{report_text}

You MUST output valid JSON ONLY.
Do NOT wrap the output in triple backticks.
Do NOT use code blocks.
Do NOT include markup such as ```json, ``` or any other fences.
Do NOT add explanations, comments, or text outside the JSON.
Your entire response MUST be a single valid JSON object and nothing else.
"""
        return template
    
    def _ground_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "grounding_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "part": {"type": "string"},
                            "support_evidence": {"type": ["string", "null"]},
                            "support_score": {"type": "number"},
                            "support_source": {"type": ["string", "null"]},
                            "contradiction": {"type": ["string", "null"]},
                            "contradiction_score": {"type": "number"},
                            "contradiction_source": {"type": ["string", "null"]}
                        },
                        "required": [
                            "claim", "part",
                            "support_evidence", "support_score", "support_source",
                            "contradiction", "contradiction_score", "contradiction_source"
                        ]
                    }
                }
            },
            "required": ["grounding_results"]
        }



