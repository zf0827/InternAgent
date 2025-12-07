"""
GroundingAgent - Two-Phase Optimized Version
Extract Phase: Extract Evidence → Ground Phase: Scoring + Explanation
Adapted for ReportAgent Output Format
"""

import json
import logging
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)

class GroundingAgent(BaseAgent):
    """
    Two-Phase Grounding Agent
    
    Process:
    1. Extract Phase: Extract best evidence for each claim from each report
    2. Ground Phase: Score the extracted evidence and provide rationale
    
    Adapted for ReportAgent Output Format:
    - web_reports: [{report_id, source_description, summary, report_content}]
    - code_reports: [{report_id, source_description, summary, report_content}]
    - paper_reports: [{paper_metadata, basic_idea, motivation, ...}]
    
    Output Format:
    {
      "grounding_results": [
        {
          "claim_index": 0,
          "part": "motivation",
          "report_type": "web_report",
          "report_id": "web_01",
          "evidence": "Evidence text",
          "score": 8,
          "rationale": "Scoring rationale...",
          "source": "WEB_REPORT:web_01"
        }
      ]
    }
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.model = model
        self.extract_temp = float(config.get("extract_temperature", 0.3))
        self.ground_temp = float(config.get("ground_temperature", 0.3))
    
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute two-phase grounding process"""
        # Validate input
        part_name, claims = self._validate_input(context)
        
        # Get reports (adapted for ReportAgent output format)
        reports_input = context.get("reports", {})
        web_reports = self._ensure_list(reports_input.get("web_reports", []))
        code_reports = self._ensure_list(reports_input.get("code_reports", []))
        paper_reports = self._ensure_list(reports_input.get("paper_reports", []))
        
        print(f"📊 Grounding Input Statistics:")
        print(f"  - Claims: {len(claims)}")
        print(f"  - Web Reports: {len(web_reports)}")
        print(f"  - Code Reports: {len(code_reports)}")
        print(f"  - Paper Reports: {len(paper_reports)}")
        
        # Phase 1: Extract evidence from all reports
        all_evidence = await self._extract_evidence_from_all_reports(
            part_name, claims, paper_reports, web_reports, code_reports
        )
        
        print(f"\n✅ Evidence extraction completed: {len(all_evidence)} evidence fragments")
        
        # Phase 2: Score the evidence
        grounded_results = await self._ground_evidence(
            part_name, claims, all_evidence
        )
        
        print(f"\n✅ Evidence scoring completed: {len(grounded_results)} scoring results")
        
        return {"grounding_results": grounded_results}
    
    def _validate_input(self, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Validate input format"""
        claims_input = context.get("claims")
        if not claims_input or not isinstance(claims_input, dict):
            raise AgentExecutionError("'claims' dictionary required")
        
        if len(claims_input) != 1:
            raise AgentExecutionError("Only one part can be processed at a time")
        
        part_name, claims = next(iter(claims_input.items()))
        if not isinstance(claims, list) or len(claims) == 0:
            raise AgentExecutionError("Non-empty claims list required")
        
        return part_name, claims
    
    def _ensure_list(self, data):
        """Ensure data is a list"""
        if not isinstance(data, list):
            return [data] if data else []
        return data
    
    async def _extract_evidence_from_all_reports(self, part_name: str, claims: List[str],
                                               paper_reports: List[Dict],
                                               web_reports: List[Dict],
                                               code_reports: List[Dict]) -> List[Dict]:
        """Extract evidence from all reports"""
        all_evidence = []
        
        print(f"\n📄 Extracting evidence from reports...")
        
        # Process paper reports
        print(f"  - Processing {len(paper_reports)} paper reports")
        for idx, paper in enumerate(paper_reports):
            if not paper:
                continue
            
            evidence = await self._extract_from_paper(part_name, claims, paper, idx)
            if evidence:
                all_evidence.extend(evidence)
                print(f"    ✅ Paper {idx}: Extracted {len(evidence)} evidence")
        
        # Process web reports
        print(f"  - Processing {len(web_reports)} web reports")
        for idx, web_report in enumerate(web_reports):
            if not web_report:
                continue
            
            evidence = await self._extract_from_web(part_name, claims, web_report, idx)
            if evidence:
                all_evidence.extend(evidence)
                report_id = web_report.get("report_id", f"web_{idx:02d}")
                print(f"    ✅ {report_id}: Extracted {len(evidence)} evidence")
        
        # Process code reports
        print(f"  - Processing {len(code_reports)} code reports")
        for idx, code_report in enumerate(code_reports):
            if not code_report:
                continue
            
            evidence = await self._extract_from_code(part_name, claims, code_report, idx)
            if evidence:
                all_evidence.extend(evidence)
                report_id = code_report.get("report_id", f"code_{idx:02d}")
                print(f"    ✅ {report_id}: Extracted {len(evidence)} evidence")
        
        return all_evidence
    
    async def _extract_from_paper(self, part_name: str, claims: List[str],
                                paper: Dict, paper_idx: int) -> List[Dict]:
        """Extract evidence from paper report"""
        metadata = paper.get("paper_metadata", {})
        title = metadata.get("title", f"Paper_{paper_idx}").replace("\n", " ").strip()[:100]
        report_id = f"paper:{paper_idx}_{title}"
        
        # Build paper content
        paper_content = self._build_paper_content(paper)
        
        if not paper_content.strip():
            return []
        
        # Extract evidence
        return await self._extract_from_single_report(
            part_name=part_name,
            claims=claims,
            report_content=paper_content,
            report_type="paper_report",
            report_id=report_id,
            source_prefix="PAPER_REPORT"
        )
    
    async def _extract_from_web(self, part_name: str, claims: List[str],
                              web_report: Dict, web_idx: int) -> List[Dict]:
        """Extract evidence from web report (adapted for ReportAgent format)"""
        if not isinstance(web_report, dict):
            return []
        
        # Get report ID
        report_id = web_report.get("report_id", f"web_{web_idx:02d}")
        
        # Get report_content (ReportAgent format)
        if "content" in web_report and isinstance(web_report["content"], dict):
            # Old format: content is a dictionary
            report_text = web_report["content"].get("report_content", "")
        else:
            # ReportAgent format: directly contains summary and report_content
            report_text = web_report.get("report_content", "")
        
        # If no report_content, try using summary
        if not report_text:
            report_text = web_report.get("summary", "")
        
        if not report_text or not report_text.strip():
            return []
        
        # Extract evidence
        return await self._extract_from_single_report(
            part_name=part_name,
            claims=claims,
            report_content=report_text,
            report_type="web_report",
            report_id=report_id,
            source_prefix="WEB_REPORT"
        )
    
    async def _extract_from_code(self, part_name: str, claims: List[str],
                               code_report: Dict, code_idx: int) -> List[Dict]:
        """Extract evidence from code report (adapted for ReportAgent format)"""
        if not isinstance(code_report, dict):
            return []
        
        # Get report ID
        report_id = code_report.get("report_id", f"code_{code_idx:02d}")
        
        # Get report_content (ReportAgent format)
        if "content" in code_report and isinstance(code_report["content"], dict):
            # Old format: content is a dictionary
            report_text = code_report["content"].get("report_content", "")
        else:
            # ReportAgent format: directly contains summary and report_content
            report_text = code_report.get("report_content", "")
        
        # If no report_content, try using summary
        if not report_text:
            report_text = code_report.get("summary", "")
        
        if not report_text or not report_text.strip():
            return []
        
        # Extract evidence
        return await self._extract_from_single_report(
            part_name=part_name,
            claims=claims,
            report_content=report_text,
            report_type="code_report",
            report_id=report_id,
            source_prefix="CODE_REPORT"
        )
    
    async def _extract_from_single_report(self, part_name: str, claims: List[str],
                                        report_content: str, report_type: str,
                                        report_id: str, source_prefix: str) -> List[Dict]:
        """Extract evidence from a single report"""
        try:
            # Extract Phase: Select best evidence
            extract_prompt = self._build_extract_prompt(
                part_name=part_name,
                claims=claims,
                report_content=report_content,
                report_type=report_type,
                report_id=report_id,
                source_prefix=source_prefix
            )
            
            extract_response = await self._call_model(
                prompt=extract_prompt,
                system_prompt=self._get_extract_system_prompt(report_type),
                schema=self._extract_schema(),
                temperature=self.extract_temp,
            )
            
            # Convert results to unified format
            evidences = extract_response.get("evidences", [])
            if not isinstance(evidences, list):
                evidences = []
            
            # Add metadata
            for evidence_item in evidences:
                evidence_item.update({
                    "report_type": report_type,
                    "report_id": report_id,
                    "part": part_name
                })
            
            return evidences
            
        except Exception as e:
            logger.error(f"Evidence extraction failed [{report_type}:{report_id}]: {e}")
            # Return empty evidence list
            return self._create_empty_evidence(claims, report_type, report_id, part_name)
    
    async def _ground_evidence(self, part_name: str, claims: List[str],
                             all_evidence: List[Dict]) -> List[Dict]:
        """Score the extracted evidence"""
        if not all_evidence:
            return self._create_empty_results(claims)
        
        try:
            # Prepare grounding input
            grounding_input = self._prepare_grounding_input(claims, all_evidence)
            
            # Ground Phase: Scoring
            ground_prompt = self._build_ground_prompt(
                part_name=part_name,
                claims=claims,
                evidence_by_claim=grounding_input
            )
            
            ground_response = await self._call_model(
                prompt=ground_prompt,
                system_prompt=self._get_ground_system_prompt(),
                schema=self._ground_schema(),
                temperature=self.ground_temp,
            )
            
            grounded_results = ground_response.get("grounding_results", [])
            
            # Ensure each claim has a result
            return self._ensure_complete_results(claims, grounded_results, all_evidence)
            
        except Exception as e:
            logger.error(f"Evidence scoring failed: {e}")
            return self._create_empty_results(claims)
    
    # ----------------- Phase 1: Extract -----------------
    
    def _get_extract_system_prompt(self, report_type: str) -> str:
        """Extract phase system prompt"""
        type_name = report_type.replace("_", " ").title()
        
        return f"""You are an evidence extraction expert. Task: Select the most relevant supporting evidence for each claim from one {type_name}.

Rules:
1. Analyze only this specific {type_name}
2. For each claim, select one evidence fragment from the report that best supports it
3. Evidence must be complete, containing full source identification
4. If no relevant evidence exists in the report, set evidence field to null
5. Output should use claim_index to reference claims, do not repeat claim text
6. For PAPER report：
- This is a BACKGROUND/RELATED PAPER, not our research idea paper
- The paper's sections (motivation, method, etc.) describe THAT PAPER'S own research
- Evidence should show how this related paper's content is RELEVANT to our research idea
Important: Only select one best evidence for each claim!"""
    
    def _build_extract_prompt(self, part_name: str, claims: List[str],
                            report_content: str, report_type: str,
                            report_id: str, source_prefix: str) -> str:
        """Build extract prompt"""
        
        claims_list = "\n".join([f"{i}. {claim}" for i, claim in enumerate(claims)])
        
        return f"""
## Evidence Extraction Task
Part: {part_name}
Report Type: {report_type}
Report ID: {report_id}

## Claims List ({len(claims)} total):
{claims_list}

## Report Content:
{report_content} 

## Extraction Requirements:
1. For each claim, extract the most supporting evidence from the above report
2. Evidence should be complete semantic units:
   - Can be several related sentences
   - Can be an entire paragraph
   - Can be complete bullet points (including all sub-points)
   - Ensure evidence is self-contained and meaningful
3. Do not extract only single short sentences or fragments
4. Evidence must be complete
5. If no relevant evidence exists in the report, set evidence to null
6. Use claim_index to reference claim (0-based index)

## Output Format:
{{
  "evidences": [
    {{
      "claim_index": 0,
      "evidence": "Evidence text",
      "source": "{source_prefix}:{report_id}"
    }},
    // Other claims...
  ]
}}

Note: Output only JSON, no additional content."""
    
    def _extract_schema(self) -> Dict[str, Any]:
        """Extract phase output schema"""
        return {
            "type": "object",
            "properties": {
                "evidences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_index": {
                                "type": "integer",
                                "minimum": 0,
                                "description": "Claim index, starting from 0"
                            },
                            "evidence": {
                                "type": ["string", "null"],
                                "description": "Evidence text, containing source identification"
                            },
                            "source": {
                                "type": ["string", "null"],
                                "description": "Source identification"
                            }
                        },
                        "required": ["claim_index", "evidence", "source"]
                    }
                }
            },
            "required": ["evidences"]
        }
    
    # ----------------- Phase 2: Ground -----------------
    
    def _prepare_grounding_input(self, claims: List[str], all_evidence: List[Dict]) -> Dict[int, List[Dict]]:
        """Organize evidence by claim for grounding phase"""
        evidence_by_claim = {}
        
        for claim_idx in range(len(claims)):
            evidence_by_claim[claim_idx] = []
        
        for evidence in all_evidence:
            claim_idx = evidence.get("claim_index")
            if claim_idx is not None and 0 <= claim_idx < len(claims):
                evidence_by_claim[claim_idx].append(evidence)
        
        return evidence_by_claim
    
    def _get_ground_system_prompt(self) -> str:
        """Ground phase system prompt"""
        return """You are an evidence scoring expert. Task: Evaluate the strength of evidence supporting each claim.

Scoring Criteria (0-10 points):
10 points: Direct experimental verification, quantitative results, clear confirmation
9 points: Strong experimental support, significant results
8 points: Clear theoretical support, detailed implementation
7 points: Strong correlation, reasonable support
6 points: Moderate support, relevant but not direct
5 points: Some correlation, limited support
4 points: Weak correlation, indirect support
3 points: Minimal correlation, very weak support
2 points: Barely relevant
1 point: Almost irrelevant
0 points: Completely irrelevant or no evidence

You must provide clear rationale for each score, explaining why you gave that rating."""
    
    def _build_ground_prompt(self, part_name: str, claims: List[str],
                           evidence_by_claim: Dict[int, List[Dict]]) -> str:
        """Build ground prompt"""
        
        # Build evidence list for each claim
        evidence_sections = []
        
        for claim_idx in range(len(claims)):
            evidence_list = evidence_by_claim.get(claim_idx, [])
            
            # Truncate claim text for display
            claim_preview = claims[claim_idx]
            # if len(claim_preview) > 100:
            #     claim_preview = claim_preview[:100] + "..."
            
            section = f"\n### Claim {claim_idx}: {claim_preview}"
            
            if not evidence_list:
                section += "\nNo relevant evidence found."
            else:
                section += f"\nFound {len(evidence_list)} evidence:"
                for i, ev in enumerate(evidence_list):
                    evidence_text = ev.get('evidence', 'No evidence')
                    # # Truncate evidence text for display
                    # if evidence_text and len(evidence_text) > 200:
                    #     evidence_text = evidence_text[:200] + "..."
                    
                    section += f"\n{i+1}. From {ev.get('report_type')} {ev.get('report_id')}:"
                    section += f"\n   Evidence: {evidence_text}"
            
            evidence_sections.append(section)
        
        evidence_text = "\n".join(evidence_sections)
        
        return f"""
## Evidence Scoring Task
Part: {part_name}

## Evidence to be Scored:
{evidence_text}

## Scoring Requirements:
1. For each claim, evaluate the support strength of all evidence (0-10 points)
2. If no evidence exists, score is 0, rationale should state "No relevant evidence"
3. If multiple evidence exists, select the most relevant one for scoring (take the highest score)
4. Must provide detailed scoring rationale, explaining why it supports and why this score was given
5. Preserve the complete original evidence text and source

## Output Format:
{{
  "grounding_results": [
    {{
      "claim_index": 0,
      "evidence": "Original evidence text",
      "score": 8,
      "rationale": "Scoring rationale...",
      "source": "WEB_REPORT:web_01"
    }},
    // Other claims...
  ]
}}

Note: Output only one result per claim (best evidence). Output only JSON."""
    
    def _ground_schema(self) -> Dict[str, Any]:
        """Ground phase output schema"""
        return {
            "type": "object",
            "properties": {
                "grounding_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_index": {
                                "type": "integer",
                                "minimum": 0
                            },
                            "evidence": {
                                "type": ["string", "null"]
                            },
                            "score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 10
                            },
                            "rationale": {
                                "type": "string"
                            },
                            "source": {
                                "type": ["string", "null"]
                            }
                        },
                        "required": ["claim_index", "evidence", "score", "rationale", "source"]
                    }
                }
            },
            "required": ["grounding_results"]
        }
    
    # ----------------- Utility Methods -----------------
    
    def _build_paper_content(self, paper: Dict) -> str:
        """Build paper content"""
        sections = []
        
        field_mapping = {
            "basic_idea": "BASIC IDEA",
            "motivation": "MOTIVATION", 
            "research_question": "RESEARCH QUESTION",
            "method": "METHOD",
            "experimental_setting": "EXPERIMENTAL SETTING",
            "expected_results": "EXPECTED RESULTS"
        }
        
        for field, label in field_mapping.items():
            content = paper.get(field, [])
            if content:
                sections.append(f"{label}:")
                sections.extend([f"- {item}" for item in content])
                sections.append("")
        
        return "\n".join(sections)
    
    def _create_empty_evidence(self, claims: List[str], report_type: str,
                             report_id: str, part_name: str) -> List[Dict]:
        """Create empty evidence list"""
        evidences = []
        for claim_idx in range(len(claims)):
            evidences.append({
                "claim_index": claim_idx,
                "evidence": None,
                "source": None,
                "report_type": report_type,
                "report_id": report_id,
                "part": part_name
            })
        return evidences
    
    def _create_empty_results(self, claims: List[str]) -> List[Dict]:
        """Create empty results list"""
        results = []
        for claim_idx, claim in enumerate(claims):
            results.append({
                "claim_index": claim_idx,
                "evidence": None,
                "score": 0,
                "rationale": "No evidence found in any report",
                "report_type": None,
                "report_id": None,
                "source": None
            })
        return results
    
    def _ensure_complete_results(self, claims: List[str], grounded_results: List[Dict],
                               all_evidence: List[Dict]) -> List[Dict]:
        """Ensure each claim has a result"""
        # Create result index
        results_by_index = {}
        for result in grounded_results:
            idx = result.get("claim_index")
            if idx is not None:
                results_by_index[idx] = result
        
        # Ensure each claim has a result
        complete_results = []
        for claim_idx, claim in enumerate(claims):
            if claim_idx in results_by_index:
                result = results_by_index[claim_idx]
                # Add missing metadata
                if "report_type" not in result or "report_id" not in result:
                    # Find from original evidence
                    for evidence in all_evidence:
                        if evidence.get("claim_index") == claim_idx and evidence.get("evidence"):
                            result.setdefault("report_type", evidence.get("report_type"))
                            result.setdefault("report_id", evidence.get("report_id"))
                            result.setdefault("part", evidence.get("part"))
                            break
                complete_results.append(result)
            else:
                # Create default result
                complete_results.append({
                    "claim_index": claim_idx,
                    "evidence": None,
                    "score": 0,
                    "rationale": "No evidence was grounded for this claim",
                    "report_type": None,
                    "report_id": None,
                    "source": None
                })
        
        return complete_results

# # ------------------ Test Code ------------------
# if __name__ == "__main__":
#     import asyncio
#     from ..models.r1_model import R1Model
    
#     model = R1Model(
#         model_name="DeepSeek-R1",
#         api_key="sk-8UBXNvFTwUajsLaXzWW2Ge9WcuZ1ZGPSk0yr3tVYpZCJDy6t",
#         base_url="https://www.dmxapi.cn/v1"
#     )
    
#     config = {
#         "extract_temperature": 0.3,
#         "ground_temperature": 0.3
#     }
    
#     agent = GroundingAgent(model=model, config=config)
    
#     async def demo():
#         claims = {"motivation": [
#             "Standard chain-of-thought reasoning improves model performance on complex tasks.",
#             "Explanations produced by standard CoT are often not causally connected to the model's actual decision process."
#         ]}
        
#         # Test data
#         try:
#             with open('/home/wys/wys/InternAgent-main/report_paper.json', 'r', encoding='utf-8') as f:
#                 paper_report = json.load(f)
#             with open('/home/wys/wys/InternAgent-main/internagent/mas/agents/test_webreport.json', 'r', encoding='utf-8') as f:
#                 web_report = json.load(f)
#         except FileNotFoundError:
#             print("Using sample data")
#             paper_report = [{
#                 "paper_metadata": {"title": "Test Paper"},
#                 "basic_idea": ["Test concept"],
#                 "motivation": ["Test motivation"]
#             }]
#             web_report = [{
#                 "report_id": "test_web",
#                 "content": {"report_content": "Test web content"}
#             }]
    
#         reports = {
#             "web_reports": web_report,
#             "paper_reports": paper_report,
#             "code_reports": []
#         }

#         context = {
#             "claims": claims,
#             "reports": reports
#         }
        
#         print("Starting two-phase GroundingAgent execution...")
#         try:
#             out = await agent.execute(context, params={})
#             print("\nFinal Results:")
#             print(json.dumps(out, indent=2, ensure_ascii=False))
#         except Exception as e:
#             print(f"Execution failed: {e}")
#             import traceback
#             traceback.print_exc()
    
#     asyncio.run(demo())