
"""
GroundingAgent
- Works on ONE part at a time (e.g., "basic_idea" or "motivation").
- Accepts three reports for the part: web_report, code_report, paper_report.
- Stage 1: LLM extracts evidences for each claim from the labeled reports.
- Stage 2: LLM selects the best supporting and contradicting evidence and scores them.
- OUTPUT: grounding_results list where each item contains support_source and contradiction_source fields.

Dependencies:
- BaseAgent (provided in the repo)
- extract_text_from_pdf from tools.utils
- python-docx

This file is written to integrate with the project's BaseAgent._call_model method which expects
(prompt, system_prompt, schema, temperature) and returns a dict matching the schema.

"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

import docx

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.utils import extract_text_from_pdf

logger = logging.getLogger(__name__)


class GroundingAgent(BaseAgent):
    """
    Unified Grounding Agent (two-stage) but works on ONE PART at a time.

    Context input (one of these forms must be provided):
    {
      "claims": { "part_name": ["claim1", "claim2", ...] },  # EXACTLY one part
      "report_paths": ["/path/a.pdf", "/path/b.docx", ...],  # optional
      "report_texts": ["raw text 1", "raw text 2", ...]     # optional
    }

    Output:
    {
      "grounding_results": [
        {
          "claim": "...",
          "part": "motivation",
          "support_evidence": "...",
          "support_score": 8,
          "support_source": "web_report|code_report|paper_report",
          "contradiction": "...",
          "contradiction_score": -3,
          "contradiction_source": "web_report|code_report|paper_report"
        }
      ]
    }
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.model = model
        self.extract_temperature = float(config.get("extract_temperature", 0.0))
        self.ground_temperature = float(config.get("ground_temperature", 0.0))
        self.top_k = int(config.get("top_k_evidence", 20))

    # ---------------- main execution ----------------
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        # Validate claims input (must be dict with exactly one part)
        claims_input = context.get("claims")
        if not claims_input or not isinstance(claims_input, dict):
            raise AgentExecutionError("UnifiedGroundingAgent requires 'claims' as a dict with one part.")

        if len(claims_input) != 1:
            raise AgentExecutionError("UnifiedGroundingAgent expects exactly one part in 'claims' (e.g., {'basic_idea': [...]}).")

        part_name, claims = next(iter(claims_input.items()))
        if not isinstance(claims, list) or len(claims) == 0:
            raise AgentExecutionError("The part must contain a non-empty list of claims.")

        # Collect report texts from paths and text list
        report_texts: List[str] = []
        # If the input includes a dict with keys web_report, code_report, paper_report, accept that too
        reports_dict = {}
        if context.get("reports") and isinstance(context.get("reports"), dict):
            # direct dict with named reports
            reports_dict = context.get("reports")
        else:
            # gather from separate lists
            if context.get("report_paths"):
                paths = context.get("report_paths") or []
                for p in paths:
                    report_texts.append(self._read_report_file(p))
            if context.get("report_texts"):
                texts = context.get("report_texts") or []
                for t in texts:
                    if isinstance(t, str) and t.strip():
                        report_texts.append(t.strip())

        # Expected keys: web_report, code_report, paper_report
        if reports_dict:
            # copy strings or empty
            web = reports_dict.get("web_report", "") or ""
            code = reports_dict.get("code_report", "") or ""
            paper = reports_dict.get("paper_report", "") or ""
            report_map = {"web_report": web, "code_report": code, "paper_report": paper}
        else:
            # If user provided free texts or file reads, try to map them in order
            # Prefer explicit labeling by user; if three texts provided, assume order web/code/paper
            if len(report_texts) == 3:
                report_map = {
                    "web_report": report_texts[0] or "",
                    "code_report": report_texts[1] or "",
                    "paper_report": report_texts[2] or "",
                }
            else:
                raise ValueError(
                    "GroundingAgent requires exactly 3 reports: web_report, code_report, and paper_report. "
                    "But received an unknown or insufficient report structure."
                )


        # Stage 1: Evidence extraction for the provided claims (LLM1)
        extract_prompt = self._build_extract_prompt(part_name, claims, report_map)
        extract_schema = self._extract_schema()
        try:
            extract_response = await self._call_model(
                prompt=extract_prompt,
                system_prompt=self._system_extract_prompt(),
                schema=extract_schema,
                temperature=self.extract_temperature,
            )
            print(json.dumps(extract_response, indent=2, ensure_ascii=False))
        except Exception as e:
            raise AgentExecutionError(f"Evidence extraction model call failed: {e}")

        evidences_obj = extract_response.get("evidences")
        if not isinstance(evidences_obj, dict):
            raise AgentExecutionError("Evidence extraction returned invalid structure (no 'evidences' dict).")

        # Trim and normalize evidences
        for c in list(evidences_obj.keys()):
            if isinstance(evidences_obj[c], list):
                evidences_obj[c] = [str(x).strip() for x in evidences_obj[c] if x and str(x).strip()][: self.top_k]
            else:
                evidences_obj[c] = []

        # Stage 2: Scoring (LLM2)
        ground_prompt = self._build_ground_prompt(part_name, claims, evidences_obj, report_map)
        ground_schema = self._ground_schema()
        try:
            ground_response = await self._call_model(
                prompt=ground_prompt,
                system_prompt=self._system_ground_prompt(),
                schema=ground_schema,
                temperature=self.ground_temperature,
            )
        except Exception as e:
            raise AgentExecutionError(f"Grounding model call failed: {e}")

        results = ground_response.get("grounding_results")
        if not isinstance(results, list):
            raise AgentExecutionError("Grounding response missing 'grounding_results' list.")

        return {"grounding_results": results}

    # ---------------- helpers: file reading ----------------
    def _read_report_file(self, path: str) -> str:
        path = str(path)
        if not os.path.exists(path):
            raise AgentExecutionError(f"Report file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                text = extract_text_from_pdf(path)
                return text or ""
            elif ext == ".docx":
                doc = docx.Document(path)
                return "".join(p.text for p in doc.paragraphs)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            raise AgentExecutionError(f"Failed to read report file {path}: {e}")

    # ---------------- PROMPTS & SCHEMAS ----------------
    def _system_extract_prompt(self) -> str:
        return (
            "You are an expert extractor: find semantic supporting and contradicting evidences for "
            "each provided scientific claim from the provided reports. Use ONLY report content and label source blocks."
        )

    def _build_extract_prompt(self, part_name: str, claims: List[str], report_map: Dict[str, str]) -> str:
        # Build claims block
        lines = [f"PART: {part_name}"]
        for i, c in enumerate(claims, start=1):
            lines.append(f"{i}. {c}")
        claims_block = "".join(lines)

        # We present three labeled report blocks so the extractor can reference sources.
        report_block = "".join([f"[WEB_REPORT]{report_map.get('web_report','')}",f"[CODE_REPORT]{report_map.get('code_report','')}",f"[PAPER_REPORT]{report_map.get('paper_report','')}"])

        template = """
You will be given a set of claims (belonging to one part) and three labeled reports.
For each claim, extract ALL relevant evidences found in the reports that either SUPPORT or CONTRADICT the claim.
- Evidence should be detailed, complete and must strictly follow the report content (no hallucination).
- For each evidence string, indicate which labeled block it came from by appending the source report name in parentheses after the text (e.g., "evidence text here" (WEB_REPORT))
- If no evidence is found for a claim, return an empty list for that claim.
Return a JSON object with a single top-level key "evidences" mapping each claim to a list of evidence strings.
You MUST output valid JSON ONLY.
Do NOT wrap the output in triple backticks.
Do NOT use code blocks.
Do NOT include markup such as ```json, ``` or any other fences.
Do NOT add explanations, comments, or text outside the JSON.
Your entire response MUST be a single valid JSON object and nothing else.
for example:

{{
  "evidences": {{
    "CLAIM TEXT 1": ["evidence snippet A", "evidence snippet B"],
    "CLAIM TEXT 2": []
  }}
}}

--- CLAIMS ---
{claims_block}

--- REPORTS ---
{report_block}
"""
        return template.format(claims_block=claims_block, report_block=report_block)

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

    def _system_ground_prompt(self) -> str:
        return (
            "You are an expert scientific adjudicator. For each claim and candidate evidences, "
            "select the strongest supporting and contradicting evidence, assign numeric scores, and provide sources."
        )

    def _build_ground_prompt(self, part_name: str, claims: List[str], evidences_dict: Dict[str, List[str]], report_map: Dict[str, str]) -> str:
        # build block of claims + enumerated evidences
        blocks = []
        for idx, claim in enumerate(claims, start=1):
            evs = evidences_dict.get(claim) or []
            blocks.append(f"ITEM {idx} - CLAIM:{claim} EVIDENCES:")
            if not evs:
                blocks.append("  (no evidences found)")
            else:
                for j, ev in enumerate(evs, start=1):
                    blocks.append(f"  [{j}] {ev}")
            blocks.append("")  # blank line

        block_text = "".join(blocks)

        # Provide the three reports (labeled) so model can identify source by matching text
        report_block = "".join([f"[WEB_REPORT]{report_map.get('web_report','')}", f"[CODE_REPORT]{report_map.get('code_report','')}", f"[PAPER_REPORT]{report_map.get('paper_report','')}"])

        template = """
You will be given multiple items; each item contains a claim in the part {part} and a list of candidate evidences (extracted from the report(s)).

For each item:
1) For every candidate evidence, judge whether it SUPPORTS, CONTRADICTS, or is NEUTRAL with respect to the claim.
2) Choose the SINGLE best supporting evidence (if any). Provide its text with its source (specifically from which report) and a numeric support score using the following strict scale:
   - 10: Direct experimental validation, quantitative results, or explicit confirmation
   - 8-9: Strong theoretical support, detailed implementation description, or authoritative citation  
   - 6-7: Clear conceptual alignment, reasonable inference, or moderate evidence
   - 4-5: Weak or indirect support, tangential relevance
   - 1-3: Very weak connection, barely relevant
   - 0: No supporting evidence
   
3) Choose the SINGLE best contradicting evidence (if any). Provide its text with its source and a numeric contradiction score using the following strict scale:
   - -10: Direct experimental contradiction or explicit refutation
   - -8 to -9: Strong theoretical contradiction or conflicting evidence
   - -6 to -7: Clear conceptual misalignment or reasonable counter-argument
   - -4 to -5: Weak or indirect contradiction
   - -1 to -3: Very weak counter-evidence
   - 0: No contradicting evidence

4) **Score Distribution**: Ensure scores reflect meaningful distinctions. Reserve high scores (8-10) for the strongest evidence only.
5）Verify the source of evidences again.
Return STRICT JSON in the following shape (array keeps same order as the items above):
You MUST output valid JSON ONLY.
Do NOT wrap the output in triple backticks.
Do NOT use code blocks.
Do NOT include markup such as ```json, ``` or any other fences.
Do NOT add explanations, comments, or text outside the JSON.
Your entire response MUST be a single valid JSON object and nothing else.
for example:
{{
  "grounding_results": [
    {{
      "claim": "...",
      "part": "{part}",
      "support_evidence": "...",
      "support_score": 0,
      "support_source": "web_report | code_report | paper_report",
      "contradiction": "...",
      "contradiction_score": 0,
      "contradiction_source": "web_report | code_report | paper_report"
    }}
  ]
}}

--- ITEMS ---
{block_text}

"""
        # Use format with explicit part insertion and blocks
        return template.format(part=part_name, block_text=block_text, report_block=report_block)

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
                            "contradiction_source": {"type": ["string", "null"]},
                        },
                        "required": [
                            "claim",
                            "part",
                            "support_evidence",
                            "support_score",
                            "support_source",
                            "contradiction",
                            "contradiction_score",
                            "contradiction_source",
                        ]
                    }
                }
            },
            "required": ["grounding_results"]
        }