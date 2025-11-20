"""
Extraction Agent for InternAgent

Extracts key research components (motivation, research question, method,
experimental setting, expected results) from ideas or papers.

Supports:
- Plain text input (idea)
- Local file input (PDF / DOCX)
- DOI input (auto-download via utils.download_pdf_by_doi)
- Semantic Scholar paper_id input (fetches open-access PDF)
"""

import os
import json
import logging
import random
import asyncio
import yaml
from typing import Dict, Any
from .base_agent import BaseAgent, AgentExecutionError
from ..models.base_model import BaseModel
from ..models.openai_model import OpenAIModel
from ..models.r1_model import R1Model
# Reuse functions from utils.py
from ..tools.utils import (
    extract_text_from_pdf,
    download_pdf_by_doi,
    get_pdf_url,
    download_pdf
)
import docx

logger = logging.getLogger(__name__)


class ExtractionAgent(BaseAgent):
    """
    Agent that extracts structured scientific information from various inputs.

    Input options:
        - idea (string)
        - paper_path (PDF/DOCX file)
        - doi (string)
        - paper_id (string, Semantic Scholar ID)
    """

    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.temperature = 0.3

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the extraction pipeline.

        Args:
            context: Input dictionary — may include:
                - idea: str
                - paper_path: str
                - doi: str
                - paper_id: str
            params: Additional runtime parameters (unused)
        """
        input_text = ""
        temp_pdf_path = None
        source_type = None

        # Handle plain text idea
        if "idea" in context and context["idea"]:
            input_text = context["idea"].strip()
            source_type = "idea"
            logger.info("ExtractionAgent: processing idea text input.")

        # Handle local paper file
        elif "paper_path" in context and context["paper_path"]:
            paper_path = context["paper_path"]
            if not os.path.exists(paper_path):
                raise AgentExecutionError(f"File not found: {paper_path}")
            input_text = self._read_paper_file(paper_path)
            source_type = "paper"

        # Handle DOI
        elif "url" in context and context["url"]:
            url = context["url"].strip()
            logger.info(f"ExtractionAgent: downloading paper via URL {url}")
            temp_pdf_path = download_pdf(url, save_folder="downloaded_papers")
            if not temp_pdf_path or not os.path.exists(temp_pdf_path):
                raise AgentExecutionError(f"Failed to download PDF for URL {url}")
            input_text = extract_text_from_pdf(temp_pdf_path)
            source_type = url

        elif "doi" in context and context["doi"]:
            doi = context["doi"].strip()
            logger.info(f"ExtractionAgent: downloading paper via DOI {doi}")
            temp_pdf_path = download_pdf_by_doi(doi, download_dir="downloaded_papers")
            if not temp_pdf_path or not os.path.exists(temp_pdf_path):
                raise AgentExecutionError(f"Failed to download PDF for DOI {doi}")
            input_text = extract_text_from_pdf(temp_pdf_path)
            source_type = "doi"

        # Handle Semantic Scholar paper_id
        elif "paper_id" in context and context["paper_id"]:
            paper_id = context["paper_id"].strip()
            logger.info(f"ExtractionAgent: fetching open-access PDF via Semantic Scholar ID {paper_id}")
            pdf_url = get_pdf_url(paper_id)
            if not pdf_url:
                raise AgentExecutionError(f"No open-access PDF found for paper ID {paper_id}")
            temp_pdf_path = download_pdf(pdf_url, save_folder="downloaded_papers")
            if not temp_pdf_path or not os.path.exists(temp_pdf_path):
                raise AgentExecutionError("Failed to download open-access PDF.")
            input_text = extract_text_from_pdf(temp_pdf_path)
            source_type = "paper_id"

        else:
            raise AgentExecutionError(
                "No valid input provided — expected one of: 'idea', 'paper_path', 'doi', 'paper_id'."
            )

        if not input_text or len(input_text.strip()) < 50:
            raise AgentExecutionError("Extracted text is too short or empty.")

        # Build prompt and schema
        prompt = self._build_extraction_prompt(input_text)
        schema = self._build_output_schema()

        # Call model
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=self._build_system_prompt(),
                schema=schema,
                temperature=self.temperature,
            )
            # Cleanup temporary file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                    logger.info(f"Deleted temporary file: {temp_pdf_path}")
                except Exception as e:
                    logger.warning(f"Could not delete temp file {temp_pdf_path}: {e}")

            # Print JSON for logging
            print("\n=== Extraction Results ===")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            return response
            # return {
            #     "extracted_info": response,
            #     "metadata": {
            #         "source_type": source_type,
            #         "input_length": len(input_text),
            #     },
            # }

        except Exception as e:
            logger.error(f"ExtractionAgent failed: {str(e)}")
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            raise AgentExecutionError(f"Extraction failed: {str(e)}")



    def _read_paper_file(self, path: str) -> str:
        """Read text from local PDF or DOCX file."""
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                return extract_text_from_pdf(path)
            elif ext == ".docx":
                doc = docx.Document(path)
                return "\n".join(p.text for p in doc.paragraphs)
            else:
                raise AgentExecutionError(f"Unsupported file format: {ext}")
        except Exception as e:
            raise AgentExecutionError(f"Failed to read file {path}: {e}")

    def _build_output_schema(self) -> Dict[str, Any]:
    # Define expected JSON structure for atomic claims
        return {
            "type": "object",
            "properties": {
                "basic_idea": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "a summary of the core concept"
                },
                "motivation": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of atomic motivation claims - each as a separate, verifiable statement"
                },
                "research_question": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of atomic research questions - each as a separate, focused question"
                },
                "method": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of atomic method steps/features - each as a separate technical element"
                },
                "experimental_setting": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of atomic experimental details - each as a separate setup component"
                },
                "expected_results": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of atomic expected outcomes - each as a separate measurable result"
                }
            },
            "required": [
                "basic_idea",
                "motivation", 
                "research_question",
                "method",
                "experimental_setting",
                "expected_results"
            ]
        }


    def _build_extraction_prompt(self, text: str) -> str:
        """
        Construct prompt for detailed, atomic scientific claim extraction.

        The goal is to extract granular, verifiable research elements from a research idea or paper.
        """
        # truncated_text = text[:30000]  # safety limit
        truncated_text = text
        return f"""
    You are a scientific information extraction agent.

    Your task is to extract key research components as **ATOMIC CLAIMS** — each representing one independent, verifiable scientific statement.

    Please analyze the following input (which may be a research idea or full paper) and extract the following sections:

    1. **BASIC IDEA**
    - A concise summary of the central concept or innovation.
    - What is the core contribution or novel idea?

    2. **MOTIVATION**
    - Why is this research important or necessary?
    - Break into separate, testable claims (e.g., problem statements, limitations of prior work, gaps in current methods).
    - Each motivation should express *a single rationale or need*.

    3. **RESEARCH QUESTION**
    - What specific scientific questions or hypotheses are being addressed?
    - Break into distinct, focused research questions.
    - Each question should be precise, and independently answerable.

    4. **METHOD**
    - Describe the proposed approach, technique, or algorithm.
    - Break into *atomic method components*: model architecture, training strategy, optimization techniques, theoretical formulations, etc.
    - Include both procedural steps and core design ideas, each as a separate claim.

    5. **EXPERIMENTAL SETTING**
    - Summarize the experimental setup in atomic details like below:
        - **Datasets:** which datasets or benchmarks are used?
        - **Baselines:** what existing methods are compared against?
        - **Metrics:** what evaluation metrics are applied?
        - **Ablation Studies:** what components or hyperparameters are varied?
        - **Implementation Details:** environment, training config, hardware, or key hyperparameters.
    - Each item above should appear as a *separate atomic statement* or small list.

    6. **EXPECTED RESULTS**
    - Summarize the main findings, expected outcomes, or hypotheses about results.
    - Break into *independent claims* — each one measurable or testable (e.g., “Our method improves accuracy by X% over baseline”, “The ablation study shows the attention module improves recall”).
    - Avoid narrative commentary; report factual expectations or results.

    **RULES FOR ATOMIC CLAIMS**
    - Each claim must be a single, self-contained statement.
    - Avoid compound sentences with "and", "but", or "while".
    - Focus on clarity, factual precision, and scientific verifiability.
    - Use **arrays/lists** for every section (even if only one claim).

    **OUTPUT FORMAT**
    Strict JSON with arrays for all sections:
    {{
    "basic_idea": [ ... ],
    "motivation": [ ... ],
    "research_question": [ ... ],
    "method": [ ... ],
    "experimental_setting": [ ... ],
    "expected_results": [ ... ]
    }}
    --- Input Text ---
    {truncated_text}
    -------------------
    """

    def _build_system_prompt(self) -> str:
        """Define system-level model behavior."""
        return (
            "You are an expert in scientific document analysis. "
            "Your primary goals are:\n"
            "1. Identify and separate *atomic claims* — small, self-contained facts or hypotheses.\n"
            "2. Distinguish between motivation, research questions, methods, experimental settings, and expected results.\n"
            "3. When describing experiments, explicitly include datasets, baselines, metrics, and ablation studies if mentioned.\n"
            "4. When describing methods, emphasize step-by-step components, not narrative summaries.\n"
            "5. Produce JSON strictly matching the provided schema — no extra commentary, no markdown, no text outside JSON.\n\n"
            "Style and quality requirements:\n"
            "- Be objective and fact-based; never speculate.\n"
            "- Write clear, scientific English.\n"
            "- Each claim or list item must be short (one sentence) and independently meaningful.\n"
            "- Do not merge unrelated ideas; split them into separate entries.\n"
            "- Output must be strictly machine-readable JSON."
    )


    @classmethod
    def from_config(cls, config: Dict[str, Any], model: 'BaseModel') -> 'ExtractionAgent':
        """Factory constructor."""
        return cls(model, config)

