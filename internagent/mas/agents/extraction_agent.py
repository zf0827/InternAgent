"""
Extraction Agent for InternAgent

Extracts key research components (basic idea, motivation, research question, method,
experimental setting, expected results) from ideas or papers.

Supports:
- Plain text input (idea)
- Local file input (PDF / DOCX)
- DOI input (auto-download via utils.download_pdf_by_doi)
- Semantic Scholar paper_id input (fetches open-access PDF)
- PDF URL input (direct PDF download link)
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
        - pdf url
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
            temp_pdf_path = download_pdf(url)
            if not temp_pdf_path or not os.path.exists(temp_pdf_path):
                raise AgentExecutionError(f"Failed to download PDF for URL {url}")
            input_text = extract_text_from_pdf(temp_pdf_path)
            source_type = url

        elif "doi" in context and context["doi"]:
            doi = context["doi"].strip()
            logger.info(f"ExtractionAgent: downloading paper via DOI {doi}")
            temp_pdf_path = download_pdf_by_doi(doi)
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
            temp_pdf_path = download_pdf(pdf_url)
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

            # Print JSON for logging
            print("\n=== Extraction Results ===")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            return response

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
    - A concise summary of the central innovation, the main technique and the key expected effect.
    - Must NOT be vague, generic, or over-abstract.

    2. **MOTIVATION**
    - Why is this research important or necessary? Extract all reasons why the research is needed.
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
    - **MUST FOLLOW THIS EXACT STRUCTURE** - First list the core experimental components, then describe the experiments:

    **Step 1: Core Experimental Components**
        - "Datasets: [comma-separated list of all datasets used in the experiments]"
        - "Baselines: [comma-separated list of all baseline models/methods compared against, grouped by proprietary and open-source]"
        - "Metrics: [comma-separated list of all evaluation metrics used]"
        - "Hardware: [specific hardware configuration used for experiments]"

    **Step 2: Experiment Descriptions**
    - Split into **TWO** categories (Main Experiments and Analysis Experiments):

    **Main Experiments**
        - **Group similar experimental setups together**: If multiple experiments share the same evaluation protocol, methodology, and purpose (e.g., benchmarking on different datasets), describe them as a SINGLE comprehensive experimental setup.
        - Each merged main experiment item should describe:
        • The unified experimental methodology and evaluation protocol
        • All datasets/benchmarks evaluated (list them together)
        • All baseline methods compared against (group by model type)
        • Evaluation metrics applied across all evaluations
        • Shared implementation details (environment, hardware, hyperparameters)
        - **Example of merged description**: "Main Experiment: Benchmark evaluation on DABench, TableBench, and BIRD datasets comparing with proprietary models (GPT-4o, o4-mini, DeepSeek-R1, DeepSeek-V3.1, GPT-5) and open-source models (QwQ-32B, Qwen2.5-Coder-32B, Llama-3.3-70B, Qwen2.5-72B, TableLLM, Table-R1, OmniSQL, SQL-R1) using pass@1 and pass@3 metrics with GPT-4o-mini judge model on 8 A100 GPUs"
        - **Only create separate items** when experiments have fundamentally different purposes, methodologies, or evaluation frameworks.

   **Analysis Experiments** 
        - Each item describes one analysis experiment in a single sentence containing:
        • Type of analysis (ablation, sensitivity, diagnostic, etc.)
        • Variable being changed/tested
        • Measurement being taken
        • Experimental setup context
        - **Ensure distinct analysis dimensions**: Each analysis experiment should focus on a unique aspect (data scaling, training strategy, hyperparameters, evaluation methodology, etc.)
        - **Avoid overlapping variables**: If multiple experiments test similar factors (e.g., both data volume and training epochs affect training dynamics), group them logically or ensure clear distinction
        - **Cover all major analysis types mentioned in the paper like:
        • Data scaling and volume effects
        • Training strategy comparisons  
        • Hyperparameter sensitivity
        • Component contribution (filtering methods, reward design, etc.)
        • Evaluation methodology robustness
        • Model capability and scaling laws
        • Training stability and convergence

    **OUTPUT FORMAT FOR EXPERIMENTAL_SETTING**:
    The array MUST start with the core components, then the experiment descriptions:
    "Datasets: DABench, TableBench, BIRD, QRData",
    "Baselines: proprietary models (GPT-4o, o4-mini, DeepSeek-R1, DeepSeek-V3.1, GPT-5), open-source models (QwQ-32B, Qwen2.5-Coder-32B, Llama-3.3-70B, Qwen2.5-72B, TableLLM, Table-R1, OmniSQL, SQL-R1)",
    "Metrics: pass@1, pass@3, Rouge-L, exact match",
    "Hardware: 8 A100 80G GPUs",
    "Main Experiment: Benchmark evaluation on DABench, TableBench, and BIRD datasets comparing with proprietary models (GPT-4o, o4-mini, DeepSeek-R1, DeepSeek-V3.1, GPT-5) and open-source models (QwQ-32B, Qwen2.5-Coder-32B, Llama-3.3-70B, Qwen2.5-72B, TableLLM, Table-R1, OmniSQL, SQL-R1) using pass@1 and pass@3 metrics with GPT-4o-mini judge model on 8 A100 GPUs",
    "Analysis Experiment: Ablation study testing the effect of training data volume (2K, 4K, 8K, 12K) on model performance across all benchmarks",
    "Analysis Experiment: Comparison of different training strategies (SFT-only, zero-RL, SFT-then-RL, SFT-and-RL) on 7B model performance",
    "... other analysis experiments ..."
    ]

   6. **EXPECTED RESULTS**
    - Describe the **anticipated outcomes** and **hypothetical benefits** of this research idea.
    - Focus on qualitative expectations about potential performance and advantages.
    - Must NOT contain actual experimental results or numerical data from papers.
    - Must NOT reuse exact quantitative findings from existing research.
    - Break into *independent claims* — each describing one expected qualitative benefit.
    - Use qualitative comparative expressions like:
        • "Superior performance compared to state-of-the-art methods"
        • "Improved efficiency with lower computational cost"
        • "Enhanced stability during training process"
        • "Better generalization across different domains"
        • "Higher robustness to input variations"
        • "Reduced training time and resource requirements"
        • "Increased interpretability and transparency"
        • "Stronger scalability for larger datasets"
    - Split into:
        • Expected qualitative benefits for Main Experiments
        • Expected qualitative benefits for Analysis Experiments

    **RULES FOR ATOMIC CLAIMS**
    - Each claim must be a single, self-contained statement.
    - Focus on clarity, factual precision, and scientific verifiability.
    - Do NOT infer, extrapolate, or add any information not present in the source.
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
           """You are an expert scientific analysis agent. Your job is to transform a research idea or paper into structured, atomic, and evaluation-ready research components.

            GENERAL RULES:
            - Every output must be factual, concise, and mechanically verifiable.
            - Split all content into *atomic items* — no compound sentences.
            - Use formal academic English.
            - Output MUST strictly match the JSON schema required by the user (arrays only).

            SECTION-SPECIFIC RULES:

            1. BASIC IDEA
            - Must be ONE detailed sentence.
            - Must summarize the *core innovation*, capturing the key novelty, the main method, and the key expected impact.
            - Must NOT be vague, generic, or over-abstract.
            2. MOTIVATION
            - Extract distinct reasons why the research is needed.
            - Each motivation must be a single factual need/gap/limitation.

            3. RESEARCH QUESTION
            - Each item must be a clearly answerable scientific question.

            4. METHOD
            - Break down the method into atomic technical components.
            - Each item should describe *what* is done, not *why*.

            5. EXPERIMENTAL SETTING
           Must follow this exact structure:

            **Step 1: Core Experimental Components such as** 
            - "Datasets: [list all datasets]"
            - "Baselines: [list all baselines, grouped by proprietary/open-source]"
            - "Metrics: [list all evaluation metrics]"
            - "Hardware: [specific hardware configuration]"

            **Step 2: Experiment Descriptions**
            Split into two categories:

                **Main Experiments**
                - **GROUP SIMILAR SETUPS**: Merge experiments that share the same evaluation protocol, methodology, and purpose into single comprehensive descriptions.
                - Each merged item should describe: all datasets evaluated, all baseline comparisons, shared evaluation metrics, and common implementation details.
                - **Only create separate items** for experiments with fundamentally different purposes or methodologies.
                - Example: "Main Experiment: Benchmark evaluation on DatasetA, DatasetB, and DatasetC comparing with ModelX, ModelY using Accuracy and F1 metrics on 8 GPUs"

                **Analysis Experiments**
                - Ablation, sensitivity, diagnostic experiments and so on.
                - What variable is changed and what is measured.
                - One experiment per item.
                - Examples: 
                "Analysis Experiment: Ablation study testing the effect of training data volume on model performance"
                "Analysis Experiment: Comparison of different training strategies on model performance"

            **OUTPUT ORDER IS CRITICAL**: Always start with the core components, then main experiments, then analysis experiments.

            6. EXPECTED RESULTS
            - Must also be separated into two categories:
                **Main Experiments: expected outcomes**
                **Analysis Experiments: expected outcomes**
            - Must NOT contain numerical results.
            - Must NOT reuse exact results from papers.
            - Should describe expected outcomes like:
                - “Analysis Experiments:greater stability...”
                - “Main Experiments: state-of-the-art performance...”

            RESTRICTIONS:
            - No speculation unrelated to the text.
            - Each sentence must stand alone.
            - Do NOT infer, extrapolate, or add any information not present in the source.
            - Output MUST be pure JSON only, without any code block markers like ```json or ```, strictly matching the provided schema — no extra commentary, no markdown, no text outside JSON.
            """

    )


    @classmethod
    def from_config(cls, config: Dict[str, Any], model: 'BaseModel') -> 'ExtractionAgent':
        """Factory constructor."""
        return cls(model, config)

