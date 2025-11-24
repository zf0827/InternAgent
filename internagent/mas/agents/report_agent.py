import random
import json
from typing import Dict, Any, List

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.searchers.models import SearchResults


class ReportAgent(BaseAgent):
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "ReportAgent"
        self.temperature = config.get("temperature")

    def _pick_descriptions(self, items: List[Dict[str, Any]], k: int) -> List[str]:
        descs = []
        for i in items:
            if not isinstance(i, dict):
                continue
            text = i.get("page_raw_text","")
            summary = i.get("page_structured_summary","")
            evidence = i.get("page_structured_evidence","")
            rational = i.get("page_structured_rational","")
            description = i.get("description","")
            
            # Check if this is a GitHub repository with file_tree
            platform = i.get("platform", "")
            file_tree = i.get("file_tree", "")
            
            block = "Source: " + i.get("title","") + "\n"
            if description!="":
                block += f"description: \n{description}\n"
            if summary!="":
                block += f"summary: \n{summary}\n"
            if evidence!="":
                block += f"evidence: \n{evidence}\n"
            if rational!="":
                block += f"rational: \n{rational}\n"
            
            # Add file tree information for GitHub repositories
            if platform == "github" and file_tree:
                block += f"file_tree: \n{file_tree}\n"
            
            descs.append(block)
        if not descs:
            return []
        if len(descs) <= k:
            return descs
        return random.sample(descs, k)
    def _build_web_prompt(self, idea_text: str, descriptions: List[str]) -> str:
        joined = "\n".join([f"- {d}" for d in descriptions])
        return (
            "You are generating an evidence-based analytical report about a research idea.\n\n"
            "=== Input Specification ===\n"
            "Idea Text:\n"
            f"{idea_text}\n\n"
            "Web Resource Descriptions:\n"
            f"{joined}\n\n"
            "These descriptions summarize webpages that may contain viewpoints, discussions, claims, opinions, "
            "supporting/contradicting evidence, example use cases, concerns, limitations, or debates relevant to the idea.\n\n"
            "=== Task Requirements ===\n"
            "Your goal is to synthesize ALL content from the descriptions that is relevant to evaluating the idea—"
            "including evidence that supports, challenges, contradicts, or complicates the idea.\n\n"
            "You MUST:\n"
            "1. Extract and group all viewpoints relevant to the idea (both positive and negative).\n"
            "2. Highlight consensus, recurring themes, or widely accepted arguments.\n"
            "3. Identify controversies, conflicts, disagreements, risks, or negative evidence.\n"
            "4. Only use information present in the descriptions—do not invent content.\n"
            "5. Focus strictly on relevance to the idea: arguments, implications, feasibility insights, criticisms, etc.\n"
            "6. Remain objective and avoid bias toward supporting the idea.\n\n"
            "=== Output Format ===\n"
            "Produce a structured report with the following sections:\n"
            "1. Idea Overview — concise restatement of the idea.\n"
            "2. Extracted Viewpoints — grouped themes of claims, arguments, evidence (supportive and contradictory).\n"
            "3. Consensus Patterns — viewpoints multiple sources agree on.\n"
            "4. Conflicts & Contradictions — challenges, risks, negative findings, disagreements.\n"
            "5. Implications for the Idea — what the above suggests for feasibility, novelty, risks.\n"
            "6. Notable Links or Entities (if explicitly mentioned in descriptions).\n\n"
            "Ensure clarity, completeness, and neutrality."
        )


    def _build_code_prompt(self, idea_text: str, descriptions: List[str]) -> str:
        joined = "\n".join([f"- {d}" for d in descriptions])
        return (
            "You are generating a technical tool-integration report for a research idea.\n\n"
            "=== Input Specification ===\n"
            "Idea Text:\n"
            f"{idea_text}\n\n"
            "Code Resource Descriptions (e.g., repos, libraries, toolkits):\n"
            f"{joined}\n\n"
            "These descriptions summarize repositories and codebases that may include models, frameworks, pipelines, "
            "utilities, datasets, training code, evaluation scripts, or configurations relevant to implementing the idea.\n\n"
            "=== Task Requirements ===\n"
            "Your goal is to extract ALL implementation-relevant information from the descriptions, including:\n"
            "- Useful components / modules / tools\n"
            "- Repository structure and organization (file tree analysis)\n"
            "- Typical architectures and pipelines\n"
            "- Integration opportunities\n"
            "- Known issues, constraints, or incompatibilities (negative evidence)\n"
            "- Dependencies and environment considerations\n"
            "- Practical challenges in adopting or adapting these tools\n\n"
            "For GitHub repositories, PAY SPECIAL ATTENTION to the file_tree field, which reveals the repository structure. "
            "Analyze the file tree to determine:\n"
            "- Whether the repository contains relevant code modules for your idea\n"
            "- The overall architecture and organization of the codebase\n"
            "- Key directories and files that could be leveraged or extended\n"
            "- Missing components that would need to be implemented\n"
            "- How the existing code structure aligns with your idea's requirements\n\n"
            "You MUST:\n"
            "1. Identify capabilities and components relevant to implementing the idea.\n"
            "2. Analyze repository file structures to assess code reusability and architectural fit.\n"
            "3. Extract both strengths (useful tools) and limitations/challenges.\n"
            "4. Outline typical workflows/patterns that can inform implementation.\n"
            "5. Analyze how these resources could be integrated into the idea, including potential obstacles.\n"
            "6. Only use information present in the descriptions.\n"
            "7. Maintain technical precision and objectivity.\n\n"
            "=== Output Format ===\n"
            "Produce a structured technical report with the following sections:\n"
            "1. Idea Overview — concise restatement of the idea.\n"
            "2. Useful Components — tools, modules, models, functions applicable to the idea.\n"
            "3. Repository Structure Analysis — file tree examination and architectural assessment.\n"
            "4. Typical Pipelines — common workflows from the repos that can support implementation.\n"
            "5. Integration Strategy & Considerations — how to combine these tools with the idea; include potential hurdles.\n"
            "6. Limitations & Risks — constraints, missing components, incompatibilities, maintenance risks.\n\n"
            "Ensure the output is technically grounded, balanced, and directly tied to the idea. "
            "Focus on practical code-level analysis rather than just README-level descriptions."
        )


    def _build_paper_prompt(self, idea_text: str, descriptions: List[str]) -> str:
        joined = "\n".join([f"- {d}" for d in descriptions])
        return (
            "You are generating a literature-based analytic report for a research idea.\n\n"
            "=== Input Specification ===\n"
            "Idea Text:\n"
            f"{idea_text}\n\n"
            "Paper & Scholar Descriptions:\n"
            f"{joined}\n\n"
            "These descriptions summarize academic papers, related work, baseline methods, theoretical analyses, "
            "evaluation results, and scholarly commentary that may relate to the idea.\n\n"
            "=== Task Requirements ===\n"
            "Your goal is to extract ALL relevant academic evidence—both supportive and contradictory—including:\n"
            "- Prior work with similar ideas (novelty threats)\n"
            "- Baseline methods and limitations\n"
            "- Evaluation methodologies and results\n"
            "- Theoretical or empirical support for components of the idea\n"
            "- Critical perspectives or contradictory findings\n\n"
            "You MUST:\n"
            "1. Summarize the core findings, methods, and results from the described papers.\n"
            "2. Explicitly highlight connections to the idea (conceptual, methodological, or empirical).\n"
            "3. Extract both supporting evidence and challenging/contradicting evidence.\n"
            "4. Identify gaps, unresolved questions, or weaknesses in prior work.\n"
            "5. Avoid inventing content beyond what is in the descriptions.\n"
            "6. Provide an objective assessment of relevance and implications.\n\n"
            "=== Output Format ===\n"
            "Produce a structured literature report with the following sections:\n"
            "1. Idea Overview — concise restatement of the idea.\n"
            "2. Core Findings from the Papers — grouped by themes.\n"
            "3. Methods & Experimental Results — summarized and categorized.\n"
            "4. Alignment with the Idea — supportive evidence & methodological alignment.\n"
            "5. Contradictions & Novelty Challenges — competing methods, similar ideas, negative evidence.\n"
            "6. References Mentioned — only if explicitly named in descriptions.\n\n"
            "Ensure the report is scholarly, rigorous, objective, and comprehensive."
        )

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

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        sr = context.get("search_result") or context.get("search_results")
        
        if not isinstance(sr, dict):
            raise AgentExecutionError("search_result must be a dict from SearchResults.to_dict(). Please provide it in the context.")

        web_pages = sr.get("web_pages", [])
        github_repos = sr.get("github_repos", [])
        kaggle_results = sr.get("kaggle_results", [])

        web_desc = self._pick_descriptions(web_pages, 10)
        code_desc = self._pick_descriptions(github_repos + kaggle_results, 10)
        paper_desc = self._pick_descriptions(sr.get("papers", []) + sr.get("scholar_results", []), 10)

        idea_text = self._get_idea_text(sr)

        # web_prompt = self._build_web_prompt_contradictory(idea_text, web_desc) if web_desc else "No web descriptions available. Produce a brief empty report."
        # code_prompt = self._build_code_prompt_contradictory(idea_text, code_desc) if code_desc else "No code descriptions available. Produce a brief empty report."
        # paper_prompt = self._build_paper_prompt_contradictory(idea_text, paper_desc) if paper_desc else "No paper descriptions available. Produce a brief empty report."
        web_prompt = self._build_web_prompt(idea_text, web_desc) if web_desc else "No web descriptions available. Produce a brief empty report."
        code_prompt = self._build_code_prompt(idea_text, code_desc) if code_desc else "No code descriptions available. Produce a brief empty report."
        paper_prompt = self._build_paper_prompt(idea_text, paper_desc) if paper_desc else "No paper descriptions available. Produce a brief empty report."
        print("=" * 40 + " Web Prompt " + "=" * 40)
        print(web_prompt)
        print("\n" + "=" * 40 + " Code Prompt " + "=" * 40)
        print(code_prompt)
        print("\n" + "=" * 40 + " Paper Prompt " + "=" * 40)
        print(paper_prompt)
        try:
            web_report = await self._call_model(
                prompt=web_prompt,
                system_prompt=self.system_prompt,
                temperature=params.get("temperature", self.temperature),
            )
            web_report = ""
            code_report = await self._call_model(
                prompt=code_prompt,
                system_prompt=self.system_prompt,
                temperature=params.get("temperature", self.temperature),
            )
            paper_report = await self._call_model(
                prompt=paper_prompt,
                system_prompt=self.system_prompt,
                temperature=params.get("temperature", self.temperature),
            )
            paper_report = ""
        except Exception as e:
            raise AgentExecutionError(str(e))
        print("=" * 40 + " Web Report " + "=" * 40)
        print(web_report)
        print("\n" + "=" * 40 + " Code Report " + "=" * 40)
        print(code_report)
        print("\n" + "=" * 40 + " Paper Report " + "=" * 40)
        print(paper_report)
        
        return {
            "web_report": web_report,
            "code_report": code_report,
            "paper_report": paper_report,
            "params": params,
        }
    # def _build_web_prompt_contradictory(self, idea_text: str, descriptions: List[str]) -> str:
    #     joined = "\n".join([f"- {d}" for d in descriptions])
    #     return (
    #         "You are generating a strictly critical and contradiction-focused report on a research idea. "
    #         "Your sole purpose is to expose every piece of negative, challenging, or contradictory evidence from the provided sources.\n\n"
    #         "=== Input Specification ===\n"
    #         "Idea Text:\n"
    #         f"{idea_text}\n\n"
    #         "Web Resource Descriptions (discussions, claims, opinions, critiques):\n"
    #         f"{joined}\n\n"
    #         "=== Task Requirements ===\n"
    #         "You are explicitly instructed to be maximally critical and skeptical. "
    #         "You MUST only extract and discuss information that weakens, contradicts, complicates, or casts doubt on the idea. "
    #         "Ignore or downplay any positive statements.\n\n"
    #         "Focus on:\n"
    #         "- Direct contradictions to the idea's assumptions or claims\n"
    #         "- Widely expressed doubts, criticisms, or rejection in communities\n"
    #         "- Practical failures, negative experiences, or abandoned attempts reported\n"
    #         "- Ethical, legal, privacy, or societal risks highlighted\n"
    #         "- Performance limitations, scalability issues, or real-world deployment failures mentioned\n"
    #         "- Strong counter-arguments or alternative viewpoints that undermine the idea\n\n"
    #         "You are forbidden from mentioning any supporting evidence unless it is immediately followed by a stronger rebuttal or limitation.\n\n"
    #         "=== Output Format ===\n"
    #         "Produce a harshly critical structured report:\n"
    #         "1. Idea Restatement (for context only)\n"
    #         "2. Core Contradictions & Refutations — direct logical or factual contradictions to the idea\n"
    #         "3. Major Criticisms & Community Skepticism — recurring negative opinions and debates\n"
    #         "4. Practical Failures & Negative Experiences — reported real-world issues or failed attempts\n"
    #         "5. Risks & Undesirable Consequences — ethical, legal, societal, economic, or technical risks\n"
    #         "6. Fundamental Feasibility Challenges — reasons why the idea is likely impractical or doomed\n"
    #         "7. Notable Counter-Examples or Discredited Similar Ideas (if mentioned)\n\n"
    #         "Write in an objective but relentlessly negative tone. Every section must contain concrete evidence from the descriptions."
    #     )


    # def _build_code_prompt_contradictory(self, idea_text: str, descriptions: List[str]) -> str:
    #     joined = "\n".join([f"- {d}" for d in descriptions])
    #     return (
    #         "You are generating a purely negative technical feasibility report that highlights why existing codebases make the research idea difficult or impossible to implement effectively.\n\n"
    #         "=== Input Specification ===\n"
    #         "Idea Text:\n"
    #         f"{idea_text}\n\n"
    #         "Code Resource Descriptions (repos, libraries, toolkits):\n"
    #         f"{joined}\n\n"
    #         "=== Task Requirements ===\n"
    #         "Your only goal is to mine every limitation, incompatibility, missing feature, maintenance nightmare, and practical obstacle present in these codebases. "
    #         "Act as a harsh code reviewer whose job is to prove the idea is not realistically implementable with current tools.\n\n"
    #         "You MUST focus exclusively on negative aspects:\n"
    #         "- Missing critical components or functionality\n"
    #         "- Outdated, abandoned, or unmaintained repositories\n"
    #         "- Severe version conflicts or dependency hell\n"
    #         "- Known bugs, performance bottlenecks, or scalability failures\n"
    #         "- Inflexible architectures that cannot be reasonably extended\n"
    #         "- Poor documentation, lack of examples, or steep adoption barriers\n"
    #         "- Failed attempts by others to achieve similar goals using these tools\n"
    #         "- License incompatibilities or restrictive licensing\n\n"
    #         "Do not mention any strengths unless immediately followed by a fatal flaw.\n\n"
    #         "=== Output Format ===\n"
    #         "Produce a devastating technical critique:\n"
    #         "1. Idea Restatement (for contrast)\n"
    #         "2. Critical Missing Components — functionality the idea requires but no codebase provides\n"
    #         "3. Architectural Mismatches & Inflexibility — why existing designs cannot accommodate the idea\n"
    #         "4. Dependency & Environment Nightmares — conflicts, versions, setup difficulties\n"
    #         "5. Performance & Scalability Failures — documented bottlenecks or crashes\n"
    #         "6. Maintenance & Sustainability Red Flags — abandoned repos, no updates, dead communities\n"
    #         "7. Historical Implementation Failures — evidence of others trying and failing with these tools\n"
    #         "8. Overall Implementation Verdict — why the idea is practically unbuildable with current code ecosystem\n\n"
    #         "Be ruthless, precise, and evidence-based."
    #     )


    # def _build_paper_prompt_contradictory(self, idea_text: str, descriptions: List[str]) -> str:
    #     joined = "\n".join([f"- {d}" for d in descriptions])
    #     return (
    #         "You are generating a strictly adversarial literature review whose sole purpose is to demonstrate that existing academic work either already solves, disproves, or severely undermines the research idea.\n\n"
    #         "=== Input Specification ===\n"
    #         "Idea Text:\n"
    #         f"{idea_text}\n\n"
    #         "Paper & Scholar Descriptions:\n"
    #         f"{joined}\n\n"
    #         "=== Task Requirements ===\n"
    #         "You must adopt a completely hostile stance toward the idea's novelty and viability. "
    #         "Only extract and emphasize evidence that damages the idea. Ignore or minimize any gaps that could favor the idea.\n\n"
    #         "Focus exclusively on:\n"
    #         "- Prior work that already implements very similar or superior ideas\n"
    #         "- Theoretical proofs or strong evidence against core assumptions\n"
    #         "- Empirical results showing the idea's approach fails or underperforms\n"
    #         "- Identified fundamental limitations of methods the idea relies on\n"
    #         "- Harsh criticisms or refutations in follow-up work or citations\n"
    #         "- Negative results, ablation studies, or failure analyses relevant to the idea\n\n"
    #         "You are not allowed to claim any novelty or contribution unless immediately refuted.\n\n"
    #         "=== Output Format ===\n"
    #         "Produce a crushing literature-based takedown:\n"
    #         "1. Idea Restatement (to highlight what is being attacked)\n"
    #         "2. Prior Art That Solves or Supersedes the Idea\n"
    #         "3. Direct Refutations & Negative Theoretical Results\n"
    #         "4. Damning Empirical Evidence & Failed Experiments\n"
    #         "5. Fundamental Limitations of Underlying Methods\n"
    #         "6. Critical Commentary & Citation Impact (negative reception)\n"
    #         "7. Conclusion on Novelty and Viability — why the idea is redundant, flawed, or already disproven\n"
    #         "8. Key References That Undermine the Idea (explicitly listed if named)\n\n"
    #         "Maintain scholarly tone but be mercilessly critical and comprehensive in exposing weaknesses."
    #     )