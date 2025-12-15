#!/usr/bin/env python3
"""
Comparison Pipeline - 多 Idea 比较 Pipeline

支持三种模式：
- point-wise (n=1): 单个 idea，正常处理
- pair-wise (n=2): 两个 idea，比较并选出最好的
- group-wise (n=4): 四个 idea，比较并选出最好的

流程：
1. 对每个 PDF，运行 SingleIdeaPipeline
2. 收集所有 single_final_report 和 evaluation_results
3. 使用 dspy 生成比较报告
4. 输出最终报告（包含各 idea 的报告 + 比较分析 + 最佳选择）
"""

import asyncio
import json
import logging
import os
import sys
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import dspy

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from internagent.tester.test_agent_pipelinev2 import (
    SingleIdeaPipeline,
    load_environment_variables,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PDF 标识生成
# --------------------------------------------------------------------------- #
def extract_pdf_identifier(pdf_url: str) -> str:
    """
    从 PDF URL 提取唯一标识符。

    Args:
        pdf_url: PDF URL

    Returns:
        唯一标识符字符串
    """
    # OpenReview: https://openreview.net/pdf?id=wLR9d5ZFpY -> wLR9d5ZFpY
    if "openreview.net" in pdf_url:
        match = re.search(r"id=([A-Za-z0-9_-]+)", pdf_url)
        if match:
            return match.group(1)

    # arXiv: https://arxiv.org/pdf/2301.12345.pdf -> 2301.12345
    if "arxiv.org" in pdf_url:
        match = re.search(r"arxiv\.org/pdf/(\d+\.\d+)", pdf_url)
        if match:
            return match.group(1).replace(".", "_")

    # 其他 URL: 使用 hash
    url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:16]
    return f"hash_{url_hash}"


def get_cache_path_for_pdf(pdf_url: str, cache_dir: Path) -> Path:
    """根据 PDF URL 生成 cache 文件路径。"""
    identifier = extract_pdf_identifier(pdf_url)
    return cache_dir / f"pipeline_result_{identifier}.json"


# --------------------------------------------------------------------------- #
# DSPy Comparison Module
# --------------------------------------------------------------------------- #
def create_comparison_signature(num_ideas: int) -> Type[dspy.Signature]:
    """
    动态创建比较 Signature 类。

    Args:
        num_ideas: Idea 数量

    Returns:
        dspy.Signature 类
    """
    docstring = f"""You are an experienced research reviewer and meta-evaluator. Your task is to compare {num_ideas} research ideas and select the best one based on comprehensive multi-dimensional analysis.

## Input Data:
"""
    input_fields = {}
    for i in range(1, num_ideas + 1):
        docstring += f"- idea_{i}_evaluation: Evaluation results for Idea {i}, containing scores (0-10) and detailed reasoning for five dimensions: clarity, novelty, validity, feasibility, and significance. Each dimension includes reviewer assessments from multiple personas.\n"
        docstring += f"- idea_{i}_report: Complete evaluation report for Idea {i}, including the research idea description, searched resources (papers, web pages, code repositories), evaluation results from multiple reviewers, final decision, and revision advice.\n"
        input_fields[f"idea_{i}_evaluation"] = dspy.InputField(
            desc=f"Evaluation results for Idea {i} (scores and reasoning for clarity, novelty, validity, feasibility, significance)"
        )
        input_fields[f"idea_{i}_report"] = dspy.InputField(
            desc=f"Complete evaluation report for Idea {i} (full assessment including idea, resources, evaluations, decision)"
        )

    docstring += f"""
## Task Requirements:

1. **Comprehensive Multi-Dimensional Comparison**: Analyze all {num_ideas} ideas across five key dimensions:
   - **Clarity**: How well-defined and understandable is the research idea?
   - **Novelty**: How original and innovative is the contribution?
   - **Validity**: How sound and well-grounded is the methodology and reasoning?
   - **Feasibility**: How realistic and achievable is the proposed approach?
   - **Significance**: How important and impactful would the results be?

2. **Detailed Analysis**: For each idea, identify:
   - Strengths and unique contributions
   - Weaknesses and potential limitations
   - Key differentiators compared to other ideas
   - Risk factors and implementation challenges

3. **Comparative Assessment**: 
   - Highlight relative advantages and disadvantages
   - Identify trade-offs between different ideas
   - Note any complementary aspects or synergies
   - Consider reviewer consensus and divergence

4. **Best Idea Selection**: 
   - Synthesize all evidence to select the best idea
   - Provide clear, well-justified reasoning
   - Acknowledge any limitations or uncertainties in the selection

## Output Format Requirements:

### comparison_analysis (Markdown format):
The comparison analysis report MUST follow this exact structure with the following sections:

#### 1. Executive Summary
- Brief overview of all ideas being compared
- High-level comparison highlighting key differences
- Summary of the comparative assessment

#### 2. Dimensional Comparison
For each of the five dimensions (clarity, novelty, validity, feasibility, significance):
- **Clarity Comparison**: Compare how clearly each idea is presented and understood
- **Novelty Comparison**: Compare the originality and innovation level of each idea
- **Validity Comparison**: Compare the soundness and rigor of methodologies
- **Feasibility Comparison**: Compare the practicality and achievability
- **Significance Comparison**: Compare the potential impact and importance

For each dimension, provide:
- Relative rankings or scores
- Key differences between ideas
- Notable strengths or weaknesses

#### 3. Individual Idea Analysis
For each idea (Idea 1, Idea 2, etc.):
- **Strengths**: List 3-5 key strengths
- **Weaknesses**: List 3-5 key weaknesses or concerns
- **Unique Contributions**: What makes this idea distinctive
- **Risk Assessment**: Potential challenges and mitigation strategies

#### 4. Comparative Insights
- **Trade-offs**: Key trade-offs between ideas (e.g., novelty vs. feasibility)
- **Complementarity**: How ideas might complement each other
- **Reviewer Consensus**: Areas where reviewers agree or disagree
- **Critical Differences**: Most significant factors differentiating the ideas

#### 5. Overall Assessment
- Synthesized view of all ideas
- Relative positioning of each idea
- Key factors influencing the comparison

### best_idea_index (integer):
- Must be an integer between 1 and {num_ideas} (inclusive)
- Represents the index of the best idea based on comprehensive analysis

### selection_reason (string):
- Clear, concise explanation (2-4 sentences) for why this idea was selected
- Should reference specific strengths, dimensions, or comparative advantages
- Should acknowledge any limitations or close alternatives
"""

    output_fields = {
        "comparison_analysis": dspy.OutputField(
            desc="Detailed comparison analysis report in Markdown format with required sections: Executive Summary, Dimensional Comparison, Individual Idea Analysis, Comparative Insights, Overall Assessment"
        ),
        "best_idea_index": dspy.OutputField(
            desc=f"Index of the best idea (integer between 1 and {num_ideas})"
        ),
        "selection_reason": dspy.OutputField(
            desc="Clear, concise explanation (2-4 sentences) for why this idea was selected, referencing specific strengths and comparative advantages"
        ),
    }

    # 创建 Signature 类
    class_name = f"ComparisonSignature{num_ideas}"
    signature_dict = {
        "__doc__": docstring,
        **input_fields,
        **output_fields,
    }

    signature_class = type(class_name, (dspy.Signature,), signature_dict)
    return signature_class


class ComparisonModule(dspy.Module):
    """使用 dspy 进行多 idea 比较的模块。"""

    def __init__(self, num_ideas: int, model_config: Optional[Dict[str, Any]] = None):
        """
        初始化比较模块。

        Args:
            num_ideas: Idea 数量
            model_config: 模型配置（可选）
        """
        super().__init__()
        self.num_ideas = num_ideas

        # 配置 dspy LM
        if model_config is None:
            ds_api_key = os.getenv("DS_API_KEY")
            if ds_api_key:
                model_config = {
                    "model": "openai/DeepSeek-V3.2",
                    "api_key": ds_api_key,
                    "api_base": os.getenv("DS_API_BASE_URL"),
                }
            else:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    raise ValueError("No API keys found. Please set DS_API_KEY or OPENAI_API_KEY in environment variables.")
                model_config = {
                    "model": "gpt-4o-mini",
                    "api_key": openai_api_key,
                    "api_base": os.getenv("OPENAI_API_BASE_URL"),
                }

        try:
            self.lm = dspy.LM(
                model=model_config.get("model", "gpt-4o-mini"),
                api_key=model_config["api_key"],
                api_base=model_config.get("api_base"),
            )
            logger.info(f"Initialized ComparisonModule with model: {model_config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy: {e}")
            raise

        # 创建 Signature 和 Predict 模块
        signature_class = create_comparison_signature(num_ideas)
        self.compare = dspy.ChainOfThought(signature_class)

    def forward(self, **kwargs) -> Dict[str, Any]:
        """
        执行比较。

        Args:
            **kwargs: 包含 idea_1_evaluation, idea_1_report, idea_2_evaluation, ... 等字段

        Returns:
            包含 comparison_analysis, best_idea_index, selection_reason 的字典
        """
        with dspy.settings.context(lm=self.lm):
            result = self.compare(**kwargs)

        return {
            "comparison_analysis": getattr(result, "comparison_analysis", ""),
            "best_idea_index": getattr(result, "best_idea_index", 1),
            "selection_reason": getattr(result, "selection_reason", ""),
        }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def format_evaluation_summary(evaluation_result: Dict[str, Any]) -> str:
    """格式化评估结果为字符串摘要。"""
    evaluation_results = evaluation_result.get("evaluation_results", [])
    if not evaluation_results:
        return "No evaluation results available."

    summaries = []
    for idx, item in enumerate(evaluation_results, 1):
        evaluation = item.get("evaluation", {})
        persona = item.get("persona", {})
        persona_tag = persona.get("background") or persona.get("goal") or f"Reviewer {idx}"

        parts = []
        for key in ["clarity", "novelty", "validity", "feasibility", "significance"]:
            data = evaluation.get(key, {}) or {}
            score = data.get("score", "N/A")
            reason = data.get("reason", "")
            parts.append(f"{key.title()}: {score}/10 – {reason[:100]}")

        overall = evaluation.get("overall", {})
        overall_txt = overall.get("summary", "")
        parts.append(f"Overall: {overall_txt[:200]}")

        summaries.append(f"Reviewer {idx} ({persona_tag}):\n" + "\n".join(parts))

    return "\n\n".join(summaries)


async def run_comparison_pipeline(
    pdf_urls: List[str],
    cache_dir: Path,
    persona_path: Path,
    research_params: Dict[str, Any],
    future_cutoff: str = "2025-05-31",
    num_personas: int = 3,
) -> Dict[str, Any]:
    """
    运行比较 pipeline。

    Args:
        pdf_urls: PDF URL 列表（1, 2, 或 4 个）
        cache_dir: Cache 目录
        persona_path: Persona 文件路径
        research_params: ResearchAgent 参数
        future_cutoff: 未来论文时间截止点
        num_personas: Persona 数量

    Returns:
        包含所有结果的字典
    """
    num_ideas = len(pdf_urls)
    if num_ideas not in [1, 2, 4]:
        raise ValueError(f"Unsupported number of ideas: {num_ideas}. Must be 1, 2, or 4.")

    mode = "point-wise" if num_ideas == 1 else ("pair-wise" if num_ideas == 2 else "group-wise")
    logger.info(f"Running {mode} comparison with {num_ideas} ideas")

    # 1. 为每个 PDF 创建并运行 SingleIdeaPipeline
    results = []
    for idx, pdf_url in enumerate(pdf_urls, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing Idea {idx}/{num_ideas}: {pdf_url}")
        logger.info(f"{'='*80}")

        cache_path = get_cache_path_for_pdf(pdf_url, cache_dir)

        pipeline = SingleIdeaPipeline(
            pdf_url=pdf_url,
            cache_path=cache_path,
            persona_path=persona_path,
            research_params=research_params,
            future_cutoff=future_cutoff,
            num_personas=num_personas,
        )

        try:
            result = await pipeline.run()
            result["pdf_url"] = pdf_url
            result["idea_index"] = idx
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process idea {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        raise RuntimeError("No ideas were successfully processed.")

    # 2. 如果只有一个 idea，直接返回结果
    if num_ideas == 1:
        return {
            "mode": mode,
            "ideas": results,
            "final_report": results[0]["final_report"],
            "best_idea_index": 1,
        }

    # 3. 使用 dspy 生成比较报告
    logger.info(f"\n{'='*80}")
    logger.info("Generating Comparison Report")
    logger.info(f"{'='*80}")

    comparison_module = ComparisonModule(num_ideas=num_ideas)

    # 准备输入数据
    comparison_inputs = {}
    for idx, result in enumerate(results, 1):
        evaluation_summary = format_evaluation_summary(result["evaluation_result"])
        comparison_inputs[f"idea_{idx}_evaluation"] = evaluation_summary
        comparison_inputs[f"idea_{idx}_report"] = result["final_report"]

    # 生成比较报告
    comparison_result = comparison_module.forward(**comparison_inputs)

    # 4. 组装最终报告
    final_report_parts = ["# Idea Comparison Report\n"]

    for idx, result in enumerate(results, 1):
        pdf_identifier = extract_pdf_identifier(result["pdf_url"])
        final_report_parts.append(f"## Idea {idx}: {pdf_identifier}\n")
        final_report_parts.append(result["final_report"])
        final_report_parts.append("\n")

    final_report_parts.append("## Comparison Analysis\n")
    final_report_parts.append(comparison_result["comparison_analysis"])
    final_report_parts.append("\n")

    final_report_parts.append("## Best Idea Selection\n")
    final_report_parts.append(f"Selected: Idea {comparison_result['best_idea_index']}\n")
    final_report_parts.append(f"Reason: {comparison_result['selection_reason']}")

    final_report = "\n".join(final_report_parts)

    return {
        "mode": mode,
        "ideas": results,
        "comparison_result": comparison_result,
        "final_report": final_report,
        "best_idea_index": comparison_result["best_idea_index"],
    }


async def main() -> None:
    """主函数。"""
    print("\n" + "=" * 80)
    print("COMPARISON PIPELINE TEST")
    print("=" * 80)

    # 加载环境变量
    if not load_environment_variables():
        logger.warning("Failed to load environment variables, continuing anyway...")

    # 配置参数（硬编码在 main 中）
    # 示例：point-wise (n=1)
    pdf_urls = [
        "https://openreview.net/pdf?id=wLR9d5ZFpY",
    ]

    # 示例：pair-wise (n=2) - 取消注释以使用
    # pdf_urls = [
    #     "https://openreview.net/pdf?id=wLR9d5ZFpY",
    #     "https://openreview.net/pdf?id=ANOTHER_ID",
    # ]

    # 示例：group-wise (n=4) - 取消注释以使用
    # pdf_urls = [
    #     "https://openreview.net/pdf?id=wLR9d5ZFpY",
    #     "https://openreview.net/pdf?id=ID2",
    #     "https://openreview.net/pdf?id=ID3",
    #     "https://openreview.net/pdf?id=ID4",
    # ]

    cache_dir = project_root / "cache"

    # 查找 persona 文件
    persona_path = cache_dir / "reviewer_personas.json"
    if not persona_path.exists():
        env_personas = os.getenv("PERSONAS_FILE_PATH")
        if env_personas and Path(env_personas).exists():
            persona_path = Path(env_personas)
        else:
            alt_cache_dir = project_root.parent / "cache"
            alt_personas_file = alt_cache_dir / "reviewer_personas_redistributed.json"
            if alt_personas_file.exists():
                persona_path = alt_personas_file

    research_params = {
        "after": "2024-01-01",
        "web_temperature": 0.5,
        "code_temperature": 0.5,
        "title": "NO TRAINING DATA, NO CRY: MODEL EDITING WITHOUT TRAINING DATA OR FINETUNING",
        "depth": 3,
    }

    # 运行比较 pipeline
    result = await run_comparison_pipeline(
        pdf_urls=pdf_urls,
        cache_dir=cache_dir,
        persona_path=persona_path,
        research_params=research_params,
        future_cutoff="2025-05-31",
        num_personas=3,
    )

    # 打印最终报告
    print("\n" + "=" * 80)
    print("FINAL COMPARISON REPORT")
    print("=" * 80)
    print(result["final_report"])
    print("=" * 80)

    # 保存结果
    output_path = cache_dir / f"comparison_result_{result['mode']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved comparison result to {output_path}")

    print("\n" + "=" * 80)
    print("COMPARISON PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

