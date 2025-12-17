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

from internagent.tester.test_agent_pipelinev2 import load_environment_variables

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


def create_rank_signature(num_ideas: int) -> Type[dspy.Signature]:
    """
    动态创建排序 Signature 类。

    输出是对所有 idea 的一个全排序，最高分 -> 最低分。
    """
    docstring = f"""You are an experienced research reviewer and meta-evaluator. Your task is to rank {num_ideas} research ideas from best to worst based on comprehensive multi-dimensional analysis.

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

1. **Global Ranking**: Analyze all {num_ideas} ideas jointly and produce a single global ranking from best to worst.
2. **Multi-Dimensional Evaluation**: Consider clarity, novelty, validity, feasibility, and significance.
3. **Relative Comparison**: Focus on relative strengths/weaknesses and trade-offs between ideas.

## Output Format Requirements:

### ranking_analysis (Markdown format):
- Provide a detailed explanation of why the final ranking was chosen.
- Highlight key strengths and weaknesses for each idea.
- Emphasize the most important factors that drive the ordering.

### index_list (string):
- A comma-separated list of integers between 1 and {num_ideas} (inclusive), without additional text.
- It MUST contain each idea index exactly once.
- The order MUST be from best (highest-ranked) to worst (lowest-ranked).
- Example (for {num_ideas} ideas): "2, 1, 3, 4"
"""

    output_fields = {
        "ranking_analysis": dspy.OutputField(
            desc="Detailed ranking analysis report in Markdown format explaining the ordering of ideas"
        ),
        "index_list": dspy.OutputField(
            desc=f"Comma-separated list of idea indices (1..{num_ideas}) from best to worst, e.g. '2, 1, 3, 4'"
        ),
    }

    class_name = f"RankSignature{num_ideas}"
    signature_dict = {
        "__doc__": docstring,
        **input_fields,
        **output_fields,
    }
    signature_class = type(class_name, (dspy.Signature,), signature_dict)
    return signature_class


class RankingModule(dspy.Module):
    """使用 dspy 进行多 idea 排序的模块。"""

    def __init__(self, num_ideas: int, model_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.num_ideas = num_ideas

        # 配置 dspy LM（与 ComparisonModule 保持一致）
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
            logger.info(f"Initialized RankingModule with model: {model_config.get('model', 'gpt-4o-mini')}")
        except Exception as e:
            logger.error(f"Failed to initialize dspy in RankingModule: {e}")
            raise

        signature_class = create_rank_signature(num_ideas)
        self.rank = dspy.ChainOfThought(signature_class)

    def _parse_index_list(self, raw: str) -> List[int]:
        """将模型输出的 index_list 字符串解析为去重且合法的整数列表。"""
        if not raw:
            return list(range(1, self.num_ideas + 1))

        tokens = re.split(r"[,\s]+", raw.strip())
        seen = set()
        indices: List[int] = []
        for tok in tokens:
            if not tok:
                continue
            try:
                v = int(tok)
            except ValueError:
                continue
            if 1 <= v <= self.num_ideas and v not in seen:
                seen.add(v)
                indices.append(v)

        # 如果缺少某些 index，则按升序补齐到完整排列
        missing = [i for i in range(1, self.num_ideas + 1) if i not in seen]
        indices.extend(missing)
        return indices

    def forward(self, **kwargs) -> Dict[str, Any]:
        """
        执行排序，返回 ranking_analysis 和 index_list（整数列表）。
        """
        with dspy.settings.context(lm=self.lm):
            result = self.rank(**kwargs)

        raw_index_list = getattr(result, "index_list", "") or ""
        parsed_indices = self._parse_index_list(str(raw_index_list))

        return {
            "ranking_analysis": getattr(result, "ranking_analysis", ""),
            "index_list": parsed_indices,
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
    dataset_name: str,
    paper_ids: List[str],
    num_paper: int,
    mode: str = "best",
    cache_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    使用离线 cache 结果运行比较 pipeline（不再实际跑 SingleIdeaPipeline）。

    Args:
        dataset_name: 数据集名称（与 dataset jsonl 文件同名，无扩展名）
        paper_ids: paper_id 列表
        num_paper: 本次比较中实际使用的 paper 数量；
            - 当 num_paper == 1 时，直接返回对应 idea 的结果（point-wise）
            - 当 num_paper >= 2 时，根据 mode 选择比较方式
        mode: 比较模式：
            - "best": 使用 ComparisonModule 选择最优 idea
            - "rank": 使用 RankingModule 对所有 idea 排序
        cache_root: cache 根目录（默认为项目下的 cache）

    Returns:
        包含所有结果的字典
    """
    if num_paper <= 0:
        raise ValueError(f"num_paper must be > 0, got {num_paper}")

    if cache_root is None:
        cache_root = project_root / "cache"

    # 与 DatasetPipeline 的约定保持一致：cache_root / dataset_{dataset_name} / {paper_id}.json
    dataset_cache_dir = cache_root / f"dataset_{dataset_name}"
    if not dataset_cache_dir.exists():
        raise FileNotFoundError(
            f"Dataset cache directory not found: {dataset_cache_dir}. "
            "请先运行 test_dataset_pipeline 生成离线结果。"
        )

    # 根据 num_paper 截断 / 使用 paper_ids
    if num_paper > len(paper_ids):
        logger.warning(
            "num_paper=%d 大于提供的 paper_ids 数量=%d，将使用全部 paper_ids。",
            num_paper,
            len(paper_ids),
        )
        effective_ids = paper_ids
    else:
        effective_ids = paper_ids[:num_paper]

    num_ideas = len(effective_ids)
    if num_ideas not in [1, 2, 4]:
        raise ValueError(f"Unsupported number of ideas: {num_ideas}. Must be 1, 2, or 4.")

    internal_mode = "point-wise" if num_ideas == 1 else ("pair-wise" if num_ideas == 2 else "group-wise")
    logger.info(f"Running {internal_mode} comparison with {num_ideas} ideas (user mode={mode})")

    # 1. 从离线 cache 读取每个 paper 的结果
    results = []
    for idx, paper_id in enumerate(effective_ids, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Loading Idea {idx}/{num_ideas}: paper_id={paper_id}")
        logger.info(f"{'='*80}")

        try:
            cache_path = dataset_cache_dir / f"{paper_id}.json"
            if not cache_path.exists():
                logger.error(f"Cache file not found for paper_id={paper_id}: {cache_path}")
                continue

            with open(cache_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            # 附加少量元信息，方便后续使用
            result["paper_id"] = paper_id
            result["idea_index"] = idx
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to load idea {idx} (paper_id={paper_id}): {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        raise RuntimeError("No ideas were successfully processed.")

    # 2. num_paper == 1 -> 直接 point-wise 返回
    if num_ideas == 1:
        return {
            "mode": internal_mode,
            "ideas": results,
            "final_report": results[0]["final_report"],
            "best_idea_index": 1,
        }

    # 3. 使用 dspy 生成比较 / 排序报告
    logger.info(f"\n{'='*80}")
    logger.info("Generating Comparison / Ranking Report")
    logger.info(f"{'='*80}")

    # 准备输入数据
    comparison_inputs: Dict[str, Any] = {}
    for idx, result in enumerate(results, 1):
        evaluation_summary = format_evaluation_summary(result["evaluation_result"])
        comparison_inputs[f"idea_{idx}_evaluation"] = evaluation_summary
        comparison_inputs[f"idea_{idx}_report"] = result["final_report"]

    final_report_parts = ["# Idea Comparison Report\n"]
    for idx, result in enumerate(results, 1):
        paper_id = result.get("paper_id", f"idea_{idx}")
        final_report_parts.append(f"## Idea {idx}: {paper_id}\n")
        final_report_parts.append(result["final_report"])
        final_report_parts.append("\n")

    if mode == "best":
        comparison_module = ComparisonModule(num_ideas=num_ideas)
        comparison_result = comparison_module.forward(**comparison_inputs)

        final_report_parts.append("## Comparison Analysis\n")
        final_report_parts.append(comparison_result["comparison_analysis"])
        final_report_parts.append("\n")

        final_report_parts.append("## Best Idea Selection\n")
        final_report_parts.append(f"Selected: Idea {comparison_result['best_idea_index']}\n")
        final_report_parts.append(f"Reason: {comparison_result['selection_reason']}")

        final_report = "\n".join(final_report_parts)

        return {
            "mode": internal_mode,
            "ideas": results,
            "comparison_result": comparison_result,
            "final_report": final_report,
            "best_idea_index": comparison_result["best_idea_index"],
        }
    elif mode == "rank":
        ranking_module = RankingModule(num_ideas=num_ideas)
        ranking_result = ranking_module.forward(**comparison_inputs)

        final_report_parts.append("## Ranking Analysis\n")
        final_report_parts.append(ranking_result["ranking_analysis"])
        final_report_parts.append("\n")

        final_report_parts.append("## Idea Ranking\n")
        final_report_parts.append(f"Order (best -> worst): {ranking_result['index_list']}")

        final_report = "\n".join(final_report_parts)

        return {
            "mode": internal_mode,
            "ideas": results,
            "ranking_result": ranking_result,
            "final_report": final_report,
            "index_list": ranking_result["index_list"],
        }
    else:
        raise ValueError(f"Unsupported comparison mode: {mode}. Supported modes: 'best', 'rank'.")


async def main() -> None:
    """主函数。"""
    print("\n" + "=" * 80)
    print("COMPARISON PIPELINE TEST")
    print("=" * 80)

    # 加载环境变量
    if not load_environment_variables():
        logger.warning("Failed to load environment variables, continuing anyway...")

    # 离线比较 / 排序配置（硬编码示例，可按需修改）
    # dataset_name 与 dataset jsonl 文件名（去掉扩展名）一致，例如 my_test_iclr.jsonl -> my_test_iclr
    dataset_name = "my_test_iclr"

    # 需要进行比较 / 排序的 paper_id 列表（来自数据集中的 paper_id 字段）
    # 这里只需要提供 id，所有计算结果都从 cache 中读取
    paper_ids = [
        "wLR9d5ZFpY",
        # "ANOTHER_PAPER_ID",
        # "THIRD_PAPER_ID",
        # "FOURTH_PAPER_ID",
    ]

    # 显式指定本次要使用的 paper 数量
    num_paper = len(paper_ids)

    # 可在 "best"（选出最优）和 "rank"（全序排序）之间切换
    user_mode = "rank"

    # 运行比较 / 排序 pipeline（离线模式）
    result = await run_comparison_pipeline(
        dataset_name=dataset_name,
        paper_ids=paper_ids,
        num_paper=num_paper,
        mode=user_mode,
        cache_root=project_root / "cache",
    )

    # 打印最终报告
    print("\n" + "=" * 80)
    print("FINAL COMPARISON REPORT")
    print("=" * 80)
    print(result["final_report"])
    print("=" * 80)

    # 保存结果
    cache_dir = project_root / "cache"
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

