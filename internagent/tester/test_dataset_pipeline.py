#!/usr/bin/env python3
"""
Dataset Pipeline - 批量处理整份 idea 数据集（point-wise）

功能：
1. 从 JSONL 数据集读取若干 paper（paper_id, title, decision）
2. 根据 paper_id 构造 pdf_url（OpenReview）
3. 为每个 paper 调用 SingleIdeaPipeline，产出结果并写入各自的 cache json
4. 支持并行处理一个 batch 的 idea（受并发上限控制）
5. 在运行结束后，根据 final_decision 与真实 decision 对比，计算 accuracy

使用方式：
    python -m internagent.tester.test_dataset_pipeline

必要配置：
    - 在 LLM.env 中配置相关的 API key（路径与 test_agent_pipelinev2 保持一致）
    - 确保 reviewer_personas.json 或对应的 personas 文件存在
"""

import asyncio
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from internagent.tester.test_agent_pipelinev2 import (  # noqa: E402
    SingleIdeaPipeline,
    load_environment_variables,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class PaperItem:
    paper_id: str
    title: str
    decision: Optional[str]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def load_dataset(jsonl_path: Path, num: Optional[int] = None, seed: int = 42) -> List[PaperItem]:
    """从 JSONL 文件加载数据，并随机采样 num 个样本（如果指定）。"""
    logger.info(f"Loading dataset from {jsonl_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    items: List[PaperItem] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to parse line as JSON, skipping. Error: {e}")
                continue

            paper_id = obj.get("paper_id") or obj.get("id")
            title = obj.get("title", "")
            decision = obj.get("decision")

            if not paper_id:
                logger.warning(f"Missing paper_id in line: {obj}")
                continue

            items.append(PaperItem(paper_id=str(paper_id), title=title, decision=decision))

    if not items:
        raise ValueError(f"No valid items found in dataset: {jsonl_path}")

    if num is not None and num > 0:
        random.seed(seed)
        if num < len(items):
            items = random.sample(items, num)
        else:
            logger.info(
                f"Requested num={num} >= dataset size={len(items)}, "
                "using full dataset instead."
            )

    logger.info(f"Loaded {len(items)} items from dataset")
    return items


def build_pdf_url_from_paper_id(paper_id: str) -> str:
    """根据 paper_id 构造 OpenReview PDF URL。"""
    return f"https://openreview.net/pdf?id={paper_id}"


def build_cache_path_for_paper(
    dataset_path: Path,
    paper_id: str,
    cache_root: Optional[Path] = None,
) -> Path:
    """
    为指定 paper 构造 cache 路径。

    约定：
        cache_root / dataset_name / {paper_id}.json
    其中 dataset_name = dataset 文件名（不含扩展名）
    """
    if cache_root is None:
        cache_root = project_root / "cache"

    dataset_name = dataset_path.stem  # e.g. my_test_iclr
    dataset_cache_dir = cache_root / f"dataset_{dataset_name}"
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = dataset_cache_dir / f"{paper_id}.json"
    return cache_path


def find_persona_path() -> Path:
    """只从 cache 目录下读取 reviewer_personas.json，取不到则直接失败。"""
    cache_dir = project_root / "cache"
    persona_path = cache_dir / "reviewer_personas.json"
    if not persona_path.exists():
        raise FileNotFoundError(
            f"Persona file not found: {persona_path}. "
            "Please put reviewer_personas.json under cache directory."
        )
    return persona_path


def normalize_decision(value: Optional[str]) -> Optional[str]:
    """
    规范化决策标签到四类：
        - oral
        - spotlight
        - poster
        - reject
    支持从较长的 decision 文本中提取上述类别。
    """
    if value is None:
        return None

    v = str(value).strip().lower()

    # 先做精确映射
    exact_map = {
        "accept (oral)": "oral",
        "oral": "oral",
        "accept (spotlight)": "spotlight",
        "spotlight": "spotlight",
        "accept (poster)": "poster",
        "poster": "poster",
        "reject": "reject",
        "reject.": "reject",
    }
    if v in exact_map:
        return exact_map[v]

    # 再做包含匹配（处理更长的描述文本）
    if "oral" in v:
        return "oral"
    if "spotlight" in v:
        return "spotlight"
    if "poster" in v:
        return "poster"
    if "reject" in v:
        return "reject"

    return None


# --------------------------------------------------------------------------- #
# 单个 paper 的处理
# --------------------------------------------------------------------------- #
async def process_single_paper(
    item: PaperItem,
    dataset_path: Path,
    persona_path: Path,
    base_research_params: Dict[str, Any],
    num_personas: int,
) -> Tuple[PaperItem, Optional[Dict[str, Any]]]:
    """
    处理单个 paper：
        - 构造 pdf_url 和 cache_path
        - 创建 SingleIdeaPipeline
        - 让 pipeline 内部负责 cache 逻辑（包括增量 / 续跑）
        - 返回 pipeline 的整体 result
    """
    pdf_url = build_pdf_url_from_paper_id(item.paper_id)
    cache_path = build_cache_path_for_paper(dataset_path, item.paper_id)

    logger.info(f"Starting pipeline for paper {item.paper_id} | cache={cache_path}")

    try:
        # 为当前 paper 定制 research_params（特别是 title）
        research_params = dict(base_research_params)
        if item.title:
            research_params["title"] = item.title

        pipeline = SingleIdeaPipeline(
            pdf_url=pdf_url,
            cache_path=cache_path,
            persona_path=persona_path,
            research_params=research_params,
            num_personas=num_personas,
        )
        result = await pipeline.run()
        return item, result
    except Exception as e:  # noqa: BLE001
        logger.error(f"Pipeline failed for paper {item.paper_id}: {e}")
        return item, None


async def process_papers_in_parallel(
    items: List[PaperItem],
    dataset_path: Path,
    persona_path: Path,
    research_params: Dict[str, Any],
    num_personas: int,
    max_concurrent: int = 3,
) -> List[Tuple[PaperItem, Optional[Dict[str, Any]]]]:
    """
    并行处理多个 paper，使用 semaphore 控制最大并发数。
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results: List[Tuple[PaperItem, Optional[Dict[str, Any]]]] = []

    async def _worker(item: PaperItem) -> None:
        async with semaphore:
            paper, res = await process_single_paper(
                item=item,
                dataset_path=dataset_path,
                persona_path=persona_path,
                base_research_params=research_params,
                num_personas=num_personas,
            )
            results.append((paper, res))

    tasks = [asyncio.create_task(_worker(it)) for it in items]
    await asyncio.gather(*tasks)
    return results


# --------------------------------------------------------------------------- #
# 评估与主流程
# --------------------------------------------------------------------------- #
def evaluate_accuracy(
    processed_results: List[Tuple[PaperItem, Optional[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """
    根据 final_decision 与真实 decision 计算 accuracy。
    """
    total = 0
    matched = 0
    detailed: List[Dict[str, Any]] = []

    for item, result in processed_results:
        if result is None:
            # pipeline 失败，不计入 accuracy，但保留记录
            detailed.append(
                {
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "label_decision": item.decision,
                    "pred_decision": None,
                    "match": None,
                    "status": "failed",
                }
            )
            continue

        label_decision = normalize_decision(item.decision)
        pred_decision = normalize_decision(result.get("final_decision"))

        match: Optional[bool]
        if label_decision is None or pred_decision is None:
            match = None
        else:
            match = label_decision == pred_decision

        if match is not None:
            total += 1
            if match:
                matched += 1

        detailed.append(
            {
                "paper_id": item.paper_id,
                "title": item.title,
                "label_decision": label_decision,
                "pred_decision": pred_decision,
                "match": match,
                "status": "ok",
            }
        )

    acc = matched / total if total > 0 else 0.0
    return {
        "accuracy": acc,
        "total_evaluable": total,
        "matched": matched,
        "num_items": len(processed_results),
        "details": detailed,
    }


async def main() -> None:
    """主入口：批量运行 point-wise SingleIdeaPipeline 并评估 accuracy。"""
    print("\n" + "=" * 80)
    print("DATASET PIPELINE TEST - POINT-WISE BATCH EVALUATION")
    print("=" * 80)

    # 加载环境变量
    if not load_environment_variables():
        logger.warning("Failed to load environment variables, continuing anyway...")

    # 数据集路径（可按需修改或参数化）
    dataset_path = Path(
        "/home/weiyunxiang/yunx/IdeaEvaluation/InternAgent/dataset/my_test_iclr.jsonl"
    )

    # 要评测的 paper 数量；为 None 或 <=0 时表示使用整个数据集
    num_papers: Optional[int] = None  # 比如改成 10 只评测 10 个

    # SingleIdeaPipeline 的 research_params
    research_params: Dict[str, Any] = {
        "after": "2024-01-01",
        "before": "2025-05-31",  # 用于划分 future papers 的时间点
        "web_temperature": 0.5,
        "code_temperature": 0.5,
        "title": "NO TRAINING DATA, NO CRY: MODEL EDITING WITHOUT TRAINING DATA OR FINETUNING",
        "depth": 3,
    }

    num_personas = 3
    max_concurrent = 3

    # persona 文件
    persona_path = find_persona_path()
    logger.info(f"Using persona file: {persona_path}")

    # 加载数据集
    items = load_dataset(dataset_path, num=num_papers)

    # 并行运行 pipelines
    processed_results = await process_papers_in_parallel(
        items=items,
        dataset_path=dataset_path,
        persona_path=persona_path,
        research_params=research_params,
        num_personas=num_personas,
        max_concurrent=max_concurrent,
    )

    # 评估 accuracy
    eval_result = evaluate_accuracy(processed_results)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(
        f"Num items: {eval_result['num_items']}\n"
        f"Evaluable: {eval_result['total_evaluable']}\n"
        f"Matched: {eval_result['matched']}\n"
        f"Accuracy: {eval_result['accuracy']:.4f}"
    )
    print("=" * 80)

    # 将详细结果保存到 cache 目录下，方便后续分析
    cache_root = project_root / "cache"
    dataset_name = dataset_path.stem
    output_path = cache_root / f"dataset_eval_{dataset_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved evaluation result to {output_path}")

    print("\n" + "=" * 80)
    print("DATASET PIPELINE COMPLETED SUCCESSFULLY!")
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



