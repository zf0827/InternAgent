#!/usr/bin/env python3
"""
Agent Pipeline Test - 串联测试五个Agent

测试流程:
1. ExtractionAgent: PDF -> idea (提取结构化信息)
2. ResearchAgent: idea -> SearchResults (深度搜索)
3. ReportAgent: SearchResults -> reports (生成报告)
4. GroundingAgent: reports + claims -> grounding_results (证据验证)
5. EvaluationAgent: idea + reports + SearchResults -> evaluation (评估研究想法)

参考: quick_test_deepresearch.py 和 test_report_agent.py
"""

import logging
import sys
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from internagent.mas.models.model_factory import ModelFactory
from internagent.mas.agents.agent_factory import AgentFactory
from internagent.mas.tools.searchers.models import Idea, SearchResults

# 配置日志 - 确保所有组件的logger都可以正常展示
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_environment_variables():
    """Load environment variables from LLM.env file."""
    # Try multiple possible paths
    possible_paths = [
        project_root / "internagent" / "LLM.env",
        project_root / "IdeaEvaluation" / "LLM.env",
        Path(__file__).parent.parent.parent / "internagent" / "LLM.env",
    ]
    
    llm_env_path = None
    for path in possible_paths:
        if path.exists():
            llm_env_path = path
            break
    
    if not llm_env_path:
        logger.warning("LLM.env file not found in any expected location")
        return False
    
    logger.info(f"Loading environment variables from {llm_env_path}")
    
    # Manually load the .env file by parsing it
    try:
        with open(llm_env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    os.environ[key] = value
        logger.info("Successfully loaded environment variables")
        return True
    except Exception as e:
        logger.error(f"Error loading LLM.env file: {e}")
        return False


def extraction_to_idea(extraction_output: Dict[str, Any]) -> Idea:
    """
    将ExtractionAgent的输出转换为Idea对象。
    
    Args:
        extraction_output: ExtractionAgent的输出字典，包含数组格式的字段
        
    Returns:
        Idea对象
    """
    logger.info("Converting ExtractionAgent output to Idea object...")
    
    def join_list(items: List[str]) -> str:
        """将字符串数组连接成字符串"""
        if not items:
            return ""
        return "\n".join(items)
    
    # 提取各个字段（忽略basic_idea）
    motivation = join_list(extraction_output.get("motivation", []))
    research_question = join_list(extraction_output.get("research_question", []))
    method = join_list(extraction_output.get("method", []))
    experimental_setting = join_list(extraction_output.get("experimental_setting", []))
    expected_results = join_list(extraction_output.get("expected_results", [])) if extraction_output.get("expected_results") else None
    
    # 构建raw_text：合并所有字段（忽略basic_idea）
    parts = []
    if motivation:
        parts.append(f"Motivation: {motivation}")
    if research_question:
        parts.append(f"Research Question: {research_question}")
    if method:
        parts.append(f"Method: {method}")
    if experimental_setting:
        parts.append(f"Experimental Setting: {experimental_setting}")
    if expected_results:
        parts.append(f"Expected Results: {expected_results}")
    raw_text = "\n\n".join(parts) if parts else None
    
    idea = Idea(
        motivation=motivation,
        research_question=research_question,
        method=method,
        experimental_setting=experimental_setting,
        expected_results=expected_results,
        raw_text=raw_text
    )
    
    logger.info("Successfully converted extraction output to Idea")
    return idea


def load_personas(personas_file_path: Path, num_personas: int = 3) -> List[Dict[str, Any]]:
    """
    Load personas from reviewer_personas.json file.
    
    Args:
        personas_file_path: Path to the personas JSON file
        num_personas: Number of personas to select (default: 3)
        
    Returns:
        List of persona dictionaries
    """
    logger.info(f"Loading personas from {personas_file_path}")
    
    if not personas_file_path.exists():
        logger.warning(f"Personas file not found: {personas_file_path}")
        return []
    
    try:
        with open(personas_file_path, 'r', encoding='utf-8') as f:
            personas_data = json.load(f)
        
        if not isinstance(personas_data, list):
            raise ValueError("reviewer_personas.json must be a list")
        
        # Extract persona field from each item
        personas = []
        for item in personas_data:
            persona = item.get("persona")
            if persona:
                personas.append(persona)
        
        # Select only the first num_personas
        selected_personas = personas[:num_personas]
        
        logger.info(f"Loaded {len(selected_personas)} personas (from {len(personas)} total)")
        return selected_personas
    except Exception as e:
        logger.error(f"Error loading personas: {e}")
        return []

# python3 -m internagent.tester.test_agent_pipeline
async def main():
    print("\n" + "=" * 80)
    print("AGENT PIPELINE TEST - 串联测试")
    print("=" * 80)
    
    # 1. 加载环境变量
    if not load_environment_variables():
        logger.warning("Failed to load environment variables, continuing anyway...")
    
    # 2. 创建ModelFactory和AgentFactory
    model_factory = ModelFactory()
    agent_factory = AgentFactory()
    
    # 3. 配置各个Agent
    # 使用默认的模型配置
    default_model_config = {
        "models": {
            "default_provider": "dsr1",
            "dsr1": {
                "model_name": "deepseek-v3",
                "api_key": os.getenv("DS_API_KEY", ""),
                "base_url": os.getenv("DS_API_BASE_URL", ""),
                "max_tokens": 4096,
                "temperature": 0.7
            }
        }
    }
    
    # ExtractionAgent配置
    extraction_config = {
        "name": "ExtractionAgent",
        "model_provider": "dsr1",
        "extract_temperature": 0.3,
        "_global_config": default_model_config
    }
    
    # ResearchAgent配置
    research_config = {
        "name": "ResearchAgent",
        "model_provider": "dsr1",
        "max_iters": 0,  # 快速测试用0次迭代
        "max_results_per_source": 3,
        "enable_code_search": True,
        "enable_web_search": True,
        "enable_scholar_search": True,
        "paper_sources": ["arxiv", "semantic_scholar"],
        "enable_filtering": True,
        "enable_file_tree": True,  # 快速测试禁用文件树
        "filter_top_k_papers": 10,
        "filter_top_k_code": 6,
        "filter_top_k_web": 10,
        "top_k_readpage": 3,
        "_global_config": default_model_config
    }
    
    # ReportAgent配置
    report_config = {
        "name": "ReportAgent",
        "model_provider": "dsr1",
        "temperature": 0.7,
        "system_prompt": "You are a helpful assistant that generates comprehensive reports.",
        "_global_config": default_model_config
    }
    
    # GroundingAgent配置
    grounding_config = {
        "name": "GroundingAgent",
        "model_provider": "dsr1",
        "extract_temperature": 0.0,
        "ground_temperature": 0.0,
        "top_k_evidence": 20,
        "_global_config": default_model_config
    }
    
    # EvaluationAgent配置
    evaluation_config = {
        "name": "EvaluationAgent",
        "description": "Evaluates research ideas from multiple aspects",
        "model_provider": "dsr1",
        "temperature": 0.7,
        "_global_config": default_model_config,
        "max_retries": 10,
    }
    
    # 4. 创建Agent实例
    logger.info("Creating agent instances...")
    try:
        extraction_agent = agent_factory.create_agent("extraction", extraction_config, model_factory)
        logger.info("ExtractionAgent created successfully")
        
        research_agent = agent_factory.create_agent("research", research_config, model_factory)
        logger.info("ResearchAgent created successfully")
        
        report_agent = agent_factory.create_agent("report", report_config, model_factory)
        logger.info("ReportAgent created successfully")
        
        grounding_agent = agent_factory.create_agent("grounding", grounding_config, model_factory)
        logger.info("GroundingAgent created successfully")
        
        evaluation_agent = agent_factory.create_agent("evaluation", evaluation_config, model_factory)
        logger.info("EvaluationAgent created successfully")
    except Exception as e:
        logger.error(f"Failed to create agents: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 5. 执行ExtractionAgent (PDF -> idea)
    print("\n" + "=" * 80)
    print("STEP 1: ExtractionAgent - PDF -> Idea")
    print("=" * 80)
    
    pdf_url = "https://arxiv.org/pdf/2301.13379"
    extraction_context = {
        "url": pdf_url
    }
    extraction_params = {}
    
    try:
        logger.info(f"Executing ExtractionAgent with URL: {pdf_url}")
        extraction_result = await extraction_agent.execute(extraction_context, extraction_params)
        
        # 打印ExtractionAgent的中间结果
        logger.info("=" * 80)
        logger.info("ExtractionAgent Output:")
        logger.info("=" * 80)
        logger.info(json.dumps(extraction_result, indent=2, ensure_ascii=False))
        print("\n" + json.dumps(extraction_result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"ExtractionAgent execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 6. 转换ExtractionAgent输出为Idea对象
    logger.info("Converting extraction output to Idea object...")
    idea = extraction_to_idea(extraction_result)
    
    logger.info("=" * 80)
    logger.info("Converted Idea Object:")
    logger.info("=" * 80)
    logger.info(f"Motivation: {idea.motivation[:200]}..." if len(idea.motivation) > 200 else f"Motivation: {idea.motivation}")
    logger.info(f"Research Question: {idea.research_question[:200]}..." if len(idea.research_question) > 200 else f"Research Question: {idea.research_question}")
    print(f"\nIdea Summary:")
    print(f"  Motivation: {idea.motivation[:100]}..." if len(idea.motivation) > 100 else f"  Motivation: {idea.motivation}")
    print(f"  Research Question: {idea.research_question[:100]}..." if len(idea.research_question) > 100 else f"  Research Question: {idea.research_question}")
    
    # 7. 执行ResearchAgent (idea -> SearchResults)
    print("\n" + "=" * 80)
    print("STEP 2: ResearchAgent - Idea -> SearchResults")
    print("=" * 80)
    
    research_context = {
        "idea": idea.to_dict()
    }
    research_params = {
        "max_iters": 1  # 快速测试
    }
    
    try:
        logger.info("Executing ResearchAgent...")
        research_result = await research_agent.execute(research_context, research_params)
        
        search_results_dict = research_result.get("search_results", {})
        search_results = SearchResults.from_dict(search_results_dict)
        
        # 打印ResearchAgent的中间结果
        logger.info("=" * 80)
        logger.info("ResearchAgent Output Summary:")
        logger.info("=" * 80)
        logger.info(search_results.summary())
        print("\n" + search_results.summary())
        
    except Exception as e:
        logger.error(f"ResearchAgent execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 8. 执行ReportAgent (SearchResults -> reports)
    print("\n" + "=" * 80)
    print("STEP 3: ReportAgent - SearchResults -> Reports")
    print("=" * 80)
    
    report_context = {
        "search_result": search_results_dict
    }
    report_params = {
        "temperature": 0.7
    }
    
    try:
        logger.info("Executing ReportAgent...")
        report_result = await report_agent.execute(report_context, report_params)
        
        # 打印ReportAgent的中间结果
        logger.info("=" * 80)
        logger.info("ReportAgent Output:")
        logger.info("=" * 80)
        logger.info(f"Web Report Length: {len(report_result.get('web_report', ''))} characters")
        logger.info(f"Code Report Length: {len(report_result.get('code_report', ''))} characters")
        logger.info(f"Paper Report Length: {len(report_result.get('paper_report', ''))} characters")
        print(f"\nReport Lengths:")
        print(f"  Web Report: {len(report_result.get('web_report', ''))} characters")
        print(f"  Code Report: {len(report_result.get('code_report', ''))} characters")
        print(f"  Paper Report: {len(report_result.get('paper_report', ''))} characters")
        
        # 打印报告预览
        if report_result.get('web_report'):
            logger.info(f"Web Report Preview: {report_result['web_report'][:500]}...")
        if report_result.get('code_report'):
            logger.info(f"Code Report Preview: {report_result['code_report'][:500]}...")
        if report_result.get('paper_report'):
            logger.info(f"Paper Report Preview: {report_result['paper_report'][:500]}...")
        
    except Exception as e:
        logger.error(f"ReportAgent execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 9. 循环执行GroundingAgent (对每个part)
    print("\n" + "=" * 80)
    print("STEP 4: GroundingAgent - Reports + Claims -> Grounding Results")
    print("=" * 80)
    
    # 准备parts列表（忽略basic_idea）
    parts = ["motivation", "research_question", "method", "experimental_setting", "expected_results"]
    
    all_grounding_results = {}
    grounding_params = {
        "extract_temperature": 0.0,
        "ground_temperature": 0.0
    }
    
    for part in parts:
        if part in extraction_result and extraction_result[part]:
            logger.info(f"Processing part: {part}")
            
            grounding_context = {
                "claims": {part: extraction_result[part]},
                "reports": {
                    "web_report": report_result.get("web_report", ""),
                    "code_report": report_result.get("code_report", ""),
                    "paper_report": report_result.get("paper_report", "")
                }
            }
            
            try:
                grounding_result = await grounding_agent.execute(grounding_context, grounding_params)
                all_grounding_results[part] = grounding_result.get("grounding_results", [])
                
                # 打印每个part的grounding结果
                logger.info("=" * 80)
                logger.info(f"GroundingAgent Output for part '{part}':")
                logger.info("=" * 80)
                logger.info(json.dumps(grounding_result, indent=2, ensure_ascii=False))
                print(f"\nGrounding Results for '{part}': {len(grounding_result.get('grounding_results', []))} claims processed")
                
            except Exception as e:
                logger.error(f"GroundingAgent execution failed for part '{part}': {e}")
                import traceback
                traceback.print_exc()
                # 继续处理其他parts
                continue
    
    # 10. 执行EvaluationAgent (idea + reports + SearchResults -> evaluation)
    print("\n" + "=" * 80)
    print("STEP 5: EvaluationAgent - Idea + Reports + SearchResults -> Evaluation")
    print("=" * 80)
    
    # 加载personas（只选3个）
    cache_dir = project_root / "cache"
    personas_file = cache_dir / "reviewer_personas_redistributed.json"
    # 如果文件不存在，尝试从环境变量或相对路径查找
    if not personas_file.exists():
        # 尝试从环境变量获取路径
        env_personas_path = os.getenv("PERSONAS_FILE_PATH")
        if env_personas_path and Path(env_personas_path).exists():
            personas_file = Path(env_personas_path)
        else:
            # 尝试项目根目录下的 cache 目录
            alt_cache_dir = project_root.parent / "cache"
            alt_personas_file = alt_cache_dir / "reviewer_personas_redistributed.json"
            if alt_personas_file.exists():
                personas_file = alt_personas_file
    
    personas = load_personas(personas_file, num_personas=3)
    
    if not personas:
        logger.warning("No personas loaded, skipping EvaluationAgent")
        all_evaluation_results = []
    else:
        logger.info(f"Loaded {len(personas)} personas for evaluation")
        
        # 准备基础context
        base_evaluation_context = {
            "idea": idea.to_dict(),
            "search_results": search_results_dict,
            "web_report": report_result.get("web_report", ""),
            "code_report": report_result.get("code_report", ""),
            "paper_report": report_result.get("paper_report", "")
        }
        
        evaluation_params = {
            "temperature": 0.7
        }
        
        all_evaluation_results = []
        
        try:
            for idx, persona in enumerate(personas, 1):
                logger.info(f"Evaluating with persona {idx}/{len(personas)}")
                print(f"\n[{idx}/{len(personas)}] Evaluating with persona {idx}...")
                
                # 为当前人格创建context
                eval_context = base_evaluation_context.copy()
                eval_context["persona"] = persona
                
                # 执行评估
                eval_result = await evaluation_agent.execute(eval_context, evaluation_params)
                
                # 保存结果
                result_with_persona = {
                    "persona_index": idx,
                    "persona": persona,
                    "evaluation": eval_result
                }
                all_evaluation_results.append(result_with_persona)
                
                # 打印当前人格的评估结果摘要
                clarity_score = eval_result.get("clarity", {}).get("score", "N/A")
                novelty_score = eval_result.get("novelty", {}).get("score", "N/A")
                feasibility_score = eval_result.get("feasibility", {}).get("score", "N/A")
                print(f"  Persona {idx} Results: Clarity={clarity_score}/10, "
                      f"Novelty={novelty_score}/10, Feasibility={feasibility_score}/10")
                
        except Exception as e:
            logger.error(f"EvaluationAgent execution failed: {e}")
            import traceback
            traceback.print_exc()
            # 继续执行，不中断整个流程
    
    # 11. 保存和打印最终结果
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    # 构建search_results_summary，包含每一类的第一个resource的详细资料
    search_results_summary = {
        "total_count": search_results.total_count,
        "papers": {
            "count": len(search_results.papers),
            "first_item": search_results.papers[0].to_dict() if search_results.papers else None
        },
        "github_repos": {
            "count": len(search_results.github_repos),
            "first_item": search_results.github_repos[0].to_dict() if search_results.github_repos else None
        },
        "kaggle_results": {
            "count": len(search_results.kaggle_results),
            "first_item": search_results.kaggle_results[0].to_dict() if search_results.kaggle_results else None
        },
        "web_pages": {
            "count": len(search_results.web_pages),
            "first_item": search_results.web_pages[0].to_dict() if search_results.web_pages else None
        },
        "scholar_results": {
            "count": len(search_results.scholar_results),
            "first_item": search_results.scholar_results[0].to_dict() if search_results.scholar_results else None
        }
    }
    
    final_results = {
        "extraction_output": extraction_result,
        "idea": idea.to_dict(),
        "search_results_summary": search_results_summary,
        "reports": {
            "web_report": report_result.get("web_report", ""),
            "code_report": report_result.get("code_report", ""),
            "paper_report": report_result.get("paper_report", "")
        },
        "grounding_results": all_grounding_results,
        "evaluation_results": all_evaluation_results
    }
    
    logger.info("=" * 80)
    logger.info("Final Results Summary:")
    logger.info("=" * 80)
    logger.info(json.dumps(final_results, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    # 打印Pipeline执行摘要
    print("\nPipeline Execution Summary:")
    print(f"  - Extraction: {len(extraction_result.get('motivation', []))} motivation claims, "
          f"{len(extraction_result.get('research_question', []))} research questions")
    print(f"  - Research: {search_results.total_count} total sources found")
    print(f"  - Reports: Generated 3 reports")
    print(f"  - Grounding: Processed {len(all_grounding_results)} parts")
    print(f"  - Evaluation: Processed {len(all_evaluation_results)} personas")
    
    # 打印SearchResults详细资料（每一类的第一个resource）
    print("\n" + "-" * 80)
    print("SEARCH RESULTS SUMMARY (First Item of Each Category)")
    print("-" * 80)
    
    if search_results.papers:
        print(f"\nPapers ({len(search_results.papers)} total):")
        first_paper = search_results.papers[0]
        print(f"  Title: {first_paper.title}")
        print(f"  URL: {first_paper.url}")
        print(f"  Authors: {', '.join(first_paper.authors) if first_paper.authors else 'N/A'}")
        if first_paper.description:
            print(f"  Description: {first_paper.description[:200]}..." if len(first_paper.description) > 200 else f"  Description: {first_paper.description}")
        if first_paper.year:
            print(f"  Year: {first_paper.year}")
        if first_paper.citations is not None:
            print(f"  Citations: {first_paper.citations}")
    
    if search_results.github_repos:
        print(f"\nGitHub Repositories ({len(search_results.github_repos)} total):")
        first_repo = search_results.github_repos[0]
        print(f"  Title: {first_repo.title}")
        print(f"  URL: {first_repo.url}")
        print(f"  Description: {first_repo.description[:200]}..." if len(first_repo.description) > 200 else f"  Description: {first_repo.description}")
    
    if search_results.kaggle_results:
        print(f"\nKaggle Results ({len(search_results.kaggle_results)} total):")
        first_kaggle = search_results.kaggle_results[0]
        print(f"  Title: {first_kaggle.title}")
        print(f"  URL: {first_kaggle.url}")
        print(f"  Description: {first_kaggle.description[:200]}..." if len(first_kaggle.description) > 200 else f"  Description: {first_kaggle.description}")
    
    if search_results.web_pages:
        print(f"\nWeb Pages ({len(search_results.web_pages)} total):")
        first_web = search_results.web_pages[0]
        print(f"  Title: {first_web.title}")
        print(f"  URL: {first_web.url}")
        if first_web.description:
            print(f"  Description: {first_web.description[:200]}..." if len(first_web.description) > 200 else f"  Description: {first_web.description}")
    
    if search_results.scholar_results:
        print(f"\nScholar Results ({len(search_results.scholar_results)} total):")
        first_scholar = search_results.scholar_results[0]
        print(f"  Title: {first_scholar.title}")
        print(f"  URL: {first_scholar.url}")
        print(f"  Authors: {', '.join(first_scholar.authors) if first_scholar.authors else 'N/A'}")
        if first_scholar.description:
            print(f"  Description: {first_scholar.description[:200]}..." if len(first_scholar.description) > 200 else f"  Description: {first_scholar.description}")
        if first_scholar.year:
            print(f"  Year: {first_scholar.year}")
        if first_scholar.citations is not None:
            print(f"  Citations: {first_scholar.citations}")
    
    # 打印报告全长
    print("\n" + "-" * 80)
    print("REPORTS (Full Content)")
    print("-" * 80)
    
    web_report = report_result.get("web_report", "")
    code_report = report_result.get("code_report", "")
    paper_report = report_result.get("paper_report", "")
    
    print(f"\nWeb Report ({len(web_report)} characters):")
    print(web_report)
    
    print(f"\nCode Report ({len(code_report)} characters):")
    print(code_report)
    
    print(f"\nPaper Report ({len(paper_report)} characters):")
    print(paper_report)
    
    # 打印三个人的评分、理由等信息
    if all_evaluation_results:
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS (All Personas)")
        print("=" * 80)
        
        for idx, result_item in enumerate(all_evaluation_results, 1):
            persona = result_item["persona"]
            evaluation = result_item["evaluation"]
            
            print("\n" + "-" * 80)
            print(f"PERSONA {idx}: {persona.get('background', 'N/A')[:80]}...")
            print("-" * 80)
            
            clarity = evaluation.get("clarity", {})
            novelty = evaluation.get("novelty", {})
            feasibility = evaluation.get("feasibility", {})
            overall = evaluation.get("overall", {})
            
            print(f"\nClarity Score: {clarity.get('score', 'N/A')}/10")
            print(f"Clarity Reason: {clarity.get('reason', 'N/A')}")
            
            print(f"\nNovelty Score: {novelty.get('score', 'N/A')}/10")
            print(f"Novelty Reason: {novelty.get('reason', 'N/A')}")
            
            print(f"\nFeasibility Score: {feasibility.get('score', 'N/A')}/10")
            print(f"Feasibility Reason: {feasibility.get('reason', 'N/A')}")
            if feasibility.get('pseudocode'):
                print(f"Feasibility Pseudocode:\n{feasibility.get('pseudocode', 'N/A')}")
            
            print(f"\nOverall Summary: {overall.get('summary', 'N/A')}")
            print(f"Overall Recommendation: {overall.get('recommendation', 'N/A')}")
    
    
    print("\n" + "=" * 80)
    print("Agent Pipeline Test Completed Successfully!")
    print("=" * 80)
    
    # 保存结果到文件（注意：reports可能很长，保存完整内容）
    output_file = cache_dir / "agent_pipeline_results.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {output_file}")
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

