#!/usr/bin/env python3
"""
Agent Pipeline Test - 串联测试Agent工作流

测试流程:
1. ExtractionAgent: PDF -> idea (提取结构化信息，使用Idea.from_lists()创建Idea对象)
2. ResearchAgentV2: idea -> SearchResults (深度搜索，返回三个平台的结果)
3. ReportAgent (Research Mode): SearchResults -> research reports (生成研究报告，返回三个报告列表)
4. GroundingAgent: reports + claims -> grounding_results (证据验证，对六个部分分别处理)
5. EvaluationAgent: idea + reports + SearchResults -> evaluation (评估，返回三个维度的评分)
6. ReportAgent (Review Mode): evaluation_results -> review reports (生成评审报告，基于评估结果)

参考: quick_test_deepresearch.py 和 test_report_agent.py
"""

import logging
import sys
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from internagent.mas.models.model_factory import ModelFactory
from internagent.mas.agents.agent_factory import AgentFactory
from internagent.mas.tools.searchersv2.models import Idea, SearchResults

# 配置日志 - 确保所有组件的logger都可以正常展示
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pipeline_result(file_path: Path) -> Dict[str, Any]:
    """
    Load existing pipeline_result.json if it exists, otherwise return empty dict.
    
    Returns:
        Dict with keys: search_results_dict, reports_data, grounding_result, evaluation_result, final_report
    """
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure all required keys exist
                if "search_results_dict" not in data:
                    data["search_results_dict"] = None
                if "reports_data" not in data:
                    data["reports_data"] = None
                if "grounding_result" not in data:
                    data["grounding_result"] = {}
                if "evaluation_result" not in data:
                    data["evaluation_result"] = None
                if "final_report" not in data:
                    data["final_report"] = None
                return data
        except Exception as e:
            logger.warning(f"Failed to load existing pipeline_result.json: {e}, creating new file")
            return {
                "search_results_dict": None,
                "reports_data": None,
                "grounding_result": {},
                "evaluation_result": None,
                "final_report": None
            }
    else:
        return {
            "search_results_dict": None,
            "reports_data": None,
            "grounding_result": {},
            "evaluation_result": None,
            "final_report": None
        }


def save_pipeline_result(file_path: Path, data: Dict[str, Any]):
    """
    Save pipeline_result.json with proper formatting.
    
    Args:
        file_path: Path to save the file
        data: Dict with keys: search_results_dict, reports_data, grounding_result, evaluation_result
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved pipeline_result to {file_path}")


def update_pipeline_result(file_path: Path, 
                           search_results_dict: Optional[Dict[str, Any]] = None,
                           reports_data: Optional[Dict[str, Any]] = None,
                           grounding_result_part: Optional[str] = None,
                           grounding_result_claims: Optional[List[Dict[str, Any]]] = None,
                           evaluation_result: Optional[Dict[str, Any]] = None,
                           final_report: Optional[str] = None):
    """
    Update pipeline_result.json with new data (append mode).
    
    Args:
        file_path: Path to pipeline_result.json
        search_results_dict: New search_results_dict to save
        reports_data: New reports_data to save
        grounding_result_part: Part name for grounding_result (e.g., "basic_idea")
        grounding_result_claims: List of claim dicts for the part
        evaluation_result: New evaluation_result to save
        final_report: Final report string to save
    """
    # Load existing data
    data = load_pipeline_result(file_path)
    
    # Update search_results_dict
    if search_results_dict is not None:
        data["search_results_dict"] = search_results_dict
    
    # Update reports_data
    if reports_data is not None:
        data["reports_data"] = reports_data
    
    # Update grounding_result for a specific part
    if grounding_result_part is not None and grounding_result_claims is not None:
        if "grounding_result" not in data:
            data["grounding_result"] = {}
        data["grounding_result"][grounding_result_part] = grounding_result_claims
    
    # Update evaluation_result
    if evaluation_result is not None:
        data["evaluation_result"] = evaluation_result
    
    # Update final_report
    if final_report is not None:
        data["final_report"] = final_report
    
    # Save updated data
    save_pipeline_result(file_path, data)


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


def has_cache(cached_data: Dict[str, Any], cache_key: str, check_func: Optional[Callable[[Any], bool]] = None) -> bool:
    """
    检查缓存中是否存在指定key的有效数据。
    
    Args:
        cached_data: 从pipeline_result.json加载的缓存数据
        cache_key: 要检查的key（如 "search_results_dict", "reports_data"）
        check_func: 可选的验证函数，用于进一步检查数据有效性
        
    Returns:
        bool: 如果缓存存在且有效返回True，否则返回False
    """
    if cache_key not in cached_data:
        return False
    
    value = cached_data[cache_key]
    
    # 检查是否为None或空
    if value is None:
        return False
    
    # 如果是字典，检查是否为空
    if isinstance(value, dict) and len(value) == 0:
        return False
    
    # 如果是列表，检查是否为空
    if isinstance(value, list) and len(value) == 0:
        return False
    
    # 如果提供了验证函数，使用它进一步检查
    if check_func is not None:
        return check_func(value)
    
    return True

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
                "model_name": "DeepSeek-V3.2",
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
    
    # ResearchAgentV2配置
    research_config = {
        "name": "ResearchAgentV2",
        "model_provider": "dsr1",
        "max_results_per_query": 8,
        "enable_paper_filtering": True,
        "paper_batch_size": 8,
        "web_max_results": 8,
        "topk_papers": 10,
        "topk_web_pages": 10,
        "_global_config": default_model_config
    }
    
    # ReportAgent配置（支持research reports和review reports两种模式）
    report_config = {
        "name": "ReportAgent",
        "model_provider": "dsr1",
        "temperature": 0.7,
        "extraction_config": {
            "name": "ExtractionAgent",
            "model_provider": "dsr1",
            "extract_temperature": 0.3,
            "_global_config": default_model_config
        },
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
    
    
    # 4. 初始化pipeline_result.json文件路径
    pipeline_result_path = Path(project_root / "cache" / "pipeline_result_v3.json")
    pipeline_result_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 4.5. 加载缓存数据（用于加速跳过已完成的阶段）
    cached_data = load_pipeline_result(pipeline_result_path)
    logger.info("Loaded pipeline cache for acceleration check")
    
    # 5. 创建Agent实例
    logger.info("Creating agent instances...")
    try:
        extraction_agent = agent_factory.create_agent("extraction", extraction_config, model_factory)
        logger.info("ExtractionAgent created successfully")
        
        research_agent = agent_factory.create_agent("researchv2", research_config, model_factory)
        logger.info("ResearchAgentV2 created successfully")
        
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
    
    # 6. 执行ExtractionAgent (PDF -> idea)
    print("\n" + "=" * 80)
    print("STEP 1: ExtractionAgent - PDF -> Idea")
    print("=" * 80)
    
    pdf_url = "https://openreview.net/pdf?id=wLR9d5ZFpY"
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
    
    # 7. 转换ExtractionAgent输出为Idea对象
    logger.info("Converting extraction output to Idea object using Idea.from_lists()...")
    idea = Idea.from_lists(
        basic_idea_list=extraction_result.get("basic_idea", []),
        motivation_list=extraction_result.get("motivation", []),
        research_question_list=extraction_result.get("research_question", []),
        method_list=extraction_result.get("method", []),
        experimental_setting_list=extraction_result.get("experimental_setting", []),
        expected_results_list=extraction_result.get("expected_results", [])
    )
    
    logger.info("=" * 80)
    logger.info("Converted Idea Object:")
    logger.info("=" * 80)
    logger.info(f"Basic Idea: {idea.basic_idea[:200]}..." if len(idea.basic_idea) > 200 else f"Basic Idea: {idea.basic_idea}")
    logger.info(f"Motivation: {idea.motivation[:200]}..." if len(idea.motivation) > 200 else f"Motivation: {idea.motivation}")
    logger.info(f"Research Question: {idea.research_question[:200]}..." if len(idea.research_question) > 200 else f"Research Question: {idea.research_question}")
    logger.info(f"Method: {idea.method[:200]}..." if len(idea.method) > 200 else f"Method: {idea.method}")
    logger.info(f"Experimental Setting: {idea.experimental_setting[:200]}..." if len(idea.experimental_setting) > 200 else f"Experimental Setting: {idea.experimental_setting}")
    logger.info(f"Expected Results: {idea.expected_results[:200]}..." if len(idea.expected_results) > 200 else f"Expected Results: {idea.expected_results}")
    print(f"\nIdea Summary:")
    print(f"  Basic Idea: {idea.basic_idea[:100]}..." if len(idea.basic_idea) > 100 else f"  Basic Idea: {idea.basic_idea}")
    print(f"  Motivation: {idea.motivation[:100]}..." if len(idea.motivation) > 100 else f"  Motivation: {idea.motivation}")
    print(f"  Research Question: {idea.research_question[:100]}..." if len(idea.research_question) > 100 else f"  Research Question: {idea.research_question}")
    print(f"  Method: {idea.method[:100]}..." if len(idea.method) > 100 else f"  Method: {idea.method}")
    print(f"  Experimental Setting: {idea.experimental_setting[:100]}..." if len(idea.experimental_setting) > 100 else f"  Experimental Setting: {idea.experimental_setting}")
    print(f"  Expected Results: {idea.expected_results[:100]}..." if len(idea.expected_results) > 100 else f"  Expected Results: {idea.expected_results}")
   
    # 8. 执行ResearchAgentV2 (idea -> SearchResults)
    print("\n" + "=" * 80)
    print("STEP 2: ResearchAgentV2 - Idea -> SearchResults")
    print("=" * 80)
    
    # 检查缓存
    if has_cache(cached_data, "search_results_dict"):
        logger.info("✓ Found cached search_results_dict, skipping ResearchAgentV2 execution")
        search_results_dict = cached_data["search_results_dict"]
        search_results = SearchResults.from_dict(search_results_dict)
        print("✓ Using cached search results")
        logger.info(search_results.summary())
        print("\n" + search_results.summary())
        
        # 从缓存中获取 future_papers（如果存在）
        if "future_papers" in cached_data:
            logger.info(f"✓ Found cached future_papers: {len(cached_data['future_papers'])} papers")
    else:
        research_context = {
            "idea": idea.to_dict()
        }
        research_params = {"before": "2025-10-01", "after": "2024-01-01"}
        
        try:
            logger.info("Executing ResearchAgentV2...")
            research_result = await research_agent.execute(research_context, research_params)
            
            search_results_dict = research_result.get("search_results", {})
            search_results = SearchResults.from_dict(search_results_dict)
            
            # 分离 future_papers：根据 after 日期划分
            after_date = research_params.get("after")
            papers = search_results.papers
            future_papers = []
            regular_papers = []
            
            if after_date:
                for paper in papers:
                    # 检查 paper 的 timestamp（日期格式 yyyy-mm-dd）
                    paper_date = paper.timestamp
                    if paper_date and paper_date >= after_date:
                        future_papers.append(paper.to_dict())
                    else:
                        regular_papers.append(paper)
                logger.info(f"Separated papers: {len(regular_papers)} regular, {len(future_papers)} future (after {after_date})")
            else:
                regular_papers = papers
                logger.info("No after date provided, all papers are regular papers")
            
            # 更新 search_results，只保留 regular_papers
            search_results.papers = regular_papers
            search_results_dict = search_results.to_dict()
            
            # 保存search_results到pipeline_result.json（第一部分）
            update_pipeline_result(
                pipeline_result_path,
                search_results_dict=search_results_dict
            )
            logger.info(f"Saved search_results_dict to pipeline_result.json")
            
            # 更新缓存数据
            cached_data["search_results_dict"] = search_results_dict
            if future_papers:
                cached_data["future_papers"] = future_papers
            
            # 打印ResearchAgentV2的中间结果
            logger.info("=" * 80)
            logger.info("ResearchAgentV2 Output Summary:")
            logger.info("=" * 80)
            logger.info(search_results.summary())
            print("\n" + search_results.summary())
            
        except Exception as e:
            logger.error(f"ResearchAgentV2 execution failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # 9. 执行ReportAgent (SearchResults -> research reports)
    print("\n" + "=" * 80)
    print("STEP 3: ReportAgent - SearchResults -> Research Reports")
    print("=" * 80)
    
    # 检查缓存
    if has_cache(cached_data, "reports_data", lambda x: isinstance(x, dict) and 
                 (x.get("web_reports") or x.get("code_reports") or x.get("paper_reports"))):
        logger.info("✓ Found cached reports_data, skipping ReportAgent execution")
        reports_data = cached_data["reports_data"]
        web_reports = reports_data.get("web_reports", [])
        code_reports = reports_data.get("code_reports", [])
        paper_reports = reports_data.get("paper_reports", [])
        print("✓ Using cached research reports")
        print(f"\nResearch Report Counts:")
        print(f"  Web Reports: {len(web_reports)}")
        print(f"  Code Reports: {len(code_reports)}")
        print(f"  Paper Reports: {len(paper_reports)}")
    else:
        research_report_context = {
            "search_result": search_results_dict
        }
        research_report_params = {
            "temperature": 0.7
        }
        
        try:
            logger.info("Executing ReportAgent for research reports...")
            research_report_result = await report_agent.execute(research_report_context, research_report_params)
            
            # 获取三个报告列表（research reports模式）
            web_reports = research_report_result.get("web_reports", [])
            code_reports = research_report_result.get("code_reports", [])
            paper_reports = research_report_result.get("paper_reports", [])
            
            # 保存reports_data到pipeline_result.json（第二部分）
            reports_data = {
                "web_reports": web_reports,
                "code_reports": code_reports,
                "paper_reports": paper_reports
            }
            update_pipeline_result(
                pipeline_result_path,
                reports_data=reports_data
            )
            logger.info(f"Saved research reports_data to pipeline_result.json")
            
            # 更新缓存数据
            cached_data["reports_data"] = reports_data
            
            # 打印ReportAgent的中间结果
            logger.info("=" * 80)
            logger.info("ReportAgent Output (Research Reports):")
            logger.info("=" * 80)
            logger.info(f"Web Reports: {len(web_reports)} reports")
            logger.info(f"Code Reports: {len(code_reports)} reports")
            logger.info(f"Paper Reports: {len(paper_reports)} reports")
            print(f"\nResearch Report Counts:")
            print(f"  Web Reports: {len(web_reports)}")
            print(f"  Code Reports: {len(code_reports)}")
            print(f"  Paper Reports: {len(paper_reports)}")
            
            # 打印报告预览
            if web_reports:
                # 新格式：直接包含 report_content 字段
                report_content = web_reports[0].get("report_content", "")
                if not report_content:
                    # 兼容旧格式：content 是字典
                    content_dict = web_reports[0].get("content", {})
                    if isinstance(content_dict, dict):
                        report_content = content_dict.get("report_content", "")
                logger.info(f"First Web Report Preview: {report_content[:500]}..." if len(report_content) > 500 else f"First Web Report Preview: {report_content}")
            if code_reports:
                # 新格式：直接包含 report_content 字段
                report_content = code_reports[0].get("report_content", "")
                if not report_content:
                    # 兼容旧格式：content 是字典
                    content_dict = code_reports[0].get("content", {})
                    if isinstance(content_dict, dict):
                        report_content = content_dict.get("report_content", "")
                logger.info(f"First Code Report Preview: {report_content[:500]}..." if len(report_content) > 500 else f"First Code Report Preview: {report_content}")
            if paper_reports:
                logger.info(f"First Paper Report Title: {paper_reports[0].get('paper_metadata', {}).get('title', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"ReportAgent execution failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # 10. 循环执行GroundingAgent (对每个part)
    print("\n" + "=" * 80)
    print("STEP 4: GroundingAgent - Reports + Claims -> Grounding Results")
    print("=" * 80)
    
    # 准备六个部分
    parts = ["basic_idea", "motivation", "research_question", "method", "experimental_setting", "expected_results"]
    
    all_grounding_results = {}
    grounding_params = {
        "extract_temperature": 0.0,
        "ground_temperature": 0.0
    }
    
    # 确保grounding_result在缓存中存在
    if "grounding_result" not in cached_data:
        cached_data["grounding_result"] = {}
    
    for part in parts:
        # 从idea对象获取对应的*_list字段作为claims
        claims_list = getattr(idea, f"{part}_list", [])
        if not claims_list:
            logger.info(f"Skipping part '{part}': no claims found")
            continue
        
        logger.info(f"Processing part: {part} ({len(claims_list)} claims)")
        
        # 检查该part的缓存
        if has_cache(cached_data["grounding_result"], part, lambda x: isinstance(x, list) and len(x) > 0):
            logger.info(f"✓ Found cached grounding_result for part '{part}', skipping execution")
            grounding_claims = cached_data["grounding_result"][part]
            all_grounding_results[part] = grounding_claims
            print(f"✓ Using cached grounding results for '{part}': {len(grounding_claims)} claims")
        else:
            grounding_context = {
                "claims": {part: claims_list},
                "reports": {
                    "web_reports": web_reports,
                    "code_reports": code_reports,
                    "paper_reports": paper_reports
                }
            }
            
            try:
                grounding_result = await grounding_agent.execute(grounding_context, grounding_params)
                grounding_claims = grounding_result.get("grounding_results", [])
                all_grounding_results[part] = grounding_claims
                
                # 保存grounding_result到pipeline_result.json（第三部分，按part保存）
                update_pipeline_result(
                    pipeline_result_path,
                    grounding_result_part=part,
                    grounding_result_claims=grounding_claims
                )
                logger.info(f"Saved grounding_result for part '{part}' to pipeline_result.json")
                
                # 更新缓存数据
                cached_data["grounding_result"][part] = grounding_claims
                
                # 打印每个part的grounding结果
                logger.info("=" * 80)
                logger.info(f"GroundingAgent Output for part '{part}':")
                logger.info("=" * 80)
                logger.info(json.dumps(grounding_result, indent=2, ensure_ascii=False))
                print(f"\nGrounding Results for '{part}': {len(grounding_claims)} claims processed")
                
            except Exception as e:
                logger.error(f"GroundingAgent execution failed for part '{part}': {e}")
                import traceback
                traceback.print_exc()
                # 继续处理其他parts
                continue
    
    # 11. 执行EvaluationAgent (idea + reports + SearchResults -> evaluation)
    print("\n" + "=" * 80)
    print("STEP 5: EvaluationAgent - Idea + Reports + SearchResults -> Evaluation")
    print("=" * 80)
    
    # 初始化变量
    evaluation_result = None
    all_evaluation_results = []
    
    # 检查缓存
    if has_cache(cached_data, "evaluation_result", 
                 lambda x: isinstance(x, dict) and x.get("evaluation_results")):
        logger.info("✓ Found cached evaluation_result, skipping EvaluationAgent execution")
        evaluation_result = cached_data["evaluation_result"]
        all_evaluation_results = evaluation_result.get("evaluation_results", [])
        print("✓ Using cached evaluation results")
        print(f"\nEvaluation Results Summary:")
        print(f"  Total personas evaluated: {len(all_evaluation_results)}")
        for idx, result in enumerate(all_evaluation_results, 1):
            eval_data = result.get("evaluation", {})
            clarity = eval_data.get("clarity", {}).get("score", 0)
            novelty = eval_data.get("novelty", {}).get("score", 0)
            feasibility = eval_data.get("feasibility", {}).get("score", 0)
            avg_score = (clarity + novelty + feasibility) / 3.0
            print(f"  Persona {idx}: Avg Score = {avg_score:.2f}/10")
    else:
        # 加载personas（如果没有personas，使用空列表）
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
            logger.warning("No personas loaded, evaluation will be limited")
            personas = []
        
        if not personas:
            logger.warning("No personas available, skipping EvaluationAgent")
        else:
            logger.info(f"Loaded {len(personas)} personas for evaluation")
            
            # 准备基础context
            base_evaluation_context = {
                "idea": idea.to_dict(),
                "search_results": search_results_dict,
                "web_reports": web_reports,
                "code_reports": code_reports,
                "paper_reports": paper_reports
            }
            
            evaluation_params = {
                "temperature": 0.7
            }
            
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
                
                # 保存evaluation_result到pipeline_result.json（第四部分）
                evaluation_result = {
                    "evaluation_results": all_evaluation_results
                }
                update_pipeline_result(
                    pipeline_result_path,
                    evaluation_result=evaluation_result
                )
                logger.info(f"Saved evaluation_result to pipeline_result.json")
                
                # 更新缓存数据
                cached_data["evaluation_result"] = evaluation_result
                
                # 打印EvaluationAgent的中间结果
                logger.info("=" * 80)
                logger.info("EvaluationAgent Output:")
                logger.info("=" * 80)
                logger.info(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
                print(f"\nEvaluation Results Summary:")
                print(f"  Total personas evaluated: {len(all_evaluation_results)}")
                for idx, result in enumerate(all_evaluation_results, 1):
                    eval_data = result["evaluation"]
                    clarity = eval_data.get("clarity", {}).get("score", 0)
                    novelty = eval_data.get("novelty", {}).get("score", 0)
                    feasibility = eval_data.get("feasibility", {}).get("score", 0)
                    avg_score = (clarity + novelty + feasibility) / 3.0
                    print(f"  Persona {idx}: Avg Score = {avg_score:.2f}/10")
                    
            except Exception as e:
                logger.error(f"EvaluationAgent execution failed: {e}")
                import traceback
                traceback.print_exc()
                evaluation_result = None
                all_evaluation_results = []
    
    # 12. 执行ReportAgent (EvaluationResults -> review reports)
    print("\n" + "=" * 80)
    print("STEP 6: ReportAgent - EvaluationResults -> Review Reports")
    print("=" * 80)
    
    if all_evaluation_results:
        # 检查review_reports缓存
        evaluation_reports = None
        if evaluation_result and "review_reports" in evaluation_result:
            if isinstance(evaluation_result["review_reports"], list) and len(evaluation_result["review_reports"]) > 0:
                logger.info("✓ Found cached review_reports, skipping ReportAgent execution")
                evaluation_reports = evaluation_result["review_reports"]
                print("✓ Using cached review reports")
                print(f"\nReview Report Counts:")
                print(f"  Total Review Reports: {len(evaluation_reports)}")
                for idx, review_report in enumerate(evaluation_reports, 1):
                    full_report = review_report.get("full_review_report", "")
                    if full_report:
                        print(f"  Review Report {idx}: {len(full_report)} characters")
        
        # 检查final_report缓存（从pipeline_result.json中读取）
        final_report = None
        pipeline_data = load_pipeline_result(pipeline_result_path)
        if "final_report" in pipeline_data and pipeline_data["final_report"]:
            final_report = pipeline_data["final_report"]
            logger.info("✓ Found cached final_report")
            print("\n" + "=" * 80)
            print("FINAL REPORT (from cache)")
            print("=" * 80)
            print(final_report)
            print("=" * 80)
            # 更新缓存数据
            cached_data["final_report"] = final_report
        
        if evaluation_reports is None:
            try:
                logger.info("Executing ReportAgent for review reports...")
                
                # 准备review report的context（模式1：只有evaluation_results）
                # 如果有 future_papers，也添加到 context 中
                review_report_context = {
                    "evaluation_results": all_evaluation_results  # 直接传入evaluation_results列表
                }
                # 从缓存中获取 future_papers（如果有）
                future_papers = cached_data.get("future_papers", [])
                if future_papers:
                    review_report_context["future_papers"] = future_papers
                    logger.info(f"Added {len(future_papers)} future papers to review report context")
                
                review_report_params = {
                    "temperature": 0.7
                }
                
                review_report_result = await report_agent.execute(review_report_context, review_report_params)
                
                # 获取review reports
                evaluation_reports = review_report_result.get("evaluation_reports", [])
                
                # 获取final_report（如果有）
                final_report = review_report_result.get("final_report")
                
                # 保存review_reports到pipeline_result.json（第五部分）
                if "review_reports" not in load_pipeline_result(pipeline_result_path).get("evaluation_result", {}):
                    # 更新evaluation_result，添加review_reports
                    updated_evaluation_result = evaluation_result.copy()
                    updated_evaluation_result["review_reports"] = evaluation_reports
                    update_pipeline_result(
                        pipeline_result_path,
                        evaluation_result=updated_evaluation_result
                    )
                    # 更新缓存数据
                    cached_data["evaluation_result"] = updated_evaluation_result
                else:
                    # 如果已经有review_reports字段，单独更新
                    data = load_pipeline_result(pipeline_result_path)
                    if "evaluation_result" in data:
                        data["evaluation_result"]["review_reports"] = evaluation_reports
                        save_pipeline_result(pipeline_result_path, data)
                        # 更新缓存数据
                        cached_data["evaluation_result"] = data["evaluation_result"]
                
                logger.info(f"Saved review_reports to pipeline_result.json")
                
                # 打印Review Report的中间结果
                logger.info("=" * 80)
                logger.info("ReportAgent Output (Review Reports):")
                logger.info("=" * 80)
                logger.info(f"Review Reports: {len(evaluation_reports)} reports generated")
                print(f"\nReview Report Counts:")
                print(f"  Total Review Reports: {len(evaluation_reports)}")
                
                # 打印review report预览
                for idx, review_report in enumerate(evaluation_reports, 1):
                    full_report = review_report.get("full_review_report", "")
                    if full_report:
                        logger.info(f"Review Report {idx} Preview: {full_report[:300]}..." if len(full_report) > 300 else f"Review Report {idx} Preview: {full_report}")
                        print(f"  Review Report {idx}: {len(full_report)} characters")
                
                # 处理final_report（如果有）
                if final_report:
                    logger.info("=" * 80)
                    logger.info("Final Report Generated")
                    logger.info("=" * 80)
                    logger.info(f"Final Report Length: {len(final_report)} characters")
                    
                    # 打印final_report
                    print("\n" + "=" * 80)
                    print("FINAL REPORT")
                    print("=" * 80)
                    print(final_report)
                    print("=" * 80)
                    
                    # 保存final_report到pipeline_result.json
                    update_pipeline_result(
                        pipeline_result_path,
                        final_report=final_report
                    )
                    logger.info(f"Saved final_report to pipeline_result.json")
                    
                    # 更新缓存数据
                    cached_data["final_report"] = final_report
                else:
                    logger.info("No final_report generated (no future_papers provided)")
                
            except Exception as e:
                logger.error(f"ReportAgent execution failed for review reports: {e}")
                import traceback
                traceback.print_exc()
                # 不抛出异常，继续执行
    else:
        logger.warning("No evaluation results available, skipping review report generation")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)



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

