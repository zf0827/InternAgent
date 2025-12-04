#!/usr/bin/env python3
"""
Agent Pipeline Test - 串联测试Agent工作流

测试流程:
1. ExtractionAgent: PDF -> idea (提取结构化信息，使用Idea.from_lists()创建Idea对象)
2. ResearchAgentV2: idea -> SearchResults (深度搜索，返回三个平台的结果)
3. ReportAgent: SearchResults -> reports (生成报告，返回三个报告列表)
4. GroundingAgent: reports + claims -> grounding_results (证据验证，对六个部分分别处理)
5. EvaluationAgent: SKIPPED (暂时跳过)

参考: quick_test_deepresearch.py 和 test_report_agent.py
"""

import logging
import sys
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

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
        Dict with keys: search_results_dict, reports_data, grounding_result
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
                return data
        except Exception as e:
            logger.warning(f"Failed to load existing pipeline_result.json: {e}, creating new file")
            return {
                "search_results_dict": None,
                "reports_data": None,
                "grounding_result": {}
            }
    else:
        return {
            "search_results_dict": None,
            "reports_data": None,
            "grounding_result": {}
        }


def save_pipeline_result(file_path: Path, data: Dict[str, Any]):
    """
    Save pipeline_result.json with proper formatting.
    
    Args:
        file_path: Path to save the file
        data: Dict with keys: search_results_dict, reports_data, grounding_result
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved pipeline_result to {file_path}")


def update_pipeline_result(file_path: Path, 
                           search_results_dict: Optional[Dict[str, Any]] = None,
                           reports_data: Optional[Dict[str, Any]] = None,
                           grounding_result_part: Optional[str] = None,
                           grounding_result_claims: Optional[List[Dict[str, Any]]] = None):
    """
    Update pipeline_result.json with new data (append mode).
    
    Args:
        file_path: Path to pipeline_result.json
        search_results_dict: New search_results_dict to save
        reports_data: New reports_data to save
        grounding_result_part: Part name for grounding_result (e.g., "basic_idea")
        grounding_result_claims: List of claim dicts for the part
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
    
    # ReportAgent配置
    report_config = {
        "name": "ReportAgent",
        "model_provider": "dsr1",
        "temperature": 0.7,
        "system_prompt": "You are a helpful assistant that generates comprehensive reports.",
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
    
    
    # 4. 初始化pipeline_result.json文件路径
    pipeline_result_path = Path("/home/weiyunxiang/yunx/IdeaEvaluation/InternAgent/cache/pipeline_result.json")
    pipeline_result_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    research_context = {
        "idea": idea.to_dict()
    }
    research_params = {}
    
    try:
        logger.info("Executing ResearchAgentV2...")
        research_result = await research_agent.execute(research_context, research_params)
        
        search_results_dict = research_result.get("search_results", {})
        search_results = SearchResults.from_dict(search_results_dict)
        
        # 保存search_results到pipeline_result.json（第一部分）
        update_pipeline_result(
            pipeline_result_path,
            search_results_dict=search_results_dict
        )
        logger.info(f"Saved search_results_dict to pipeline_result.json")
        
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
    
    # 9. 执行ReportAgent (SearchResults -> reports)
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
        
        # 获取三个报告列表
        web_reports = report_result.get("web_reports", [])
        code_reports = report_result.get("code_reports", [])
        paper_reports = report_result.get("paper_reports", [])
        
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
        logger.info(f"Saved reports_data to pipeline_result.json")
        
        # 打印ReportAgent的中间结果
        logger.info("=" * 80)
        logger.info("ReportAgent Output:")
        logger.info("=" * 80)
        logger.info(f"Web Reports: {len(web_reports)} reports")
        logger.info(f"Code Reports: {len(code_reports)} reports")
        logger.info(f"Paper Reports: {len(paper_reports)} reports")
        print(f"\nReport Counts:")
        print(f"  Web Reports: {len(web_reports)}")
        print(f"  Code Reports: {len(code_reports)}")
        print(f"  Paper Reports: {len(paper_reports)}")
        
        # 打印报告预览
        if web_reports:
            first_web_report = web_reports[0].get("content", {})
            report_content = first_web_report.get("report_content", "")
            logger.info(f"First Web Report Preview: {report_content[:500]}..." if len(report_content) > 500 else f"First Web Report Preview: {report_content}")
        if code_reports:
            first_code_report = code_reports[0].get("content", {})
            report_content = first_code_report.get("report_content", "")
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
    
    for part in parts:
        # 从idea对象获取对应的*_list字段作为claims
        claims_list = getattr(idea, f"{part}_list", [])
        if not claims_list:
            logger.info(f"Skipping part '{part}': no claims found")
            continue
        
        logger.info(f"Processing part: {part} ({len(claims_list)} claims)")
        
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
    
    # 11. 跳过EvaluationAgent
    print("\n" + "=" * 80)
    print("STEP 5: EvaluationAgent - SKIPPED (not implemented yet)")
    print("=" * 80)
    logger.info("EvaluationAgent execution skipped as per requirements")
    print("EvaluationAgent execution skipped. Pipeline completed up to grounding step.")



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

