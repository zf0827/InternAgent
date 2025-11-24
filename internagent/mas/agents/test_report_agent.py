#!/usr/bin/env python3
"""
Test program to run ReportAgent's execute function.
This script loads environment variables, creates a ReportAgent instance,
and runs the execute function to print generated reports.
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add the parent directories to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add mas/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add internagent/ to path

# Now we can import from the local modules
from ..models.model_factory import ModelFactory
from .agent_factory import AgentFactory


def load_environment_variables():
    """Load environment variables from LLM.env file."""
    llm_env_path = "/home/weiyunxiang/yunx/InternAgent/IdeaEvaluation/LLM.env"
    
    if not os.path.exists(llm_env_path):
        print(f"Warning: LLM.env file not found at {llm_env_path}")
        return False
    
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
    except Exception as e:
        print(f"Error loading LLM.env file: {e}")
        return False
    
    # Verify that required API keys are set
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Warning: OPENAI_API_KEY not found in environment")
        return False
    ds_key = os.getenv("DS_API_KEY")
    if not ds_key:
        print("Warning: DS_API_KEY not found in environment")
        return False
    print(f"Loaded environment variables from {llm_env_path}")
    print(f"OPENAI_API_KEY is set: {'Yes' if openai_key else 'No'}")
    print(f"OPENAI_API_BASE_URL: {os.getenv('OPENAI_API_BASE_URL')}")
    print(f"DS_API_KEY is set: {'Yes' if ds_key else 'No'}")
    print(f"DS_API_BASE_URL: {os.getenv('DS_API_BASE_URL')}")
    
    return True


async def create_report_agent():
    """Create a ReportAgent instance with proper configuration."""
    
    # Model configuration
    model_config = {
        "provider": "openai",
        "model_name": "deepseek-v3",
        "api_key": os.getenv("DS_API_KEY"),
        "base_url": os.getenv("DS_API_BASE_URL"),
        # "model_name": "gpt-4o-mini",
        # "api_key": os.getenv("OPENAI_API_KEY"),
        # "base_url": os.getenv("OPENAI_API_BASE_URL"),
        "temperature": 0.7,
        "max_tokens": 10000
    }
    
    # Agent configuration
    agent_config = {
        "name": "ReportAgent",
        "description": "Agent for generating research reports",
        "system_prompt": "You are a helpful assistant that generates a long and comprehensive report of the sources attached to the idea.",
        "temperature": 0.7,
        "max_tokens": 10000,
        "model_provider": "openai",
        "max_retries": 3,
        # "model_name": "deepseek-v3",
        # "api_key": os.getenv("DS_API_KEY"),
        # "base_url": os.getenv("DS_API_BASE_URL"),
        # "model_name": "gpt-4o-mini",
        # "api_key": os.getenv("OPENAI_API_KEY"),
        # "base_url": os.getenv("OPENAI_API_BASE_URL"),
    }
    
    # Create model factory and model
    model_factory = ModelFactory()
    model = ModelFactory.create_model(model_config)
    
    # Create ReportAgent
    agent = AgentFactory.create_agent("report", agent_config, model_factory)
    
    return agent


def load_json_array(file_path: str) -> list:
    """加载json array文件"""
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"错误: 文件内容不是json array格式")
            return []
        
        return data
    except Exception as e:
        print(f"错误: 读取文件失败: {e}")
        return []


async def run_report_agent():
    """Run the ReportAgent execute function for each SearchResults in the json array."""
    try:
        # Load environment variables
        if not load_environment_variables():
            print("Failed to load environment variables")
            return
        
        # Create ReportAgent (reuse the same agent for all parts)
        print("Creating ReportAgent...")
        agent = await create_report_agent()
        print(f"Created agent: {agent.name}")
        
        # 读取过滤后的json array文件
        filtered_file = "/home/weiyunxiang/yunx/InternAgent/IdeaEvaluation/deep_search_results_by_parts_filtered_with_trees.json"
        print(f"\n正在加载过滤后的结果文件: {filtered_file}")
        
        json_array = load_json_array(filtered_file)
        
        if not json_array:
            print("错误: 无法加载过滤后的结果文件或文件为空")
            print("请先运行 test_filter_agent.py 生成过滤后的结果")
            return
        
        print(f"成功加载 {len(json_array)} 个SearchResults")
        print()
        
        # 准备参数
        params = {
            "temperature": 0.7
        }
        
        # 处理每个SearchResults
        all_results = []
        
        for idx, json_data in enumerate(json_array, 1):
            print("\n" + "="*80)
            print(f"处理第 {idx}/{len(json_array)} 个SearchResults")
            print("="*80)
            
            # 显示当前处理的part信息
            part_info = json_data.get("idea", {}).get("part", [])
            print(f"Part字段: {part_info}")
            print()
            
            # 准备context，包含search_result
            context = {
                "search_result": json_data
            }
            
            print("正在调用ReportAgent生成报告...")
            
            # Execute the agent
            result = await agent.execute(context, params)
            
            # 保存结果
            result_with_part = {
                "part": part_info,
                "web_report": result.get("web_report", ""),
                "code_report": result.get("code_report", ""),
                "paper_report": result.get("paper_report", ""),
                "metadata": result.get("metadata", {})
            }
            all_results.append(result_with_part)
            
            # 打印报告长度信息
            print("\n" + "-"*80)
            print(f"第 {idx} 个SearchResults的报告生成完成")
            print("-"*80)
            if "web_report" in result:
                print(f"Web report length: {len(result['web_report'])} characters")
            if "code_report" in result:
                print(f"Code report length: {len(result['code_report'])} characters")
            if "paper_report" in result:
                print(f"Paper report length: {len(result['paper_report'])} characters")
        
        # 打印总结
        print("\n" + "="*80)
        print("所有报告生成完成")
        print("="*80)
        print(f"共处理 {len(all_results)} 个SearchResults")
        
        # 可选：保存所有结果到文件
        output_file = "/home/weiyunxiang/yunx/InternAgent/IdeaEvaluation/report_results_con.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n所有报告已保存到: {output_file}")
        except Exception as e:
            print(f"\n警告: 保存报告结果失败: {e}")
        
    except Exception as e:
        print(f"Error running ReportAgent: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("ReportAgent Test Program")
    print("="*80)
    print("该程序将读取过滤后的json array文件，为每个SearchResults生成报告")
    print("="*80)
    
    # Check if the filtered file exists
    filtered_file = "/home/weiyunxiang/yunx/InternAgent/IdeaEvaluation/deep_search_results_by_parts_filtered.json"
    if os.path.exists(filtered_file):
        print(f"\n过滤后的结果文件 found: {filtered_file}")
        try:
            with open(filtered_file, 'r') as f:
                filtered_data = json.load(f)
            if isinstance(filtered_data, list):
                print(f"文件包含 {len(filtered_data)} 个SearchResults")
            else:
                print(f"警告: 文件内容不是json array格式")
        except Exception as e:
            print(f"警告: 无法读取文件: {e}")
    else:
        print(f"\n警告: 过滤后的结果文件不存在: {filtered_file}")
        print("请先运行 test_filter_agent.py 生成过滤后的结果")
    
    print("\nStarting ReportAgent execution...")
    
    # Run the async function
    asyncio.run(run_report_agent())