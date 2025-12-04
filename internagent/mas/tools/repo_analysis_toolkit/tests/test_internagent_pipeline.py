"""
测试 InternAgent 仓库的 Pipeline 分析

这个测试程序演示了如何：
1. 从 GitHub 下载仓库
2. 配置自定义 LLM 客户端
3. 设计任务并运行 Pipeline 生成上下文
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from repo_analysis import RepoContextPipeline, create_task_dict


def download_github_repo(repo_url: str, target_dir: str) -> str:
    """
    从 GitHub 下载仓库到指定目录
    
    Args:
        repo_url: GitHub 仓库 URL (例如: https://github.com/user/repo)
        target_dir: 目标目录路径
        
    Returns:
        下载后的仓库本地路径
    """
    # 从 URL 中提取仓库路径
    # 处理 tree/main 或 tree/branch 的情况
    if '/tree/' in repo_url:
        repo_url = repo_url.split('/tree/')[0]
    
    # 确保 URL 以 .git 结尾
    if not repo_url.endswith('.git'):
        repo_url = repo_url + '.git'
    
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    local_path = os.path.join(target_dir, repo_name)
    
    # 如果目录已存在，先删除
    if os.path.exists(local_path):
        print(f"⚠️  目录 {local_path} 已存在，正在删除...")
        shutil.rmtree(local_path)
    
    print(f"📥 正在从 {repo_url} 下载仓库到 {local_path}...")
    
    try:
        # 使用 git clone 下载
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, local_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ 仓库下载成功: {local_path}")
        return local_path
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        print(f"错误输出: {e.stderr}")
        raise
    except FileNotFoundError:
        print("❌ 错误: 未找到 git 命令。请确保已安装 git。")
        raise


def create_dmx_llm_client(api_key: str, base_url: str, model_name: str):
    """
    创建 DMX API 的 LLM 客户端函数
    
    Args:
        api_key: API 密钥（从环境变量获取）
        base_url: API 基础 URL
        model_name: 模型名称
        
    Returns:
        LLM 客户端函数
    """
    import requests
    
    def llm_client(messages: List[Dict], json_format: bool = False) -> Dict:
        """
        LLM 客户端函数
        
        Args:
            messages: 消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}]
            json_format: 是否要求返回 JSON 格式
            
        Returns:
            Dict 或解析后的 JSON 对象
        """
        url = f"{base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.7,
        }
        
        # 如果要求 JSON 格式，添加 response_format
        if json_format:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            print(f"🤖 调用 LLM API: {model_name}...")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                print(f"⚠️  LLM 返回空内容")
                return {}
            
            # 如果要求 JSON 格式，尝试解析
            if json_format:
                try:
                    parsed = json.loads(content)
                    return parsed
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON 解析失败: {e}")
                    print(f"原始内容: {content[:200]}...")
                    # 尝试从内容中提取 JSON（可能被代码块包裹）
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group())
                        except:
                            pass
                    return {"error": "Failed to parse JSON", "content": content}
            
            # 非 JSON 格式，返回内容字符串（包装成字典以保持兼容性）
            return {"content": content}
            
        except requests.exceptions.RequestException as e:
            print(f"❌ LLM API 调用失败: {e}")
            raise
        except Exception as e:
            print(f"❌ 处理 LLM 响应时出错: {e}")
            raise
    
    return llm_client


def test_internagent_pipeline():
    """
    测试 InternAgent 仓库的 Pipeline 分析
    """
    print("=" * 80)
    print("测试 InternAgent 仓库 Pipeline 分析")
    print("=" * 80)
    
    # 1. 配置参数
    repo_url = "https://github.com/Alpha-Innovator/InternAgent/tree/main"
    
    "https://github.com/FranxYao/chain-of-thought-hub"
    "https://github.com/Timothyxxx/Chain-of-ThoughtsPapers"
    "https://github.com/venkatesh-kulkarni/Ante-hoc-Explanations"
    "https://github.com/HanjaeKim98/CoT"
    "https://github.com/arumpuri/prompting_with_gemini"
    "https://github.com/Aiden0526/SymbCoT"
    "https://github.com/mbzuai-nlp/finchain"
    "https://github.com/atfortes/LLMSymbolicReasoningBench"
    "https://github.com/teacherpeterpan/Logic-LLM"
    "https://github.com/yaotingwangofficial/Awesome-MCoT"
    "https://github.com/atfortes/Awesome-LLM-Reasoning"
    
    # 2. 获取 LLM 配置
    api_key = os.environ.get("DS_API_KEY")
    if not api_key:
        raise ValueError("❌ 未找到环境变量 DS_API_KEY，请先设置该环境变量")
    
    base_url = "https://www.dmxapi.cn/v1"
    model_name = "DeepSeek-V3.2"
    
    print(f"\n📋 配置信息:")
    print(f"  - 仓库 URL: {repo_url}")
    print(f"  - LLM API: {base_url}")
    print(f"  - 模型: {model_name}")
    print(f"  - API Key: {'*' * 20}...{api_key[-4:] if len(api_key) > 4 else '****'}")
    
    # 3. 在 result 目录下创建临时目录用于下载仓库
    script_dir = Path(__file__).parent
    result_dir = script_dir.parent / "result"
    result_dir.mkdir(exist_ok=True)  # 确保 result 目录存在
    
    # 创建带时间戳的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = result_dir / f"internagent_analysis_{timestamp}"
    temp_dir.mkdir(exist_ok=True)
    temp_dir = str(temp_dir)
    
    print(f"\n📁 结果目录: {temp_dir}")
    
    try:
        # 4. 下载仓库
        repo_path = download_github_repo(repo_url, temp_dir)
        
        # 5. 创建 LLM 客户端
        print(f"\n🔧 创建 LLM 客户端...")
        llm_client = create_dmx_llm_client(api_key, base_url, model_name)
        
        # 6. 设计任务
        # 基于 InternAgent 的特点，这是一个科学研究 agent 系统
        task = create_task_dict(
            description="分析一个用于科学研究的 AI Agent 系统，该系统能够从假设生成到验证的闭环自动化流程",
            keywords=[
                "agent", "scientific research", "hypothesis", "verification",
                "automation", "multi-agent", "experiment", "pipeline",
                "machine learning", "deep learning", "code generation"
            ],
            task_type="scientific_research_agent"
        )
        
        print(f"\n📝 任务描述:")
        print(f"  {task['task_description']}")
        print(f"  关键词: {', '.join(task['keywords'][:5])}...")
        
        # 7. 创建 Pipeline
        print(f"\n🚀 创建 Pipeline...")
        pipeline = RepoContextPipeline(
            repo_path=repo_path,
            llm_client=llm_client
        )
        
        # 8. 运行 Pipeline
        print(f"\n⚙️  运行 Pipeline...")
        print("  (这可能需要几分钟时间，请耐心等待...)")
        
        results = pipeline.run(
            task=task,
            max_tokens=10000,  # 稍微增加 token 限制以获取更多上下文
            output_file=None,  # 不保存到文件，只在内存中处理
            format='json'
        )
        
        # 9. 显示结果摘要
        print("\n" + "=" * 80)
        print("Pipeline 执行完成！")
        print("=" * 80)
        print("\n" + pipeline.get_summary())
        
        # 10. 显示任务相关性评分
        if results.get('task_relevance'):
            print("\n" + "=" * 80)
            print("任务相关性评分")
            print("=" * 80)
            task_score = results['task_relevance']
            print(f"  总体相关性分数: {task_score.get('relevance_score', 0):.3f}")
            print(f"  评分方法: {task_score.get('scoring_method', 'N/A')}")
            
            if 'dimensions' in task_score:
                print(f"\n  各维度评分:")
                dimensions = task_score['dimensions']
                if isinstance(dimensions, dict):
                    for dim, score in dimensions.items():
                        if dim != 'Overall Score':
                            print(f"    - {dim}: {score}")
                    if 'Overall Score' in dimensions:
                        print(f"    - Overall Score: {dimensions['Overall Score']}/10")
        
        # 11. 显示关键模块（前5个）
        if results.get('key_modules'):
            print("\n" + "=" * 80)
            print("关键模块 (Top 5)")
            print("=" * 80)
            for i, module in enumerate(results['key_modules'][:5], 1):
                print(f"\n  {i}. {module.get('path', 'N/A')}")
                print(f"     重要性分数: {module.get('importance_score', 0):.2f}")
        
        # 12. 保存完整结果到文件
        output_file = os.path.join(temp_dir, "pipeline_results.json")
        print(f"\n💾 保存完整结果到: {output_file}")
        
        # 准备导出数据（不包含完整的 context 字符串，因为可能很大）
        export_data = {
            'repo_path': results['repo_path'],
            'analysis': results['analysis'],
            'key_modules': results['key_modules'],
            'task_relevance': results['task_relevance'],
            'context_summary': {
                'total_components': len(results.get('context', {}).get('core_components', [])),
                'overview': results.get('context', {}).get('repository_overview', {})
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存到: {output_file}")
        
        # 13. 导出上下文字符串到文件
        context_file = os.path.join(temp_dir, "context_string.txt")
        print(f"\n💾 导出上下文字符串到: {context_file}")
        
        if pipeline.context_builder:
            context_str = pipeline.context_builder.export_to_string()
            with open(context_file, 'w', encoding='utf-8') as f:
                f.write(context_str)
            print(f"✅ 上下文已保存到: {context_file}")
            print(f"   上下文长度: {len(context_str)} 字符")
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        print(f"\n📁 所有文件保存在: {temp_dir}")
        print(f"  - 仓库路径: {repo_path}")
        print(f"  - 结果文件: {output_file}")
        print(f"  - 上下文文件: {context_file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 结果目录保留，不自动删除
        print(f"\n💡 提示: 所有结果已保存在 {temp_dir}")


def load_env_from_file(env_file: str) -> None:
    """
    从 .env 文件中读取环境变量并设置到 os.environ
    
    Args:
        env_file: .env 文件路径
    """
    env_path = Path(env_file)
    
    if not env_path.exists():
        print(f"⚠️  环境变量文件不存在: {env_file}")
        return
    
    print(f"📖 从 {env_file} 读取环境变量...")
    
    loaded_count = 0
    with open(env_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 去除首尾空白
            line = line.strip()
            
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            
            # 解析 KEY=VALUE 格式
            if '=' in line:
                # 分割键值对
                parts = line.split('=', 1)  # 只分割第一个 =，因为值中可能包含 =
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # 去除值两端的引号（如果有）
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    
                    # 设置环境变量
                    if key and value:
                        # 如果环境变量已存在，打印提示但仍然覆盖（文件配置优先）
                        if key in os.environ:
                            old_value = os.environ[key]
                            os.environ[key] = value
                            # 只显示前几个字符，避免敏感信息泄露
                            masked_old = old_value[:10] + "..." if len(old_value) > 10 else old_value
                            masked_new = value[:10] + "..." if len(value) > 10 else value
                            print(f"  🔄 {key}: 已覆盖 ({masked_old} → {masked_new})")
                            loaded_count += 1
                        else:
                            os.environ[key] = value
                            masked_value = value[:20] + "..." if len(value) > 20 else value
                            print(f"  ✅ {key} = {masked_value}")
                            loaded_count += 1
                else:
                    print(f"  ⚠️  第 {line_num} 行格式错误，跳过: {line[:50]}")
            else:
                print(f"  ⚠️  第 {line_num} 行格式错误，跳过: {line[:50]}")
    
    print(f"✅ 成功加载 {loaded_count} 个环境变量")


def main():
    """主函数"""
    # 从 LLM.env 文件加载环境变量
    script_dir = Path(__file__).parent
    env_file = script_dir / "LLM.env"
    load_env_from_file(str(env_file))
    
    try:
        test_internagent_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

