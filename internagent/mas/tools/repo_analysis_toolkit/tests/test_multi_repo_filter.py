"""
测试多仓库筛选机制

这个测试程序演示了如何：
1. 从 GitHub 下载多个仓库
2. 使用 FilterAndRankRepos 进行筛选和排序
3. 配置自定义 LLM 客户端进行任务匹配评分
"""

import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from repo_analysis import FilterAndRankRepos, create_task_dict
from repo_analysis.multi_repo_filter import print_ranking_results


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
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                print(f"⚠️  LLM 返回空内容")
                return {}
            
            # 如果要求 JSON 格式，尝试解析
            if json_format:
                import re
                # 先处理 content，去除 markdown 代码块标记
                cleaned_content = content.strip()
                # 去除开头的 ```json 或 ``` 标记
                cleaned_content = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_content, flags=re.MULTILINE)
                # 去除结尾的 ``` 标记
                cleaned_content = re.sub(r'\n?```\s*$', '', cleaned_content, flags=re.MULTILINE)
                cleaned_content = cleaned_content.strip()
                
                # 尝试解析清理后的内容
                try:
                    parsed = json.loads(cleaned_content)
                    return parsed
                except json.JSONDecodeError as e:
                    # 如果清理后还是失败，尝试用正则表达式提取 JSON
                    print(f"⚠️  JSON 解析失败: {e}")
                    json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group())
                        except:
                            pass
                    print(f"原始内容: {cleaned_content[:200]}...")
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


def test_multi_repo_filter():
    """
    测试多仓库筛选机制
    """
    print("=" * 80)
    print("测试多仓库筛选和排序机制")
    print("=" * 80)
    
    # 1. 配置仓库列表
    repo_urls = [
        "https://github.com/FranxYao/chain-of-thought-hub",
        "https://github.com/Timothyxxx/Chain-of-ThoughtsPapers",
        "https://github.com/venkatesh-kulkarni/Ante-hoc-Explanations",
        "https://github.com/HanjaeKim98/CoT",
        "https://github.com/arumpuri/prompting_with_gemini",
        "https://github.com/Aiden0526/SymbCoT",
        "https://github.com/mbzuai-nlp/finchain",
        "https://github.com/atfortes/LLMSymbolicReasoningBench",
        "https://github.com/teacherpeterpan/Logic-LLM",
        "https://github.com/yaotingwangofficial/Awesome-MCoT",
        "https://github.com/atfortes/Awesome-LLM-Reasoning"
    ]
    
    # 2. 任务描述 - Faithful CoT
    task_description = """Faithful CoT uses a two-stage approach: (1) a language model translates the natural language query into a hybrid chain combining brief explanations with executable symbolic code, and (2) an external deterministic solver executes that code to produce the answer. 
    The symbolic components—written in languages such as Python, Datalog, or PDDL—are crafted so that executing them fully determines the final output. 
    Because the solver, not the model, computes the answer, the chain becomes faithful by construction. 
    This framework is modular, allowing different symbolic languages and solvers to be swapped in depending on the task."""
    
    # 创建任务字典
    task = create_task_dict(
        description=task_description,
        keywords=[
            "faithful", "chain of thought", "CoT", "symbolic reasoning",
            "executable code", "solver", "Python", "Datalog", "PDDL",
            "deterministic", "modular", "reasoning", "symbolic language"
        ],
        task_type="faithful_cot_reasoning"
    )
    
    # 3. 获取 LLM 配置
    api_key = os.environ.get("DS_API_KEY")
    if not api_key:
        raise ValueError("❌ 未找到环境变量 DS_API_KEY，请先设置该环境变量")
    
    base_url = "https://www.dmxapi.cn/v1"
    model_name = "DeepSeek-V3.2"
    
    print(f"\n📋 配置信息:")
    print(f"  - 仓库数量: {len(repo_urls)}")
    print(f"  - LLM API: {base_url}")
    print(f"  - 模型: {model_name}")
    print(f"  - API Key: {'*' * 20}...{api_key[-4:] if len(api_key) > 4 else '****'}")
    
    print(f"\n📝 任务描述:")
    print(f"  {task_description[:200]}...")
    print(f"  关键词: {', '.join(task['keywords'][:5])}...")
    
    # 4. 在 result 目录下创建临时目录用于下载仓库
    script_dir = Path(__file__).parent
    result_dir = script_dir.parent / "result"
    result_dir.mkdir(exist_ok=True)  # 确保 result 目录存在
    
    # 创建带时间戳的子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = result_dir / f"multi_repo_filter_{timestamp}"
    temp_dir.mkdir(exist_ok=True)
    temp_dir = str(temp_dir)
    
    print(f"\n📁 结果目录: {temp_dir}")
    
    try:
        # 5. 下载所有仓库
        print(f"\n📥 开始下载 {len(repo_urls)} 个仓库...")
        repo_paths = []
        failed_repos = []
        
        for i, repo_url in enumerate(repo_urls, 1):
            print(f"\n[{i}/{len(repo_urls)}] 处理仓库: {repo_url}")
            try:
                repo_path = download_github_repo(repo_url, temp_dir)
                repo_paths.append(repo_path)
            except Exception as e:
                print(f"❌ 下载失败，跳过: {e}")
                failed_repos.append(repo_url)
                continue
        
        print(f"\n✅ 成功下载 {len(repo_paths)}/{len(repo_urls)} 个仓库")
        if failed_repos:
            print(f"⚠️  失败的仓库 ({len(failed_repos)} 个):")
            for repo in failed_repos:
                print(f"  - {repo}")
        
        if not repo_paths:
            raise ValueError("❌ 没有成功下载任何仓库")
        
        # 6. 创建 LLM 客户端
        print(f"\n🔧 创建 LLM 客户端...")
        llm_client = create_dmx_llm_client(api_key, base_url, model_name)
        
        # 7. 创建筛选器
        print(f"\n🚀 创建多仓库筛选器...")
        filter_ranker = FilterAndRankRepos(
            llm_client=llm_client,
            max_workers=3,  # 并行处理3个仓库（避免API限流）
            max_tokens_per_repo=4000  # 每个仓库的最大token数
        )
        
        # 8. 执行筛选和排序
        print(f"\n⚙️  开始分析和排序仓库...")
        print("  (这可能需要较长时间，请耐心等待...)")
        print(f"  - 将分析 {len(repo_paths)} 个仓库")
        print(f"  - 使用 LLM 进行任务匹配评分")
        print(f"  - 返回 top-5 最相关的仓库\n")
        
        top_repos = filter_ranker.filter_and_rank(
            task=task,
            repo_paths=repo_paths,
            top_k=5,  # 返回前5名
            use_llm=True,  # 使用LLM评分
            batch_size=3  # 每次LLM调用评分3个仓库
        )
        
        # 9. 显示结果
        print("\n" + "=" * 80)
        print("筛选和排序完成！")
        print("=" * 80)
        
        print_ranking_results(top_repos)
        
        # 10. 保存结果到文件
        output_file = os.path.join(temp_dir, "filtered_ranked_results.json")
        print(f"\n💾 保存筛选结果到: {output_file}")
        
        filter_ranker.save_results(
            results=top_repos,
            output_file=output_file,
            include_full_context=False  # 不包含完整上下文以节省空间
        )
        
        print(f"✅ 结果已保存到: {output_file}")
        
        # 11. 保存详细结果（包含完整上下文）
        detailed_output_file = os.path.join(temp_dir, "detailed_results.json")
        print(f"\n💾 保存详细结果（包含上下文）到: {detailed_output_file}")
        
        filter_ranker.save_results(
            results=top_repos,
            output_file=detailed_output_file,
            include_full_context=True  # 包含完整上下文
        )
        
        print(f"✅ 详细结果已保存到: {detailed_output_file}")
        
        # 12. 保存任务和仓库信息摘要
        summary = {
            'task': {
                'description': task['task_description'],
                'keywords': task['keywords'],
                'task_type': task['task_type']
            },
            'repositories': {
                'total': len(repo_urls),
                'successful': len(repo_paths),
                'failed': len(failed_repos),
                'failed_urls': failed_repos
            },
            'results': {
                'top_k': len(top_repos),
                'ranking': [
                    {
                        'rank': i + 1,
                        'repo_name': repo.get('repo_name'),
                        'repo_path': repo.get('repo_path'),
                        'relevance_score': repo.get('relevance_score'),
                        'scoring_method': repo.get('scoring_method')
                    }
                    for i, repo in enumerate(top_repos)
                ]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(temp_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 摘要已保存到: {summary_file}")
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        print(f"\n📁 所有文件保存在: {temp_dir}")
        print(f"  - 筛选结果: {output_file}")
        print(f"  - 详细结果: {detailed_output_file}")
        print(f"  - 摘要: {summary_file}")
        print(f"  - 仓库目录: {temp_dir}")
        
        return top_repos
        
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
        test_multi_repo_filter()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()



