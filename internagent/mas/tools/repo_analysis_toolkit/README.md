# Repo Analysis Toolkit

一个用于分析代码仓库并提取上下文信息的工具包，从 RepoMaster 项目中提取和重构而来。

## 功能特性

- **静态分析与视图生成**：自动分析代码仓库，生成三种视图
  - HCT (层级组件树): Package → Module → Class → Function
  - MCG (模块调用图): 模块间的依赖关系
  - FCG (函数调用图): 函数间的调用关系

- **组件重要性评分**：基于多个维度对代码组件进行评分
  - 使用频率 (被导入/调用次数)
  - 模块间引用关系 (PageRank, 介数中心性)
  - 代码复杂度
  - 语义重要性 (关键词匹配)
  - Git历史 (提交频率、最近修改时间)

- **上下文构建**：提取核心模块摘要和调用链，构建LLM友好的上下文

- **任务关联评分**：评估仓库与特定任务的相关性
  - 支持LLM评分 (精确但需要API)
  - 支持启发式评分 (快速但相对粗糙)

- **多仓库筛选**：批量处理多个仓库，按相关性排序

## 安装

### 基础安装

```bash
cd repo_analysis_toolkit
pip install -e .
```

### 完整安装 (包含代码结构提取功能)

```bash
pip install -e ".[full]"
```

### 开发安装

```bash
pip install -e ".[dev]"
```

## 快速开始

### 1. 简单的仓库分析

```python
from repo_analysis import SimplePipeline

# 创建简单pipeline
pipeline = SimplePipeline('/path/to/your/repo')

# 获取上下文
context = pipeline.get_context(max_tokens=8000, format='dict')

# 获取关键模块
key_modules = pipeline.get_key_modules(top_k=10)

for module in key_modules:
    print(f"{module['path']}: {module['importance_score']:.2f}")
```

### 2. 完整的Pipeline工作流

```python
from repo_analysis import RepoContextPipeline

# 初始化pipeline
pipeline = RepoContextPipeline('/path/to/your/repo')

# 运行完整分析
results = pipeline.run(
    task={'task_description': '实现一个机器学习模型'},
    max_tokens=8000,
    output_file='output.json'
)

# 查看摘要
print(pipeline.get_summary())
```

### 3. 任务关联评分

```python
from repo_analysis import RepoContextPipeline, TaskMatcher, create_task_dict

# 分析仓库
pipeline = RepoContextPipeline('/path/to/repo')
pipeline.analyze()
pipeline.score_importance()
context = pipeline.build_context()

# 创建任务
task = create_task_dict(
    description="构建一个用于图像分类的深度学习模型",
    keywords=["deep learning", "image", "classification", "CNN"],
    task_type="machine_learning"
)

# 评分 (启发式方法，无需LLM)
matcher = TaskMatcher()
score = matcher.match_single_repo(task, context_str, use_llm=False)
print(f"相关性分数: {score['relevance_score']:.2f}")
```

### 4. 多仓库筛选和排序

```python
from repo_analysis import FilterAndRankRepos

# 创建筛选器
filter_ranker = FilterAndRankRepos(max_workers=4)

# 仓库列表
repo_paths = [
    '/path/to/repo1',
    '/path/to/repo2',
    '/path/to/repo3',
]

# 筛选和排序
task = "实现一个PyTorch图像分类模型"
top_repos = filter_ranker.filter_and_rank(
    task=task,
    repo_paths=repo_paths,
    top_k=3,
    use_llm=False  # 使用启发式评分
)

# 打印结果
from repo_analysis.multi_repo_filter import print_ranking_results
print_ranking_results(top_repos)
```

### 5. 使用LLM进行精确评分

如果您有LLM API，可以获得更精确的评分：

```python
from repo_analysis import RepoContextPipeline

# 定义LLM客户端函数
def my_llm_client(messages, json_format=False):
    """
    您的LLM客户端实现
    
    Args:
        messages: List[Dict] 格式的消息列表
        json_format: 是否返回JSON格式
        
    Returns:
        Dict 或其他响应格式
    """
    # 调用您的LLM API
    # response = your_llm_api.call(messages)
    # return response
    pass

# 使用LLM client创建pipeline
pipeline = RepoContextPipeline(
    '/path/to/repo',
    llm_client=my_llm_client
)

# 运行时会自动使用LLM进行任务匹配
results = pipeline.run(
    task="构建一个机器学习模型",
    max_tokens=8000
)
```

## 模块说明

### RepoAnalyzer
静态分析模块，生成HCT、MCG、FCG三种视图。

```python
from repo_analysis import RepoAnalyzer

analyzer = RepoAnalyzer('/path/to/repo')
results = analyzer.analyze()

# 访问分析结果
modules = results['modules']
classes = results['classes']
functions = results['functions']
call_graph = results['call_graph']  # NetworkX DiGraph
```

### ImportanceScorer
组件重要性评分模块。

```python
from repo_analysis import ImportanceScorer

scorer = ImportanceScorer(
    repo_path='/path/to/repo',
    modules=results['modules'],
    classes=results['classes'],
    functions=results['functions'],
    imports=results['imports'],
    code_tree=results['code_tree'],
    call_graph=results['call_graph']
)

# 获取关键模块
key_modules = scorer.get_key_modules(top_k=20)
```

### ContextBuilder
上下文构建模块。

```python
from repo_analysis import ContextBuilder

builder = ContextBuilder(
    repo_path='/path/to/repo',
    analysis_results=results,
    key_modules=key_modules
)

# 构建上下文
context = builder.build_context(max_tokens=8000)

# 导出为字符串 (LLM友好格式)
context_str = builder.export_to_string()
```

### TaskMatcher
任务关联评分模块。

```python
from repo_analysis import TaskMatcher, create_task_dict

task = create_task_dict("实现图像分类模型")
matcher = TaskMatcher(llm_client=your_llm_client)

# 单仓库匹配
score = matcher.match_single_repo(task, repo_context, use_llm=True)

# 多仓库匹配
scored_repos = matcher.match_multiple_repos(task, repos, batch_size=5)
```

## 配置

您可以通过配置文件自定义行为：

```python
from repo_analysis.config import PipelineConfig, ScoringConfig

# 自定义评分权重
config = PipelineConfig()
config.scoring.weights = {
    'usage': 3.0,
    'imports_relationships': 2.0,
    'complexity': 1.0,
    'semantic': 0.5,
    'git_history': 3.0,
}

# 在pipeline中使用
pipeline = RepoContextPipeline(
    '/path/to/repo',
    importance_weights=config.scoring.weights
)
```

## 输出格式

### 上下文结构

```json
{
  "metadata": {
    "repo_path": "/path/to/repo",
    "analysis_timestamp": "2025-01-01T00:00:00"
  },
  "repository_overview": {
    "total_modules": 50,
    "total_classes": 120,
    "total_functions": 350,
    "key_technologies": ["torch", "numpy", "pandas"]
  },
  "hierarchical_structure": {
    "packages": ["src", "src.core", "src.utils"],
    "modules": ["src.core.model", "src.core.trainer", ...]
  },
  "dependency_graphs": {
    "module_imports": {
      "src.core.model": ["torch", "src.utils.data"]
    },
    "function_calls": {
      "src.core.model.train": ["src.utils.data.load_data"]
    }
  },
  "core_components": [
    {
      "type": "module",
      "path": "src/core/model.py",
      "importance_score": 8.5,
      "abstract": "...",
      "summary": "..."
    }
  ]
}
```

## 性能考虑

- **大型仓库**: 对于超过10万行代码的仓库，分析可能需要几分钟
- **并行处理**: 使用 `FilterAndRankRepos` 时，调整 `max_workers` 参数以优化性能
- **Token限制**: 使用 `max_tokens` 参数控制上下文大小，避免超出LLM的限制

## 依赖说明

### 核心依赖 (必需)
- `networkx`: 图结构处理
- `tiktoken`: Token计数
- `tqdm`: 进度条

### 可选依赖
- `grep-ast`: 代码结构提取 (推荐)
- `tree-sitter`: 语法解析 (推荐)
- `nbformat`: Jupyter Notebook支持

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 引用

如果您使用了此工具包，请考虑引用原始的RepoMaster项目：

```bibtex
@article{repomaster2025,
  title={RepoMaster: Leveraging GitHub Repositories for Complex Task Solving},
  author={...},
  journal={NeurIPS},
  year={2025}
}
```

