# InternAgent 快速开始指南

## 前置准备

### 1. 配置 API 密钥

在 `IdeaEvaluation/` 目录下配置以下文件：

#### 配置 Kaggle API
```bash
cd IdeaEvaluation/
cp kaggle.json.example kaggle.json
# 编辑 kaggle.json，填入你的 Kaggle 用户名和 API Key
```

#### 配置 LLM API
```bash
cp LLM.env.example LLM.env
# 编辑 LLM.env，填入所需的 API 密钥
```

### 2. 安装依赖

确保已安装项目依赖：
```bash
pip install -r requirements.txt
```

## 运行测试

在项目主目录（`InternAgent/`）下运行：

```bash
python3 -m internagent.tester.test_agent_pipeline
```

## 查看结果

运行完成后，结果会保存在 `cache/` 目录中，主要结果文件包括：
- `agent_pipeline_results.json` - Agent 流水线测试结果

