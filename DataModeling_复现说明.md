# DataModeling 复现说明

## 目标

只跑 `DataModeling / DSBench`，不跑 InfiAgentBench 和 MatplotBench。

## 1. Qwen API 填写位置

项目走的是 OpenAI-compatible 接口，实际读取的是环境变量，不是单独的 Qwen 配置文件。

你需要填写：

- `.env` 里的 `OPENAI_API_KEY`
- `.env` 里的 `OPENAI_BASE_URL`

最小示例：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```bash
export OPENAI_API_KEY="<你的 Qwen API Key>"
export OPENAI_BASE_URL="<你的 Qwen OpenAI-compatible Base URL>"
export CUSTOM_CONFIG="configs/datamodeling_qwen.yaml"
```

如果你的 provider 不是 `qwen2.5-72b-instruct`，再改：

- `configs/datamodeling_qwen.yaml` 里的 `llm.model`

## 2. Python 环境要求

当前仓库的 DataModeling 需要：

- Python 3.10
- `requirements.txt` 里的后端依赖
- `jupyter_kernel_gateway`

在 macOS 上不要直接用系统自带的 `python3 3.9`。

建议：

```bash
conda create -n datawise python=3.10 -y
conda activate datawise
pip install -r requirements.txt
```

如果 `requirements.txt` 中的 `nvidia-nccl-cu12` 在 macOS 安装失败，就从安装列表里去掉它再装；本地 no-Docker 的 DataModeling 不依赖它。

## 3. 启动后端

仓库根目录：

```bash
conda activate datawise
python main.py
```

如果你当前 shell 没有 `python` 命令，就用激活后的环境，或者改成：

```bash
python3 main.py
```

## 4. 跑 DataModeling

新开一个终端：

```bash
cd evaluation
conda activate datawise
python eval_data_modeling.py \
  --user_name qwen-dm-run1 \
  --result_path ./experimental_results/Datamodeling-DSBench/qwen-dm-run1
```

## 5. 打分

仍在 `evaluation/` 目录：

```bash
python DataModeling/scripts/score4each.py --model qwen-dm-run1
python DataModeling/scripts/show_results.py --model qwen-dm-run1
```

## 6. 结果位置

生成结果：

- `evaluation/experimental_results/Datamodeling-DSBench/qwen-dm-run1/results.jsonl`

逐任务评分：

- `evaluation/experimental_results/Datamodeling-DSBench/qwen-dm-run1/performances/*/result.txt`

汇总指标：

- 直接看 `python DataModeling/scripts/show_results.py --model qwen-dm-run1` 的终端输出

## 7. 当前已确认的信息

当前仓库里以下 DataModeling 数据目录已经存在：

- `evaluation/DataModeling/data/task/`
- `evaluation/DataModeling/data/data_resplit/`
- `evaluation/DataModeling/data/answers/`

所以现在不缺 DSBench 基础数据，主要阻塞项是本地 Python 环境和 API 配置。
