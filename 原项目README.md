<div align="center">

# DatawiseAgent: 一个以Notebook为中心的大语言模型智能体框架，用于自适应和鲁棒的数据科学自动化

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.07044) [![Github](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/zimingyou01/DatawiseAgent)
[![Huggingface](https://img.shields.io/badge/-HuggingFace-3B4252?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/JasperYOU/DatawiseAgent-benchmarkdata)
</div>


📘 DatawiseAgent 是一个基于Notebook的大语言模型（LLM）智能体框架，具有基于有限状态转换器（FST）的多阶段架构，旨在模仿人类的工作流，以实现自适应、鲁棒和端到端的数据科学自动化。


---



## 🗂️ 目录
- [⚙️ 环境配置](#️-environment-setup)
- [🚀 将 DatawiseAgent 作为后端服务端启动](#-start-datawiseagent-as-a-backend-server)
  - [⛏️ 配置](#️-configuration)
  - [▶️ 启动服务端](#️-start-the-server)
- [📊 评估](#-evaluation)
  - [📈 实验结果](#-experiment-results)
  - [📑 基准测试数据](#-benchmark-data)
  - [🔍 数据分析 (InfiAgentBench)](#-data-analysis-infiagentbench)
  - [🎨 科学可视化 (MatplotBench)](#-scientific-visualization-matplotbench)
  - [𝌭 数据建模 (DSBench)](#-datamodeling-dsbench)
- [📚 引用](#-citing)


## ⚙️ 环境配置
1. **克隆仓库** 📥
   ```bash
   git clone git@github.com:zimingyou01/DatawiseAgent.git
   cd DatawiseAgent
   ```

2. **设置 Python 环境** 🧑‍💻 
    我们建议使用 Python ≥ 3.10（已在 3.10 环境下测试）。
    ```bash
    conda create -n datawise python=3.10 -y
    conda activate datawise
    pip install -r requirements.txt
    ```

3. **构建用于沙箱代码执行的 Docker 镜像** 🐳

    DatawiseAgent 支持基于 Docker 的沙箱，用于安全执行代码。在继续之前，请确保已安装 [Docker](https://docs.docker.com/engine/install/)。
   
    *  **纯 CPU 环境**
        构建默认镜像，并在配置 YAML 文件中设置 `image_name`: `my-jupyter-image` （见 [智能体超参数](#agent-config)）：

        ```bash
        docker build -t my-jupyter-image -f datawiseagent/coding/jupyter/default_jupyter_server.dockerfile . --progress=plain
        ```
    * **支持 GPU 的环境（例如：数据建模任务）**
        构建适配 GPU 的镜像，并在配置 YAML 文件中设置 `image_name`: `my-jupyter-image-gpus` （见 [智能体超参数](#agent-config)）：

        ```bash
        docker build -t my-jupyter-image-gpus -f datawiseagent/coding/jupyter/cuda_jupyter_server.dockerfile . --progress=plain
        ```
4. **可选：使用 vLLM 部署开源 LLMs** 🤖
    DatawiseAgent 旨在与商业及开源大模型（例如 **GPT-4o, GPT-4o-mini, Qwen2.5 7B/14B/32B/72B**）无缝协作。

    对于开源模型，我们推荐使用 [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) 将其作为与 OpenAI 兼容的 API 进行服务。
    部署完成后，只需在 `.env` 中简单配置（见 [环境变量](#env-config)）：
    ```bash
    export OPENAI_API_KEY=<your_key>
    export OPENAI_BASE_URL=<your_vllm_server_url>
    ```

##  🚀 将 DatawiseAgent 作为后端服务端启动

我们建议将 DatawiseAgent 作为一个后端服务器运行。

### ⛏️ 配置
DatawiseAgent 需要 **两个层级的配置**：
1. **环境变量 (`.env`)** <a name="env-config"></a>
    * 复制模板文件并填写必要字段：
    ```bash
    cp .env.example .env
    ```
    * 至少要设置以下变量：
      * `OPENAI_API_KEY` - 你的 OpenAI（或兼容 OpenAI 的）API 密钥
      * `OPENAI_BASE_URL` - API 的基础 URL（可以指向 [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) 或者 OpenAI 的官方端点）
    * 默认情况下，DatawiseAgent 会从 `configs/default_config.yaml` 加载设置。
    如果你想使用自定义的配置文件，请在 `.env` 中指定：
    ```bash
    export CUSTOM_CONFIG=configs/my_config.yaml
    ```



2. **智能体超参数 (configs/*.yaml)** <a name="agent-config"></a>
    * `configs/` 目录中的 YAML 文件定义了 DatawiseAgent 的所有超参数。
    * 每个字段都有清晰的注释说明。
    * 我们提供了两个参考配置以供复现结果：
      * `default_config.yaml` → 用于 *InfiAgentBench* 和 *MatplotBench* 实验
      * `datamodeling.yaml` → 用于 DSBench 中的数据建模任务
    * 它们之间的主要区别在于：
      * 代码执行使用的 Docker 镜像不同
      * 对 DatawiseAgent 中有限状态转换器限制的细微调整

### ▶️ 启动服务端
将 DatawiseAgent 作为后端服务启动：
```bash
python main.py
```
启动之后，服务器将运行在：
👉 http://localhost:8000


## 📊 评估

我们在三个具有代表性的数据科学场景下评估了 DatawiseAgent：

* **数据分析** (*InfiAgentBench*)
* **科学可视化** (*MatplotBench*)
* **预测建模** (*DSBench*)

横跨 **商业模型**（GPT-4o, GPT-4o-mini）和 **开源模型**（Qwen2.5 系列）。

### 📈 实验结果
我们报告了在 **有效性** (Effectiveness)、**适应性** (Adaptability) 和 **鲁棒性** (Robustness) 方面的结果：
- **有效性** 和 **适应性**

<p align="center">
  <img src="evaluation/experimental_results/effective_result.jpg" alt="Effectiveness Results" width="100%"/>
</p>

- **鲁棒性**

<p align="center">
  <img src="evaluation/experimental_results/robust_result.jpg" alt="Robustness Results" width="60%"/>
</p>

### 📑 基准测试数据
为促进后续研究，我们将实验结果和智能体轨迹开源在：
```
evaluation/experimental_results/
```

**💡 要复现结果，请遵照以下步骤：**
1. **下载基准测试数据** 
    所有基准测试（MatplotBench, InfiAgentBench, Datamodeling）均托管在 Hugging Face 上：  
    👉 [DatawiseAgent-benchmarkdata](https://huggingface.co/datasets/JasperYOU/DatawiseAgent-benchmarkdata)

    下载之后，数据集包含以下三个文件夹：  
    - `MatplotBench/`  
    - `InfiAgentBench/`  
    - `Datamodeling/`  

    将它们的内容移动至对应的 `evaluation/*/data` 目录下（如果 `data/` 文件夹不存在，请先创建）：  
    ```bash
    mkdir -p evaluation/MatplotBench/data evaluation/InfiAgentBench/data evaluation/DataModeling/data

    mv DatawiseAgent-benchmarkdata/MatplotBench/* evaluation/MatplotBench/data/
    mv DatawiseAgent-benchmarkdata/InfiAgentBench/* evaluation/InfiAgentBench/data/
    mv DatawiseAgent-benchmarkdata/Datamodeling/* evaluation/DataModeling/data/
    ```
    🥑 替代方案：
    如果你更希望单独下载，也可以从它们的原始仓库下载基准数据集，并手动整理到 `evaluation/*/data` 目录下：
    * [MatplotBench](https://github.com/thunlp/MatPlotAgent)
    * [InfiAgentBench](https://github.com/InfiAgent/InfiAgent)
    * [DSBench](https://github.com/LiqiangJing/DSBench) 中的数据建模任务

2. **在 `evaluation/` 根目录下运行所有评估**

    ```
    cd evaluation/
    ```
3. **确保后端服务器地址与 `chat_test_asyncio.py` 一致**
    ```python
    # 在 chat_test_asyncio.py 中设置
    BASE_URL = "http://0.0.0.0:8000"
    BASE_WS_URL = "ws://localhost:8000/register_websocket"
    ```


### 🔍 数据分析 ([InfiAgentBench](https://github.com/InfiAgent/InfiAgent))
<a name="Data-Analysis"></a>
1. 运行评估：

    ```bash
    python ./eval_infiagent_bench.py --note "temperature-0_2-args-7-6-8"
    ```

2. 评估结果（遵循 [InfiAgentBench](https://github.com/InfiAgent/InfiAgent) 协议）：
    * **步骤 1：配置 API 密钥和基础 URL**
        在以下文件中设置对应值：
        * `evaluation/InfiAgentBench/scripts/api_key.txt`
        * `evaluation/InfiAgentBench/scripts/url.txt`

    * **步骤 2：重新格式化智能体轨迹**

        ```bash
        python ./InfiAgentBench/scripts/reformat.py \
            --model "gpt-4o-mini" \
            --responses_file_path ./data/results_datawise-test.jsonl \
            --output_file_path ./experimental_results/InfiAgent-Bench/reformat/results_reformat_datawise-test.jsonl
        ```
    
    * **步骤 3：对照由于标签评估重新格式化后的答案**

        ```bash
        python ./InfiAgentBench/scripts/eval_closed_form.py \
                --questions_file_path ./InfiAgentBench/data/da-dev-questions.jsonl \
                --labels_file_path ./InfiAgentBench/data/da-dev-labels.jsonl \
                --responses_file_path ./experimental_results/InfiAgent-Bench/reformat/results_datawise-test.jsonl
        ```



### 🎨 科学可视化 ([MatplotBench](https://github.com/thunlp/MatPlotAgent))
<a name="Scientific-Visualization"></a>

1. 运行 MatplotBench 任务：
    ```bash
    python ./eval_infiagent_bench.py --note "temperature-0_2-args-7-6-8" [--with_tool]
    ```

    - `--with_tool` *(可选)*：在评估阶段启用**视觉工具**。  
    - ⚠️ 在启用 `--with_tool` 之前，请务必在以下文件中设置好你的 API 密钥和基础 URL
      `datawiseagent/tools/dsbench/vision_tool.py`:
      ```python
      # TODO: explicitly write in the api_key and base_url
      api_key = "<YOUR_API_KEY>"
      base_url = "<YOUR_BASE_URL>"
      ```

2. 使用基于模型的评估来获取结果：
    在我们的实验配置中，我们使用 `gpt-4o-2024-08-06` 默认作为 VLM 的裁判。
    * 在以下文件中设置对应值：
        * `evaluation/MatplotBench/scripts/api_key.txt`
        * `evaluation/MatplotBench/scripts/url.txt`
    
    * 运行评估脚本：
    ```bash
    python ./MatplotBench/scripts/model_eval.py --dir ./experimental_results/MatplotBench/gpt-4o-mini
    ```
    

### 𝌭 数据建模 ([DSBench](https://github.com/LiqiangJing/DSBench))
<a name="DataModeling"></a>

1. 运行数据建模任务：
    ```bash
    python eval_data_modeling.py \
        --user_name "DataModeling-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop" \
        --result_path "./results/DataModeling/gpt-4o-mini/"
    ```

2. 计算评估分数：

    ```bash
    python ./DataModeling/scripts/score4each.py --model gpt-4o-mini
    ```

3. 聚合与总结结果：
    ```bash
    python ./DataModeling/scripts/show_results.py --model gpt-4o-mini
    ```

👉 各脚本的详细参数用法已记录在对应脚本的注释中。


## 📚 引用

如果你在工作中使用到了本项目，请引用相关论文：

```bibtex
@article{you2025datawiseagent,
  title={DatawiseAgent: A Notebook-Centric LLM Agent Framework for Automated Data Science},
  author={You, Ziming and Zhang, Yumiao and Xu, Dexuan and Lou, Yiwei and Yan, Yandong and Wang, Wei and Zhang, Huaming and Huang, Yu},
  journal={arXiv preprint arXiv:2503.07044},
  year={2025}
}
```
