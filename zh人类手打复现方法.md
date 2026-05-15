# 跟着一步一步复现

## 1. 环境
    跟着翻译之后的readme把环境下载好，随后需要按要求复制.env文件，写好模型和apikey，再以default_config为模板，改一下my_config，并且在.env里用export的方式引用my_config
    huggingface的数据我直接传进gihub了，再加上有代码改动，你们可以直接重新gitclone，等他下载一会。

## 2. 数据
    需要的数据都在evaluation里面，需要跑的代码也在evaluation里面，除了chat_test_asyncio以外都是用于evaluation的代码，dsbench的就先不管了，跑剩下三个

## 3. 接下来就是跑
    之前在env里设置好了api，这里基本上就不需要了。首先要先在一个终端用```python main.py```启动后端，随后你可以在这个终端看到具体的解题过程，随后要用另一个终端去执行eval的代码：

    ```
    0. 终端 A：启动后端（一直保持运行）

    cd /home/v1staz/workspace/competitions/KDDCUP2026/DatawiseAgent
    conda activate datawise
    python main.py
    1. 终端 B：进入评测目录

    cd /home/v1staz/workspace/competitions/KDDCUP2026/DatawiseAgent/evaluation
    conda activate datawise
    2. 跑 InfiAgentBench

    生成回答：
    python eval_infiagent_bench.py --note deepseek-run1
    评估：
    python InfiAgentBench/scripts/reformat.py --model deepseek-reasoner --responses_file_path ./experimental_results/InfiAgent-Bench/results_deepseek-run1.jsonl --output_file_path ./experimental_results/InfiAgent-Bench/reformat/results_reformat_deepseek-run1.jsonl
    打分：
    python InfiAgentBench/scripts/eval_closed_form.py --questions_file_path ./InfiAgentBench/data/da-dev-questions.jsonl --labels_file_path ./InfiAgentBench/data/da-dev-labels.jsonl --responses_file_path ./experimental_results/InfiAgent-Bench/reformat/results_reformat_deepseek-run1.jsonl
    3. 跑 MatplotBench

    生成图结果：
    python eval_matplotbench.py --note deepseek-matplot-run1
    如需启用视觉工具（可选）：
    python eval_matplotbench.py --note deepseek-matplot-run1 --with_tool
    模型打分：
    python MatplotBench/scripts/model_eval.py --dir ./experimental_results/MatplotBench/gpt-4o-mini
    说明：这个目录名是脚本写死路径，不代表你必须用 gpt-4o-mini 作为被测模型。

    4. 跑 DataModeling（DSBench 子集）

    生成 submission：
    python eval_data_modeling.py --user_name deepseek-dm-run1 --result_path ./experimental_results/Datamodeling-DSBench/deepseek-reasoner
    逐任务算分：
    python DataModeling/scripts/score4each.py --model deepseek-reasoner
    汇总指标：
    python DataModeling/scripts/show_results.py --model deepseek-reasoner
    ```
