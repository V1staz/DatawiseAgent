# ds 环境说明

- 环境名：`ds`
- Python：`3.10.20`
- 创建方式：以已稳定的 `DWA` 为基线克隆，再补充泛用数据分析/挖掘/探索依赖
- 完整环境导出：`environment.ds.full.yml`

## 目标

该环境面向以下场景：

- DatawiseAgent 后端运行
- 无 Docker 的本地 Jupyter / notebook 执行
- API-first 的数据分析、数据处理、可视化、探索式任务
- 为后续 InfiAgentBench / MatplotBench 复现提供统一执行环境

## 当前额外补装的泛用包

- 可视化与统计：
  - `seaborn==0.13.2`
  - `statsmodels==0.14.4`
  - `plotnine==0.14.4`
  - `bokeh==3.9.0`
  - `missingno==0.5.2`
- 表格/数据访问：
  - `openpyxl==3.1.5`
  - `pyxlsb==1.0.10`
  - `xlrd==2.0.1`
  - `lxml==5.3.0`
  - `duckdb==1.1.3`
  - `polars==1.15.0`
  - `tabulate==0.9.0`
- 机器学习与实验：
  - `catboost==1.2.7`
  - `imbalanced-learn==0.12.4`
  - `optuna==4.1.0`
  - `shap==0.46.0`
  - `category-encoders==2.6.4`
  - `SQLAlchemy==2.0.36`
- 文本/工具：
  - `sentencepiece==0.2.0`
  - `rapidfuzz==3.10.1`
  - `Unidecode==1.3.8`
- notebook 图形扩展：
  - `kaleido==0.2.1`
  - `holoviews==1.20.2`
  - `python-ternary==1.0.8`

## 明确未安装

- `torch`
- `transformers`
- `vllm`

当前路线为 API-first，且优先打通非训练型任务，因此暂不引入上述重型依赖。

## 验证结果

- `pip check`：通过
- 关键 import：通过
- 已覆盖模块：
  - FastAPI / OpenAI / Jupyter
  - pandas / numpy / scipy / scikit-learn
  - xgboost / lightgbm / catboost
  - matplotlib / seaborn / plotly / kaleido / holoviews / ternary / plotnine / bokeh
  - duckdb / polars / openpyxl / pyxlsb / xlrd / lxml
  - optuna / shap / category_encoders / imbalanced-learn

## 使用方式

```bash
conda activate ds
cd /home/yu/projects/DataWiseAgent/DatawiseAgent
python main.py
```

