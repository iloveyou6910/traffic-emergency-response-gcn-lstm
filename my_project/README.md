# 可解释交通应急快速响应系统（GCN+LSTM版）最小可演示版

目录统一位于 `D:\PGT\my_project`，三天交付的最小可演示版本包含：数据与图构建（Day 1）、模型训练与评估（后续）、推理解释与路径规划（后续）以及简易前端（后续）。

## 快速开始
- 安装依赖：
  - `& "$env:USERPROFILE\miniconda3\envs\pgt\python.exe" -m pip install -r D:\PGT\my_project\requirements.txt`
- 生成合成数据与网格图（Day 1）：
  - `& "$env:USERPROFILE\miniconda3\envs\pgt\python.exe" D:\PGT\my_project\src\prepare_data.py --config D:\PGT\my_project\config.yaml`
  - `& "$env:USERPROFILE\miniconda3\envs\pgt\python.exe" D:\PGT\my_project\src\build_grid_graph.py --config D:\PGT\my_project\config.yaml`
- 产出文件将写入 `D:\PGT\my_project\data\` 与 `D:\PGT\my_project\outputs\`。

## 目录结构
- `requirements.txt` 依赖列表（精简）
- `config.yaml` 全局配置（节点、时间步、图参数）
- `PLAN.md` 三天交付计划书
- `src/prepare_data.py` 合成/缓存数据生成与标准化
- `src/build_grid_graph.py` 网格拓扑与邻接构建
- `data/` 数据缓存与图文件
- `outputs/` 运行输出（模型、日志、报告、图表）

## 配置说明（config.yaml）
- `grid.rows/cols`：网格行列数（默认 30×30 ≈ 900 节点）
- `grid.cell_size_m`：单元格边长（米）
- `time.history_steps/forecast_steps`：历史 12 步、预测 6 步（5分钟步长）
- `synthetic.seed`：合成数据随机种子

## 后续步骤（非 Day 1）
- 训练与评估：`src/train_eval.py`
- 推理与解释：`src/inference.py` + `src/explain.py`
- 路径规划：`src/routing.py`
- 前端：`src/app_streamlit.py`

## 环境要求
- 已配置 `conda env pgt`，含 `torch 2.4.0+cpu`、`pyg 2.7.0`、`numpy 1.26.4`
- Windows PowerShell 运行上述命令