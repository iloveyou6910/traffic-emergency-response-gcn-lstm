# 交通应急响应：GCN + LSTM 最小可演示版

本仓库包含一个可解释的交通应急快速响应最小演示系统，核心在 `my_project/` 目录中。功能涵盖：数据生成与标准化、网格/道路/站点图构建、GCN+LSTM 训练与评估、批量推理与解释、路径规划与地图输出，以及简易前端原型。

## 快速开始（Windows）
- 安装依赖（CPU 环境）：
  - `pip install -r my_project/requirements.txt`
  - 如需指定 CPU 版 PyTorch：`pip install --index-url https://download.pytorch.org/whl/cpu torch`
- 运行一键演示脚本：
  - `powershell -ExecutionPolicy Bypass -File my_project/run_demo.ps1`
  - 按提示完成：数据生成 → 图构建 → 训练评估 → 推理与解释 → 地图输出（HTML 位于 `my_project/outputs/plots/`）。

## 目录结构
- `my_project/README.md`：详细说明与命令示例（推荐阅读）。
- `my_project/config.yaml`：全局配置（节点、时间步、图参数、路径）。
- `my_project/src/`：
  - `prepare_data.py`：合成/缓存数据生成与标准化。
  - `build_grid_graph.py`、`build_road_graph.py`、`build_station_graph.py`：不同拓扑构建与邻接生成。
  - `train_eval.py`：GCN+LSTM 训练与评估（含邻域聚合与时空正则化）。
  - `inference.py`、`explain.py`：批量推理与解释、邻域特征回读。
  - `routing.py`：路径规划与地图视图（坐标范围自适应）。
  - `app_streamlit.py`：简易前端原型。
- `my_project/requirements.txt`：依赖列表（建议 `numpy<2`）。
- `my_project/run_demo.ps1`：一键演示脚本。
- `my_project/data/` 与 `my_project/outputs/`：数据缓存与运行输出（模型、日志、图表、报告）。

## 近期更新
- 融合空间邻域聚合与时空正则化，提升稳定性与邻域一致性。
- 批量推理与邻域特征回读，便于解释与可视化。
- 地图视图修复：自动适配坐标范围，避免出现空白视野。

## 许可证与致谢
- 本项目采用根目录 `LICENSE`（MIT）。
- 第三方依赖：PyTorch、NumPy、Pandas、PyYAML、可选的 Folium；地图回退使用 Leaflet 与 OpenStreetMap 瓦片。
- 数据说明：演示项目使用合成数据，不包含真实个人信息或受限数据。

## 开发与协作
- 推荐以 `feature/*` 分支开发并通过 PR 合入 `main`。
- 后续将补充 CI（安装依赖、运行测试与静态检查）与基础 `tests/` 用例，保障跨环境可运行性。