# 可解释交通应急快速响应系统（GCN+LSTM版）最小可演示版

目录统一位于 `D:\PGT\my_project`，三天交付的最小可演示版本包含：数据与图构建（Day 1）、模型训练与评估（后续）、推理解释与路径规划（后续）以及简易前端（后续）。

## 快速开始
- 安装依赖（CPU环境）：
  - `pip install -r D:\PGT\my_project\requirements.txt`
  - 如需明确安装CPU版PyTorch：`pip install --index-url https://download.pytorch.org/whl/cpu torch`
- 生成合成数据：
  - `python D:\PGT\my_project\src\prepare_data.py --config D:\PGT\my_project\config.yaml`
- 构建网格图：
  - `python D:\PGT\my_project\src\build_grid_graph.py --config D:\PGT\my_project\config.yaml`
- 训练与评估：
  - `python D:\PGT\my_project\src\train_eval.py --config D:\PGT\my_project\config.yaml`
- 推理并保存预测：
  - `python D:\PGT\my_project\src\inference.py --config D:\PGT\my_project\config.yaml`
- 路径规划与地图（运行后在 `outputs/plots/` 查看HTML）：
  - `python D:\PGT\my_project\src\routing.py --config D:\PGT\my_project\config.yaml --start 0 --end 899`
- 产出文件将写入 `D:\PGT\my_project\data\` 与 `D:\PGT\my_project\outputs\`。

## 目录结构
- `requirements.txt` 依赖列表（精简）
- `config.yaml` 全局配置（节点、时间步、图参数）
- `PLAN.md` 三天交付计划书
- `src/prepare_data.py` 合成/缓存数据生成与标准化
- `src/build_grid_graph.py` 网格拓扑与邻接构建
- `data/` 数据缓存与图文件
- `outputs/` 运行输出（模型、日志、报告、图表）

## 配置项说明
- `grid.rows` / `grid.cols`：网格行列数（默认 30×30，约 900 节点）
- `grid.cell_size_m`：网格单元边长（米），用于近似边距离
- `time.history_steps`：历史窗口长度（默认 12）
- `time.forecast_steps`：预测步数（默认 6）
- `time.step_minutes`：时间步长（分钟，默认 5）
- `synthetic.seed`：合成数据随机种子，保证复现
- `adjacency.undirected` / `adjacency.four_neighborhood`：邻接构造（是否无向、是否4邻接）
- `paths.data_dir` / `paths.outputs_dir`：数据与输出目录（可按需修改绝对路径）

## 后续步骤（非 Day 1）
- 训练与评估：`src/train_eval.py`
- 推理与解释：`src/inference.py` + `src/explain.py`
- 路径规划：`src/routing.py`
- 前端：`src/app_streamlit.py`

## 环境要求
- Python 3.10+ 环境（Windows建议使用 PowerShell 执行命令）
- 已安装 `torch`、`numpy<2`、`pandas`、`pyyaml`（参考上文安装依赖）

## 许可证与致谢
- `LICENSE`：本项目遵循仓库根目录中的许可证条款。
- 第三方依赖：PyTorch、NumPy、Pandas、PyYAML、可选的 Folium；地图回退使用 Leaflet 与 OpenStreetMap 瓦片。
- 数据说明：本项目用于演示，使用合成数据，不包含真实个人信息或受限数据。
- 致谢：感谢开源社区与相关库作者的贡献。

## 近期更新
- 空间邻域聚合与时空正则化已集成到 `src/train_eval.py`，提升稳定性与邻域一致性。
- 批量推理与邻域特征回读在 `src/inference.py`，便于解释与可视化。
- 地图视图修复：`src/routing.py` 自动适配坐标范围，避免空白视野。

## 一键演示
- 运行 `D:\PGT\my_project\run_demo.ps1`，按提示生成数据、训练、推理并在 `outputs/plots/` 打开地图。

## GitHub 同步
- 计划推送到 `iloveyou6910/traffic-emergency-response-gcn-lstm` 的 `feature/pgt-sync` 分支，并可在 GitHub 上发起 PR 合入 `main`。
- 若你更偏好直接推送到 `main`，我也可以切换目标分支为 `main`。