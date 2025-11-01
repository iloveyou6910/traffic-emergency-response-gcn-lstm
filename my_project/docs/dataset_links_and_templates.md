# 数据集与代码模板

## 合成数据集（推荐用于演示）
- 生成命令：
  - `python src/prepare_data.py --config config.yaml`
- 产出目录：`my_project/data/`
  - `grid_topology.json`、`grid_meta.json`
  - `traffic_samples.csv`、`weather_samples.csv`、`events_samples.csv`
- 特性：可复现（受 `config.yaml` 参数与随机种子控制），无外网依赖，适合快速演示与验收。

## 开源数据集链接汇总（可选替代）
- METR-LA / PEMS-BAY（交通速度传感器数据）：
  - DCRNN 官方仓库（包含数据下载说明）：`https://github.com/liyaguang/DCRNN`
  - Graph WaveNet 仓库（含处理脚本与链接）：`https://github.com/laiguokun/Graph-WaveNet`
- PeMSD 系列（加州高速路网流量）：
  - STGCN/ASTGCN 相关资源（包含数据与处理说明）：`https://github.com/guoshnBJTU/ASTGCN-r1`
- Torch Geometric Temporal 内置示例数据（鸡pox、Windmill 等）：
  - 仓库路径：`torch_geometric_temporal/dataset/`（本项目已包含副本）
  - 使用方式：参考该目录下的 `*.py` 与官方文档说明。
- 说明：上述链接均为官方/主流复现仓库；具体下载方式与许可以各仓库 README 为准。若需我提供固定镜像或本地打包版本，请告知目标数据集名称。

## 代码模板包
- 位置：`my_project/templates/`
  - `config.template.yaml`：标准配置模板（训练/路由参数与路径设置）。
  - `README.md`：使用说明（如何复制模板并运行）。
- 最小依赖：见 `my_project/requirements.txt`
- 快速启动：
  - 安装依赖：`pip install -r my_project/requirements.txt`
  - 一键演示：`./my_project/run_demo.ps1`

## 推荐使用方式
- 首次演示：直接使用合成数据（无需下载外部数据）。
- 对比实验：将外部数据放入 `my_project/data_external/`，并扩展 `config.yaml` 指向该目录；训练/预测/路由脚本已支持通过 `--config` 切换配置。