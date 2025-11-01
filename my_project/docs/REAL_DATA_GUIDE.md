# 真实数据接入指南

## 现状说明
- 项目默认使用“合成网格交通数据”（可复现，便于演示与验收）。这不是“虚假数据”，而是用于验证流程的可控样例。
- 如果你需要真实数据，有两条路径：
  1) 使用公开交通数据集（如 METR-LA / PEMS-BAY 等），离线处理后接入。
  2) 通过 API/开放平台拉取实时数据（需注册密钥与遵守服务条款），如 Open-Meteo（天气）、OpenStreetMap（路网）。

## 项目所需数据文件（目标格式）
- `grid_topology.json` / `grid_meta.json`：网格行列、边界与节点数（行×列）、单元大小（米）、边界经纬度。
- `graph_edges.csv`：网格邻接边（`src,dst,weight`），默认四邻接、权重=1。
- `traffic_samples.csv`：交通时序（`time,node,speed_kmh,flow_vpm`），时间按 5 分钟对齐。
- `weather_samples.csv`：天气时序（`time,node,weather_level`），默认 1–5 级。
- `events_samples.csv`：事件时序（`time,node,event_weight`），无真实事故可设为 0。

## 推荐真实数据来源
- 公开数据集（历史数据）
  - METR-LA / PEMS-BAY（交通速度传感器）：`https://github.com/liyaguang/DCRNN`、`https://github.com/laiguokun/Graph-WaveNet`
  - PeMSD 系列（加州高速路网流量）：`https://github.com/guoshnBJTU/ASTGCN-r1`
- 开放 API（实时/历史）
  - 天气：Open-Meteo（免费，无密钥）：`https://open-meteo.com`（已在适配脚本支持）
  - 路网：OpenStreetMap（OSM，overpass API / 地理数据下载）
  - 交通：各地图/交通平台需密钥（高德、HERE、TomTom 等），请遵守 ToS 与配额。

## 使用外部数据的步骤（离线 CSV → 项目格式）
1) 准备原始 CSV：包含 `timestamp,lat,lon,[speed_kmh|speed_mps],[flow_vpm]`。
   - 时间戳可为 ISO 字符串或 Unix 秒；经纬度为 WGS84。
2) 运行数据适配脚本：
   - `python my_project/src/data_adapter.py --raw_traffic path/to/raw_traffic.csv --out_dir my_project/data --rows 30 --cols 30 --step_minutes 5`
   - 可选天气拉取（Open-Meteo）：
     - `pip install requests`
     - `python my_project/src/data_adapter.py --raw_traffic path/to/raw.csv --open_meteo --center_lat <lat> --center_lon <lon>`
3) 检查生成文件：`traffic_samples.csv`、`graph_edges.csv`、`grid_meta.json`、`grid_topology.json`、`events_samples.csv`（天气可选）。
4) 运行项目：
   - 训练：`python my_project/src/train_eval.py --config my_project/config.yaml`
   - 预测：`python my_project/src/inference.py --config my_project/config.yaml`
   - 前端：`streamlit run my_project/src/app_streamlit.py`

## 通过公开数据集接入的提示
- METR-LA/PEMS-BAY通常为 H5/NPZ 格式，包含传感器坐标与邻接。建议先转换为 CSV：每行包含传感器时间点的速度，再用坐标映射到网格节点（或直接使用传感器节点作为图，并改造项目加载逻辑）。
- 若直接使用传感器图而非网格图：将 `graph_edges.csv` 替换为传感器邻接（权重可用距离或相关性），并将 `grid_meta.json` 中的 `rows/cols` 改为适配传感器节点数（例如 `rows=1, cols=N` 或扩展为任意图结构）。

## 事件数据（真实事故）
- 来源：城市开放数据平台、交警微博/公众号公开信息、新闻抓取（需协议）。
- 最简做法：手动维护 `events_samples.csv` 在特定时间段与节点写入权重（如 0.3/0.6 表示车道缩减 1/2 条）。

## 注意事项
- 合规与配额：使用 API 抓取实时交通需遵守服务协议；建议优先使用公开历史数据进行模型验证。
- 数据质量：缺测、异常值需在适配脚本中做插值或过滤；时间对齐按 5 分钟步长。
- 性能与规模：若数据过大，请先选定区域与时间窗口以控制节点与样本规模。

## 我可以帮你做什么
- 适配你的原始 CSV 到项目格式（快速转换并校验）。
- 为 METR-LA/PEMS-BAY 编写专用适配器（读取 H5/NPZ 并输出项目需要的 CSV/JSON）。
- 增加基于 OSM 的路网构建与可选按道路段粒度的图（需要额外库与处理）。
- 若你提供 API 密钥与目标城市范围，我可以扩展脚本来抓取天气与路网数据；实时交通数据的抓取将严格遵守对应平台的 ToS。