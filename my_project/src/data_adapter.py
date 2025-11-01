import argparse
import json
from pathlib import Path
import math
from datetime import datetime

import pandas as pd

try:
    import requests  # 可选，用于 Open-Meteo 天气拉取
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


def build_grid_edges(rows: int, cols: int) -> pd.DataFrame:
    edges = []
    def nid(r, c):
        return r * cols + c
    for r in range(rows):
        for c in range(cols):
            u = nid(r, c)
            if r + 1 < rows:
                v = nid(r + 1, c)
                edges.append((u, v, 1.0))
                edges.append((v, u, 1.0))
            if c + 1 < cols:
                v = nid(r, c + 1)
                edges.append((u, v, 1.0))
                edges.append((v, u, 1.0))
    return pd.DataFrame(edges, columns=["src", "dst", "weight"]) 


def latlon_to_cell(lat, lon, bounds, rows, cols):
    lat_min, lat_max, lon_min, lon_max = bounds
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        return None
    r = int((lat - lat_min) / max(1e-9, (lat_max - lat_min)) * rows)
    c = int((lon - lon_min) / max(1e-9, (lon_max - lon_min)) * cols)
    r = max(0, min(rows - 1, r))
    c = max(0, min(cols - 1, c))
    return r, c


def parse_time(ts, step_minutes: int):
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts)
    else:
        dt = pd.to_datetime(ts)
    # 对齐到步长
    return (dt.floor(f"{step_minutes}min")).isoformat()


def adapt_traffic(raw_csv: Path, out_dir: Path, bounds, rows: int, cols: int, step_minutes: int):
    df = pd.read_csv(raw_csv)
    required_cols = {"timestamp", "lat", "lon"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"raw_traffic.csv 需要包含列: {required_cols}")
    # 速度与流量可选
    if "speed_kmh" not in df.columns and "speed_mps" in df.columns:
        df["speed_kmh"] = df["speed_mps"] * 3.6
    if "flow_vpm" not in df.columns:
        # 若无流量，生成占位（可后续替换为真实流量）
        df["flow_vpm"] = 0.0
    # 映射到网格节点
    def map_node(row):
        rc = latlon_to_cell(row["lat"], row["lon"], bounds, rows, cols)
        return None if rc is None else rc[0] * cols + rc[1]
    df["node"] = df.apply(map_node, axis=1)
    df = df.dropna(subset=["node"])  # 丢弃不在边界内的数据
    df["node"] = df["node"].astype(int)
    df["time"] = df["timestamp"].apply(lambda x: parse_time(x, step_minutes))
    # 聚合到 (time, node)
    agg = df.groupby(["time", "node"], as_index=False).agg({"speed_kmh": "mean", "flow_vpm": "sum"})
    traffic_path = out_dir / "traffic_samples.csv"
    agg.to_csv(traffic_path, index=False)
    return traffic_path


def adapt_weather_from_open_meteo(center_lat: float, center_lon: float, start: str, end: str, times: pd.Series, out_dir: Path):
    if not _HAS_REQUESTS:
        raise RuntimeError("缺少 requests 依赖，无法从 Open-Meteo 获取数据。请先 pip install requests")
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={center_lat}&longitude={center_lon}"
        f"&hourly=precipitation,rain,temperature_2m&start_date={start}&end_date={end}&timezone=auto"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    times_hourly = pd.to_datetime(hourly.get("time", []))
    precip = hourly.get("precipitation", [])
    rain = hourly.get("rain", [])
    temp = hourly.get("temperature_2m", [])
    dfw = pd.DataFrame({
        "time": times_hourly,
        "precipitation": precip,
        "rain": rain,
        "temperature": temp,
    })
    dfw["time"] = dfw["time"].dt.floor("5min").astype(str)
    # 简单天气等级映射：按降水强度划分 1-5
    def level(p):
        if p >= 10: return 5
        if p >= 5: return 4
        if p >= 2: return 3
        if p >= 0.5: return 2
        return 1
    dfw["weather_level"] = dfw["precipitation"].apply(level)
    # 对齐到现有时间索引
    df_times = pd.DataFrame({"time": times.astype(str)})
    dfw_aligned = df_times.merge(dfw[["time", "weather_level"]], on="time", how="left").fillna({"weather_level": 1})
    # 扩展到所有节点（全局天气），如需按格点天气可自行替换为格点插值
    weather_rows = []
    # 读取网格 meta
    meta = json.load(open(out_dir.parent / "data" / "grid_meta.json", "r", encoding="utf-8"))
    n_nodes = int(meta.get("n_nodes"))
    for t, w in zip(dfw_aligned["time"], dfw_aligned["weather_level"]):
        for n in range(n_nodes):
            weather_rows.append({"time": t, "node": n, "weather_level": int(w)})
    weather_df = pd.DataFrame(weather_rows)
    weather_path = out_dir / "weather_samples.csv"
    weather_df.to_csv(weather_path, index=False)
    return weather_path


def write_grid_meta_topology(out_dir: Path, bounds, rows: int, cols: int, cell_size_m: int):
    lat_min, lat_max, lon_min, lon_max = bounds
    meta = {
        "rows": rows,
        "cols": cols,
        "n_nodes": rows * cols,
        "cell_size_m": cell_size_m,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }
    topo = {"rows": rows, "cols": cols, "bounds": {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}}
    with open(out_dir / "grid_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(out_dir / "grid_topology.json", "w", encoding="utf-8") as f:
        json.dump(topo, f, ensure_ascii=False, indent=2)


def build_events_placeholder(out_dir: Path, times: pd.Series):
    # 占位：无真实事故数据时，生成全 0 的事件权重，可后续由人工或 API 替换
    meta = json.load(open(out_dir / "grid_meta.json", "r", encoding="utf-8"))
    n_nodes = int(meta.get("n_nodes"))
    rows = []
    for t in times.astype(str):
        for n in range(n_nodes):
            rows.append({"time": t, "node": n, "event_weight": 0.0})
    df = pd.DataFrame(rows)
    path = out_dir / "events_samples.csv"
    df.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="将外部交通数据转换为项目网格格式，并可选拉取天气数据")
    parser.add_argument("--raw_traffic", type=str, required=True, help="原始交通 CSV，需包含 timestamp,lat,lon,[speed_kmh|speed_mps],[flow_vpm]")
    parser.add_argument("--out_dir", type=str, default="my_project/data", help="输出目录")
    parser.add_argument("--rows", type=int, default=30, help="网格行数")
    parser.add_argument("--cols", type=int, default=30, help="网格列数")
    parser.add_argument("--bounds", type=str, default="", help="地理边界: lat_min,lat_max,lon_min,lon_max；为空则从数据推断")
    parser.add_argument("--step_minutes", type=int, default=5, help="时间步长（分钟）")
    parser.add_argument("--cell_size_m", type=int, default=200, help="网格单元边长（米，用于 meta）")
    parser.add_argument("--open_meteo", action="store_true", help="启用 Open-Meteo 天气拉取（需要 requests 依赖）")
    parser.add_argument("--center_lat", type=float, default=None, help="天气中心纬度（Open-Meteo）")
    parser.add_argument("--center_lon", type=float, default=None, help="天气中心经度（Open-Meteo）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.raw_traffic)
    # 推断边界
    if args.bounds:
        parts = args.bounds.split(",")
        if len(parts) != 4:
            raise ValueError("--bounds 需为 lat_min,lat_max,lon_min,lon_max")
        bounds = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    else:
        lat_min = float(df_raw["lat"].min())
        lat_max = float(df_raw["lat"].max())
        lon_min = float(df_raw["lon"].min())
        lon_max = float(df_raw["lon"].max())
        # 轻微扩展边界避免落在边缘
        eps_lat = (lat_max - lat_min) * 0.01
        eps_lon = (lon_max - lon_min) * 0.01
        bounds = (lat_min - eps_lat, lat_max + eps_lat, lon_min - eps_lon, lon_max + eps_lon)

    write_grid_meta_topology(out_dir, bounds, args.rows, args.cols, args.cell_size_m)
    # 边
    edges = build_grid_edges(args.rows, args.cols)
    edges.to_csv(out_dir / "graph_edges.csv", index=False)

    # 交通
    traffic_path = adapt_traffic(Path(args.raw_traffic), out_dir, bounds, args.rows, args.cols, args.step_minutes)
    print(f"traffic_samples.csv -> {traffic_path}")

    # 天气（可选）
    df_traffic = pd.read_csv(traffic_path)
    times = pd.Series(df_traffic["time"].unique()).sort_values()
    if args.open_meteo:
        if args.center_lat is None or args.center_lon is None:
            # 若未指定中心点，使用边界中点
            center_lat = (bounds[0] + bounds[1]) / 2.0
            center_lon = (bounds[2] + bounds[3]) / 2.0
        else:
            center_lat, center_lon = args.center_lat, args.center_lon
        # 日期范围
        dt_times = pd.to_datetime(times)
        start = str(dt_times.min().date())
        end = str(dt_times.max().date())
        try:
            weather_path = adapt_weather_from_open_meteo(center_lat, center_lon, start, end, times, out_dir)
            print(f"weather_samples.csv -> {weather_path}")
        except Exception as e:
            print(f"天气拉取失败，回退占位：{e}")
            events_path = build_events_placeholder(out_dir, times)
    else:
        # 无天气拉取则生成占位事件与天气（事件在下方生成）
        pass

    # 事件占位（真实事故数据可替换此文件）
    events_path = build_events_placeholder(out_dir, times)
    print(f"events_samples.csv -> {events_path}")

    print("完成：请运行 my_project/src/app_streamlit.py 或训练/预测脚本继续流程。")


if __name__ == "__main__":
    main()