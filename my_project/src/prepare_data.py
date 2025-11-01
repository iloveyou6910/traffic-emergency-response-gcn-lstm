import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import yaml


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg):
    data_dir = Path(cfg["paths"]["data_dir"]) if "paths" in cfg and "data_dir" in cfg["paths"] else Path(__file__).resolve().parent.parent / "data"
    outputs_dir = Path(cfg["paths"]["outputs_dir"]) if "paths" in cfg and "outputs_dir" in cfg["paths"] else Path(__file__).resolve().parent.parent / "outputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, outputs_dir


def generate_synthetic_data(cfg, data_dir: Path):
    rows = int(cfg["grid"].get("rows", 30))
    cols = int(cfg["grid"].get("cols", 30))
    n_nodes = rows * cols

    history_steps = int(cfg["time"].get("history_steps", 12))
    forecast_steps = int(cfg["time"].get("forecast_steps", 6))
    step_minutes = int(cfg["time"].get("step_minutes", 5))
    total_steps = history_steps + forecast_steps + 60  # 额外留出 60 步用于训练段

    seed = int(cfg.get("synthetic", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    # 交通速度（km/h）与流量（veh/min）——随时间和节点产生轻微周期与噪声
    base_speed = 40 + 10 * np.sin(np.linspace(0, 6 * np.pi, total_steps))
    base_flow = 20 + 5 * np.cos(np.linspace(0, 4 * np.pi, total_steps))

    traffic_records = []
    for t in range(total_steps):
        speed = np.clip(base_speed[t] + np.random.normal(0, 3, n_nodes), 5, 80)
        flow = np.clip(base_flow[t] + np.random.normal(0, 2, n_nodes), 0, 60)
        for node in range(n_nodes):
            traffic_records.append({
                "time": t,
                "node": node,
                "speed_kmh": float(speed[node]),
                "flow_vpm": float(flow[node])
            })

    # 天气等级（1-5）——块状扰动 + 随机抖动
    weather_records = []
    for t in range(total_steps):
        level = np.clip(np.round(3 + np.sin(t / 12) + np.random.normal(0, 0.5, n_nodes)), 1, 5)
        for node in range(n_nodes):
            weather_records.append({
                "time": t,
                "node": node,
                "weather_level": int(level[node])
            })

    # 事故事件（0-1）——稀疏脉冲
    event_records = []
    for t in range(total_steps):
        # 每步随机 0.5% 节点发生事件
        event_nodes = np.random.choice(n_nodes, size=max(1, n_nodes // 200), replace=False)
        for node in range(n_nodes):
            event_records.append({
                "time": t,
                "node": node,
                "event_weight": 1.0 if node in event_nodes else 0.0
            })

    traffic_df = pd.DataFrame(traffic_records)
    weather_df = pd.DataFrame(weather_records)
    events_df = pd.DataFrame(event_records)

    traffic_path = data_dir / "traffic_samples.csv"
    weather_path = data_dir / "weather_samples.csv"
    events_path = data_dir / "events_samples.csv"

    traffic_df.to_csv(traffic_path, index=False)
    weather_df.to_csv(weather_path, index=False)
    events_df.to_csv(events_path, index=False)

    meta = {
        "rows": rows,
        "cols": cols,
        "n_nodes": n_nodes,
        "history_steps": history_steps,
        "forecast_steps": forecast_steps,
        "step_minutes": step_minutes,
        "total_steps": total_steps,
        "seed": seed,
        "files": {
            "traffic": str(traffic_path),
            "weather": str(weather_path),
            "events": str(events_path),
        }
    }
    with open(data_dir / "grid_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[prepare_data] Done. Saved synthetic data to: {data_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic traffic/weather/events data for grid graph")
    parser.add_argument("--config", type=str, required=False, default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir, _ = ensure_dirs(cfg)
    generate_synthetic_data(cfg, data_dir)


if __name__ == "__main__":
    main()