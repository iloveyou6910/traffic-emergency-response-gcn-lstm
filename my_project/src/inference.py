import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from train_eval import load_config, ensure_dirs, GCNLSTM, build_dataset, load_pivot_df, build_adjacency_for_nodes


def run_inference(cfg_path: str):
    cfg = load_config(cfg_path)
    data_dir, outputs_dir = ensure_dirs(cfg)

    meta = json.load(open(data_dir / "grid_meta.json", "r", encoding="utf-8"))

    # 读取特征并与 traffic 的时间/节点对齐
    traffic_csv = Path(meta["files"]["traffic"])
    weather_csv = Path(meta["files"]["weather"])
    events_csv = Path(meta["files"]["events"])

    speed_df = load_pivot_df(traffic_csv, "speed_kmh")
    flow_df = load_pivot_df(traffic_csv, "flow_vpm")
    weather_df = load_pivot_df(weather_csv, "weather_level")
    events_df = load_pivot_df(events_csv, "event_weight")

    flow_df = flow_df.reindex(index=speed_df.index, columns=speed_df.columns).fillna(0.0)
    weather_df = weather_df.reindex(index=speed_df.index, columns=speed_df.columns).fillna(1.0)
    events_df = events_df.reindex(index=speed_df.index, columns=speed_df.columns).fillna(0.0)

    # 加载模型检查点并按训练时节点顺序排列
    ckpt_path = outputs_dir / "models" / "gcnlstm_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    node_ids = ckpt.get("node_ids")
    if node_ids is None:
        # 兼容旧模型：使用 traffic 的列
        node_ids = [int(c) for c in speed_df.columns]

    # 重新排序列以匹配 node_ids
    cols = [int(c) for c in speed_df.columns]
    col_set = set(cols)
    # 仅保留存在的节点并按顺序排列
    ordered_cols = [nid for nid in node_ids if nid in col_set]
    speed_df = speed_df[ordered_cols]
    flow_df = flow_df[ordered_cols]
    weather_df = weather_df[ordered_cols]
    events_df = events_df[ordered_cols]

    speed = speed_df.to_numpy()
    flow = flow_df.to_numpy()
    weather = weather_df.to_numpy()
    events = events_df.to_numpy()

    features = {"speed": speed, "flow": flow, "weather": weather, "events": events}
    X, Y = build_dataset(features, cfg)  # [S, T_h, N, F], [S, T_f, N]

    A_norm = build_adjacency_for_nodes(ordered_cols, data_dir / "graph_edges.csv")
    model = GCNLSTM(n_nodes=len(ordered_cols), in_feats=X.shape[-1], gcn_hidden=ckpt["gcn_hidden"], lstm_hidden=ckpt["lstm_hidden"],
                    forecast_steps=ckpt["forecast_steps"], A_norm=A_norm)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 选取最后一个样本推理
    x_i = torch.from_numpy(X[-1]).float()
    with torch.no_grad():
        y_hat = model(x_i)  # [T_f, N]
    # 反归一化速度
    y_speed = (y_hat.numpy() * 80.0).astype(np.float32)

    # 保存预测（节点使用真实 node_id）
    pred_path = outputs_dir / "logs" / "predictions.csv"
    records = []
    T_f = y_speed.shape[0]
    for t in range(T_f):
        for idx, nid in enumerate(ordered_cols):
            records.append({"time_ahead": t + 1, "node": nid, "pred_speed_kmh": float(y_speed[t, idx])})
    pd.DataFrame(records).to_csv(pred_path, index=False)
    print(f"[inference] Saved predictions to {pred_path}")

    return pred_path


def main():
    parser = argparse.ArgumentParser(description="Run inference using trained model and save predictions")
    parser.add_argument("--config", type=str, required=False, default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    args = parser.parse_args()
    run_inference(args.config)


if __name__ == "__main__":
    main()