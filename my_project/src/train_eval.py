import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from typing import Optional


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg):
    data_dir = Path(cfg["paths"].get("data_dir", Path(__file__).resolve().parent.parent / "data"))
    outputs_dir = Path(cfg["paths"].get("outputs_dir", Path(__file__).resolve().parent.parent / "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "models").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "logs").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "plots").mkdir(parents=True, exist_ok=True)
    return data_dir, outputs_dir


def load_long_csv_matrix(csv_path: Path, value_col: str, time_col: str = "time", node_col: str = "node") -> np.ndarray:
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index=time_col, columns=node_col, values=value_col)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return pivot.to_numpy()  # shape [T, N]


def build_adjacency(n_nodes: int, edge_csv: Path) -> torch.Tensor:
    edges_df = pd.read_csv(edge_csv)
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for _, row in edges_df.iterrows():
        u = int(row["src"]) ; v = int(row["dst"]) ; dist = float(row["dist_m"]) if "dist_m" in row else 1.0
        w = 1.0  # 简化权重为 1（卷积聚合强度），可改为 1/dist
        A[u, v] = max(A[u, v], w)
    # 加自环
    A = A + np.eye(n_nodes, dtype=np.float32)
    # 归一化 D^{-1/2} A D^{-1/2}
    d = A.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.from_numpy(A_norm)


# 新增：基于子节点集合构建邻接矩阵

def build_adjacency_for_nodes(node_ids: List[int], edge_csv: Path) -> torch.Tensor:
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    A = np.eye(n, dtype=np.float32)  # 自环
    edges_df = pd.read_csv(edge_csv)
    for _, row in edges_df.iterrows():
        u = int(row["src"]) ; v = int(row["dst"]) ; w = 1.0
        if u in id_to_idx and v in id_to_idx:
            A[id_to_idx[u], id_to_idx[v]] = max(A[id_to_idx[u], id_to_idx[v]], w)
    d = A.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.from_numpy(A_norm)


# 新增：返回透视表以便按 traffic 节点/时间对齐
def load_pivot_df(csv_path: Path, value_col: str, time_col: str = "time", node_col: str = "node") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index=time_col, columns=node_col, values=value_col)
    return pivot.sort_index(axis=0).sort_index(axis=1)


class GCNLSTM(nn.Module):
    def __init__(self, n_nodes: int, in_feats: int, gcn_hidden: int, lstm_hidden: int, forecast_steps: int, A_norm: torch.Tensor):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_feats = in_feats
        self.gcn_hidden = gcn_hidden
        self.lstm_hidden = lstm_hidden
        self.forecast_steps = forecast_steps
        self.register_buffer("A_norm", A_norm)  # [N, N]
        self.W = nn.Linear(in_feats, gcn_hidden)
        self.act = nn.ReLU()
        self.lstm = nn.LSTM(input_size=gcn_hidden, hidden_size=lstm_hidden, batch_first=False)
        self.readout = nn.Linear(lstm_hidden, forecast_steps)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [T_h, N, F]
        T_h, N, F = X.shape
        assert N == self.n_nodes
        H_seq = []
        for t in range(T_h):
            Xt = X[t]  # [N, F]
            h_g = self.A_norm @ Xt  # [N, F]
            h_g = self.W(h_g)       # [N, H]
            h_g = self.act(h_g)
            H_seq.append(h_g)
        H = torch.stack(H_seq, dim=0)  # [T_h, N, H]
        # LSTM 按节点并行，batch = N
        out, (hn, cn) = self.lstm(H)   # out: [T_h, N, H_l]
        h_last = out[-1]               # [N, H_l]
        y = self.readout(h_last)       # [N, T_f]
        y = y.permute(1, 0)            # [T_f, N]
        return y


def build_dataset(features: Dict[str, np.ndarray], cfg) -> Tuple[np.ndarray, np.ndarray]:
    # 组装特征：speed/flow/weather/event
    speed = features["speed"] / 80.0
    flow = features["flow"] / 60.0
    weather = features["weather"] / 5.0
    events = features["events"]

    X_all = np.stack([speed, flow, weather, events], axis=-1)  # [T, N, F=4]

    history_steps = int(cfg["time"].get("history_steps", 12))
    forecast_steps = int(cfg["time"].get("forecast_steps", 6))

    targets = speed  # 预测速度（归一化）
    T_total = X_all.shape[0]
    samples_X = []
    samples_Y = []
    # 滑窗构建
    for t in range(history_steps, T_total - forecast_steps):
        Xin = X_all[t - history_steps:t]            # [T_h, N, F]
        Yout = targets[t:t + forecast_steps]        # [T_f, N]
        samples_X.append(Xin)
        samples_Y.append(Yout)
    X = np.stack(samples_X, axis=0)  # [S, T_h, N, F]
    Y = np.stack(samples_Y, axis=0)  # [S, T_f, N]
    return X, Y


def _mae_norm(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    return torch.mean(torch.abs(y_hat - y_true)).item()

def train_eval(cfg_path: str, epochs: int = 12, lr: float = 1e-3, gcn_hidden: int = 32, lstm_hidden: int = 64, train_split: float = 0.7,
               clip_norm: float = 1.0, lr_scheduler: str = "none", step_size: int = 5, gamma: float = 0.5,
               patience: int = 5, min_delta: float = 1e-4, run_name: Optional[str] = None):
    cfg = load_config(cfg_path)
    data_dir, outputs_dir = ensure_dirs(cfg)
    meta = json.load(open(data_dir / "grid_meta.json", "r", encoding="utf-8"))

    # 使用透视表并按 traffic 的时间/节点对齐
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

    # 以 traffic 的列为训练节点集合
    node_ids = [int(c) for c in speed_df.columns]
    n_nodes = len(node_ids)

    speed = speed_df.to_numpy()
    flow = flow_df.to_numpy()
    weather = weather_df.to_numpy()
    events = events_df.to_numpy()

    features = {"speed": speed, "flow": flow, "weather": weather, "events": events}

    X, Y = build_dataset(features, cfg)
    S = X.shape[0]
    S_tr = int(S * train_split)

    A_norm = build_adjacency_for_nodes(node_ids, data_dir / "graph_edges.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNLSTM(n_nodes=n_nodes, in_feats=X.shape[-1], gcn_hidden=gcn_hidden, lstm_hidden=lstm_hidden,
                    forecast_steps=Y.shape[1], A_norm=A_norm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 可选 weight_decay 后续加入
    criterion = nn.MSELoss()

    scheduler = None
    if lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).float()

    history = {"train_mse": [], "eval_mse": [], "train_rmse_kmh": [], "eval_rmse_kmh": [], "train_mae_kmh": [], "eval_mae_kmh": []}

    best_eval = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    
    events_log = outputs_dir / "logs" / "train_events.log"
    with open(events_log, "w", encoding="utf-8") as lf:
        lf.write("")

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for i in range(S_tr):
            x_i = X_t[i].to(device)
            y_i = Y_t[i].to(device)
            optimizer.zero_grad()
            y_hat = model(x_i)
            loss = criterion(y_hat, y_i)
            loss.backward()
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            train_loss += loss.item()
            train_mae += _mae_norm(y_hat, y_i)
        train_loss /= max(1, S_tr)
        train_mae /= max(1, S_tr)
        train_rmse_kmh = (train_loss ** 0.5) * 80.0
        train_mae_kmh = train_mae * 80.0

        model.eval()
        with torch.no_grad():
            eval_loss = 0.0
            eval_mae = 0.0
            for i in range(S_tr, S):
                x_i = X_t[i].to(device)
                y_i = Y_t[i].to(device)
                y_hat = model(x_i)
                loss = criterion(y_hat, y_i)
                eval_loss += loss.item()
                eval_mae += _mae_norm(y_hat, y_i)
            eval_loss /= max(1, S - S_tr)
            eval_mae /= max(1, S - S_tr)
            eval_rmse_kmh = (eval_loss ** 0.5) * 80.0
            eval_mae_kmh = eval_mae * 80.0

        if scheduler is not None:
            scheduler.step()

        history["train_mse"].append(train_loss)
        history["eval_mse"].append(eval_loss)
        history["train_rmse_kmh"].append(train_rmse_kmh)
        history["eval_rmse_kmh"].append(eval_rmse_kmh)
        history["train_mae_kmh"].append(train_mae_kmh)
        history["eval_mae_kmh"].append(eval_mae_kmh)

        msg = f"[train] epoch={ep} MSE={train_loss:.6f} RMSE_kmh={train_rmse_kmh:.3f} MAE_kmh={train_mae_kmh:.3f} | " \
              f"[eval] MSE={eval_loss:.6f} RMSE_kmh={eval_rmse_kmh:.3f} MAE_kmh={eval_mae_kmh:.3f}"
        print(msg)
        try:
            with open(events_log, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")
        except Exception:
            pass

        # 早停与保存最优
        if eval_loss < best_eval - min_delta:
            best_eval = eval_loss
            best_epoch = ep
            best_state = {"state_dict": model.state_dict(),
                          "cfg": cfg,
                          "node_ids": node_ids,
                          "in_feats": X.shape[-1],
                          "gcn_hidden": gcn_hidden,
                          "lstm_hidden": lstm_hidden,
                          "forecast_steps": Y.shape[1]}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[early_stop] No improvement for {patience} epochs, stopping at epoch {ep}.")
                break

    # 保存最后与最优模型
    model_path = outputs_dir / "models" / "gcnlstm_model.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "node_ids": node_ids, "in_feats": X.shape[-1],
                "gcn_hidden": gcn_hidden, "lstm_hidden": lstm_hidden, "forecast_steps": Y.shape[1]}, model_path)

    best_path = outputs_dir / "models" / "gcnlstm_model_best.pt"
    if best_state is not None:
        torch.save(best_state, best_path)

    log_path = outputs_dir / "logs" / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "epochs": epochs,
            "epochs_done": len(history["eval_mse"]),
            "train_mse": history["train_mse"],
            "eval_mse": history["eval_mse"],
            "train_rmse_kmh": history["train_rmse_kmh"],
            "eval_rmse_kmh": history["eval_rmse_kmh"],
            "train_mae_kmh": history["train_mae_kmh"],
            "eval_mae_kmh": history["eval_mae_kmh"],
            "best": {
                "epoch": best_epoch,
                "eval_mse": best_eval,
                "eval_rmse_kmh": ((best_eval ** 0.5) * 80.0) if best_eval < float("inf") else None,
                "eval_mae_kmh": (min(history["eval_mae_kmh"]) if history["eval_mae_kmh"] else None)
            }
        }, f, ensure_ascii=False, indent=2)

    # 额外汇总到 reports（可选 run_name）
    reports_dir = outputs_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        import csv
        tagged_log = reports_dir / f"train_log__{run_name}.json"
        try:
            tagged_log.write_text(open(log_path, "r", encoding="utf-8").read(), encoding="utf-8")
        except Exception:
            pass
        summary_csv = reports_dir / "grid_summary.csv"
        header = ["run_name","epochs","epochs_done","gcn_hidden","lstm_hidden","lr_scheduler","best_eval_mse","best_eval_rmse_kmh","best_eval_mae_kmh"]
        row = [run_name, epochs, len(history["eval_mse"]), gcn_hidden, lstm_hidden, lr_scheduler, best_eval,
               ((best_eval ** 0.5) * 80.0) if best_eval < float("inf") else None,
               (min(history["eval_mae_kmh"]) if history["eval_mae_kmh"] else None)]
        write_header = not summary_csv.exists()
        with open(summary_csv, "a", newline="", encoding="utf-8") as cf:
            w = csv.writer(cf)
            if write_header:
                w.writerow(header)
            w.writerow(row)

    print(f"[train_eval] Saved model to {model_path}")
    print(f"[train_eval] Saved best model to {best_path}")
    print(f"[train_eval] Saved log to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate simplified GCN+LSTM on grid synthetic data")
    parser.add_argument("--config", type=str, required=False, default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    parser.add_argument("--epochs", type=int, required=False, default=12)
    parser.add_argument("--clip_norm", type=float, required=False, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, required=False, choices=["none", "step", "cosine"], default="none")
    parser.add_argument("--step_size", type=int, required=False, default=5)
    parser.add_argument("--gamma", type=float, required=False, default=0.5)
    # 新增：早停与运行标识
    parser.add_argument("--patience", type=int, required=False, default=5)
    parser.add_argument("--min_delta", type=float, required=False, default=1e-4)
    parser.add_argument("--run_name", type=str, required=False, default=None)
    args = parser.parse_args()
    train_eval(args.config, epochs=args.epochs, clip_norm=args.clip_norm, lr_scheduler=args.lr_scheduler, step_size=args.step_size, gamma=args.gamma,
               patience=args.patience, min_delta=args.min_delta, run_name=args.run_name)


if __name__ == "__main__":
    main()