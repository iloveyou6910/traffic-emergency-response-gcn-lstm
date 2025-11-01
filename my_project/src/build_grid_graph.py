import argparse
import json
from pathlib import Path

import yaml
import pandas as pd


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg):
    data_dir = Path(cfg["paths"].get("data_dir", Path(__file__).resolve().parent.parent / "data"))
    outputs_dir = Path(cfg["paths"].get("outputs_dir", Path(__file__).resolve().parent.parent / "outputs"))
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, outputs_dir


def node_id(r: int, c: int, cols: int) -> int:
    return r * cols + c


def build_edges(rows: int, cols: int, undirected: bool, four_neighborhood: bool, cell_size_m: int):
    edges = []
    seen = set()
    for r in range(rows):
        for c in range(cols):
            u = node_id(r, c, cols)
            neighbors = []
            if four_neighborhood:
                if r - 1 >= 0:
                    neighbors.append((r - 1, c))
                if r + 1 < rows:
                    neighbors.append((r + 1, c))
                if c - 1 >= 0:
                    neighbors.append((r, c - 1))
                if c + 1 < cols:
                    neighbors.append((r, c + 1))
            for nr, nc in neighbors:
                v = node_id(nr, nc, cols)
                dist_m = cell_size_m
                # 唯一定向边：避免重复写入
                key_uv = (u, v)
                if key_uv not in seen:
                    edges.append((u, v, dist_m))
                    seen.add(key_uv)
                if undirected:
                    key_vu = (v, u)
                    if key_vu not in seen:
                        edges.append((v, u, dist_m))
                        seen.add(key_vu)
    return edges


def save_graph(edges, data_dir: Path):
    df = pd.DataFrame(edges, columns=["src", "dst", "dist_m"])
    edge_path = data_dir / "graph_edges.csv"
    df.to_csv(edge_path, index=False)
    print(f"[build_grid_graph] Saved {len(df)} unique directed edges to {edge_path}")


def save_meta(rows: int, cols: int, cell_size_m: int, data_dir: Path):
    meta_path = data_dir / "grid_topology.json"
    meta = {
        "rows": rows,
        "cols": cols,
        "cell_size_m": cell_size_m,
        "n_nodes": rows * cols,
        "topology": "grid-four-neighborhood-undirected"
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[build_grid_graph] Saved topology meta to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Build grid graph edges and save to data directory")
    parser.add_argument("--config", type=str, required=False, default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir, _ = ensure_dirs(cfg)

    rows = int(cfg["grid"].get("rows", 30))
    cols = int(cfg["grid"].get("cols", 30))
    cell_size_m = int(cfg["grid"].get("cell_size_m", 200))

    undirected = bool(cfg["adjacency"].get("undirected", True))
    four_neighborhood = bool(cfg["adjacency"].get("four_neighborhood", True))

    edges = build_edges(rows, cols, undirected, four_neighborhood, cell_size_m)
    save_graph(edges, data_dir)
    save_meta(rows, cols, cell_size_m, data_dir)
    print("[build_grid_graph] Done.")


if __name__ == "__main__":
    main()