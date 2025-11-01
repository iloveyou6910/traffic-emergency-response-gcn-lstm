import argparse
import json
from pathlib import Path
import heapq
from typing import Tuple

import pandas as pd
import yaml

from train_eval import load_config, ensure_dirs


def build_graph(edge_csv: Path):
    df = pd.read_csv(edge_csv)
    graph = {}
    for _, row in df.iterrows():
        u = int(row["src"]) ; v = int(row["dst"]) ; dist = float(row["dist_m"]) if "dist_m" in row else 1.0
        graph.setdefault(u, []).append((v, dist))
    return graph


def load_congestion(pred_csv: Path, avg_steps: int = 1) -> dict:
    preds = pd.read_csv(pred_csv)
    step_mask = preds["time_ahead"] <= avg_steps
    subset = preds[step_mask].copy()
    subset["congestion_level"] = subset["pred_speed_kmh"].apply(lambda v: 5 if v <= 20 else (4 if v <= 30 else (3 if v <= 40 else (2 if v <= 50 else 1))))
    avg_cong = subset.groupby("node")["congestion_level"].mean().to_dict()
    return avg_cong


def load_weather_events(data_dir: Path) -> Tuple[dict, dict]:
    weather_df = pd.read_csv(data_dir / "weather_samples.csv")
    events_df = pd.read_csv(data_dir / "events_samples.csv")
    latest_t = weather_df["time"].max()
    w_latest = weather_df[weather_df["time"] == latest_t].set_index("node")["weather_level"].to_dict()
    e_latest = events_df[events_df["time"] == latest_t].set_index("node")["event_weight"].to_dict()
    return w_latest, e_latest


def dijkstra(graph, start: int, end: int, cong: dict, weather: dict, events: dict, alpha: float = 1.0, beta: float = 0.2, gamma: float = 0.5):
    NINF = float("inf")
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end:
            break
        for v, w in graph.get(u, []):
            cong_v = float(cong.get(v, 1.0))
            weather_v = float(weather.get(v, 3.0)) / 5.0
            event_v = float(events.get(v, 0.0))
            multiplier = 1.0 + alpha * (cong_v / 5.0) + beta * weather_v + gamma * event_v
            cost = w * multiplier
            nd = d + cost
            if nd < dist.get(v, NINF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    path = []
    cur = end
    if cur not in prev and cur != start:
        return [], float("inf")
    while cur != start:
        path.append(cur)
        cur = prev.get(cur, start)
    path.append(start)
    path.reverse()
    total_cost = dist.get(end, float("inf"))
    return path, total_cost


def main():
    parser = argparse.ArgumentParser(description="Plan emergency route using predicted congestion and Dijkstra (with weather/events)")
    parser.add_argument("--config", type=str, required=False, default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    parser.add_argument("--start", type=int, required=False, default=0)
    parser.add_argument("--end", type=int, required=False, default=-1)
    parser.add_argument("--alpha", type=float, required=False, default=1.0, help="congestion weight")
    parser.add_argument("--beta", type=float, required=False, default=0.2, help="weather weight")
    parser.add_argument("--gamma", type=float, required=False, default=0.5, help="event weight")
    parser.add_argument("--avg_steps", type=int, required=False, default=3, help="average over first k future steps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir, outputs_dir = ensure_dirs(cfg)

    topo = json.load(open(data_dir / "grid_topology.json", "r", encoding="utf-8"))
    rows = int(topo.get("rows", 1))
    cols = int(topo.get("cols", 1))
    n_nodes = rows * cols
    meta = json.load(open(data_dir / "grid_meta.json", "r", encoding="utf-8"))
    cell = int(meta.get("cell_size_m", 200))
    bounds = topo.get("bounds", {})
    lat_min = float(bounds.get("lat_min", 30.0))
    lat_max = float(bounds.get("lat_max", 30.5))
    lon_min = float(bounds.get("lon_min", 120.0))
    lon_max = float(bounds.get("lon_max", 120.5))

    start = int(args.start)
    end = int(args.end) if args.end != -1 else (n_nodes - 1)

    graph = build_graph(data_dir / "graph_edges.csv")
    cong = load_congestion(outputs_dir / "logs" / "predictions.csv", avg_steps=args.avg_steps)
    weather, events = load_weather_events(data_dir)

    path, cost = dijkstra(graph, start, end, cong, weather, events, alpha=args.alpha, beta=args.beta, gamma=args.gamma)

    out_json = outputs_dir / "reports" / "route.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"start": start, "end": end, "path": path, "estimated_cost": cost,
                   "params": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma, "avg_steps": args.avg_steps}}, f, ensure_ascii=False, indent=2)
    print(f"[routing] Saved route to {out_json}")

    # 生成实际地图（Folium, 使用拓扑边界经纬度）
    try:
        import folium
        def node_rc(n):
            r = n // cols
            c = n % cols
            return r, c
        def rc_to_latlon(r, c):
            lat_range = lat_max - lat_min
            lon_range = lon_max - lon_min
            r_den = max(rows - 1, 1)
            c_den = max(cols - 1, 1)
            lat = lat_min + (r / r_den) * lat_range
            lon = lon_min + (c / c_den) * lon_range
            return lat, lon
        center_lat = (lat_min + lat_max) / 2.0
        center_lon = (lon_min + lon_max) / 2.0
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        coords = []
        for n in path:
            r, c = node_rc(n)
            lat, lon = rc_to_latlon(r, c)
            coords.append((lat, lon))
        folium.PolyLine(coords, color="red", weight=5, opacity=0.8).add_to(m)
        if coords:
            folium.Marker(coords[0], tooltip=f"Start {start}").add_to(m)
            folium.Marker(coords[-1], tooltip=f"End {end}").add_to(m)
        map_path = outputs_dir / "plots" / "route_map.html"
        m.save(str(map_path))
        print(f"[routing] Saved route map to {map_path}")
    except Exception as e:
        # Fallback: generate a simple Leaflet HTML without folium (使用边界经纬度)
        def node_rc(n):
            r = n // cols
            c = n % cols
            return r, c
        def rc_to_latlon(r, c):
            lat_range = lat_max - lat_min
            lon_range = lon_max - lon_min
            r_den = max(rows - 1, 1)
            c_den = max(cols - 1, 1)
            lat = lat_min + (r / r_den) * lat_range
            lon = lon_min + (c / c_den) * lon_range
            return lat, lon
        coords = []
        for n in path:
            r, c = node_rc(n)
            lat, lon = rc_to_latlon(r, c)
            coords.append([lat, lon])
        map_path = outputs_dir / "plots" / "route_map.html"
        center_lat = (lat_min + lat_max) / 2.0
        center_lon = (lon_min + lon_max) / 2.0
        save_leaflet_route_html(map_path, coords, start, end, center_lat=center_lat, center_lon=center_lon)
        print(f"[routing] Saved route map to {map_path} (Leaflet fallback). Reason: {e}")


def save_leaflet_route_html(html_path: Path, coords: list, start: int, end: int, center_lat: float = 30.0, center_lon: float = 120.0):
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    if coords and len(coords) > 0:
        center_lat, center_lon = coords[0][0], coords[0][1]
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Route Map</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
  <style>#map {{ height: 90vh; width: 100%; }}</style>
</head>
<body>
<div id=\"map\"></div>
<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
<script>
  var map = L.map('map').setView([{center_lat}, {center_lon}], 13);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19 }}).addTo(map);
  var coords = {coords};
  if (coords.length > 0) {{
    var poly = L.polyline(coords, {{ color: 'red', weight: 5, opacity: 0.8 }}).addTo(map);
    map.fitBounds(poly.getBounds());
    L.marker(coords[0]).bindTooltip('Start {start}').addTo(map);
    L.marker(coords[coords.length - 1]).bindTooltip('End {end}').addTo(map);
  }}
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()