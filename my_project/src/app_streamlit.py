import json
from pathlib import Path
import os
import threading
import time

import pandas as pd
import streamlit as st

# Folium embed (fallback to components when streamlit-folium is unavailable)
try:
    import folium
    _FOLIUM_AVAILABLE = True
except Exception:
    folium = None
    _FOLIUM_AVAILABLE = False

try:
    from streamlit_folium import folium_static
    _SF_AVAILABLE = True
except Exception:
    _SF_AVAILABLE = False

from train_eval import load_config, ensure_dirs, train_eval
from inference import run_inference
from routing import build_graph, load_congestion, dijkstra, load_weather_events


def node_rc(n, cols):
    r = n // cols
    c = n % cols
    return r, c


def rc_to_latlon(r, c, base_lat=30.0, base_lon=120.0, scale=0.001):
    return base_lat + r * scale, base_lon + c * scale


def render_route_folium(path_nodes, cols, cong, weather, events):
    coords = [rc_to_latlon(*node_rc(n, cols)) for n in path_nodes]
    if not _FOLIUM_AVAILABLE:
        return None
    # center
    center = coords[0] if coords else (30.0, 120.0)
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    # polyline
    folium.PolyLine(locations=coords, color="blue", weight=4, opacity=0.7).add_to(m)
    # markers
    if coords:
        folium.Marker(coords[0], tooltip="起点", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(coords[-1], tooltip="终点", icon=folium.Icon(color="red")).add_to(m)
    for n, (lat, lon) in zip(path_nodes, coords):
        c = cong.get(n, 0.0)
        w = weather.get(n, 0.0)
        e = events.get(n, 0.0)
        popup = folium.Popup(html=f"节点 {n}<br/>拥堵: {c:.2f}<br/>天气: {w:.2f}<br/>事件: {e:.2f}", max_width=250)
        folium.CircleMarker(location=(lat, lon), radius=4, color="orange", fill=True, fill_opacity=0.6, popup=popup).add_to(m)
    return m


def save_routing_to_config(cfg_path: Path, alpha: float, beta: float, gamma: float, avg_steps: int, epochs: int, clip_norm: float, lr_scheduler: str, step_size: int, lr_gamma: float):
    cfg = load_config(str(cfg_path))
    if "routing" not in cfg:
        cfg["routing"] = {}
    if "training" not in cfg:
        cfg["training"] = {}
    cfg["routing"].update({
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "avg_steps": int(avg_steps),
    })
    cfg["training"].update({
        "epochs": int(epochs),
        "clip_norm": float(clip_norm),
        "lr_scheduler": str(lr_scheduler),
        "step_size": int(step_size),
        "gamma": float(lr_gamma),
    })
    with open(cfg_path, "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def main():
    st.set_page_config(page_title="交通应急快速响应（GCN+LSTM）", layout="wide")
    st.title("交通应急快速响应系统（最小演示版）")

    cfg_path = st.text_input("配置文件路径", value=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    cfg = load_config(cfg_path)
    data_dir, outputs_dir = ensure_dirs(cfg)

    # Session state defaults from config (enable Load)
    routing_cfg = cfg.get("routing", {})
    training_cfg = cfg.get("training", {})
    st.sidebar.header("参数")
    if "alpha" not in st.session_state:
        st.session_state.alpha = float(routing_cfg.get("alpha", 1.0))
    if "beta" not in st.session_state:
        st.session_state.beta = float(routing_cfg.get("beta", 0.2))
    if "gamma" not in st.session_state:
        st.session_state.gamma = float(routing_cfg.get("gamma", 0.5))
    if "avg_steps" not in st.session_state:
        st.session_state.avg_steps = int(routing_cfg.get("avg_steps", 3))
    if "epochs" not in st.session_state:
        st.session_state.epochs = int(training_cfg.get("epochs", 12))
    # new: training params session defaults
    if "clip_norm" not in st.session_state:
        st.session_state.clip_norm = float(training_cfg.get("clip_norm", 1.0))
    if "lr_scheduler" not in st.session_state:
        st.session_state.lr_scheduler = str(training_cfg.get("lr_scheduler", "none"))
    if "step_size" not in st.session_state:
        st.session_state.step_size = int(training_cfg.get("step_size", 5))
    if "lr_gamma" not in st.session_state:
        st.session_state.lr_gamma = float(training_cfg.get("gamma", 0.5))

    meta = json.load(open(data_dir / "grid_topology.json", "r", encoding="utf-8"))
    n_nodes = int(meta["n_nodes"]) ; cols = int(meta["cols"]) ; rows = int(meta["rows"]) ; cell = int(meta["cell_size_m"])

    start = st.sidebar.number_input("起点节点", min_value=0, max_value=n_nodes - 1, value=0)
    end = st.sidebar.number_input("终点节点", min_value=0, max_value=n_nodes - 1, value=n_nodes - 1)
    alpha = st.sidebar.number_input("α 拥堵权重", min_value=0.0, max_value=5.0, value=st.session_state.alpha, step=0.1, key="alpha_input")
    beta = st.sidebar.number_input("β 天气权重", min_value=0.0, max_value=2.0, value=st.session_state.beta, step=0.1, key="beta_input")
    gamma = st.sidebar.number_input("γ 事件权重", min_value=0.0, max_value=2.0, value=st.session_state.gamma, step=0.1, key="gamma_input")
    avg_steps = st.sidebar.number_input("拥堵多步平均K", min_value=1, max_value=12, value=st.session_state.avg_steps, step=1, key="avg_steps_input")
    epochs = st.sidebar.number_input("训练轮次", min_value=1, max_value=50, value=st.session_state.epochs, step=1, key="epochs_input")
    # Training robustness controls (now from session state)
    clip_norm = st.sidebar.slider("梯度裁剪 max-norm", min_value=0.0, max_value=5.0, value=st.session_state.clip_norm, step=0.1)
    lr_options = ["none", "step", "cosine"]
    lr_scheduler = st.sidebar.selectbox("学习率调度", options=lr_options, index=(lr_options.index(st.session_state.lr_scheduler) if st.session_state.lr_scheduler in lr_options else 0))
    step_size = st.sidebar.number_input("StepLR 步长", min_value=1, max_value=50, value=st.session_state.step_size)
    lr_gamma = st.sidebar.number_input("StepLR 衰减系数", min_value=0.1, max_value=0.99, value=st.session_state.lr_gamma, step=0.05)

    cols_top = st.columns(4)
    with cols_top[0]:
        if st.button("保存配置到 config.yaml"):
            save_routing_to_config(Path(cfg_path), alpha, beta, gamma, int(avg_steps), int(epochs), float(clip_norm), str(lr_scheduler), int(step_size), float(lr_gamma))
            st.success("已保存到 config.yaml")
    with cols_top[1]:
        if st.button("从 config.yaml 加载参数"):
            cfg_latest = load_config(cfg_path)
            r = cfg_latest.get("routing", {})
            t = cfg_latest.get("training", {})
            st.session_state.alpha = float(r.get("alpha", st.session_state.alpha))
            st.session_state.beta = float(r.get("beta", st.session_state.beta))
            st.session_state.gamma = float(r.get("gamma", st.session_state.gamma))
            st.session_state.avg_steps = int(r.get("avg_steps", st.session_state.avg_steps))
            st.session_state.epochs = int(t.get("epochs", st.session_state.epochs))
            # new: read training params
            st.session_state.clip_norm = float(t.get("clip_norm", st.session_state.clip_norm))
            st.session_state.lr_scheduler = str(t.get("lr_scheduler", st.session_state.lr_scheduler))
            st.session_state.step_size = int(t.get("step_size", st.session_state.step_size))
            st.session_state.lr_gamma = float(t.get("gamma", st.session_state.lr_gamma))
            st.experimental_rerun()

    col_train, col_pred, col_route = st.columns(3)

    with col_train:
        st.subheader("1) 训练与评估")
        progress_ph = st.empty()
        log_ph = st.empty()
        def _train_thread():
            try:
                train_eval(cfg_path, epochs=int(epochs), clip_norm=float(clip_norm), lr_scheduler=str(lr_scheduler), step_size=int(step_size), gamma=float(lr_gamma))
            except Exception as e:
                st.session_state.train_error = str(e)
        if st.button("开始训练"):
            st.session_state.train_error = ""
            th = threading.Thread(target=_train_thread, daemon=True)
            th.start()
            # stream logs while thread is alive
            events_log = outputs_dir / "logs" / "train_events.log"
            seen_lines = 0
            ep_total = int(epochs)
            while th.is_alive():
                time.sleep(0.5)
                try:
                    if events_log.exists():
                        with open(events_log, "r", encoding="utf-8") as lf:
                            lines = lf.readlines()
                        new_lines = lines[seen_lines:]
                        if new_lines:
                            seen_lines = len(lines)
                            # parse epoch count
                            last_ep = 0
                            for ln in new_lines:
                                log_ph.write(ln)
                                if "epoch=" in ln:
                                    try:
                                        last_ep = int(ln.split("epoch=")[1].split()[0])
                                    except Exception:
                                        pass
                            if last_ep > 0:
                                progress_ph.progress(min(1.0, last_ep / max(1, ep_total)))
                except Exception:
                    pass
            # final update
            progress_ph.progress(1.0)
            if getattr(st.session_state, "train_error", ""):
                st.error(f"训练失败: {st.session_state.train_error}")
            else:
                st.success("训练完成，模型与日志已保存。")
        # 曲线图
        log_path = outputs_dir / "logs" / "train_log.json"
        if log_path.exists():
            log = json.load(open(log_path, "r", encoding="utf-8"))
            df_log = pd.DataFrame({"epoch": list(range(1, int(log.get("epochs", 0)) + 1)),
                                   "train_mse": log.get("train_mse", []),
                                   "eval_mse": log.get("eval_mse", [])})
            st.line_chart(df_log.set_index("epoch"))
        else:
            st.info("尚无训练日志。")

    with col_pred:
        st.subheader("2) 预测生成")
        if st.button("运行预测（使用最新数据窗口）"):
            try:
                pred_path = run_inference(cfg_path)
                st.success(f"预测已生成: {pred_path}")
            except Exception as e:
                st.error(f"预测失败: {e}")
        pred_csv = outputs_dir / "logs" / "predictions.csv"
        if pred_csv.exists():
            preds = pd.read_csv(pred_csv)
            st.write("预测摘要：", preds.head())
        else:
            st.info("尚无预测结果，请点击上方按钮生成。")

    with col_route:
        st.subheader("3) 路径规划（Folium 地图）")
        graph = build_graph(data_dir / "graph_edges.csv")
        pred_csv = outputs_dir / "logs" / "predictions.csv"
        if pred_csv.exists():
            try:
                cong = load_congestion(pred_csv, avg_steps=int(avg_steps))
                weather, events = load_weather_events(data_dir)
                path, cost = dijkstra(graph, int(start), int(end), cong, weather, events, alpha=float(alpha), beta=float(beta), gamma=float(gamma))
                st.write(f"路径长度（估计成本）: {cost:.2f}")
                st.write(f"路径节点数: {len(path)}")
                # Folium render
                fmap = render_route_folium(path, cols, cong, weather, events)
                if fmap is not None:
                    if _SF_AVAILABLE:
                        folium_static(fmap, width=600, height=400)
                    else:
                        html = fmap._repr_html_()
                        st.components.v1.html(html, height=420, scrolling=True)
                else:
                    st.warning("Folium 未可用，回退至 Streamlit 原生点图。")
                    coords = [rc_to_latlon(*node_rc(n, cols)) for n in path]
                    st.map(pd.DataFrame({"lat": [lat for lat, _ in coords], "lon": [lon for _, lon in coords]}))
                # 导出路线报告
                reports_dir = outputs_dir / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                route_json = reports_dir / "route.json"
                with open(route_json, "w", encoding="utf-8") as f:
                    json.dump({"path": path, "cost": cost}, f, ensure_ascii=False, indent=2)
                st.download_button(label="下载路线JSON", data=route_json.read_bytes(), file_name="route.json")
            except Exception as e:
                st.error(f"路径规划失败: {e}")
        else:
            st.info("请先生成预测，再进行路径规划。")

    st.subheader("4) 解释报告（模板/模型/长度）")
    from explain import generate_explanation
    use_llm = st.checkbox("启用 LLM 生成（可选）", value=False)
    model_choice = st.selectbox("选择模型", [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ], index=0)
    template_choice = st.selectbox("解释模板", ["summary", "diagnostic", "route_rationale"], index=0)
    max_tokens = st.slider("响应长度（tokens）", min_value=64, max_value=1024, value=256, step=32)
    if st.button("生成解释（LLM/模板）"):
        try:
            metrics_path = Path(outputs_dir / "logs" / "train_log.json")
            route_path = Path(outputs_dir / "reports" / "route.json")
            params = {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "avg_steps": int(avg_steps),
                "epochs": int(epochs),
            }
            explanation = generate_explanation(metrics_path, route_path, params,
                                               use_llm=bool(use_llm), llm_model=str(model_choice),
                                               template=str(template_choice), max_tokens=int(max_tokens))
            # 保存并显示
            reports_dir = outputs_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            txt_path = reports_dir / "explanation.txt"
            json_path = reports_dir / "explanation.json"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(explanation)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"model": model_choice, "template": template_choice, "text": explanation}, f, ensure_ascii=False, indent=2)
            st.success("解释生成完成。")
        except Exception as e:
            st.error(f"解释生成失败: {e}")
    report_json = outputs_dir / "reports" / "explanation.json"
    if report_json.exists():
        report = json.load(open(report_json, "r", encoding="utf-8"))
        st.json(report)
    text_path = outputs_dir / "reports" / "explanation.txt"
    if text_path.exists():
        st.download_button(label="下载解释文本", data=text_path.read_bytes(), file_name="explanation.txt")


if __name__ == "__main__":
    main()