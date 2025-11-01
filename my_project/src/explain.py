import argparse
import json
from pathlib import Path
from typing import Optional

import yaml

# Optional transformers import for LLM mode
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


def load_config(cfg_path: Path) -> dict:
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _template_explanations(template: str, data: dict, params: dict, max_chars: int = 800) -> str:
    alpha = params.get("alpha", 1.0)
    beta = params.get("beta", 0.2)
    gamma = params.get("gamma", 0.5)
    avg_steps = params.get("avg_steps", 3)
    epochs = params.get("epochs", 12)

    base = {
        "summary": (
            f"模型以 {epochs} 轮训练完成；路由采用拥堵权重 alpha={alpha}, 天气权重 beta={beta}, 事件权重 gamma={gamma}, "
            f"并对未来 {avg_steps} 步进行多步平均。整体性能与解释如下：\n"),
        "diagnostic": (
            f"诊断视角：在训练与评估过程中，权重设定(alpha={alpha}, beta={beta}, gamma={gamma})影响路径代价；" 
            f"多步平均({avg_steps})平滑短时波动；若评估MSE升高，可考虑增大训练轮次({epochs}+)，或下调alpha。\n"),
        "route_rationale": (
            f"路径选择逻辑：基于基础距离乘以权重(拥堵、天气、事件)的综合代价进行Dijkstra搜索；权重为"
            f"alpha={alpha}, beta={beta}, gamma={gamma}，并使用K={avg_steps}步平均拥堵以提升稳健性。\n"),
    }
    text = base.get(template, base["summary"]) + json.dumps(data, ensure_ascii=False, indent=2)
    return text[:max_chars]


def _llm_generate(model_name: str, prompt: str, max_new_tokens: int = 256) -> Optional[str]:
    if not _TRANSFORMERS_AVAILABLE:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)
        out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        return out[0]["generated_text"]
    except Exception:
        return None


def generate_explanation(metrics_path: Path, route_path: Path, params: dict, use_llm: bool = False,
                         llm_model: Optional[str] = None, template: str = "summary", max_tokens: int = 256) -> str:
    data = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data["metrics"] = json.load(f)
        except Exception:
            pass
    if route_path.exists():
        try:
            with open(route_path, "r", encoding="utf-8") as f:
                data["route"] = json.load(f)
        except Exception:
            pass

    prompt = (
        "你是一名交通调度助手。请依据以下模型训练指标与路径信息，生成一段清晰、结构化的中文解释，"
        "包括总体表现、关键影响因素、路径选择理由与可能的改进建议。\n\n"
        f"权重: alpha={params.get('alpha')}, beta={params.get('beta')}, gamma={params.get('gamma')}, K={params.get('avg_steps')}, epochs={params.get('epochs')}\n"
        f"数据: {json.dumps(data, ensure_ascii=False)}\n"
    )

    if use_llm and llm_model:
        llm_out = _llm_generate(llm_model, prompt, max_new_tokens=max_tokens)
        if llm_out:
            return llm_out
    # fallback to templates
    return _template_explanations(template, data, params, max_chars=max_tokens * 4)  # approx chars


def main():
    parser = argparse.ArgumentParser(description="Generate explanations for traffic forecasting and routing")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    parser.add_argument("--metrics", type=str, default=str(Path(__file__).resolve().parent.parent / "outputs" / "logs" / "train_log.json"))
    parser.add_argument("--route", type=str, default=str(Path(__file__).resolve().parent.parent / "outputs" / "reports" / "route.json"))
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--template", type=str, choices=["summary", "diagnostic", "route_rationale"], default="summary")
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    params = {
        "alpha": cfg.get("routing", {}).get("alpha", 1.0),
        "beta": cfg.get("routing", {}).get("beta", 0.2),
        "gamma": cfg.get("routing", {}).get("gamma", 0.5),
        "avg_steps": cfg.get("routing", {}).get("avg_steps", 3),
        "epochs": cfg.get("training", {}).get("epochs", 12),
    }

    explanation = generate_explanation(Path(args.metrics), Path(args.route), params,
                                       use_llm=args.use_llm, llm_model=args.model,
                                       template=args.template, max_tokens=args.max_tokens)
    print(explanation)


if __name__ == "__main__":
    main()