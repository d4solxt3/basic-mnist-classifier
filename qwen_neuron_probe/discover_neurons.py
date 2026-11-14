import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


def get_layers(model):
    """Locate transformer decoder layers across Qwen3 and Qwen3.5 architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    raise AttributeError("Could not locate decoder layers on model.")


def load_prompt_texts(path: Path) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["prompt"] if isinstance(obj, dict) else str(obj))
    if not texts:
        raise ValueError(f"No prompts found in {path}")
    return texts


@torch.no_grad()
def collect_mlp_activations(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    token_pos: str = "last",
) -> np.ndarray:
    """
    Capture post-SwiGLU activations entering mlp.down_proj across all layers.
    Returns float32 numpy array of shape (N_prompts, N_layers, intermediate_size).
    """
    layers = get_layers(model)
    n_layers = len(layers)
    n_neurons = layers[0].mlp.down_proj.in_features

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]

    all_batches = []
    batch_store: dict[int, torch.Tensor] = {}
    current_mask: torch.Tensor | None = None

    def make_pre_hook(layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0]  # (B, T, intermediate_size)
            if token_pos == "last":
                vec = x[:, -1, :]
            elif token_pos == "mean":
                mask = current_mask.unsqueeze(-1).to(x.dtype)
                vec = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                raise ValueError(f"Unknown token_pos: {token_pos}")
            batch_store[layer_idx] = vec.detach().float().cpu()

        return hook

    handles = [
        layers[l].mlp.down_proj.register_forward_pre_hook(make_pre_hook(l))
        for l in range(n_layers)
    ]

    try:
        for i in range(0, len(formatted), batch_size):
            batch_txt = formatted[i : i + batch_size]
            enc = tokenizer(batch_txt, return_tensors="pt", padding=True).to(model.device)
            assert (enc["attention_mask"][:, -1] == 1).all(), "Expected left-padded batch."
            current_mask = enc["attention_mask"]
            batch_store.clear()

            model(**enc)

            stacked = torch.stack([batch_store[l] for l in range(n_layers)], dim=1)
            assert stacked.shape[1:] == (n_layers, n_neurons)
            all_batches.append(stacked.numpy())
    finally:
        for h in handles:
            h.remove()

    return np.concatenate(all_batches, axis=0)


@torch.no_grad()
def compute_down_proj_col_norms(model) -> np.ndarray:
    """Compute ||W_down[:, n]||_2 for all layers/neurons in BF16 or 4-bit."""
    layers = get_layers(model)
    n_layers = len(layers)
    n_neurons = layers[0].mlp.down_proj.in_features
    norms = np.zeros((n_layers, n_neurons), dtype=np.float32)

    for l in range(n_layers):
        down_proj = layers[l].mlp.down_proj
        if hasattr(down_proj, "weight") and down_proj.weight.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            w = down_proj.weight.detach().float()
            norms[l] = torch.linalg.vector_norm(w, dim=0).cpu().numpy()
        else:
            w_dtype = next(down_proj.parameters()).dtype
            eye = torch.eye(n_neurons, device=model.device, dtype=w_dtype).unsqueeze(1)
            cols = down_proj(eye).float().squeeze(1)
            if getattr(down_proj, "bias", None) is not None:
                cols = cols - down_proj.bias.float()
            norms[l] = torch.linalg.vector_norm(cols, dim=-1).cpu().numpy()

    return norms


def _cohen_d(X_pos: np.ndarray, X_neg: np.ndarray) -> np.ndarray:
    """Pooled-SD Cohen's d per column (Eq. 3), float64 accumulation."""
    n_pos, n_neg = X_pos.shape[0], X_neg.shape[0]
    diff = X_pos.mean(axis=0, dtype=np.float64) - X_neg.mean(axis=0, dtype=np.float64)
    var_pos = X_pos.var(axis=0, ddof=1, dtype=np.float64) if n_pos > 1 else 0.0
    var_neg = X_neg.var(axis=0, ddof=1, dtype=np.float64) if n_neg > 1 else 0.0
    pooled_sd = np.sqrt(
        ((max(n_pos - 1, 1)) * var_pos + (max(n_neg - 1, 1)) * var_neg)
        / max(n_pos + n_neg - 2, 1)
    )
    return np.where(pooled_sd > 1e-6, diff / np.maximum(pooled_sd, 1e-12), 0.0)


def rank_neurons_with_null(
    acts_pos: np.ndarray,
    acts_neg: np.ndarray,
    col_norms: np.ndarray,
    n_permutations: int = 200,
    seed: int = 42,
    min_active_rate: float = 0.25,
    alpha_fwer: float = 0.01,
    top_k: int = 20,
) -> tuple[list[dict], dict]:
    """
    Compute per-neuron Cohen's d between pos and neg prompt sets, plus a
    whole-network shuffled-label permutation null distribution.
    """
    n_pos, n_layers, n_neurons = acts_pos.shape
    n_neg = acts_neg.shape[0]
    D = n_layers * n_neurons

    X_pos = acts_pos.reshape(n_pos, D)
    X_neg = acts_neg.reshape(n_neg, D)

    mean_pos = X_pos.mean(axis=0)
    mean_neg = X_neg.mean(axis=0)
    cohen_d = _cohen_d(X_pos, X_neg)

    fire_pos = (X_pos > 0).mean(axis=0)
    fire_neg = (X_neg > 0).mean(axis=0)
    active_mask = np.maximum(fire_pos, fire_neg) >= min_active_rate

    # Eq. 4: composite ranking score S = M * |d| * ||w_down||
    masked_abs_d = np.where(active_mask, np.abs(cohen_d), 0.0)
    flat_norms = col_norms.reshape(D).astype(np.float64)
    score = masked_abs_d * flat_norms

    # Westfall-Young max-statistic null: T_b = max_{l,n} M_{l,n} |d^(pi_b)_{l,n}| (fixed mask M)
    rng = np.random.default_rng(seed)
    X_all = np.concatenate([X_pos, X_neg], axis=0)
    n_all = n_pos + n_neg
    null_max = np.empty(n_permutations, dtype=np.float64)
    for b in range(n_permutations):
        perm = rng.permutation(n_all)
        d_b = _cohen_d(X_all[perm[:n_pos]], X_all[perm[n_pos:]])
        null_max[b] = np.max(np.where(active_mask, np.abs(d_b), 0.0))

    # Eq. 5: p_FWER = (1/B) * #{b : T_b >= |d|}, for every unit at once
    null_sorted = np.sort(null_max)
    exceed = n_permutations - np.searchsorted(null_sorted, masked_abs_d, side="left")
    p_fwer = exceed / n_permutations
    passes = active_mask & (p_fwer < alpha_fwer)

    def unit(idx: int) -> dict:
        l, n = divmod(int(idx), n_neurons)
        return {
            "layer": l,
            "neuron": n,
            "cohen_d": float(cohen_d[idx]),
            "abs_d": float(abs(cohen_d[idx])),
            "score": float(score[idx]),
            "col_norm": float(flat_norms[idx]),
            "mean_pos": float(mean_pos[idx]),
            "mean_neg": float(mean_neg[idx]),
            "fire_pos": float(fire_pos[idx]),
            "fire_neg": float(fire_neg[idx]),
            "active": bool(active_mask[idx]),
            "null_exceedances": int(exceed[idx]),
            "p_fwer": float(p_fwer[idx]),
            "passes": bool(passes[idx]),
        }

    order = np.argsort(-score)
    ranked = [unit(i) for i in order[:top_k]]

    # Stage 2 candidate: highest S among passing units; else the top-S unit (exploratory only).
    passing_idx = np.flatnonzero(passes)
    if passing_idx.size:
        cand_idx = int(passing_idx[np.argmax(score[passing_idx])])
    else:
        cand_idx = int(order[0])
    top_d_idx = int(np.argmax(masked_abs_d))

    stats = {
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "n_units": int(D),
        "n_active": int(active_mask.sum()),
        "n_permutations": int(n_permutations),
        "permutation_seed": int(seed),
        "min_active_rate": float(min_active_rate),
        "alpha_fwer": float(alpha_fwer),
        "null_max_abs_d": [float(x) for x in null_max],
        "null_q95": float(np.quantile(null_max, 0.95)),
        "null_q99": float(np.quantile(null_max, 0.99)),
        "null_mean": float(null_max.mean()),
        "n_passing_units": int(passing_idx.size),
        "min_p_fwer": float(p_fwer[active_mask].min()) if active_mask.any() else 1.0,
        "candidate": unit(cand_idx),
        "candidate_passes": bool(passes[cand_idx]),
        "top_abs_d_unit": unit(top_d_idx),
    }
    return ranked, stats


def screen_verdict(min_p: float, alpha_fwer: float, alpha_inconclusive: float) -> str:
    if min_p < alpha_fwer:
        return "pass"
    if min_p < alpha_inconclusive:
        return "inconclusive"
    return "refuted"


def main():
    from common import (OUTPUTS, env_block, env_info, load_config, load_jsonl, load_model,
                        n_params, peak_rss_gb, runlog, save_json, weight_bytes)

    ap = argparse.ArgumentParser(description="Stage 1: whole-network single-neuron discovery.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--axes", nargs="*", default=None, help="Subset of axes (default: all in config).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.config).resolve().parent
    axes = args.axes or cfg["axes"]

    t0 = time.time()
    model, tok, commit, device = load_model(cfg)
    t_load = time.time() - t0
    env = env_info(device, commit)
    col_norms = compute_down_proj_col_norms(model)

    resources = {"model_load_s": t_load, "n_params": n_params(model), "weight_bytes": weight_bytes(model),
                 "dtype": cfg["dtype"], "device": str(device), "stage1_axes": {}}
    summary_lines = []
    for axis in axes:
        ta = time.time()
        rows = load_jsonl(root / cfg["datasets_dir"] / f"{axis}.jsonl")
        pos = [r["prompt"] for r in rows if r["label"] == 1]
        neg = [r["prompt"] for r in rows if r["label"] == 0]
        data_sha = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()

        acts_pos = collect_mlp_activations(model, tok, pos, cfg["batch_size"])
        acts_neg = collect_mlp_activations(model, tok, neg, cfg["batch_size"])
        t_acts = time.time() - ta
        ranked, stats = rank_neurons_with_null(
            acts_pos, acts_neg, col_norms,
            n_permutations=cfg["n_permutations"], seed=cfg["permutation_seed"],
            min_active_rate=cfg["min_active_rate"], alpha_fwer=cfg["alpha_fwer"], top_k=cfg["top_k_report"],
        )
        verdict = screen_verdict(stats["min_p_fwer"], cfg["alpha_fwer"], cfg["alpha_inconclusive"])
        wall = time.time() - ta
        out = {"axis": axis, "verdict": verdict, "dataset_sha256": data_sha, "env": env,
               "wall_s": wall, "activation_pass_s": t_acts, "stats": stats, "top_units": ranked}
        save_json(out, OUTPUTS / f"stage1_{axis}.json")
        resources["stage1_axes"][axis] = {"wall_s": wall, "activation_pass_s": t_acts}
        c = stats["candidate"]
        summary_lines.append(
            f"| {axis} | L{c['layer']} N{c['neuron']} | {c['cohen_d']:.6f} | {stats['null_q99']:.6f} | "
            f"{c['p_fwer']:.6f} ({c['null_exceedances']}/{stats['n_permutations']}) | {c['col_norm']:.6f} | "
            f"{stats['n_passing_units']} | {stats['min_p_fwer']:.6f} | {verdict} | {wall:.1f} |"
        )
        print(summary_lines[-1], flush=True)

    resources["stage1_total_s"] = time.time() - t0
    resources["stage1_peak_rss_gb"] = peak_rss_gb()
    prev = OUTPUTS / "resources.json"
    merged = json.loads(prev.read_text()) if prev.exists() else {}
    merged.update(resources)
    save_json(merged, prev)

    runlog(
        f"## Stage 1 discovery ({datetime.now(timezone.utc).isoformat(timespec='seconds')})\n\n"
        f"Command: `python discover_neurons.py --config {args.config}"
        f"{' --axes ' + ' '.join(args.axes) if args.axes else ''}`\n\n{env_block(env)}\n"
        f"- dtype: `{cfg['dtype']}`; parameters: `{resources['n_params']}`; weight bytes: `{resources['weight_bytes']}`\n"
        f"- model load: `{t_load:.1f} s`; Stage 1 total wall time: `{resources['stage1_total_s']:.1f} s`; "
        f"peak RSS: `{resources['stage1_peak_rss_gb']:.3f} GB`\n\n"
        "Candidate = highest composite score S among units with p_FWER < 0.01 (top-S unit if none pass).\n\n"
        "| Axis | Candidate | Cohen's d | Null 99% |d| | p_FWER (exceed/B) | ‖w_down‖ | # passing units | min p_FWER | Verdict | Wall s |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(summary_lines)
    )


if __name__ == "__main__":
    main()
