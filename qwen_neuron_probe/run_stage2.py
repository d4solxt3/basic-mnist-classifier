"""Stage 2: residual-norm-calibrated causal steering of Stage 1 candidates.

For each axis the unit (layer*, neuron*) comes from outputs/stage1_<axis>.json. Interventions are additive
pushes on the post-SwiGLU input to mlp.down_proj at every token position:
    z_n <- z_n + sign(d*) * alpha * rho_l / ||W_down[:, n]||,
so the residual perturbation is exactly alpha * rho_l in l2 norm (up to BF16 rounding of the pushed value).
"""
import argparse
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from common import (OUTPUTS, env_block, env_info, format_chat, get_layers, load_config, load_jsonl,
                    load_model, peak_rss_gb, runlog, save_json)


# ----------------------------------------------------------------------------- hooks
@contextmanager
def push(model, layer: int | None, neuron: int | None, value: float):
    """Add `value` to coordinate `neuron` of the mlp.down_proj input at `layer` (no-op if layer is None)."""
    if layer is None:
        yield
        return
    down = get_layers(model)[layer].mlp.down_proj

    def hook(_m, inputs):
        x = inputs[0].clone()
        x[..., neuron] = (x[..., neuron].float() + value).to(x.dtype)
        return (x,) + tuple(inputs[1:])

    h = down.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        h.remove()


def _encode(tok, texts: list[str], device):
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return enc


# ------------------------------------------------------------------------ calibration
@torch.no_grad()
def residual_norm_medians(model, tok, formatted: list[str], device) -> list[float]:
    """rho_l = median ||h^(l)||_2 over non-pad tokens excluding each sequence's first token."""
    layers = get_layers(model)
    enc = _encode(tok, formatted, device)
    mask = enc["attention_mask"].clone()
    first = mask.argmax(dim=1)  # first real token (left padding)
    mask[torch.arange(mask.shape[0]), first] = 0
    norms: dict[int, torch.Tensor] = {}

    def mk(l):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            norms[l] = torch.linalg.vector_norm(h.float(), dim=-1)[mask.bool()].cpu()
        return hook

    hs = [layers[l].register_forward_hook(mk(l)) for l in range(len(layers))]
    try:
        model(**enc)
    finally:
        for h in hs:
            h.remove()
    return [float(norms[l].median()) for l in range(len(layers))]


# ---------------------------------------------------------------------- word scoring
def build_token_map(tok, words: list[str]) -> dict[str, list[int]]:
    # The answer starts right after "<think>\n\n</think>\n\n", so words are tokenized without a leading space.
    return {w: tok.encode(w, add_special_tokens=False) for w in words}


_LM_HEAD_FP32 = {}


def fp32_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    """Vocabulary projection in float32 (BF16 logits are quantized to 0.125 near |logit| ~ 30)."""
    key = id(model)
    if key not in _LM_HEAD_FP32:
        _LM_HEAD_FP32[key] = model.get_output_embeddings().weight.detach().float()
    return hidden.float() @ _LM_HEAD_FP32[key].T


def _last_hidden(model, **kw) -> torch.Tensor:
    return model.model(**kw).last_hidden_state  # final RMSNorm already applied


@torch.no_grad()
def word_logprobs(model, tok, formatted: list[str], token_map: dict[str, list[int]], device,
                  batch_size: int = 32) -> np.ndarray:
    """Exact log P(word | prompt) for each prompt x word; multi-token words use the chain rule."""
    words = list(token_map)
    out = np.zeros((len(formatted), len(words)), dtype=np.float64)
    enc = _encode(tok, formatted, device)
    lp = torch.log_softmax(fp32_logits(model, _last_hidden(model, **enc)[:, -1, :]), dim=-1).cpu()
    for j, w in enumerate(words):
        out[:, j] = lp[:, token_map[w][0]].numpy()

    prompt_ids = [tok.encode(f, add_special_tokens=False) for f in formatted]
    jobs = [(i, j) for i in range(len(formatted)) for j, w in enumerate(words) if len(token_map[w]) > 1]
    for s in range(0, len(jobs), batch_size):
        chunk = jobs[s:s + batch_size]
        seqs = [prompt_ids[i] + token_map[words[j]][:-1] for i, j in chunk]
        T = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, T - len(x):] = torch.tensor(x)
            att[r, T - len(x):] = 1
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        kmax = max(len(token_map[words[j]]) for _, j in chunk) - 1
        hid = _last_hidden(model, input_ids=ids.to(device), attention_mask=att.to(device),
                           position_ids=pos.to(device))[:, T - kmax:, :]
        logp = torch.log_softmax(fp32_logits(model, hid), dim=-1).cpu()  # positions T-kmax .. T-1
        for r, (i, j) in enumerate(chunk):
            wt = token_map[words[j]]
            k = len(wt)
            for t in range(1, k):
                out[i, j] += float(logp[r, kmax - (k - 1) + (t - 1), wt[t]])
    return out


def delta_logp(model, tok, formatted, tmap_targ, tmap_neut, device) -> np.ndarray:
    """Per-prompt Delta log P = logsumexp_targ log P(w) - logsumexp_neut log P(w)."""
    both = {**tmap_targ, **tmap_neut}
    lp = word_logprobs(model, tok, formatted, both, device)
    words = list(both)
    it = [words.index(w) for w in tmap_targ]
    ineu = [words.index(w) for w in tmap_neut]
    lse = lambda a: np.log(np.exp(a - a.max(axis=1, keepdims=True)).sum(axis=1)) + a.max(axis=1)
    return lse(lp[:, it]) - lse(lp[:, ineu])


# -------------------------------------------------------------------------- generation
def distinct_2(texts: list[str]) -> float:
    uniq, total = set(), 0
    for t in texts:
        w = t.lower().split()
        bg = list(zip(w, w[1:]))
        uniq.update(bg)
        total += len(bg)
    return len(uniq) / total if total else 0.0


@torch.no_grad()
def generate(model, tok, formatted, cfg, seed: int, device) -> list[str]:
    outs = []
    torch.manual_seed(seed)
    bs = cfg["gen_batch_size"]
    for s in range(0, len(formatted), bs):
        enc = tok(formatted[s:s + bs], return_tensors="pt", padding=True).to(device)
        g = model.generate(**enc, do_sample=True, temperature=cfg["temperature"], top_p=cfg["top_p"],
                           max_new_tokens=cfg["max_new_tokens"], pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def msd(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0, "n": int(x.size)}


def random_neuron(seed: int, layer: int, exclude: int, n_neurons: int) -> int:
    rng = np.random.default_rng([seed, layer, 7919])
    while True:
        m = int(rng.integers(n_neurons))
        if m != exclude:
            return m


# --------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Stage 2: causal steering of Stage 1 candidates.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--axes", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(args.config).resolve().parent
    axes = args.axes or cfg["axes"]

    t0 = time.time()
    model, tok, commit, device = load_model(cfg)
    env = env_info(device, commit)
    layers = get_layers(model)
    n_neurons = layers[0].mlp.down_proj.in_features
    seeds, doses, rd = cfg["seeds"], cfg["doses"], cfg["report_dose"]

    ood = load_jsonl(root / cfg["prompts_file"])
    ood_fmt = format_chat(tok, [r["prompt"] for r in ood])
    token_sets = json.loads((root / cfg["token_sets_file"]).read_text(encoding="utf-8"))

    rho = residual_norm_medians(model, tok, ood_fmt, device)
    save_json({"rho_median_post_layer_l2": rho, "n_prompts": len(ood), "env": env}, OUTPUTS / "calibration.json")

    # Baseline generations do not depend on the axis: compute once per seed.
    t_gen0 = time.time()
    base_gen = {s: generate(model, tok, ood_fmt, cfg, s, device) for s in seeds}
    base_gen_s = time.time() - t_gen0
    gen_log = [{"condition": "baseline", "seed": s, "prompt_id": ood[i]["id"], "text": t}
               for s in seeds for i, t in enumerate(base_gen[s])]

    timings = {"baseline_generation_s": base_gen_s, "axes": {}}
    summary_rows = []
    for axis in axes:
        ta = time.time()
        s1 = json.loads((OUTPUTS / f"stage1_{axis}.json").read_text(encoding="utf-8"))
        cand = s1["stats"]["candidate"]
        L, N, d = cand["layer"], cand["neuron"], cand["cohen_d"]
        sign = 1.0 if d >= 0 else -1.0
        w_norm = float(torch.linalg.vector_norm(layers[L].mlp.down_proj.weight.detach()[:, N].float()))
        col_norm = lambda n: float(torch.linalg.vector_norm(layers[L].mlp.down_proj.weight.detach()[:, n].float()))
        v = lambda n, a: sign * a * rho[L] / col_norm(n)

        tm_t = build_token_map(tok, token_sets[axis]["target_tokens"])
        tm_n = build_token_map(tok, token_sets[axis]["neutral_tokens"])
        rand_units = {s: random_neuron(s, L, N, n_neurons) for s in seeds}

        # ---- OOD first-token margins (deterministic: seed only changes the random unit)
        base = delta_logp(model, tok, ood_fmt, tm_t, tm_n, device)
        ood_res = {"baseline": {"per_prompt": base.tolist(), **msd(base)}, "target": {}, "random": {}}
        for a in doses:
            with push(model, L, N, v(N, a)):
                tgt = delta_logp(model, tok, ood_fmt, tm_t, tm_n, device)
            ood_res["target"][str(a)] = {"push_value": v(N, a), "per_prompt": tgt.tolist(), **msd(tgt),
                                         "shift": msd(tgt - base)}
            per_seed = {}
            for s in seeds:
                m = rand_units[s]
                with push(model, L, m, v(m, a)):
                    r = delta_logp(model, tok, ood_fmt, tm_t, tm_n, device)
                per_seed[str(s)] = {"neuron": m, "col_norm": col_norm(m), "push_value": v(m, a),
                                    "per_prompt": r.tolist(), "mean": float(r.mean())}
            rmean = np.mean([per_seed[str(s)]["per_prompt"] for s in seeds], axis=0)
            ood_res["random"][str(a)] = {"per_seed": per_seed, "per_prompt_seed_mean": rmean.tolist(),
                                         **msd(rmean), "shift": msd(rmean - base)}

        # ---- sampled generations: fluency guard (distinct_2) per seed
        gen_res = {"baseline": {str(s): distinct_2(base_gen[s]) for s in seeds}, "target": {}, "random": {}}
        samples = {"baseline": base_gen[seeds[0]][:cfg["samples_to_save"]]}
        for a in doses:
            gen_res["target"][str(a)], gen_res["random"][str(a)] = {}, {}
            for s in seeds:
                with push(model, L, N, v(N, a)):
                    tg = generate(model, tok, ood_fmt, cfg, s, device)
                m = rand_units[s]
                with push(model, L, m, v(m, a)):
                    rg = generate(model, tok, ood_fmt, cfg, s, device)
                gen_res["target"][str(a)][str(s)] = distinct_2(tg)
                gen_res["random"][str(a)][str(s)] = distinct_2(rg)
                gen_log += [{"axis": axis, "condition": f"target_a{a}", "seed": s, "unit": [L, N],
                             "prompt_id": ood[i]["id"], "text": t} for i, t in enumerate(tg)]
                gen_log += [{"axis": axis, "condition": f"random_a{a}", "seed": s, "unit": [L, m],
                             "prompt_id": ood[i]["id"], "text": t} for i, t in enumerate(rg)]
                if s == seeds[0]:
                    samples[f"target_a{a}"] = tg[:cfg["samples_to_save"]]
                    samples[f"random_a{a}"] = rg[:cfg["samples_to_save"]]
        gen_summary = {
            "baseline": msd(list(gen_res["baseline"].values())),
            "target": {a: msd(list(gen_res["target"][a].values())) for a in gen_res["target"]},
            "random": {a: msd(list(gen_res["random"][a].values())) for a in gen_res["random"]},
        }

        # ---- in-distribution: elicitation on negatives (+push), suppression on positives (-push)
        rows = load_jsonl(root / cfg["datasets_dir"] / f"{axis}.jsonl")
        pos_fmt = format_chat(tok, [r["prompt"] for r in rows if r["label"] == 1])
        neg_fmt = format_chat(tok, [r["prompt"] for r in rows if r["label"] == 0])
        indist = {}
        for name, fmt, direction in (("elicitation_on_neg", neg_fmt, 1.0), ("suppression_on_pos", pos_fmt, -1.0)):
            b = delta_logp(model, tok, fmt, tm_t, tm_n, device)
            with push(model, L, N, direction * v(N, rd)):
                t_ = delta_logp(model, tok, fmt, tm_t, tm_n, device)
            rs = []
            for s in seeds:
                m = rand_units[s]
                with push(model, L, m, direction * v(m, rd)):
                    rs.append(delta_logp(model, tok, fmt, tm_t, tm_n, device))
            rmean = np.mean(rs, axis=0)
            indist[name] = {"dose": rd, "direction": direction, "baseline": msd(b), "target": msd(t_),
                            "random_seed_mean": msd(rmean), "target_shift": msd(t_ - b),
                            "random_shift": msd(rmean - b),
                            "per_prompt": {"baseline": b.tolist(), "target": t_.tolist(), "random": rmean.tolist()}}

        # ---- pre-registered decision (as amended before execution; see paper Sec. 6.1)
        k = str(rd)
        shift = ood_res["target"][k]["shift"]["mean"]
        rshift = ood_res["random"][k]["shift"]["mean"]
        sd_base = ood_res["baseline"]["sd"]
        d2 = gen_summary["target"][k]["mean"]
        crit = {
            "stage1_p_lt_alpha": bool(s1["stats"]["candidate_passes"]),
            "shift_ge_2sd": bool(shift >= cfg["shift_sd_multiplier"] * sd_base),
            "beats_random": bool(shift - rshift > 0),
            "distinct2_ok": bool(d2 >= cfg["min_distinct_2"]),
        }
        if s1["verdict"] == "refuted":
            verdict = "refuted (Stage 1)"
        elif s1["verdict"] == "inconclusive":
            verdict = "inconclusive (Stage 1)"
        elif all(crit.values()):
            verdict = "confirmed"
        elif not crit["distinct2_ok"]:
            verdict = "inconclusive (fluency collapse)"
        else:
            verdict = "refuted (Stage 2)"
        decision = {"criteria": crit, "shift_mean": shift, "random_shift_mean": rshift, "sd_baseline": sd_base,
                    "threshold": cfg["shift_sd_multiplier"] * sd_base, "target_distinct2_mean": d2,
                    "causal_effect": bool(crit["shift_ge_2sd"] and crit["beats_random"] and crit["distinct2_ok"]),
                    "verdict": verdict, "exploratory": not s1["stats"]["candidate_passes"]}

        wall = time.time() - ta
        out = {"axis": axis, "env": env, "unit": {"layer": L, "neuron": N, "cohen_d": d, "push_sign": sign,
                                                   "col_norm": w_norm, "stage1_verdict": s1["verdict"]},
               "rho_layer": rho[L], "doses": doses, "report_dose": rd, "random_units": rand_units,
               "token_map": {"target": tm_t, "neutral": tm_n,
                             "target_pieces": {w: tok.convert_ids_to_tokens(i) for w, i in tm_t.items()},
                             "neutral_pieces": {w: tok.convert_ids_to_tokens(i) for w, i in tm_n.items()}},
               "ood_delta_logp": ood_res, "distinct_2": {"per_seed": gen_res, "summary": gen_summary},
               "in_distribution": indist, "decision": decision, "samples_seed0": samples, "wall_s": wall}
        save_json(out, OUTPUTS / f"stage2_{axis}.json")
        timings["axes"][axis] = wall
        summary_rows.append(
            f"| {axis} | L{L} N{N} | {ood_res['baseline']['mean']:.6f} ± {sd_base:.6f} | "
            f"{ood_res['target'][k]['mean']:.6f} | {shift:.6f} | {rshift:.6f} | "
            f"{gen_summary['baseline']['mean']:.6f} | {d2:.6f} | {gen_summary['random'][k]['mean']:.6f} | "
            f"{indist['elicitation_on_neg']['target_shift']['mean']:.6f} | "
            f"{indist['suppression_on_pos']['target_shift']['mean']:.6f} | {verdict} | {wall:.1f} |")
        print(summary_rows[-1], flush=True)

    with open(OUTPUTS / "stage2_generations.jsonl", "w", encoding="utf-8") as f:
        for g in gen_log:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    total = time.time() - t0
    res_path = OUTPUTS / "resources.json"
    resources = json.loads(res_path.read_text()) if res_path.exists() else {}
    resources.update({"stage2_total_s": total, "stage2_peak_rss_gb": peak_rss_gb(), "stage2_timings": timings,
                      "stage2_generation_runs": len(seeds) * (1 + 2 * len(doses) * len(axes)),
                      "stage2_prompts_per_run": len(ood)})
    save_json(resources, res_path)

    runlog(
        f"## Stage 2 causal steering ({datetime.now(timezone.utc).isoformat(timespec='seconds')})\n\n"
        f"Command: `python run_stage2.py --config {args.config}"
        f"{' --axes ' + ' '.join(args.axes) if args.axes else ''}`\n\n{env_block(env)}\n"
        f"- doses: `{doses}` (decision dose α = {rd}); seeds: `{seeds}`; "
        f"decoding: T={cfg['temperature']}, top_p={cfg['top_p']}, {cfg['max_new_tokens']} new tokens\n"
        f"- Stage 2 total wall time: `{total:.1f} s`; peak RSS: `{peak_rss_gb():.3f} GB`\n"
        f"- ρ_ℓ (median post-layer residual ℓ2 norm on P_OOD) saved to `outputs/calibration.json`\n\n"
        "OOD ΔlogP values are means over the 16 OOD prompts (SD across prompts); random control = mean over "
        "3 seeds, each with its own random unit in layer ℓ*. distinct_2 = mean over seeds.\n\n"
        "| Axis | Unit | Baseline ΔlogP ± SD | Target ΔlogP | Target shift | Random shift | d2 base | d2 target | "
        "d2 random | Elicit shift (neg) | Suppress shift (pos) | Verdict | Wall s |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(summary_rows)
    )


if __name__ == "__main__":
    main()
