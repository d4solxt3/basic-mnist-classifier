"""Shared helpers for the Qwen3-1.7B single-neuron probe (Stage 1 and Stage 2)."""
import json
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RUNLOG = ROOT / "RUNLOG.md"


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_revision(model_id: str, revision: str) -> str:
    """Return the exact commit hash for `revision` (network first, then local cache)."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id, revision=revision).sha
    except Exception:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, revision=revision, local_files_only=True)).name


def pick_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def load_model(cfg: dict):
    commit = resolve_revision(cfg["model_id"], cfg["revision"])
    device = pick_device(cfg)
    dtype = getattr(torch, cfg["dtype"])
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=commit)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if cfg.get("load_in_4bit"):
        raise NotImplementedError("4-bit loading is not used in this study.")
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=commit, dtype=dtype)
    model.to(device).eval()
    return model, tok, commit, device


def format_chat(tok, prompts: list[str]) -> list[str]:
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    raise AttributeError("Could not locate decoder layers on model.")


def weight_bytes(model) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def peak_rss_gb() -> float:
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def env_info(device: torch.device, model_commit: str) -> dict:
    import transformers

    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git_commit(),
        "model_commit": model_commit,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cpu": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def runlog(text: str) -> None:
    if not RUNLOG.exists():
        RUNLOG.write_text(
            "# RUNLOG\n\nMachine-written log of every executed run. Numbers here are copied by the "
            "scripts from `outputs/*.json`; nothing in this file is typed by hand.\n",
            encoding="utf-8",
        )
    with open(RUNLOG, "a", encoding="utf-8") as f:
        f.write("\n" + text.rstrip() + "\n")


def env_block(env: dict) -> str:
    return "\n".join(f"- {k}: `{v}`" for k, v in env.items())
