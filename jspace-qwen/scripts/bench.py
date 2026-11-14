"""CPU throughput of the exact and sketch estimators on a randomly initialised
model with Qwen3-0.6B's exact shape (timing does not depend on the weights).

python scripts/bench.py --dtype bf16 --seq 128 --batch 16
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from jspace.fitting import PromptGraph  # noqa: E402
from jspace.model import load_model  # noqa: E402

QWEN3_06B = dict(vocab_size=151936, hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
                 num_attention_heads=16, num_key_value_heads=8, head_dim=128, rope_theta=1e6,
                 max_position_embeddings=40960, tie_word_embeddings=True, rms_norm_eps=1e-6)
QWEN3_17B = dict(QWEN3_06B, hidden_size=2048, intermediate_size=6144)

ap = argparse.ArgumentParser()
ap.add_argument("--dtype", default="bf16")
ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--batch", type=int, nargs="+", default=[8, 16, 32])
ap.add_argument("--size", default="0.6b")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
cfg = QWEN3_06B if a.size == "0.6b" else QWEN3_17B
model = load_model("qwen3-like", random_init=True, dtype=dt, config_overrides=cfg)
d, L = model.d_model, model.n_layers
ids = torch.randint(0, 150000, (1, a.seq))
print(f"model d={d} L={L} dtype={a.dtype} seq={a.seq} threads={torch.get_num_threads()}")
for B in a.batch:
    t0 = time.perf_counter()
    g = PromptGraph(model, ids, list(range(L - 1)), L - 1, batch=B)
    t1 = time.perf_counter()
    reps = 3
    for r in range(reps):
        cot = torch.randn(B, d)
        g.vjp_rows(cot, retain=True)
    t2 = time.perf_counter()
    per_bwd = (t2 - t1) / reps
    exact_prompt = (t1 - t0) + per_bwd * (d / B)
    print(f"B={B:>3}: forward {t1 - t0:.2f}s, one VJP block {per_bwd:.2f}s "
          f"({per_bwd / B * 1000:.0f} ms/row) -> exact prompt ~{exact_prompt / 60:.1f} min; "
          f"sketch(K={B}) prompt ~{(t1 - t0) + per_bwd:.1f}s")
    del g
