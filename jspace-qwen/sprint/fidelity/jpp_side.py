"""Fidelity check, jpp_lens side: per-cluster LRP ("rlens") Jacobian row sums for one
prompt, computed by jpp_lens's own ExpertJacobianRelPTrainer on Qwen3-0.6B (CPU, bf16),
with our router's PCA-64 k-means parameters loaded into jpp_lens's router class.

Run with the jpp_lens environment:
  /home/user/safety-research/jpp_lens/.venv/bin/python sprint/fidelity/jpp_side.py OUT.pt
"""

import os
import sys

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from workspace_lens.config import LensConfig
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
from workspace_lens.routing.router import ActivationRouterCollection, LayerActivationRouter

OUT = sys.argv[1]
LAYERS = [7, 21]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
t.set_num_threads(4)
t.manual_seed(0)

hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=t.bfloat16)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = jlens.from_hf(hf, tok)

ours = t.load(os.path.join(ROOT, "lenses", "router_E8.pt"), weights_only=True)
router = ActivationRouterCollection(num_clusters=8, projection_dim=64)
router.layer_routers_L_dict = {
    l: LayerActivationRouter(pca_mean_N=ours["mean"][l].float(), pca_components_ND=ours["comps"][l].float(),
                             centroids_ED=ours["cents"][l].float())
    for l in LAYERS
}

prompt = open(os.path.join(ROOT, "sprint", "fidelity", "prompt.txt")).read()
cfg = LensConfig(hf_model_name="Qwen/Qwen3-0.6B", checkpoint_name="fidelity", source_layers=LAYERS,
                 lrp_mode="rlens", num_clusters=8, cluster_projection_dim=64, max_seq_len=48,
                 skip_first_n_positions=16, jacobian_rows_per_pass=8, checkpoint_every_n_prompts=None,
                 artifacts_base_dir="/tmp/claude-0/jpp_fidelity_artifacts")
trainer = ExpertJacobianRelPTrainer(cfg, model, [prompt], router_collections_K_dict={8: router}, device="cpu")
seq_len, n_pos = trainer.fit_step(prompt)
sums = trainer.fit_sums_K_dict[8].layer_sums_L_dict
ids = model.encode(prompt, max_length=48)[0]
t.save({"input_ids": ids, "seq_len": seq_len, "n_pos": n_pos, "target_layer": trainer.target_layer,
        "row_sums": {l: sums[l].weighted_jacobian_row_sum_EFN.clone() for l in LAYERS},
        "counts": {l: sums[l].position_count_E.clone() for l in LAYERS},
        "patch_report": {"ln": len(trainer.last_patch_report.ln_rule), "mlp": len(trainer.last_patch_report.mlp_rule)}},
       OUT)
print("jpp side done", seq_len, n_pos, "target layer", trainer.target_layer,
      {l: sums[l].position_count_E.tolist() for l in LAYERS})
