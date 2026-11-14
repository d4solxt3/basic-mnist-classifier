# Single-neuron localization and steering in Qwen3-1.7B

Code, data, run log and manuscript for a two-stage, pre-registered test of whether behavioral contrasts
(sycophancy, epistemic uncertainty, caution, persona/register) localize onto single post-SwiGLU MLP units
of `Qwen/Qwen3-1.7B`.

| Path | What it is |
|---|---|
| `config.yaml` | All run settings (model, Stage 1 null, Stage 2 doses/seeds/decoding, decision thresholds) |
| `prompts.jsonl` | 16 neutral out-of-distribution evaluation prompts |
| `datasets/{syc,unc,cau,per}.jsonl` | 80 contrast pairs (160 prompts), extracted from the paper's Appendix B |
| `datasets/token_sets.json` | Per-axis target/neutral first-token word sets (Appendix A) |
| `extract_benchmark.py` | Re-extracts `datasets/` from `paper/main.tex` and checks the design invariants |
| `discover_neurons.py` | Stage 1: activations, Cohen's d, Westfall–Young max-statistic null |
| `run_stage2.py` | Stage 2: residual-norm-calibrated additive push, random controls, fluency guard |
| `make_tables.py` | Writes every number in the paper (`paper/generated/*.tex`) from `outputs/*.json` |
| `outputs/` | Raw results (`stage1_*.json`, `stage2_*.json`, `calibration.json`, `resources.json`, all generations) |
| `RUNLOG.md` | Machine-written log of every run: commits, versions, timings, key metrics |
| `paper/main.tex`, `paper/main.pdf` | Manuscript source (pdfLaTeX) and compiled PDF |

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build on a GPU
pip install "transformers>=4.51" accelerate safetensors numpy pyyaml huggingface_hub
python extract_benchmark.py .        # optional: rebuild datasets/ from the paper
python discover_neurons.py           # Stage 1  (~1 min on 4 CPU cores)
python run_stage2.py                 # Stage 2  (~30 min on 4 CPU cores)
python make_tables.py
cd paper && pdflatex main && pdflatex main
```

The reported run used CPU only (4-core Xeon with AMX-BF16, BF16 weights). The model revision is pinned
at runtime and recorded in `RUNLOG.md`.
