# Flip, Don't Shuffle: Watermarking LLMs at the Speed of Inference

[![Paper](https://img.shields.io/badge/EMNLP%202026-Paper-blue)](https://aclanthology.org/TODO)
[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b)](https://arxiv.org/abs/TODO)

This repository contains the paper source, experimental data, and reproduction scripts for [Flip, Don't Shuffle: Watermarking LLMs at the Speed of Inference](https://aclanthology.org/TODO) (EMNLP 2026).

## Abstract

We introduce Stateless Bernoulli Watermarking (SBW), a new statistical watermark for Large Language Models that determines green list membership through independent per-token Bernoulli trials. Unlike KGW's vocabulary permutation or SynthID's multi-layer tournament, SBW requires only a single comparison per token against a counter-based RNG, reducing membership complexity to O(1) and enabling single-kernel execution with zero intermediate allocations. We prove that this formulation preserves the same detection guarantees as fixed-size green lists: the z-score test remains N(0,1) under the null. The stateless architecture enables capabilities unavailable to existing methods: full-vocabulary self-salt watermarking (over 6000× faster than KGW's self-salt and 2× faster than SynthID despite biasing the entire vocabulary with candidate-dependent seeding) and architectural compatibility with distributed inference. In end-to-end generation benchmarks, SBW adds less than 1% overhead at all batch sizes. We additionally identify hash function design as a previously unexplored axis for watermark quality, showing that a GPU-native Jenkins hash improves null calibration by 1.8× while producing more diverse text. Experiments across two seeding schemes and eight (γ, δ) configurations confirm statistical equivalence with ROC-AUC differences below 0.01.

## Repository Structure

```
flip-dont-shuffle/
├── paper/                  # LaTeX source and compiled PDF
│   ├── paper_long.tex      # Main paper
│   ├── paper_long.pdf      # Compiled PDF
│   ├── custom.bib          # Bibliography
│   ├── figures/            # Paper figures (PDF)
│   └── tables/             # LaTeX table files
├── scripts/                # Figure/table generation scripts
│   └── synthid_ref/        # SynthID reference implementation
├── data/                   # Experimental results (waterpipe output)
│   ├── roc500_*/           # ROC analysis experiments
│   │   ├── config.json     # Experiment configuration
│   │   ├── generations.jsonl
│   │   ├── detection.jsonl
│   │   └── metrics/
│   ├── profile_*.json      # Performance profiling data
│   └── synthid_comparison.json
├── reproduce.sh            # Master reproduction script
└── README.md
```

## Dependencies

- **[sbw](https://github.com/si-mon-jinn/sbw)** — Watermark detection and generation library
- **[waterpipe](https://github.com/si-mon-jinn/waterpipe)** — Evaluation pipeline

## Reproducibility

### Figures and Tables

To regenerate all figures and tables from the included experimental data:

```bash
./reproduce.sh
```

This script clones dependencies, sets up a virtual environment, and runs all generation scripts.

### Raw Experimental Data

The experimental data in `data/` was generated using the [waterpipe](https://github.com/si-mon-jinn/waterpipe) evaluation pipeline. Each experiment directory contains:

- `config.json` — Experiment configuration
- `generations.jsonl` — Generated text samples (watermarked and non-watermarked)
- `detection.jsonl` — Detection results
- `metrics/` — Quality metrics (perplexity, diversity)

To regenerate raw data from scratch:

```bash
# 1. Start a vLLM server with SBW watermarking
pip install vllm vllm-sbw
vllm serve Qwen/Qwen3-8B --port 8008 \
    --logits-processors vllm_sbw:SBWLogitsProcessor

# 2. Install waterpipe
pip install llm-waterpipe

# 3. Run an experiment
cp -r data/roc500_selfhash/roc500_selfhash_d5_g25 my_experiment
waterpipe run my_experiment --verbose
```

See the [waterpipe documentation](https://github.com/si-mon-jinn/waterpipe#readme) for full usage instructions.

### Experiment Naming Convention

```
roc500_{scheme}[_d{delta}_g{gamma}]/
```

- `scheme`: `selfhash`, `gselfhash`, `simple1`, `gsimple1`
- `delta`: Watermark strength (0, 1, 2, 5, 10)
- `gamma`: Green list fraction (25, 50)

## Citation

```bibtex
@inproceedings{ceppi-sanchez-2026-flip,
  title={Flip, Don't Shuffle: Watermarking LLMs at the Speed of Inference},
  author={Ceppi, Simone and Sanchez, Ignacio},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026},
  publisher={Association for Computational Linguistics}
}
```

## License

- Paper content: CC BY 4.0
- Code: MIT License
