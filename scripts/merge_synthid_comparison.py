#!/usr/bin/env python3
"""Merge synthid benchmark results into data/synthid_comparison.json.

Combines:
  - results from benchmark_synthid_vs_sbw.py (option_a_full_vocab, option_b_topk40, cpu_schemes)
  - results from benchmark_synthid_top40.py (synthid_compiled_topk40, kgw_selfhash_40)

The option_b_topk40 section merges GPU-fused results from the first script
with SynthID+KGW results from the second.

Usage:
    python scripts/merge_synthid_comparison.py
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def run():
    fullvocab = json.loads((DATA_DIR / "results_synthid_fullvocab.json").read_text())
    top40 = json.loads((DATA_DIR / "results_top40.json").read_text())

    # option_b_topk40: merge both sources
    # - from benchmark_synthid_vs_sbw.py: sbw_fused_simple_4_40, sbw_fused_selfhash_40, synthid_compiled_topk40
    # - from benchmark_synthid_top40.py: synthid_compiled_topk40, kgw_selfhash_40
    # Use top40's synthid/kgw results (dedicated top-40 benchmark) and fullvocab's sbw results
    fv_topk = fullvocab["results"]["option_b_topk40"]
    option_b = {}
    for bs in fv_topk:
        option_b[str(bs)] = {
            **top40["results"][str(bs)],
            f"sbw_fused_simple_4_40": fv_topk[bs][f"sbw_fused_simple_4_40"],
            f"sbw_fused_selfhash_40": fv_topk[bs][f"sbw_fused_selfhash_40"],
        }

    output = {
        "metadata": {
            **fullvocab["metadata"],
            "synthid_top_k": 40,
            "note": "option_b uses top-k=40 (SynthID + KGW selfhash). "
                    "option_a and cpu_schemes from full-vocab run.",
        },
        "results": {
            "option_a_full_vocab": fullvocab["results"]["option_a_full_vocab"],
            "option_b_topk40": option_b,
            "cpu_schemes": fullvocab["results"]["cpu_schemes"],
        },
    }

    out_path = DATA_DIR / "synthid_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()
