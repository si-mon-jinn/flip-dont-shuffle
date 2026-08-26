#!/usr/bin/env python3
"""Plot tail latency (p99) for Appendix F."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import paper_style
paper_style.apply()


DATA_FILE = Path(__file__).parent.parent / "data" / "synthid_comparison.json"
OUTPUT_DIR = Path(__file__).parent.parent / "paper" / "figures"

with open(DATA_FILE) as f:
    data = json.load(f)

full_vocab = data["results"]["option_a_full_vocab"]
topk40 = data["results"]["option_b_topk40"]
cpu = data["results"]["cpu_schemes"]

batches = sorted(int(b) for b in full_vocab.keys())

methods = [
    ("SBW-4", lambda b: full_vocab[str(b)]["sbw_fused_simple_4"], '#377eb8', 's--'),
    ("SBW-ss-V", lambda b: full_vocab[str(b)]["sbw_fused_selfhash_fullvocab"], '#4daf4a', '^--'),
    ("SynthID (k=40)", lambda b: topk40[str(b)]["synthid_compiled_topk40"], '#e41a1c', 'D-'),
    ("KGW simple_4", lambda b: cpu[str(b)]["cpu_simple_4"], '#a65628', 'x:'),
    ("KGW selfhash (top-40)", lambda b: topk40[str(b)]["kgw_selfhash_40"], '#999999', '+:'),
]

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

for label, get_data, color, style in methods:
    p99 = [get_data(b)["latency_p99_ms"] for b in batches]
    marker = style[0]
    linestyle = style[1:]
    ax.semilogy(range(len(batches)), p99, marker=marker, linestyle=linestyle,
                label=label, color=color, linewidth=2, markersize=6)

ax.set_xlabel('Batch Size')
ax.set_ylabel('p99 Latency (ms)')
ax.set_xticks(range(len(batches)))
ax.set_xticklabels([str(b) for b in batches])
ax.legend(fontsize=8, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
paper_style.savefig(fig, OUTPUT_DIR / 'tail_latency.pdf')
print(f"Saved: {OUTPUT_DIR / 'tail_latency.pdf'}")
plt.close()
