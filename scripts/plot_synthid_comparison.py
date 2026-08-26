#!/usr/bin/env python3
"""Plot SynthID vs SBW latency comparison and speedup ratios."""

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

# Extract latencies (median p50)
synthid_full = [full_vocab[str(b)]["synthid_compiled"]["latency_p50_ms"] for b in batches]
sbw_simple4_full = [full_vocab[str(b)]["sbw_fused_simple_4"]["latency_p50_ms"] for b in batches]
sbw_selfhash_full = [full_vocab[str(b)]["sbw_fused_selfhash_fullvocab"]["latency_p50_ms"] for b in batches]
synthid_topk = [topk40[str(b)]["synthid_compiled_topk40"]["latency_p50_ms"] for b in batches]
kgw_selfhash_40 = [topk40[str(b)]["kgw_selfhash_40"]["latency_p50_ms"] for b in batches]
cpu_simple4 = [cpu[str(b)]["cpu_simple_4"]["latency_p50_ms"] for b in batches]

# --- Figure 1: Latency comparison ---
fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

ax.loglog(batches, synthid_topk, 'D-', label='SynthID (k=40)', color='#e41a1c', linewidth=2, markersize=6)
ax.loglog(batches, sbw_simple4_full, 's--', label='SBW-4', color='#377eb8', linewidth=2, markersize=6)
ax.loglog(batches, sbw_selfhash_full, '^--', label='SBW-ss-V', color='#4daf4a', linewidth=2, markersize=6)
ax.loglog(batches, cpu_simple4, 'x:', label='KGW simple_4', color='#a65628', linewidth=1.5, markersize=6)
ax.loglog(batches, kgw_selfhash_40, '+:', label='KGW selfhash (top-40)', color='#999999', linewidth=1.5, markersize=7)

ax.set_xlabel('Batch Size')
ax.set_ylabel('Latency (ms)')
ax.set_xticks(batches)
ax.set_xticklabels([str(b) for b in batches])
ax.legend(fontsize=8, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
paper_style.savefig(fig, OUTPUT_DIR / 'synthid_latency.pdf')
print(f"Saved: {OUTPUT_DIR / 'synthid_latency.pdf'}")
plt.close()

# --- Figure 2: Speedup ratios ---
fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

speedup_synthid_vs_simple4_full = [s / f for s, f in zip(synthid_topk, sbw_simple4_full)]
speedup_synthid_vs_selfhash_full = [s / f for s, f in zip(synthid_topk, sbw_selfhash_full)]
speedup_synthidfull_vs_simple4_full = [s / f for s, f in zip(synthid_full, sbw_simple4_full)]
speedup_cpu_simple4_vs_gpu = [c / g for c, g in zip(cpu_simple4, sbw_simple4_full)]
speedup_kgw_selfhash_vs_gpu = [k / g for k, g in zip(kgw_selfhash_40, sbw_selfhash_full)]

ax.loglog(batches, speedup_synthid_vs_simple4_full, 's-', label='SynthID k=40 / SBW-4', color='#377eb8', linewidth=2, markersize=6)
ax.loglog(batches, speedup_synthid_vs_selfhash_full, '^-', label='SynthID k=40 / SBW-ss-V', color='#4daf4a', linewidth=2, markersize=6)
ax.loglog(batches, speedup_synthidfull_vs_simple4_full, 'D-', label='SynthID full-V / SBW-4', color='#e41a1c', linewidth=2, markersize=6)
ax.loglog(batches, speedup_cpu_simple4_vs_gpu, 'x:', label='KGW simple_4 / SBW-4', color='#a65628', linewidth=1.5, markersize=7)
ax.loglog(batches, speedup_kgw_selfhash_vs_gpu, '+:', label='KGW selfhash / SBW-ss-V', color='#999999', linewidth=1.5, markersize=7)

ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Batch Size')
ax.set_ylabel('Speedup (×)')
ax.set_xticks(batches)
ax.set_xticklabels([str(b) for b in batches])
ax.legend(fontsize=7.5, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
paper_style.savefig(fig, OUTPUT_DIR / 'synthid_speedup.pdf')
print(f"Saved: {OUTPUT_DIR / 'synthid_speedup.pdf'}")
plt.close()
