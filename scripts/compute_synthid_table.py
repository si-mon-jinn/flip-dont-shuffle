#!/usr/bin/env python3
"""Generate LaTeX tables comparing SBW vs SynthID latency."""

import json
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "synthid_comparison.json"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"

with open(DATA_FILE) as f:
    data = json.load(f)

full_vocab = data["results"]["option_a_full_vocab"]
topk40 = data["results"]["option_b_topk40"]
cpu = data["results"]["cpu_schemes"]

all_batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Get SBW selfhash full vocab latencies as baseline (our best method)
sbw_lat = {b: full_vocab[str(b)]["sbw_fused_selfhash_fullvocab"]["latency_p50_ms"] for b in all_batches}
synthid_lat = {b: topk40[str(b)]["synthid_compiled_topk40"]["latency_p50_ms"] for b in all_batches}

methods = [
    ("SBW-4", {b: full_vocab[str(b)]["sbw_fused_simple_4"]["latency_p50_ms"] for b in all_batches}),
    ("SBW-ss-V", {b: full_vocab[str(b)]["sbw_fused_selfhash_fullvocab"]["latency_p50_ms"] for b in all_batches}),
    ("SynthID (k=40)", synthid_lat),
    ("KGW simple\\_4", {b: cpu[str(b)]["cpu_simple_4"]["latency_p50_ms"] for b in all_batches}),
    ("KGW selfhash (top-40)", {b: topk40[str(b)]["kgw_selfhash_40"]["latency_p50_ms"] for b in all_batches}),
]

def generate_table(batches):
    lines = []
    n = len(batches)
    lines.append(r"\begin{tabular}{@{}ll|" + "r" * n + r"@{}}")
    lines.append(r"\toprule")
    header = " & ".join([f"{b}" for b in batches])
    lines.append(f"Method & & {header} \\\\")
    lines.append(r"\midrule")
    for i, (label, lats) in enumerate(methods):
        lat_strs = []
        speedup_strs = []
        for b in batches:
            v = lats[b]
            lat_strs.append(f"{v:.3f}" if v < 10 else f"{v:.1f}")
            ratio = v / sbw_lat[b]
            speedup_strs.append(f"{ratio:.1f}$\\times$")
        lines.append(f"\\multirow{{2}}{{*}}{{{label}}} & ms & {' & '.join(lat_strs)} \\\\")
        lines.append(f" & ratio & {' & '.join(speedup_strs)} \\\\")
        if i < len(methods) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Main text: compact (B=1, 64, 512)
compact = generate_table([1, 64, 512])
(TABLES_DIR / "tab_synthid.tex").write_text(compact)
print(f"Saved: {TABLES_DIR / 'tab_synthid.tex'}")

# Appendix: full (all batch sizes)
full = generate_table(all_batches)
(TABLES_DIR / "tab_synthid_full.tex").write_text(full)
print(f"Saved: {TABLES_DIR / 'tab_synthid_full.tex'}")
