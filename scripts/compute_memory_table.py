#!/usr/bin/env python3
"""Generate memory overhead table for Appendix F."""

import json
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "synthid_comparison.json"
OUTPUT_FILE = Path(__file__).parent.parent / "paper" / "tables" / "tab_memory.tex"

with open(DATA_FILE) as f:
    data = json.load(f)

full_vocab = data["results"]["option_a_full_vocab"]
topk40 = data["results"]["option_b_topk40"]

target_batches = [1, 8, 32, 64, 128, 256, 512]

methods = [
    ("SBW-4", lambda b: full_vocab[str(b)]["sbw_fused_simple_4"]["peak_delta_mb"]),
    ("SBW-ss-V", lambda b: full_vocab[str(b)]["sbw_fused_selfhash_fullvocab"]["peak_delta_mb"]),
    ("SynthID (k=40)", lambda b: topk40[str(b)]["synthid_compiled_topk40"]["peak_delta_mb"]),
    ("SBW-4 (k=40)", lambda b: topk40[str(b)]["sbw_fused_simple_4_40"]["peak_delta_mb"]),
    ("SBW-ss (k=40)", lambda b: topk40[str(b)]["sbw_fused_selfhash_40"]["peak_delta_mb"]),
]

ncols = len(target_batches)
col_spec = "l|" + "r" * ncols
header_cols = " & ".join([f"B={b}" for b in target_batches])

lines = []
lines.append(r"\begin{tabular}{@{}" + col_spec + r"@{}}")
lines.append(r"\toprule")
lines.append(f"Method & {header_cols} \\\\")
lines.append(r"\midrule")

for i, (label, get_mem) in enumerate(methods):
    vals = []
    for b in target_batches:
        v = get_mem(b)
        vals.append(f"{v:.1f}" if v >= 1 else f"{v:.2f}" if v > 0 else "0")
    lines.append(f"{label} & {' & '.join(vals)} \\\\")
    if i == 1:
        lines.append(r"\midrule")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text("\n".join(lines) + "\n")
print(f"Saved: {OUTPUT_FILE}")
