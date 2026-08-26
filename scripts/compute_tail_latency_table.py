#!/usr/bin/env python3
"""Generate tail latency table (p50/p95/p99, all batch sizes) for Appendix F."""

import json
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "synthid_comparison.json"
OUTPUT_FILE = Path(__file__).parent.parent / "paper" / "tables" / "tab_tail_latency.tex"

with open(DATA_FILE) as f:
    data = json.load(f)

full_vocab = data["results"]["option_a_full_vocab"]
topk40 = data["results"]["option_b_topk40"]
cpu = data["results"]["cpu_schemes"]

target_batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

methods = [
    ("SBW-4", lambda b: full_vocab[str(b)]["sbw_fused_simple_4"]),
    ("SBW-ss-V", lambda b: full_vocab[str(b)]["sbw_fused_selfhash_fullvocab"]),
    ("SynthID (k=40)", lambda b: topk40[str(b)]["synthid_compiled_topk40"]),
    ("KGW simple\\_4", lambda b: cpu[str(b)]["cpu_simple_4"]),
    ("KGW selfhash (top-40)", lambda b: topk40[str(b)]["kgw_selfhash_40"]),
]

def fmt(v):
    if v < 10:
        return f"{v:.2f}"
    elif v < 1000:
        return f"{v:.1f}"
    else:
        return f"{v:.0f}"

ncols = len(target_batches)
col_spec = "l|" + "r" * ncols
header = " & ".join([str(b) for b in target_batches])

lines = []
lines.append(r"\begin{tabular}{@{}" + col_spec + r"@{}}")
lines.append(r"\toprule")
lines.append(f"Method & {header} \\\\")

for metric, metric_key in [("p50", "latency_p50_ms"), ("p95", "latency_p95_ms"), ("p99", "latency_p99_ms")]:
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{{ncols + 1}}}{{c}}{{\\textit{{{metric}}}}} \\\\")
    lines.append(r"\midrule")
    for i, (label, get_data) in enumerate(methods):
        vals = [fmt(get_data(b)[metric_key]) for b in target_batches]
        lines.append(f"{label} & {' & '.join(vals)} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text("\n".join(lines) + "\n")
print(f"Saved: {OUTPUT_FILE}")
