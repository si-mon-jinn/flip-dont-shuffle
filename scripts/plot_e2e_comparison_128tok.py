"""Plot combined benchmark results with error bars."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style
paper_style.apply()

d = Path(__file__).resolve().parent.parent
data = json.loads((d / "data" / "results_e2e_comparison.json").read_text())
# vllm_sbw = json.loads((d / "results_vllm_128tok_sbw.json").read_text())

bs_list = sorted(int(k) for k in data["results"] if int(k) > 8)

syn_oh, sbw_oh, syn_ci, sbw_ci = [], [], [], []
syn_pct, sbw_pct = [], []

for bs in bs_list:
    r = data["results"][str(bs)]
    base_med = r["baseline"]["median_ms"]
    base_std = r["baseline"]["std_ms"]
    syn_std = r["synthid"]["std_ms"]
    sbw_std = r["sbw"]["std_ms"]
    n = 50

    syn_oh.append(r["synthid_overhead_ms"])
    sbw_oh.append(r["sbw_overhead_ms"])
    syn_ci.append(np.sqrt(syn_std**2 + base_std**2))
    sbw_ci.append(np.sqrt(sbw_std**2 + base_std**2))
    syn_pct.append(r["synthid_overhead_ms"] / base_med * 100)
    sbw_pct.append(r["sbw_overhead_ms"] / base_med * 100)

# # vLLM SBW data (uncomment to add vLLM curve)
# bs_vllm_sbw = sorted(int(k) for k in vllm_sbw["results"] if int(k) > 8)
# vllm_sbw_oh, vllm_sbw_ci, vllm_sbw_pct = [], [], []
# for bs in bs_vllm_sbw:
#     r = vllm_sbw["results"][str(bs)]
#     base_std = r["baseline"]["std_ms"]
#     sbw_std = r["sbw"]["std_ms"]
#     vllm_sbw_oh.append(r["sbw_overhead_ms"])
#     vllm_sbw_ci.append(1.96 * np.sqrt(sbw_std**2 + base_std**2) / np.sqrt(50))
#     vllm_sbw_pct.append(r["sbw_overhead_ms"] / r["baseline"]["median_ms"] * 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5))

# Absolute overhead (ms) with error bars
ax1.errorbar(bs_list, syn_oh, yerr=syn_ci, fmt="s-", label="SynthID", color="tab:red", capsize=4)
ax1.errorbar(bs_list, sbw_oh, yerr=sbw_ci, fmt="o-", label="SBW-ss-V", color="tab:blue", capsize=4)
# ax1.errorbar(bs_vllm_sbw, vllm_sbw_oh, yerr=vllm_sbw_ci, fmt="D-", label="SBW (vLLM native)", color="tab:green", capsize=4)
ax1.set_xscale("log", base=2)
ax1.set_xlabel("Batch size"); ax1.set_ylabel("Overhead (ms)")
ax1.set_title("(a) Absolute overhead")
ax1.grid(True, alpha=0.3)
ax1.set_xticks(bs_list); ax1.set_xticklabels(bs_list)
ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)

# Percentage overhead
ax2.plot(bs_list, syn_pct, "s-", label="SynthID", color="tab:red")
ax2.plot(bs_list, sbw_pct, "o-", label="SBW-ss-V", color="tab:blue")
# ax2.plot(bs_vllm_sbw, vllm_sbw_pct, "D-", label="SBW (vLLM native)", color="tab:green")
ax2.set_xscale("log", base=2)
ax2.set_xlabel("Batch size"); ax2.set_ylabel("Overhead (%)")
ax2.set_title("(b) Relative overhead")
ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.set_xticks(bs_list); ax2.set_xticklabels(bs_list)

plt.tight_layout()
out = d / "paper" / "figures" / "e2e_overhead_comparison_128tok.pdf"
paper_style.savefig(fig, out)
