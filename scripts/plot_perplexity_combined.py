#!/usr/bin/env python3
"""Generate combined perplexity plot: left=simple_1, right=selfhash, both gammas overlaid."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import paper_style
paper_style.apply()

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = Path(__file__).parent.parent / "paper" / "figures" / "perplexity_combined.pdf"


def load_perplexity(scheme, gamma, delta):
    if "selfhash" in scheme:
        prefix = "roc500_gselfhash" if "gpu" in scheme else "roc500_selfhash"
        g = "g" if "gpu" in scheme else ""
        exp = f"roc500_{g}selfhash_d{int(delta)}_g{int(gamma*100)}"
    else:
        prefix = "roc500_gsimple1" if "gpu" in scheme else "roc500_simple1"
        g = "g" if "gpu" in scheme else ""
        exp = f"roc500_{g}simple_1_d{int(delta)}_g{int(gamma*100)}"
    ppl_file = DATA_DIR / prefix / exp / "metrics" / "perplexity.jsonl"
    if not ppl_file.exists():
        return None
    return [json.loads(l) for l in ppl_file.read_text().strip().split('\n')]

def compute_delta_ppl(scheme, gamma, deltas):
    means, sems = [], []
    for delta in deltas:
        data = load_perplexity(scheme, gamma, delta)
        if data:
            wm = np.array([d["watermarked"] for d in data])
            nowm = np.array([d["non_watermarked"] for d in data])
            diff = wm - nowm
            means.append(diff.mean())
            sems.append(diff.std() / np.sqrt(len(diff)))
        else:
            means.append(np.nan)
            sems.append(np.nan)
    return means, sems

def main():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    deltas = [1, 2, 5, 10]
    x = np.arange(len(deltas))
    width = 0.2
    
    # Left: Simple-1
    ax = axes[0]
    for i, (gamma, color_cpu, color_gpu) in enumerate([(0.5, '#2ecc71', '#27ae60'), 
                                                         (0.25, '#3498db', '#2980b9')]):
        cpu_m, cpu_s = compute_delta_ppl("simple_1", gamma, deltas)
        gpu_m, gpu_s = compute_delta_ppl("gpu-simple_1", gamma, deltas)
        offset = (i - 0.5) * width * 2
        ax.bar(x + offset - width/2, cpu_m, width, yerr=cpu_s, label=f'simple_1 γ={gamma}',
               color=color_cpu, capsize=2, alpha=0.8)
        ax.bar(x + offset + width/2, gpu_m, width, yerr=gpu_s, label=f'SBW-1 γ={gamma}',
               color=color_gpu, capsize=2, alpha=0.8, hatch='//')
    ax.set_xlabel('δ (bias strength)')
    ax.set_ylabel('Δ Perplexity (WM − NoWM)')
    ax.set_title('(a) Without self-salt')
    ax.set_xticks(x)
    ax.set_xticklabels(deltas)
    ax.legend(loc='upper left', ncol=1)
    ax.grid(axis='y', alpha=0.3)
    
    # Right: Selfhash
    ax = axes[1]
    for i, (gamma, color_cpu, color_gpu) in enumerate([(0.5, '#e74c3c', '#c0392b'),
                                                         (0.25, '#9b59b6', '#8e44ad')]):
        cpu_m, cpu_s = compute_delta_ppl("selfhash", gamma, deltas)
        gpu_m, gpu_s = compute_delta_ppl("gpu-selfhash", gamma, deltas)
        offset = (i - 0.5) * width * 2
        ax.bar(x + offset - width/2, cpu_m, width, yerr=cpu_s, label=f'selfhash γ={gamma}',
               color=color_cpu, capsize=2, alpha=0.8)
        ax.bar(x + offset + width/2, gpu_m, width, yerr=gpu_s, label=f'SBW-ss γ={gamma}',
               color=color_gpu, capsize=2, alpha=0.8, hatch='//')
    ax.set_xlabel('δ (bias strength)')
    ax.set_ylabel('Δ Perplexity (WM − NoWM)')
    ax.set_title('(b) With self-salt')
    ax.set_xticks(x)
    ax.set_xticklabels(deltas)
    ax.legend(loc='upper left', ncol=1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    paper_style.savefig(fig, OUTPUT_FILE)
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
