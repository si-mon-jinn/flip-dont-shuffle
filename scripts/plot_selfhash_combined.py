#!/usr/bin/env python3
"""Generate combined Selfhash figure: z-score (top), ROC (bottom), 2x2 grid."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import paper_style
paper_style.apply()

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = Path(__file__).parent.parent / "paper" / "figures" / "selfhash_combined.pdf"


def load_detection(scheme, gamma, delta):
    prefix = "roc500_gselfhash" if "gpu" in scheme else "roc500_selfhash"
    g = "g" if "gpu" in scheme else ""
    exp = f"roc500_{g}selfhash_d{int(delta)}_g{int(gamma*100)}"
    det_file = DATA_DIR / prefix / exp / "detection.jsonl"
    data = [json.loads(l) for l in det_file.read_text().strip().split('\n')]
    return ([d["watermarked"]["z_score"] for d in data],
            [d["non_watermarked"]["z_score"] for d in data])

def main():
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6))
    configs = [(0.5, 1.0), (0.5, 2.0)]
    gamma, delta = 0.5, 2.0
    
    # Top row: z-score distributions (left=watermarked, right=non-watermarked)
    # Top-left: watermarked
    ax = axes[0, 0]
    cpu_wm, _ = load_detection("selfhash", gamma, delta)
    gpu_wm, _ = load_detection("gpu-selfhash", gamma, delta)
    bins = np.linspace(-2, 12, 35)
    ax.hist(cpu_wm, bins=bins, alpha=0.6, density=True, label='selfhash')
    ax.hist(gpu_wm, bins=bins, alpha=0.6, density=True, label='SBW-ss')
    ax.plot([], [], 'k--', lw=1.5, label='$\\mathcal{N}(0,1)$')
    ax.set_xlabel('z-score')
    ax.set_ylabel('Density')
    ax.set_title(f'Watermarked (γ={gamma}, δ={int(delta)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: non-watermarked with N(0,1)
    ax = axes[0, 1]
    _, cpu_nowm = load_detection("selfhash", gamma, delta)
    _, gpu_nowm = load_detection("gpu-selfhash", gamma, delta)
    bins = np.linspace(-4, 4, 35)
    ax.hist(cpu_nowm, bins=bins, alpha=0.6, density=True)
    ax.hist(gpu_nowm, bins=bins, alpha=0.6, density=True)
    x = np.linspace(-4, 4, 100)
    ax.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), 'k--', lw=1.5)
    ax.set_xlabel('z-score')
    ax.set_ylabel('Density')
    ax.set_title(f'Non-Watermarked (γ={gamma})')
    ax.grid(True, alpha=0.3)
    
    # Bottom row: ROC curves
    for ax, (gamma, delta) in zip(axes[1], configs):
        for scheme, color, ls, label in [("selfhash", "b", "-", "selfhash"),
                                          ("gpu-selfhash", "r", "--", "SBW-ss")]:
            wm, nowm = load_detection(scheme, gamma, delta)
            y = [1]*len(wm) + [0]*len(nowm)
            scores = wm + nowm
            fpr, tpr, _ = roc_curve(y, scores)
            auc = roc_auc_score(y, scores)
            ax.plot(fpr, tpr, color=color, ls=ls, lw=1.5, label=f'{label} ({auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k:', lw=0.8, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC (γ={gamma}, δ={int(delta)})')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    paper_style.savefig(fig, OUTPUT_FILE)
    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
