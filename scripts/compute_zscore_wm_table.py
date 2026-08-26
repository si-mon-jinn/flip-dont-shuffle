#!/usr/bin/env python3
"""Generate LaTeX table for watermarked z-score statistics (tab:zscore_wm)."""

import json
from pathlib import Path
import numpy as np
from scipy import stats

DATA_DIR = Path(__file__).parent.parent / "data"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
TABLE_FILE = TABLES_DIR / "tab_zscore_wm.tex"

CONFIGS = [(0.25, 1.0), (0.25, 2.0), (0.25, 5.0), (0.25, 10.0),
           (0.50, 1.0), (0.50, 2.0), (0.50, 5.0), (0.50, 10.0)]

def load_wm_zscores(data_dir, exp_prefix, gamma, delta):
    exp_name = f"roc500_{exp_prefix}_d{int(delta)}_g{int(gamma*100)}"
    det_file = data_dir / exp_name / "detection.jsonl"
    if not det_file.exists():
        return None
    data = [json.loads(l) for l in det_file.read_text().strip().split('\n')]
    return np.array([d["watermarked"]["z_score"] for d in data])

def fmt_delta(v):
    sign = "+" if v >= 0 else "$-$"
    return f"{sign}{abs(v):.2f}"

def fmt_p(p):
    if p < 1e-5: return "$< 10^{-5}$"
    if p < 1e-3: return "$< 10^{-3}$"
    return f"{p:.3f}"

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    lines = [
        r"\begin{tabular}{@{}ccccccc@{}}",
        r"\toprule",
        r"$\gamma$ & $\delta$ & KGW (std) & Ours (std) & $\Delta$ & t-stat & p \\",
        r"\midrule",
        r"\multicolumn{7}{c}{\textit{Without self-salt}} \\",
        r"\midrule",
    ]
    
    # Simple-1
    for gamma, delta in CONFIGS:
        cpu = load_wm_zscores(DATA_DIR / "roc500_simple1", "simple_1", gamma, delta)
        gpu = load_wm_zscores(DATA_DIR / "roc500_gsimple1", "gsimple_1", gamma, delta)
        if cpu is None or gpu is None:
            continue
        t, p = stats.ttest_ind(cpu, gpu)
        diff = gpu.mean() - cpu.mean()
        delta_str = "10.0" if delta == 10.0 else f"{delta:.1f}"
        lines.append(f"{gamma:.2f} & {delta_str} & {cpu.mean():.2f} ({cpu.std():.2f}) & {gpu.mean():.2f} ({gpu.std():.2f}) & {fmt_delta(diff)} & {t:.2f} & {fmt_p(p)} \\\\")
    
    lines.extend([
        r"\midrule",
        r"\multicolumn{7}{c}{\textit{With self-salt}} \\",
        r"\midrule",
    ])
    
    # Selfhash
    for gamma, delta in CONFIGS:
        cpu = load_wm_zscores(DATA_DIR / "roc500_selfhash", "selfhash", gamma, delta)
        gpu = load_wm_zscores(DATA_DIR / "roc500_gselfhash", "gselfhash", gamma, delta)
        if cpu is None or gpu is None:
            continue
        t, p = stats.ttest_ind(cpu, gpu)
        diff = gpu.mean() - cpu.mean()
        delta_str = "10.0" if delta == 10.0 else f"{delta:.1f}"
        lines.append(f"{gamma:.2f} & {delta_str} & {cpu.mean():.2f} ({cpu.std():.2f}) & {gpu.mean():.2f} ({gpu.std():.2f}) & {fmt_delta(diff)} & {t:.2f} & {fmt_p(p)} \\\\")
    
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    
    content = "\n".join(lines)
    TABLE_FILE.write_text(content + "\n")
    print(content)
    print(f"\nUpdated {TABLE_FILE}")

if __name__ == "__main__":
    main()
