#!/usr/bin/env python3
"""Generate LaTeX table for two-sample KS tests on watermarked z-scores (tab:ks_twosample_wm)."""

import json
from pathlib import Path
import numpy as np
from scipy import stats

DATA_DIR = Path(__file__).parent.parent / "data"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
TABLE_FILE = TABLES_DIR / "tab_ks_twosample_wm.tex"

CONFIGS = [(0.25, 1.0), (0.25, 2.0), (0.25, 5.0), (0.25, 10.0),
           (0.50, 1.0), (0.50, 2.0), (0.50, 5.0), (0.50, 10.0)]

def load_wm_zscores(data_dir, exp_prefix, gamma, delta):
    exp_name = f"roc500_{exp_prefix}_d{int(delta)}_g{int(gamma*100)}"
    det_file = data_dir / exp_name / "detection.jsonl"
    if not det_file.exists():
        return None
    data = [json.loads(l) for l in det_file.read_text().strip().split('\n')]
    return np.array([d["watermarked"]["z_score"] for d in data])

def fmt_p(p):
    if p < 1e-5: return f"$<10^{{-5}}$"
    return f"{p:.3f}"

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    lines = [
        r"\begin{tabular}{@{}cc|cc|cc@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c|}{\textit{Without self-salt}} & \multicolumn{2}{c}{\textit{With self-salt}} \\",
        r"$\gamma$ & $\delta$ & KS & p & KS & p \\",
        r"\midrule",
    ]
    
    for gamma, delta in CONFIGS:
        # Simple-1
        cpu_s = load_wm_zscores(DATA_DIR / "roc500_simple1", "simple_1", gamma, delta)
        gpu_s = load_wm_zscores(DATA_DIR / "roc500_gsimple1", "gsimple_1", gamma, delta)
        # Selfhash
        cpu_h = load_wm_zscores(DATA_DIR / "roc500_selfhash", "selfhash", gamma, delta)
        gpu_h = load_wm_zscores(DATA_DIR / "roc500_gselfhash", "gselfhash", gamma, delta)
        
        ks_s, p_s = stats.ks_2samp(cpu_s, gpu_s) if cpu_s is not None and gpu_s is not None else (0, 1)
        ks_h, p_h = stats.ks_2samp(cpu_h, gpu_h) if cpu_h is not None and gpu_h is not None else (0, 1)
        
        delta_str = "10" if delta == 10.0 else f"{delta:.1f}"
        lines.append(f"{gamma:.2f} & {delta_str} & {ks_s:.3f} & {fmt_p(p_s)} & {ks_h:.3f} & {fmt_p(p_h)} \\\\")
    
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    
    content = "\n".join(lines)
    TABLE_FILE.write_text(content + "\n")
    print(content)
    print(f"\nUpdated {TABLE_FILE}")

if __name__ == "__main__":
    main()
