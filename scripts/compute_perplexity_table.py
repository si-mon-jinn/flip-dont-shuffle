#!/usr/bin/env python3
"""Generate/update LaTeX table for perplexity comparison (Simple-1 columns).

Updates tab_ppl.tex with simple_1 vs gpu-simple_1 data.
"""

import json
import re
from pathlib import Path
import numpy as np
from scipy import stats

DATA_DIR = Path(__file__).parent.parent / "data"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
TABLE_FILE = TABLES_DIR / "tab_ppl.tex"

CONFIGS = [
    (0.25, 1.0), (0.25, 2.0), (0.25, 5.0), (0.25, 10.0),
    (0.50, 1.0), (0.50, 2.0), (0.50, 5.0), (0.50, 10.0),
]

TEMPLATE = r"""\begin{tabular}{@{}cc|ccc|ccc@{}}
\toprule
& & \multicolumn{3}{c|}{\textit{Simple-1}} & \multicolumn{3}{c}{\textit{Selfhash}} \\
$\gamma$ & $\delta$ & $\Delta$KGW & $\Delta$Ours & p & $\Delta$KGW & $\Delta$Ours & p \\
\midrule
0.25 & 1.0 & -- & -- & -- & -- & -- & -- \\
0.25 & 2.0 & -- & -- & -- & -- & -- & -- \\
0.25 & 5.0 & -- & -- & -- & -- & -- & -- \\
0.25 & 10 & -- & -- & -- & -- & -- & -- \\
0.50 & 1.0 & -- & -- & -- & -- & -- & -- \\
0.50 & 2.0 & -- & -- & -- & -- & -- & -- \\
0.50 & 5.0 & -- & -- & -- & -- & -- & -- \\
0.50 & 10 & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
"""

def load_perplexity(scheme: str, gamma: float, delta: float):
    prefix = "roc500_gsimple1" if scheme == "gpu" else "roc500_simple1"
    exp_name = f"roc500_{'g' if scheme == 'gpu' else ''}simple_1_d{int(delta)}_g{int(gamma*100)}"
    ppl_file = DATA_DIR / prefix / exp_name / "metrics" / "perplexity.jsonl"
    if not ppl_file.exists():
        return None
    return [json.loads(l) for l in ppl_file.read_text().strip().split('\n')]

def fmt_delta(v):
    sign = "+" if v >= 0 else "$-$"
    if abs(v) >= 10:
        return f"{sign}{abs(v):.1f}"
    return f"{sign}{abs(v):.2f}"

def fmt_p(p):
    if p < 1e-7: return "$<10^{-7}$"
    if p < 0.01: return f".{int(p*1000):03d}".lstrip('0') or ".000"
    return f"{p:.2f}"

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    content = TABLE_FILE.read_text() if TABLE_FILE.exists() else TEMPLATE
    
    for gamma, delta in CONFIGS:
        cpu = load_perplexity("cpu", gamma, delta)
        gpu = load_perplexity("gpu", gamma, delta)
        if not cpu or not gpu:
            continue
        
        cpu_wm = np.array([d["watermarked"] for d in cpu])
        cpu_nowm = np.mean([d["non_watermarked"] for d in cpu])
        gpu_wm = np.array([d["watermarked"] for d in gpu])
        gpu_nowm = np.mean([d["non_watermarked"] for d in gpu])
        
        delta_cpu = cpu_wm.mean() - cpu_nowm
        delta_gpu = gpu_wm.mean() - gpu_nowm
        _, p_val = stats.ttest_ind(cpu_wm, gpu_wm)
        
        delta_str = "10" if delta == 10.0 else f"{delta:.1f}"
        pattern = rf"^({gamma:.2f}) & ({delta_str}) & [^&]+ & [^&]+ & [^&]+ & (.*)$"
        replacement = rf"\1 & \2 & {fmt_delta(delta_cpu)} & {fmt_delta(delta_gpu)} & {fmt_p(p_val)} & \3"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    TABLE_FILE.write_text(content)
    print(content)
    print(f"\nUpdated {TABLE_FILE}")

if __name__ == "__main__":
    main()
