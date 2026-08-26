#!/usr/bin/env python3
"""Generate/update LaTeX table for ROC-AUC comparison (Selfhash columns).

Updates tab_roc.tex with selfhash vs gpu-selfhash data.
"""

import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).parent.parent / "data"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
TABLE_FILE = TABLES_DIR / "tab_roc.tex"

CONFIGS = [
    (0.25, 1.0), (0.25, 2.0), (0.25, 5.0), (0.25, 10.0),
    (0.50, 1.0), (0.50, 2.0), (0.50, 5.0), (0.50, 10.0),
]

TEMPLATE = r"""\begin{tabular}{@{}ccccc|ccc@{}}
\toprule
\multicolumn{5}{c|}{\textit{Simple-1}} & \multicolumn{3}{c}{\textit{Selfhash (Top-40)}} \\
$\gamma$ & $\delta$ & KGW & Ours & $\Delta$ & KGW & Ours & $\Delta$ \\
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

def load_detection_results(scheme: str, gamma: float, delta: float) -> list:
    prefix = "roc500_gselfhash" if scheme == "gpu-selfhash" else "roc500_selfhash"
    exp_name = f"roc500_{'g' if 'gpu' in scheme else ''}selfhash_d{int(delta)}_g{int(gamma*100)}"
    det_file = DATA_DIR / prefix / exp_name / "detection.jsonl"
    if not det_file.exists():
        return None
    return [json.loads(l) for l in det_file.read_text().strip().split('\n')]

def compute_auc(detections: list) -> float:
    wm_z = [d["watermarked"]["z_score"] for d in detections]
    nowm_z = [d["non_watermarked"]["z_score"] for d in detections]
    y_true = [1]*len(wm_z) + [0]*len(nowm_z)
    y_scores = wm_z + nowm_z
    return roc_auc_score(y_true, y_scores)

def fmt_auc(v):
    if v >= 0.9995: return "1.00"
    return f".{int(v*1000):03d}".rstrip('0') or ".000"

def fmt_delta(v):
    if abs(v) < 0.0005: return ".000"
    sign = "+" if v >= 0 else "$-$"
    val = f".{int(abs(v)*1000):03d}".rstrip('0')
    if not val or val == ".": val = ".000"
    return f"{sign}{val}"

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    content = TABLE_FILE.read_text() if TABLE_FILE.exists() else TEMPLATE
    
    for gamma, delta in CONFIGS:
        cpu_det = load_detection_results("selfhash", gamma, delta)
        gpu_det = load_detection_results("gpu-selfhash", gamma, delta)
        
        if not cpu_det or not gpu_det:
            continue
        
        cpu_auc = compute_auc(cpu_det)
        gpu_auc = compute_auc(gpu_det)
        diff = gpu_auc - cpu_auc
        
        delta_str = "10" if delta == 10.0 else f"{delta:.1f}"
        # Match row and update last 3 columns (Selfhash)
        pattern = rf"^({gamma:.2f}) & ({delta_str}) & ([^&]+) & ([^&]+) & ([^&]+) & [^&]+ & [^&]+ & [^\\\n]+ \\\\$"
        replacement = rf"\1 & \2 & \3 & \4 & \5 & {fmt_auc(cpu_auc)} & {fmt_auc(gpu_auc)} & {fmt_delta(diff)} \\\\"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    TABLE_FILE.write_text(content)
    print(content)
    print(f"\nUpdated {TABLE_FILE}")

if __name__ == "__main__":
    main()
