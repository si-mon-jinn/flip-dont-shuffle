#!/usr/bin/env python3
"""Generate LaTeX table for text diversity comparison (tab:diversity)."""

import json
from pathlib import Path
import numpy as np
from scipy import stats

DATA_DIR = Path(__file__).parent.parent / "data"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
TABLE_FILE = TABLES_DIR / "tab_diversity.tex"

def load_diversity(div_file):
    if not div_file.exists():
        return None
    data = [json.loads(l) for l in div_file.read_text().strip().split('\n')]
    wm = np.array([d["watermarked"] for d in data])
    wm = wm[np.isfinite(wm)]
    return wm if len(wm) > 0 else None

def fmt_p(p):
    if p < 0.0001: return "$<$.0001"
    return f"{p:.2f}"

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    lines = [
        r"\begin{tabular}{@{}cccc@{}}",
        r"\toprule",
        r"$\delta$ & selfhash & SBW-ss & p \\",
        r"\midrule",
    ]
    
    for delta in [1, 2, 5, 10]:
        sh = load_diversity(DATA_DIR / f"roc500_selfhash/roc500_selfhash_d{delta}_g25/metrics/diversity.jsonl")
        gsh = load_diversity(DATA_DIR / f"roc500_gselfhash/roc500_gselfhash_d{delta}_g25/metrics/diversity.jsonl")
        if sh is not None and gsh is not None:
            _, p = stats.ttest_ind(sh, gsh)
            lines.append(f"{delta} & {sh.mean():.3f} & {gsh.mean():.3f} & {fmt_p(p)} \\\\")
        else:
            lines.append(f"{delta} & -- & -- & -- \\\\")
    
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    
    content = "\n".join(lines)
    TABLE_FILE.write_text(content + "\n")
    print(content)
    print(f"\nUpdated {TABLE_FILE}")

if __name__ == "__main__":
    main()
