#!/usr/bin/env python3
"""Generate LaTeX table for perplexity hash isolation (tab:ppl_hash_isolation)."""

import json
from pathlib import Path
import numpy as np
from scipy import stats

DATA_DIR = Path(__file__).parent.parent / "data"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
TABLE_FILE = TABLES_DIR / "tab_ppl_hash_isolation.tex"

def load_perplexity(ppl_file):
    if not ppl_file.exists():
        return None
    data = [json.loads(l) for l in ppl_file.read_text().strip().split('\n')]
    wm = np.array([d["watermarked"] for d in data])
    nowm = np.array([d["non_watermarked"] for d in data])
    return wm, nowm

def fmt_p(p):
    if p < 1e-8: return "$<10^{-8}$"
    return f"{p:.3f}" if p >= 0.01 else f".{int(p*1000):03d}"

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    lines = [
        r"\begin{tabular}{@{}ll|cc|cc@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c|}{$\delta=5$} & \multicolumn{2}{c}{$\delta=10$} \\",
        r"Scheme & Hash & $\Delta$ppl & p & $\Delta$ppl & p \\",
        r"\midrule",
    ]
    
    # selfhash (baseline)
    sh5 = load_perplexity(DATA_DIR / "roc500_selfhash/roc500_selfhash_d5_g25/metrics/perplexity.jsonl")
    sh10 = load_perplexity(DATA_DIR / "roc500_selfhash/roc500_selfhash_d10_g25/metrics/perplexity.jsonl")
    if sh5 and sh10:
        d5 = sh5[0].mean() - sh5[1].mean()
        d10 = sh10[0].mean() - sh10[1].mean()
        lines.append(f"selfhash & CPU perm & {d5:.2f} & --- & {d10:.2f} & --- \\\\")
        sh5_wm, sh10_wm = sh5[0], sh10[0]
    
    # gpu-selfhash-cpuhash
    gshcpu5 = load_perplexity(DATA_DIR / "roc500_gselfhash_d5_g25_cpuhash/metrics/perplexity.jsonl")
    gshcpu10 = load_perplexity(DATA_DIR / "roc500_gselfhash_d10_g25_cpuhash/metrics/perplexity.jsonl")
    if gshcpu5 and gshcpu10:
        d5 = gshcpu5[0].mean() - gshcpu5[1].mean()
        d10 = gshcpu10[0].mean() - gshcpu10[1].mean()
        _, p5 = stats.ttest_ind(sh5_wm, gshcpu5[0])
        _, p10 = stats.ttest_ind(sh10_wm, gshcpu10[0])
        lines.append(f"SBW-ss-cpu & CPU perm & {d5:.2f} & {fmt_p(p5)} & {d10:.2f} & {fmt_p(p10)} \\\\")
    
    # gpu-selfhash (Jenkins)
    gsh5 = load_perplexity(DATA_DIR / "roc500_gselfhash/roc500_gselfhash_d5_g25/metrics/perplexity.jsonl")
    gsh10 = load_perplexity(DATA_DIR / "roc500_gselfhash/roc500_gselfhash_d10_g25/metrics/perplexity.jsonl")
    if gsh5 and gsh10:
        d5 = gsh5[0].mean() - gsh5[1].mean()
        d10 = gsh10[0].mean() - gsh10[1].mean()
        _, p5 = stats.ttest_ind(sh5_wm, gsh5[0])
        _, p10 = stats.ttest_ind(sh10_wm, gsh10[0])
        lines.append(f"SBW-ss & Jenkins & {d5:.2f} & {fmt_p(p5)} & {d10:.2f} & {fmt_p(p10)} \\\\")
    
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    
    content = "\n".join(lines)
    TABLE_FILE.write_text(content + "\n")
    print(content)
    print(f"\nUpdated {TABLE_FILE}")

if __name__ == "__main__":
    main()
