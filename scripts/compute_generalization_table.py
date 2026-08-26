#!/usr/bin/env python3
"""Generate LaTeX table for cross-model generalization experiments (tab:generalization).

Compares KGW vs SBW on Falcon-7B with Alpaca prompts, different sampling (T=0.7, top_p=0.8),
Mistral judge, and A6000 hardware — 5 variables changed vs main paper.

Input: detection.jsonl files from falcon_alpaca experiments
Output: LaTeX table with two-sample KS tests
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

# Default data directory (can be overridden via command line)
DATA_DIR = Path(__file__).parent.parent / "data" / "generalization"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
TABLE_FILE = TABLES_DIR / "tab_generalization.tex"


def load_zscores(detection_file: Path):
    """Load z-scores from detection.jsonl, filtering degenerate samples."""
    wm_z, nowm_z = [], []
    with open(detection_file) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            # Filter degenerate samples (z=0 means empty/single-token output)
            if d["watermarked"]["z_score"] != 0:
                wm_z.append(d["watermarked"]["z_score"])
            if d["non_watermarked"]["z_score"] != 0:
                nowm_z.append(d["non_watermarked"]["z_score"])
    return np.array(wm_z), np.array(nowm_z)


def fmt_p(p: float) -> str:
    """Format p-value for LaTeX."""
    if p < 1e-5:
        return f"$<10^{{-5}}$"
    return f"{p:.3f}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing falcon_alpaca_* subdirectories")
    parser.add_argument("--output", type=Path, default=TABLE_FILE,
                        help="Output LaTeX table path")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    # Load non-self-salt experiments
    simple1_wm, simple1_nowm = load_zscores(
        data_dir / "falcon_alpaca_simple_1_d2_g50" / "detection.jsonl")
    gsimple1_wm, gsimple1_nowm = load_zscores(
        data_dir / "falcon_alpaca_gsimple_1_d2_g50" / "detection.jsonl")
    
    # Load self-salt experiments (using cpuhash for fair comparison)
    selfhash_wm, selfhash_nowm = load_zscores(
        data_dir / "falcon_alpaca_selfhash_d2_g50" / "detection.jsonl")
    gselfhash_wm, gselfhash_nowm = load_zscores(
        data_dir / "falcon_alpaca_gselfhash_cpuhash_d2_g50" / "detection.jsonl")
    
    # Compute KS tests
    ks_simple_wm = stats.ks_2samp(simple1_wm, gsimple1_wm)
    ks_simple_nowm = stats.ks_2samp(simple1_nowm, gsimple1_nowm)
    ks_self_wm = stats.ks_2samp(selfhash_wm, gselfhash_wm)
    ks_self_nowm = stats.ks_2samp(selfhash_nowm, gselfhash_nowm)
    
    # Print summary
    print("=== Cross-Model Generalization: Falcon-7B + Alpaca + T=0.7 ===\n")
    print(f"Non-self-salt (simple_1 vs gsimple_1):")
    print(f"  Watermarked:     n={len(simple1_wm):3d} vs {len(gsimple1_wm):3d}, "
          f"D={ks_simple_wm.statistic:.3f}, p={ks_simple_wm.pvalue:.3f}")
    print(f"  Non-watermarked: n={len(simple1_nowm):3d} vs {len(gsimple1_nowm):3d}, "
          f"D={ks_simple_nowm.statistic:.3f}, p={ks_simple_nowm.pvalue:.3f}")
    print()
    print(f"Self-salt (selfhash vs gselfhash-cpuhash):")
    print(f"  Watermarked:     n={len(selfhash_wm):3d} vs {len(gselfhash_wm):3d}, "
          f"D={ks_self_wm.statistic:.3f}, p={ks_self_wm.pvalue:.3f}")
    print(f"  Non-watermarked: n={len(selfhash_nowm):3d} vs {len(gselfhash_nowm):3d}, "
          f"D={ks_self_nowm.statistic:.3f}, p={ks_self_nowm.pvalue:.3f}")
    print()
    
    # Summary stats
    print("=== Summary Statistics ===\n")
    print(f"simple_1 wm:       mean={simple1_wm.mean():.2f} ± {simple1_wm.std():.2f}")
    print(f"gsimple_1 wm:      mean={gsimple1_wm.mean():.2f} ± {gsimple1_wm.std():.2f}")
    print(f"selfhash wm:       mean={selfhash_wm.mean():.2f} ± {selfhash_wm.std():.2f}")
    print(f"gselfhash-cpu wm:  mean={gselfhash_wm.mean():.2f} ± {gselfhash_wm.std():.2f}")
    print()
    
    # Generate LaTeX table
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    lines = [
        r"\begin{tabular}{@{}l|cc|cc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{\textit{Watermarked}} & \multicolumn{2}{c}{\textit{Non-watermarked}} \\",
        r"Seeding & KS $D$ & $p$ & KS $D$ & $p$ \\",
        r"\midrule",
        f"Without self-salt & {ks_simple_wm.statistic:.3f} & {fmt_p(ks_simple_wm.pvalue)} & "
        f"{ks_simple_nowm.statistic:.3f} & {fmt_p(ks_simple_nowm.pvalue)} \\\\",
        f"With self-salt & {ks_self_wm.statistic:.3f} & {fmt_p(ks_self_wm.pvalue)} & "
        f"{ks_self_nowm.statistic:.3f} & {fmt_p(ks_self_nowm.pvalue)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    
    content = "\n".join(lines)
    args.output.write_text(content + "\n")
    print(f"LaTeX table written to: {args.output}")
    print()
    print(content)


if __name__ == "__main__":
    main()
