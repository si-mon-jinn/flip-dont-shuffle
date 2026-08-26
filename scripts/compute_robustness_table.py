#!/usr/bin/env python3
"""Generate LaTeX table for robustness experiments (tab:robustness).

Compares KGW vs SBW z-score distributions under 19 attack configurations.

Input: attacks/*.jsonl files from roc500_simple_1_d2_g50 and roc500_gsimple_1_d2_g50
Output: LaTeX table with two-sample KS tests
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "robustness"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
TABLE_FILE = TABLES_DIR / "tab_robustness.tex"

# Attack configurations in display order
ATTACKS = [
    # (attack_id, display_name, category)
    ("random_char_1pct", "Char subst. 1\\%", "Character"),
    ("random_char_5pct", "Char subst. 5\\%", "Character"),
    ("random_char_10pct", "Char subst. 10\\%", "Character"),
    ("random_char_20pct", "Char subst. 20\\%", "Character"),
    ("char_delete_5pct", "Char delete 5\\%", "Character"),
    ("char_delete_10pct", "Char delete 10\\%", "Character"),
    ("char_delete_20pct", "Char delete 20\\%", "Character"),
    ("truncation_75pct", "Truncation 25\\%", "Structural"),  # keep 75% = remove 25%
    ("truncation_50pct", "Truncation 50\\%", "Structural"),
    ("truncation_25pct", "Truncation 75\\%", "Structural"),  # keep 25% = remove 75%
    ("word_delete_10pct", "Word delete 10\\%", "Word"),
    ("word_delete_20pct", "Word delete 20\\%", "Word"),
    ("word_delete_30pct", "Word delete 30\\%", "Word"),
    ("word_reorder_full", "Word reorder", "Word"),
    ("paraphrase_light", "Paraphrase (light)", "Semantic"),
    ("paraphrase_moderate", "Paraphrase (moderate)", "Semantic"),
    ("paraphrase_heavy", "Paraphrase (heavy)", "Semantic"),
    ("mlm_10pct", "MLM subst. 10\\%", "Semantic"),
    ("mlm_20pct", "MLM subst. 20\\%", "Semantic"),
]


def load_attack_zscores(attack_file: Path):
    """Load z-scores from attack jsonl file."""
    z_scores = []
    with open(attack_file) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            z = d.get("detection", {}).get("z_score", 0)
            if z != 0:  # filter degenerate
                z_scores.append(z)
    return np.array(z_scores)


def fmt_p(p: float) -> str:
    """Format p-value for LaTeX."""
    if p < 0.001:
        return f"$<$0.001"
    return f"{p:.3f}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing experiment subdirectories")
    parser.add_argument("--output", type=Path, default=TABLE_FILE,
                        help="Output LaTeX table path")
    args = parser.parse_args()

    data_dir = args.data_dir
    kgw_dir = data_dir / "roc500_simple_1_d2_g50" / "attacks"
    sbw_dir = data_dir / "roc500_gsimple_1_d2_g50" / "attacks"

    print("=== Robustness Experiments: KGW vs SBW under Attacks ===\n")

    results = []
    equiv_count = 0
    sbw_better_count = 0

    for attack_id, display_name, category in ATTACKS:
        kgw_file = kgw_dir / f"{attack_id}.jsonl"
        sbw_file = sbw_dir / f"{attack_id}.jsonl"

        if not kgw_file.exists() or not sbw_file.exists():
            print(f"  {attack_id}: MISSING")
            continue

        kgw_z = load_attack_zscores(kgw_file)
        sbw_z = load_attack_zscores(sbw_file)

        ks = stats.ks_2samp(kgw_z, sbw_z)
        equiv = ks.pvalue > 0.05
        sbw_better = sbw_z.mean() > kgw_z.mean() + 0.1  # meaningful difference

        if equiv:
            equiv_count += 1
        elif sbw_better:
            sbw_better_count += 1

        results.append({
            "id": attack_id,
            "name": display_name,
            "category": category,
            "kgw_mean": kgw_z.mean(),
            "kgw_std": kgw_z.std(),
            "sbw_mean": sbw_z.mean(),
            "sbw_std": sbw_z.std(),
            "ks_d": ks.statistic,
            "ks_p": ks.pvalue,
            "equiv": equiv,
            "sbw_better": sbw_better,
            "n_kgw": len(kgw_z),
            "n_sbw": len(sbw_z),
        })

        status = "✓" if equiv else ("SBW+" if sbw_better else "KGW+")
        print(f"  {display_name:25s}: KGW={kgw_z.mean():.2f}±{kgw_z.std():.2f}, "
              f"SBW={sbw_z.mean():.2f}±{sbw_z.std():.2f}, "
              f"D={ks.statistic:.3f}, p={ks.pvalue:.3f} [{status}]")

    print(f"\n=== Summary ===")
    print(f"Equivalent (p>0.05): {equiv_count}/19")
    print(f"Non-equivalent, SBW better: {sbw_better_count}")
    print(f"Non-equivalent, KGW better: {19 - equiv_count - sbw_better_count}")

    # Generate LaTeX table
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Category & Attack & KGW $\bar{z}$ & SBW $\bar{z}$ & KS $p$ & Equiv. \\",
        r"\midrule",
    ]

    current_category = None
    for r in results:
        cat_str = r["category"] if r["category"] != current_category else ""
        current_category = r["category"]
        
        equiv_str = r"\checkmark" if r["equiv"] else ("SBW+" if r["sbw_better"] else "KGW+")
        
        lines.append(
            f"{cat_str} & {r['name']} & {r['kgw_mean']:.2f} & {r['sbw_mean']:.2f} & "
            f"{fmt_p(r['ks_p'])} & {equiv_str} \\\\"
        )
        
        # Add midrule between categories
        if r != results[-1]:
            next_cat = results[results.index(r) + 1]["category"]
            if next_cat != current_category:
                lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    content = "\n".join(lines)
    args.output.write_text(content + "\n")
    print(f"\nLaTeX table written to: {args.output}")
    print()
    print(content)


if __name__ == "__main__":
    main()
