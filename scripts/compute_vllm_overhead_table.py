#!/usr/bin/env python3
"""Generate LaTeX table for vLLM serving benchmark (tab:vllm_overhead).

Compares request throughput with and without SBW watermark at various concurrency levels.

Input: vllm_benchmark/*.txt files (output from benchmark_vllm_overhead.sh)
Output: LaTeX table with throughput and overhead percentages
"""

import re
from pathlib import Path

# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "vllm_benchmark"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
TABLE_FILE = TABLES_DIR / "tab_vllm_overhead.tex"

CONCURRENCIES = [1, 8, 32, 64, 128]


def parse_throughput(filepath: Path) -> float:
    """Extract request throughput from benchmark output file."""
    text = filepath.read_text()
    match = re.search(r"Request throughput \(req/s\):\s+([\d.]+)", text)
    if match:
        return float(match.group(1))
    return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing benchmark result files")
    parser.add_argument("--output", type=Path, default=TABLE_FILE,
                        help="Output LaTeX table path")
    args = parser.parse_args()

    data_dir = args.data_dir

    print("=== vLLM Serving Benchmark: SBW Overhead ===\n")
    print(f"{'Conc':>5} | {'No proc':>8} | {'SBW δ=2':>8} | {'Overhead':>8}")
    print("-" * 40)

    results = []
    for conc in CONCURRENCIES:
        no_proc_file = data_dir / f"no_processor_conc{conc}.txt"
        delta2_file = data_dir / f"delta2_conc{conc}.txt"

        if not no_proc_file.exists() or not delta2_file.exists():
            print(f"{conc:>5} | MISSING")
            continue

        no_proc = parse_throughput(no_proc_file)
        delta2 = parse_throughput(delta2_file)

        # Overhead as throughput reduction percentage
        overhead_pct = (no_proc - delta2) / no_proc * 100 if no_proc > 0 else 0

        results.append({
            "conc": conc,
            "no_proc": no_proc,
            "delta2": delta2,
            "overhead": overhead_pct,
        })

        print(f"{conc:>5} | {no_proc:>8.2f} | {delta2:>8.2f} | {overhead_pct:>+7.1f}%")

    # Summary for high concurrency
    high_conc_results = [r for r in results if r["conc"] >= 32]
    if high_conc_results:
        max_overhead = max(r["overhead"] for r in high_conc_results)
        print(f"\nAt concurrency ≥32: max overhead = {max_overhead:.1f}%")

    # Generate LaTeX table
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{tabular}{@{}rccc@{}}",
        r"\toprule",
        r"Concurrency & Baseline (req/s) & SBW (req/s) & Overhead \\",
        r"\midrule",
    ]

    for r in results:
        overhead_str = f"{r['overhead']:.1f}\\%" if r['overhead'] >= 0.05 else "$<$0.1\\%"
        lines.append(
            f"{r['conc']} & {r['no_proc']:.2f} & {r['delta2']:.2f} & {overhead_str} \\\\"
        )

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
