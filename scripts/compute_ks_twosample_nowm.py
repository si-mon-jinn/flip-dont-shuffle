#!/usr/bin/env python3
"""Compute pooled two-sample KS tests on non-watermarked z-scores.

Loads non-watermarked texts from all experiments (4 deltas x 2 gammas = 4000 texts),
scores them with each detector gamma, then runs two-sample KS between KGW and SBW.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sbw import WatermarkDetector

DATA_DIR = Path(__file__).parent.parent / "data"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
OUTPUT_TABLE = TABLES_DIR / "tab_ks_twosample_nowm.tex"
HASH_KEY = 15485863
TOKENIZER_NAME = "Qwen/Qwen3-8B"
DELTAS = [1.0, 2.0, 5.0, 10.0]
GEN_GAMMAS = [0.25, 0.50]
DET_GAMMAS = [0.25, 0.50]


def collect_nowm_texts(scheme_dir, exp_prefix):
    """Collect non-watermarked texts from all experiments (4 deltas x 2 gammas)."""
    texts = []
    for gamma in GEN_GAMMAS:
        for delta in DELTAS:
            exp_name = f"roc500_{exp_prefix}_d{int(delta)}_g{int(gamma*100)}"
            gen_file = DATA_DIR / scheme_dir / exp_name / "generations.jsonl"
            if gen_file.exists():
                with open(gen_file) as f:
                    for line in f:
                        if line.strip():
                            texts.append(json.loads(line)["non_watermarked"])
    return texts


def score_texts(texts, detector):
    """Score texts with detector, return z-scores."""
    z_scores = []
    for text in tqdm(texts, desc="Scoring", leave=False):
        result = detector.detect(text=text)
        z_scores.append(result.get("z_score", 0.0))
    return np.array(z_scores)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    vocab = [0] * tokenizer.vocab_size

    schemes = [
        ("simple_1", "roc500_simple1", "simple_1", "roc500_gsimple1", "gpu-simple_1"),
        ("selfhash", "roc500_selfhash", "selfhash", "roc500_gselfhash", "gpu-selfhash"),
    ]

    print("Pooled Two-Sample KS Tests (non-watermarked z-scores)")
    print("Pooling: 4 deltas x 2 gammas x 500 samples = 4000 texts per scheme")
    print("=" * 70)

    results = []
    for name, kgw_dir, kgw_seeding, sbw_dir, sbw_seeding in schemes:
        # Collect texts
        kgw_texts = collect_nowm_texts(kgw_dir, kgw_seeding.replace("-", "_"))
        sbw_texts = collect_nowm_texts(sbw_dir, "g" + kgw_seeding.replace("-", "_"))
        print(f"\n--- {name} (N_KGW={len(kgw_texts)}, N_SBW={len(sbw_texts)}) ---")
        print(f"{'det_γ':>6} | {'KS':>8} | {'p-value':>12} | {'KGW mean':>9} | {'SBW mean':>9}")
        print("-" * 60)

        for det_gamma in DET_GAMMAS:
            # Score with KGW detector
            det_kgw = WatermarkDetector(
                device=device, tokenizer=tokenizer, vocab=vocab,
                gamma=det_gamma, seeding_scheme=kgw_seeding, hash_key=HASH_KEY, z_threshold=4.0
            )
            z_kgw = score_texts(kgw_texts, det_kgw)

            # Score with SBW detector
            det_sbw = WatermarkDetector(
                device=device, tokenizer=tokenizer, vocab=vocab,
                gamma=det_gamma, seeding_scheme=sbw_seeding, hash_key=HASH_KEY, z_threshold=4.0
            )
            z_sbw = score_texts(sbw_texts, det_sbw)

            # Two-sample KS
            ks, p = stats.ks_2samp(z_kgw, z_sbw)
            results.append((name, det_gamma, len(z_kgw), ks, p))
            print(f"{det_gamma:>6.2f} | {ks:>8.4f} | {p:>12.2e} | {z_kgw.mean():>+9.3f} | {z_sbw.mean():>+9.3f}")

    # Generate LaTeX table
    def fmt_p(p):
        if p > 0.01:
            return f"{p:.3f}"
        exp = int(np.floor(np.log10(p)))
        return f"$< 10^{{{exp}}}$"

    lines = [
        r"\begin{tabular}{@{}ccccl@{}}",
        r"\toprule",
        r"Scheme & $\gamma_{\text{det}}$ & $N$ & KS & p-value \\",
        r"\midrule",
    ]
    prev_name = None
    for name, det_gamma, n, ks, p in results:
        if prev_name and name != prev_name:
            lines.append(r"\midrule")
        name_escaped = name.replace('_', r'\_')
        lines.append(f"\\texttt{{{name_escaped}}} & {det_gamma:.2f} & {n} & {ks:.3f} & {fmt_p(p)} \\\\")
        prev_name = name
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
