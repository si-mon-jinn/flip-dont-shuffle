#!/usr/bin/env python3
"""Generate LaTeX table for one-sample KS tests (tab:ks_onesample).

Tests non-watermarked z-scores vs N(0,1) for both Simple-1 and Selfhash.
Loads texts from generations.jsonl and scores on-the-fly with detectors.
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
TABLE_FILE = TABLES_DIR / "tab_ks_onesample.tex"
HASH_KEY = 15485863
TOKENIZER_NAME = "Qwen/Qwen3-8B"


def collect_texts(data_dir, exp_prefix):
    """Collect non-watermarked texts from all experiments (excluding d0)."""
    texts = []
    for gen_gamma in [0.25, 0.50]:
        for delta in [1.0, 2.0, 5.0, 10.0]:
            exp_name = f"roc500_{exp_prefix}_d{int(delta)}_g{int(gen_gamma*100)}"
            gen_file = data_dir / exp_name / "generations.jsonl"
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


def fmt_mean(v):
    sign = "+" if v >= 0 else "$-$"
    return f"{sign}{abs(v):.3f}"


def fmt_p(p):
    if p < 1e-200: return "$< 10^{-267}$"
    if p < 1e-50:
        exp = int(np.floor(np.log10(p)))
        return f"$< 10^{{{exp}}}$"
    return f"{p:.2e}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    vocab = [0] * tokenizer.vocab_size
    
    # Collect texts for each scheme family
    simple1_texts = collect_texts(DATA_DIR / "roc500_simple1", "simple_1")
    gsimple1_texts = collect_texts(DATA_DIR / "roc500_gsimple1", "gsimple_1")
    selfhash_texts = collect_texts(DATA_DIR / "roc500_selfhash", "selfhash")
    gselfhash_texts = collect_texts(DATA_DIR / "roc500_gselfhash", "gselfhash")
    
    print(f"Collected: simple_1={len(simple1_texts)}, gpu-simple_1={len(gsimple1_texts)}, "
          f"selfhash={len(selfhash_texts)}, gpu-selfhash={len(gselfhash_texts)}")
    
    results = []
    
    for det_gamma in [0.25, 0.50]:
        print(f"\n=== Detection γ={det_gamma} ===")
        
        # Simple-1 schemes (gpu-simple_1 = SBW-1 in paper)
        for scheme, seeding, texts in [
            ("simple\\_1", "simple_1", simple1_texts),
            ("SBW-1", "gpu-simple_1", gsimple1_texts),
        ]:
            if not texts:
                continue
            detector = WatermarkDetector(
                device=device, tokenizer=tokenizer, vocab=vocab,
                gamma=det_gamma, seeding_scheme=seeding, hash_key=HASH_KEY, z_threshold=4.0
            )
            z = score_texts(texts, detector)
            mean, std = z.mean(), z.std()
            ks, p = stats.kstest(z, 'norm')
            results.append(("Simple-1", det_gamma, scheme, mean, std, ks, p, len(z)))
            print(f"  {scheme}: n={len(z)}, mean={mean:.3f}, std={std:.3f}, KS={ks:.3f}, p={p:.2e}")
        
        # Selfhash schemes (gpu-selfhash = SBW-ss in paper)
        for scheme, seeding, texts in [
            ("selfhash", "selfhash", selfhash_texts),
            ("SBW-ss", "gpu-selfhash", gselfhash_texts),
        ]:
            if not texts:
                continue
            detector = WatermarkDetector(
                device=device, tokenizer=tokenizer, vocab=vocab,
                gamma=det_gamma, seeding_scheme=seeding, hash_key=HASH_KEY, z_threshold=4.0
            )
            z = score_texts(texts, detector)
            mean, std = z.mean(), z.std()
            ks, p = stats.kstest(z, 'norm')
            results.append(("Selfhash", det_gamma, scheme, mean, std, ks, p, len(z)))
            print(f"  {scheme}: n={len(z)}, mean={mean:.3f}, std={std:.3f}, KS={ks:.3f}, p={p:.2e}")
    
    # Generate table
    lines = [
        r"\begin{tabular}{@{}cccccc@{}}",
        r"\toprule",
        r"$\gamma$ & Scheme & Mean & Std & KS & p-value \\",
        r"\midrule",
        r"\multicolumn{6}{c}{\textit{Without self-salt}} \\",
        r"\midrule",
    ]
    for family, gamma, scheme, mean, std, ks, p, n in results:
        if family == "Simple-1":
            lines.append(f"{gamma:.2f} & \\texttt{{{scheme}}} & {fmt_mean(mean)} & {std:.3f} & {ks:.3f} & {fmt_p(p)} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{c}{\textit{With self-salt}} \\")
    lines.append(r"\midrule")
    for family, gamma, scheme, mean, std, ks, p, n in results:
        if family == "Selfhash":
            lines.append(f"{gamma:.2f} & \\texttt{{{scheme}}} & {fmt_mean(mean)} & {std:.3f} & {ks:.3f} & {fmt_p(p)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    
    content = "\n".join(lines)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_FILE.write_text(content)
    print(f"\n{content}")
    print(f"\nUpdated {TABLE_FILE}")


if __name__ == "__main__":
    main()
