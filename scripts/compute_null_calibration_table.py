#!/usr/bin/env python3
"""Generate LaTeX table for null calibration (tab:null_calibration).

Computes null calibration on-the-fly by scoring non-watermarked texts.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from sbw import WatermarkDetector

DATA_DIR = Path(__file__).parent.parent / "data"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
TABLE_FILE = TABLES_DIR / "tab_null_calibration.tex"
HASH_KEY = 15485863
TOKENIZER_NAME = "Qwen/Qwen3-8B"

def collect_d0_texts(n=500):
    """Collect non-watermarked texts from d0 experiments."""
    for exp in ["roc500_gselfhash/roc500_gselfhash_d0_g50", "roc500_selfhash/roc500_selfhash_d0_g50"]:
        f = DATA_DIR / exp / "generations.jsonl"
        if f.exists():
            data = [json.loads(l) for l in f.read_text().strip().split('\n')]
            texts = [d["non_watermarked"] for d in data[:n]]
            if texts:
                return texts
    return []

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    texts = collect_d0_texts(500)
    print(f"Collected {len(texts)} texts")
    if not texts:
        print("No texts found")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    vocab = list(range(tokenizer.vocab_size))
    
    results = {}
    for gamma in [0.25, 0.50]:
        detectors = {
            'selfhash': WatermarkDetector(device=device, tokenizer=tokenizer, vocab=vocab,
                                          gamma=gamma, seeding_scheme="selfhash", hash_key=HASH_KEY, z_threshold=4.0),
            'gpu-selfhash-cpuhash': WatermarkDetector(device=device, tokenizer=tokenizer, vocab=vocab,
                                                      gamma=gamma, seeding_scheme="gpu-selfhash-cpuhash", hash_key=HASH_KEY, z_threshold=4.0),
            'gpu-selfhash': WatermarkDetector(device=device, tokenizer=tokenizer, vocab=vocab,
                                              gamma=gamma, seeding_scheme="gpu-selfhash", hash_key=HASH_KEY, z_threshold=4.0),
        }
        
        scores = {k: [] for k in detectors}
        for text in tqdm(texts, desc=f"Scoring (γ={gamma})"):
            for name, det in detectors.items():
                z = det.detect(text=text).get("z_score", 0.0)
                scores[name].append(z)
        
        results[gamma] = {}
        for name in detectors:
            z = np.array(scores[name])
            ks, _ = stats.kstest(z, 'norm')
            results[gamma][name] = (z.mean(), ks)
    
    lines = [
        r"\begin{tabular}{@{}l|cc|cc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{$\gamma=0.25$} & \multicolumn{2}{c}{$\gamma=0.50$} \\",
        r"Detector & Mean & KS & Mean & KS \\",
        r"\midrule",
    ]
    
    for name, label in [('selfhash', 'selfhash (CPU hash)'), 
                        ('gpu-selfhash-cpuhash', 'SBW-ss-cpu'),
                        ('gpu-selfhash', 'SBW-ss (Jenkins)')]:
        r25 = results[0.25][name]
        r50 = results[0.50][name]
        lines.append(f"{label} & $-${abs(r25[0]):.3f} & {r25[1]:.3f} & $-${abs(r50[0]):.3f} & {r50[1]:.3f} \\\\")
    
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    
    content = "\n".join(lines)
    TABLE_FILE.write_text(content + "\n")
    print(content)
    print(f"\nUpdated {TABLE_FILE}")

if __name__ == "__main__":
    main()
