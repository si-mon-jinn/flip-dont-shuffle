#!/usr/bin/env python3
"""SynthID (top-40) vs KGW selfhash (top-40) benchmark.

Usage:
    cd /home/simone/repos/watermark-tests/vllm-watermarking
    source ~/vllm_venv/bin/activate
    python benchmarks/synth-id-comparison/benchmark_top40.py
"""
import sys, json, time, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from sbw import WatermarkBatch

DEVICE = torch.device("cuda")
V = 151936
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
WARMUP = 20
ITERS = 200
DEPTH = 30
TOP_K = 40
MULT, INC, SHIFT = 6364136223846793005, 1, 64 // 12


def time_fn(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))
    return timings


def measure_memory(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return {"peak_delta_mb": (torch.cuda.max_memory_allocated() - mem_before) / 1e6}


def compute_stats(timings, B):
    arr = np.array(timings)
    return {
        "latency_mean_ms": round(float(np.mean(arr)), 4),
        "latency_std_ms": round(float(np.std(arr)), 4),
        "latency_p50_ms": round(float(np.percentile(arr, 50)), 4),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 4),
        "latency_p99_ms": round(float(np.percentile(arr, 99)), 4),
        "per_token_ms": round(float(np.mean(arr)) / B, 4),
    }


# --- SynthID compiled pipeline ---
keys_tensor = torch.tensor(list(range(DEPTH)), device=DEVICE)
hash_iv = int.from_bytes(
    hashlib.sha256(keys_tensor.cpu().numpy().tobytes()).digest(),
    byteorder="big") % torch.iinfo(torch.int64).max


def _synthid_pipeline(scores_in, context_in, indices_in):
    h = torch.full((context_in.shape[0],), hash_iv, dtype=torch.long, device=DEVICE)
    for i in range(context_in.shape[1]):
        h = (h + context_in[:, i]) * MULT + INC
    h = h.unsqueeze(1).expand(-1, indices_in.shape[1])
    h = (h + indices_in) * MULT + INC
    h = h.unsqueeze(2).expand(-1, -1, keys_tensor.shape[0])
    ngram_k = (h + keys_tensor.view(1, 1, -1)) * MULT + INC
    for _ in range(12):
        ngram_k = ((ngram_k + 1) * MULT + INC) >> SHIFT
    g_vals = ((ngram_k >> 30) % 2).float()
    probs = torch.softmax(scores_in, dim=1)
    for i in range(g_vals.shape[2]):
        g = g_vals[:, :, i]
        g_mass = (g * probs).sum(dim=1, keepdim=True)
        probs = probs * (1 + g - g_mass)
    return torch.log(probs.clamp(min=1e-30))


synthid_compiled = torch.compile(_synthid_pipeline, mode="max-autotune")


def run():
    results = {}

    print("=" * 65)
    print(f"SynthID (top-{TOP_K}) vs KGW selfhash (top-{TOP_K})")
    print(f"GPU: {torch.cuda.get_device_name()}, V={V}, depth={DEPTH}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print("=" * 65)

    for B in BATCH_SIZES:
        scores = torch.randn(B, V, device=DEVICE)
        context = torch.randint(0, V, (B, 4), device=DEVICE)
        torch.cuda.empty_cache()

        # SynthID compiled top-40
        def step_synthid(scores=scores, context=context):
            top_s, top_i = torch.topk(scores, TOP_K, dim=-1)
            synthid_compiled(top_s, context, top_i)

        s_syn = compute_stats(time_fn(step_synthid), B)
        s_syn.update(measure_memory(step_synthid))
        torch.cuda.empty_cache()

        # KGW selfhash (CPU, top-40)
        wm = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0,
                            seeding_scheme="selfhash", device=DEVICE)

        def step_kgw(context=context, scores=scores):
            mask = wm.get_greenlist_masks(context, logits=scores)
            scores.add_(mask.to(scores.dtype), alpha=2.0)

        s_kgw = compute_stats(time_fn(step_kgw), B)
        s_kgw.update(measure_memory(step_kgw))

        results[B] = {"synthid_compiled_topk40": s_syn, "kgw_selfhash_40": s_kgw}
        ratio = s_syn["latency_mean_ms"] / s_kgw["latency_mean_ms"]
        print(f"  B={B:>3}: SynthID(k=40)={s_syn['latency_mean_ms']:.3f}ms  "
              f"KGW selfhash={s_kgw['latency_mean_ms']:.3f}ms  ratio={ratio:.2f}x")

    # Save
    output = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "vocab_size": V,
            "synthid_depth": DEPTH,
            "top_k": TOP_K,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "warmup_iterations": WARMUP,
            "measurement_iterations": ITERS,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }
    out_path = Path(__file__).parent / "results_top40.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run()
