#!/usr/bin/env python3
"""SynthID vs SBW comparative watermark overhead benchmark.

Comparisons:
  Option A (full vocab): SynthID compiled (full V) vs gpu-fused-simple_4
  Option B (top-k=40): SynthID compiled (k=40) vs gpu-fused-simple_4 vs gpu-fused-selfhash-40

Usage:
    cd /home/simone/repos/watermark-tests/vllm-watermarking
    source ~/vllm_venv/bin/activate
    python benchmarks/synth-id-comparison/benchmark.py
"""
import sys, json, time, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from synthid_ref import SynthIDLogitsProcessor
from sbw import WatermarkBatch

# --- Config ---
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
def _make_synthid_pipeline():
    keys_tensor = torch.tensor(list(range(DEPTH)), device=DEVICE)
    hash_iv = int.from_bytes(
        hashlib.sha256(keys_tensor.cpu().numpy().tobytes()).digest(),
        byteorder="big"
    ) % torch.iinfo(torch.int64).max

    def pipeline(scores_in, context_in, indices_in):
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

    return torch.compile(pipeline, mode="max-autotune"), keys_tensor, hash_iv


_synthid_compiled, _keys_tensor, _hash_iv = _make_synthid_pipeline()


def bench_synthid_compiled(B, apply_top_k):
    scores = torch.randn(B, V, device=DEVICE)
    context = torch.randint(0, V, (B, 4), device=DEVICE)

    if apply_top_k:
        def step():
            top_scores, top_idx = torch.topk(scores, TOP_K, dim=-1)
            _synthid_compiled(top_scores, context, top_idx)
    else:
        all_idx = torch.arange(V, device=DEVICE).unsqueeze(0).expand(B, -1)
        def step():
            _synthid_compiled(scores, context, all_idx)

    timings = time_fn(step)
    mem = measure_memory(step)
    s = compute_stats(timings, B)
    s.update(mem)
    return s


def bench_sbw(B, scheme):
    torch.cuda.empty_cache()
    wm = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0, seeding_scheme=scheme, device=DEVICE)
    context = torch.randint(0, V, (B, wm.context_width), device=DEVICE)
    logits = torch.randn(B, V, device=DEVICE)

    if scheme == "gpu-fused-simple_4":
        def step():
            wm.apply_watermark_simple_fused(context, logits, 0.5, 2.0)
    elif wm.is_fused and not wm.self_salt and wm.selfsalt_num_candidates:
        def step():
            wm.apply_watermark_topk(context, logits, 0.5, 2.0)
    elif wm.self_salt and wm.is_fused:
        if wm.selfsalt_num_candidates == 0 or (wm.selfsalt_num_candidates and wm.selfsalt_num_candidates >= V):
            # Fullvocab: in-place mutation, no copy needed
            def step():
                wm.apply_watermark_selfsalt_fused(context, logits, 0.5, 2.0)
        else:
            # Top-k: scatter_add_ mutates in-place, no copy needed
            def step():
                wm.apply_watermark_selfsalt_fused(context, logits, 0.5, 2.0)
    elif wm.seeding_scheme.startswith("gpu-"):
        def step():
            wm.apply_watermark_fused(context, logits.clone(), 0.5, 2.0)
    else:
        # CPU schemes: get_greenlist_masks + manual bias
        def step():
            mask = wm.get_greenlist_masks(context, logits)
            logits.add_(mask.to(logits.dtype), alpha=2.0)

    timings = time_fn(step)
    mem = measure_memory(step)
    s = compute_stats(timings, B)
    s.update(mem)
    return s


def run():
    results = {"option_a_full_vocab": {}, f"option_b_topk{TOP_K}": {}}

    print("=" * 75)
    print("SynthID (compiled) vs SBW Comparative Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Vocab: {V}, SynthID depth: {DEPTH}, Top-k: {TOP_K}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print("=" * 75)

    # === Option A: Full Vocabulary ===
    print("\n--- OPTION A: Full Vocabulary ---")
    print("  SynthID compiled (full V) vs gpu-fused-simple_4 vs gpu-fused-selfhash-fullvocab")

    for B in BATCH_SIZES:
        print(f"\n  B={B}:")
        torch.cuda.empty_cache()
        results["option_a_full_vocab"][B] = {}

        # SynthID compiled full vocab
        try:
            s = bench_synthid_compiled(B, apply_top_k=False)
            results["option_a_full_vocab"][B]["synthid_compiled"] = s
            print(f"    SynthID compiled (full V):     {s['latency_mean_ms']:8.3f} ms  (per-tok: {s['per_token_ms']:.4f}, mem: {s['peak_delta_mb']:.1f} MB)")
        except Exception as e:
            print(f"    SynthID compiled (full V):     FAILED ({type(e).__name__})")
            results["option_a_full_vocab"][B]["synthid_compiled"] = {"error": str(e)[:100]}

        torch.cuda.empty_cache()

        # SBW gpu-fused-simple_4
        s = bench_sbw(B, "gpu-fused-simple_4")
        results["option_a_full_vocab"][B]["sbw_fused_simple_4"] = s
        print(f"    SBW gpu-fused-simple_4:       {s['latency_mean_ms']:8.3f} ms  (per-tok: {s['per_token_ms']:.4f}, mem: {s['peak_delta_mb']:.1f} MB)")

        # SBW gpu-fused-selfhash-fullvocab
        s = bench_sbw(B, "gpu-fused-selfhash-fullvocab")
        results["option_a_full_vocab"][B]["sbw_fused_selfhash_fullvocab"] = s
        print(f"    SBW fused-selfhash-fullvocab: {s['latency_mean_ms']:8.3f} ms  (per-tok: {s['per_token_ms']:.4f}, mem: {s['peak_delta_mb']:.1f} MB)")

    # === Option B: Top-k ===
    print(f"\n--- OPTION B: Top-k={TOP_K} ---")
    print(f"  SynthID compiled (k={TOP_K}) vs gpu-fused-simple_4-{TOP_K} vs gpu-fused-selfhash-{TOP_K}")

    for B in BATCH_SIZES:
        print(f"\n  B={B}:")
        torch.cuda.empty_cache()
        results[f"option_b_topk{TOP_K}"][B] = {}

        # SynthID compiled top-k
        s = bench_synthid_compiled(B, apply_top_k=True)
        results[f"option_b_topk{TOP_K}"][B][f"synthid_compiled_topk{TOP_K}"] = s
        print(f"    SynthID compiled (k={TOP_K}):      {s['latency_mean_ms']:8.3f} ms  (per-tok: {s['per_token_ms']:.4f}, mem: {s['peak_delta_mb']:.1f} MB)")

        torch.cuda.empty_cache()

        # SBW gpu-fused-simple_4-{TOP_K}
        s = bench_sbw(B, f"gpu-fused-simple_4-{TOP_K}")
        results[f"option_b_topk{TOP_K}"][B][f"sbw_fused_simple_4_{TOP_K}"] = s
        print(f"    SBW fused-simple_4-{TOP_K}:       {s['latency_mean_ms']:8.3f} ms  (per-tok: {s['per_token_ms']:.4f}, mem: {s['peak_delta_mb']:.1f} MB)")

        # SBW gpu-fused-selfhash-{TOP_K}
        s = bench_sbw(B, f"gpu-fused-selfhash-{TOP_K}")
        results[f"option_b_topk{TOP_K}"][B][f"sbw_fused_selfhash_{TOP_K}"] = s
        print(f"    SBW fused-selfhash-{TOP_K}:       {s['latency_mean_ms']:8.3f} ms  (per-tok: {s['per_token_ms']:.4f}, mem: {s['peak_delta_mb']:.1f} MB)")

    # === CPU Schemes ===
    print("\n--- CPU SCHEMES: simple_1, selfhash-100 ---")
    results["cpu_schemes"] = {}

    for B in BATCH_SIZES:
        torch.cuda.empty_cache()
        results["cpu_schemes"][B] = {}

        s = bench_sbw(B, "ff-additive_prf-4-False")
        results["cpu_schemes"][B]["cpu_simple_4"] = s
        print(f"  B={B:>3}: simple_4={s['latency_mean_ms']:.3f}ms", end="")

        s = bench_sbw(B, "selfhash-100")
        results["cpu_schemes"][B]["cpu_selfhash_100"] = s
        print(f"  selfhash-100={s['latency_mean_ms']:.3f}ms")

    # === Top-k Sweep ===
    print("\n--- TOP-K SWEEP: k=100, 500, 1000 + fullvocab ---")
    results["topk_sweep"] = {}
    for K in [100, 500, 1000]:
        print(f"\n  k={K}:")
        results["topk_sweep"][K] = {}
        for B in BATCH_SIZES:
            torch.cuda.empty_cache()
            scores = torch.randn(B, V, device=DEVICE)
            ctx = torch.randint(0, V, (B, 4), device=DEVICE)

            def _synthid_step(scores=scores, ctx=ctx, K=K):
                top_s, top_i = torch.topk(scores, K, dim=-1)
                _synthid_compiled(top_s, ctx, top_i)
            t_s = np.mean(time_fn(_synthid_step))
            torch.cuda.empty_cache()

            wm_simple = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0, seeding_scheme=f'gpu-fused-simple_4-{K}', device=DEVICE)
            t_simple = np.mean(time_fn(lambda: wm_simple.apply_watermark_topk(ctx, scores, 0.5, 2.0)))
            torch.cuda.empty_cache()

            wm_self = WatermarkBatch(vocab=[0]*V, gamma=0.5, delta=2.0, seeding_scheme=f'gpu-fused-selfhash-{K}', device=DEVICE)
            t_self = np.mean(time_fn(lambda: wm_self.apply_watermark_selfsalt_fused(ctx, scores, 0.5, 2.0)))

            results["topk_sweep"][K][B] = {"synthid": t_s, "simple_4": t_simple, "selfhash": t_self}
            print(f"    B={B:>3}: S={t_s:.3f} simple={t_simple:.3f} self={t_self:.3f}")

    # Fullvocab (already in Option A, just copy references)
    results["topk_sweep"]["fullvocab"] = {}
    for B in BATCH_SIZES:
        r_a = results["option_a_full_vocab"][B]
        results["topk_sweep"]["fullvocab"][B] = {
            "simple_4": r_a["sbw_fused_simple_4"]["latency_mean_ms"],
            "selfhash": r_a["sbw_fused_selfhash_fullvocab"]["latency_mean_ms"],
        }

    # --- Save ---
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "vocab_size": V,
        "synthid_depth": DEPTH,
        "synthid_top_k": TOP_K,
        "sbw_gamma": 0.5,
        "sbw_delta": 2.0,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup_iterations": WARMUP,
        "measurement_iterations": ITERS,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output = {"metadata": metadata, "results": results}
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    # --- Summary ---
    print("\n" + "=" * 75)
    print("SUMMARY — Option A: Full Vocabulary")
    print(f"{'B':>4} | {'SynthID':>10} | {'simple_4':>10} | {'selfhash-fv':>11} | {'S/simple':>8} | {'S/self':>7}")
    print("-" * 65)
    for B in BATCH_SIZES:
        r = results["option_a_full_vocab"][B]
        sc = r["synthid_compiled"].get("latency_mean_ms", float('nan'))
        f4 = r["sbw_fused_simple_4"]["latency_mean_ms"]
        fs = r["sbw_fused_selfhash_fullvocab"]["latency_mean_ms"]
        print(f"{B:>4} | {sc:>7.3f} ms | {f4:>7.3f} ms | {fs:>8.3f} ms | {sc/f4:>7.1f}x | {sc/fs:>6.1f}x")

    print(f"\nSUMMARY — Option B: Production (all top-{TOP_K})")
    print(f"{'B':>4} | {f'SynthID k={TOP_K}':>13} | {f'simple_4-{TOP_K}':>12} | {f'selfhash-{TOP_K}':>12} | {'S/simple':>8} | {'S/self':>7}")
    print("-" * 75)
    for B in BATCH_SIZES:
        r = results[f"option_b_topk{TOP_K}"][B]
        sc = r[f"synthid_compiled_topk{TOP_K}"]["latency_mean_ms"]
        f4 = r[f"sbw_fused_simple_4_{TOP_K}"]["latency_mean_ms"]
        fs = r[f"sbw_fused_selfhash_{TOP_K}"]["latency_mean_ms"]
        print(f"{B:>4} | {sc:>10.3f} ms | {f4:>9.3f} ms | {fs:>9.3f} ms | {sc/f4:>7.2f}x | {sc/fs:>6.2f}x")


if __name__ == "__main__":
    run()
