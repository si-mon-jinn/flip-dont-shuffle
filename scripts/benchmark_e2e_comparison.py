#!/usr/bin/env python3
"""E2E comparison: SynthID vs SBW watermarking overhead.

Setup:
    python3 -m venv combined_venv
    ./combined_venv/bin/pip install -r benchmarks/end-to-end/requirements-e2e.txt

Run:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./sbw ./combined_venv/bin/python \
        benchmarks/end-to-end/benchmark_e2e_comparison.py --batch-sizes "16,128"
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from synthid_text import synthid_mixin
from sbw.batch import WatermarkBatch


class SBWLogitsProcessor:
    """SBW logits processor for HuggingFace generate()."""

    def __init__(self, vocab_size, device):
        self.wm = WatermarkBatch(
            vocab=[0] * vocab_size, gamma=0.5, delta=2.0,
            seeding_scheme="gpu-fused-selfhash-fullvocab", device=device,
        )
        self.context_width = self.wm.context_width
        self.device = device
        self._gamma_cache, self._delta_cache = {}, {}

    def __call__(self, input_ids, scores):
        B = input_ids.shape[0]
        if input_ids.shape[1] < self.context_width:
            return scores
        context = input_ids[:, -self.context_width :].contiguous()
        if B not in self._gamma_cache:
            self._gamma_cache[B] = torch.full((B,), 0.5, device=self.device)
            self._delta_cache[B] = torch.full((B,), 2.0, device=self.device)
        V = min(scores.shape[1], len(self.wm.vocab))
        self.wm.apply_watermark_selfsalt_fused(
            context, scores[:, :V], self._gamma_cache[B], self._delta_cache[B]
        )
        return scores


def make_synthid_model(base_model):
    """Patch model with SynthID mixin."""
    cls = type(
        "SynthID" + base_model.__class__.__name__,
        (synthid_mixin.SynthIDSparseTopKMixin, base_model.__class__),
        {},
    )
    base_model.__class__ = cls
    return base_model


def benchmark(name, run_fn, warmup, iters):
    """Run benchmark with warmup."""
    for _ in range(warmup):
        with torch.no_grad():
            run_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            run_fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    med = np.median(times)
    std = np.std(times)
    print(f"  {name:12s} {med:8.0f} ms  (std={std:.1f})")
    return {"median_ms": float(med), "std_ms": float(std), "times": times}


def run_benchmark(args):
    device = torch.device("cuda")
    print(f"torch: {torch.__version__}, triton: {__import__('triton').__version__}")
    print(f"Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    vocab_size = model.config.vocab_size

    orig_cls = model.__class__
    make_synthid_model(model)
    synthid_cls = model.__class__

    gen_config = GenerationConfig(
        do_sample=True, temperature=1.0, top_k=args.top_k,
        max_new_tokens=args.max_tokens, pad_token_id=tokenizer.pad_token_id,
    )

    results = {}
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"\nBenchmark: {args.max_tokens} tokens, top_k={args.top_k}, "
          f"{args.iterations} iters, {args.warmup} warmup")
    print("=" * 70)

    for bs in batch_sizes:
        print(f"\nB={bs}:")
        torch.cuda.empty_cache()

        inputs = tokenizer(
            ["Hello, how are you?"] * bs, return_tensors="pt", padding=True
        ).to(device)

        # Baseline
        model.__class__ = orig_cls
        base = benchmark("Baseline", lambda: model.generate(**inputs, generation_config=gen_config),
                         args.warmup, args.iterations)

        # SynthID
        model.__class__ = synthid_cls
        synth = benchmark("SynthID", lambda: model.generate(**inputs, generation_config=gen_config),
                          args.warmup, args.iterations)

        # SBW
        model.__class__ = orig_cls
        sbw_proc = SBWLogitsProcessor(vocab_size, device)
        sbw = benchmark("SBW", lambda: model.generate(**inputs, generation_config=gen_config,
                                                       logits_processor=[sbw_proc]),
                        args.warmup, args.iterations)

        base_med = base["median_ms"]
        synth_oh = synth["median_ms"] - base_med
        sbw_oh = sbw["median_ms"] - base_med

        print(f"  {'Overhead:':<12} SynthID: {synth_oh:+.0f}ms ({synth_oh/base_med*100:+.2f}%)  "
              f"SBW: {sbw_oh:+.0f}ms ({sbw_oh/base_med*100:+.2f}%)")

        results[bs] = {
            "baseline": base, "synthid": synth, "sbw": sbw,
            "synthid_overhead_ms": synth_oh, "sbw_overhead_ms": sbw_oh,
        }

    # Summary
    print("\n" + "=" * 70)
    print(f"{'B':>4} | {'Baseline':>10} | {'SynthID':>14} | {'SBW':>14} | {'Speedup':>8}")
    print("-" * 70)
    for bs, r in results.items():
        base = r["baseline"]["median_ms"]
        synth_oh = r["synthid_overhead_ms"]
        sbw_oh = r["sbw_overhead_ms"]
        speedup = synth_oh / sbw_oh if sbw_oh > 0 else float("inf")
        print(f"{bs:>4} | {base:>8.0f}ms | {synth_oh:>+12.0f}ms | {sbw_oh:>+12.0f}ms | {speedup:>7.1f}x")

    # Save
    output = {
        "metadata": {
            "model": args.model, "max_tokens": args.max_tokens, "top_k": args.top_k,
            "iterations": args.iterations, "warmup": args.warmup,
            "torch": torch.__version__, "triton": __import__("triton").__version__,
        },
        "results": {str(k): v for k, v in results.items()},
    }
    out_path = Path(__file__).parent / f"results_comparison_{args.max_tokens}tok.json"
    out_path.write_text(json.dumps(output, indent=2, default=lambda x: None))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2-7B")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--batch-sizes", default="16,128")
    run_benchmark(p.parse_args())
