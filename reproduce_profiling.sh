#!/bin/bash
# Reproduce all profiling data for the "Flip, Don't Shuffle" paper.
#
# This script generates:
#   - profile_simple1.json      (CPU vs GPU simple_1 scheme comparison)
#   - profile_selfhash_nV.json  (GPU selfhash fullvocab profiling)
#   - synthid_comparison.json   (SynthID vs SBW latency comparison)
#   - results_e2e_comparison.json (end-to-end model inference comparison)
#
# Requirements:
#   - Python 3.10+
#   - NVIDIA GPU with CUDA 12.x+ driver
#   - ~16GB GPU memory for e2e benchmark (Qwen2-7B model)
#   - Run ./reproduce.sh first to clone sbw dependency
#
# Usage:
#   cd flip-dont-shuffle
#   ./reproduce_profiling.sh [--skip-e2e]
#
# Virtual environments:
#   This script uses two virtual environments:
#   - profile_venv: torch 2.10.0 + vllm 0.19.1 (for profiling benchmarks 1-4)
#   - e2e_venv: torch 2.10.0 + synthid-text (for e2e benchmark 5)
#
#   The environments are created automatically on first run.
#   Frozen requirements are in requirements_profile_venv.txt and requirements_e2e_venv.txt.
#
# Output files are saved to data/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
SBW_DIR="$SCRIPT_DIR/sbw"
SBW_BENCHMARKS="$SBW_DIR/benchmarks"
PROFILE_VENV_DIR="$SCRIPT_DIR/profile_venv"
E2E_VENV_DIR="$SCRIPT_DIR/e2e_venv"

SKIP_E2E=false
if [[ "${1:-}" == "--skip-e2e" ]]; then
    SKIP_E2E=true
fi

echo "=============================================="
echo "Profiling Data Reproduction Script"
echo "=============================================="
echo "Output directory: $DATA_DIR"
echo "Date: $(date)"
echo ""

# ============================================================================
# Check prerequisites
# ============================================================================

# Check sbw is available (cloned by reproduce.sh)
if [ ! -d "$SBW_BENCHMARKS" ]; then
    echo "ERROR: sbw/benchmarks not found."
    echo "       Run ./reproduce.sh first to clone dependencies."
    exit 1
fi

# ============================================================================
# Set up profile_venv if needed
# ============================================================================
if [ ! -d "$PROFILE_VENV_DIR" ]; then
    echo "Setting up profile_venv (torch 2.10.0 + vllm 0.19.1)..."
    echo "This may take several minutes."
    echo ""
    
    python3 -m venv "$PROFILE_VENV_DIR"
    source "$PROFILE_VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install torch==2.10.0 numpy -q
    pip install vllm==0.19.1 -q
    pip install -e "$SBW_DIR/sbw/" -q
    pip install -e "$SBW_DIR/vllm_sbw/" -q
    
    echo "✓ profile_venv created"
    echo ""
else
    source "$PROFILE_VENV_DIR/bin/activate"
fi

# Check CUDA availability
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
    echo "ERROR: CUDA is required for profiling benchmarks"
    exit 1
}

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "GPU: $GPU_NAME"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "vLLM: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo ""

# ============================================================================
# 1. profile_simple1.json
# ============================================================================
echo "=============================================="
echo "[1/5] Generating profile_simple1.json"
echo "      Schemes: simple_1, gpu-simple_1"
echo "      Batch sizes: 1,4,8,16,32,64,128,256,512"
echo "      Vocab size: 151669 (Qwen3)"
echo "=============================================="

python3 "$SBW_BENCHMARKS/profile_logits_processor.py" \
    --schemes "simple_1,gpu-simple_1" \
    --batch-sizes "1,4,8,16,32,64,128,256,512" \
    --vocab-size 151669 \
    --iterations 100 \
    --warmup 10 \
    --output "$DATA_DIR/profile_simple1.json"

echo "✓ Generated profile_simple1.json"
echo ""

# ============================================================================
# 2. profile_selfhash_nV.json
# ============================================================================
echo "=============================================="
echo "[2/5] Generating profile_selfhash_nV.json"
echo "      Schemes: gpu-selfhash-fullvocab"
echo "      Batch sizes: 1,4,8,16,32,64,128,256,512"
echo "      Vocab size: 151669 (Qwen3)"
echo "=============================================="

python3 "$SBW_BENCHMARKS/profile_logits_processor.py" \
    --schemes "gpu-selfhash-fullvocab" \
    --batch-sizes "1,4,8,16,32,64,128,256,512" \
    --vocab-size 151669 \
    --iterations 100 \
    --warmup 10 \
    --output "$DATA_DIR/profile_selfhash_nV.json"

echo "✓ Generated profile_selfhash_nV.json"
echo ""

# ============================================================================
# 3. synthid_comparison.json (Part 1: SynthID vs SBW full vocab + top-k)
# ============================================================================
echo "=============================================="
echo "[3/5] Generating synthid_comparison.json (part 1)"
echo "      Running benchmark_synthid_vs_sbw.py"
echo "      - Option A: full vocab comparison"
echo "      - Option B: top-k=40 comparison (SBW variants)"
echo "      - CPU schemes"
echo "=============================================="

cd "$SCRIPTS_DIR"
python3 benchmark_synthid_vs_sbw.py
# Script outputs to scripts/results.json, move to data/
mv results.json "$DATA_DIR/results_synthid_fullvocab.json"

echo "✓ Generated results_synthid_fullvocab.json"
echo ""

# ============================================================================
# 4. synthid_comparison.json (Part 2: SynthID vs KGW CPU top-40)
# ============================================================================
echo "=============================================="
echo "[4/5] Generating synthid_comparison.json (part 2)"
echo "      Running benchmark_synthid_top40.py"
echo "      - SynthID compiled top-k=40"
echo "      - KGW selfhash CPU top-40"
echo "=============================================="

python3 benchmark_synthid_top40.py
# Script outputs to scripts/results_top40.json, move to data/
mv results_top40.json "$DATA_DIR/results_top40.json"

echo "✓ Generated results_top40.json"
echo ""

# ============================================================================
# 4b. Merge into synthid_comparison.json
# ============================================================================
echo "=============================================="
echo "      Merging into synthid_comparison.json"
echo "=============================================="

python3 merge_synthid_comparison.py

echo "✓ Generated synthid_comparison.json"
echo ""

cd "$SCRIPT_DIR"

# ============================================================================
# 5. results_e2e_comparison.json (end-to-end model inference)
# ============================================================================
if [[ "$SKIP_E2E" == "true" ]]; then
    echo "=============================================="
    echo "[5/5] SKIPPED: results_e2e_comparison.json"
    echo "      (run without --skip-e2e to include)"
    echo "=============================================="
else
    echo "=============================================="
    echo "[5/5] Generating results_e2e_comparison.json"
    echo "      Model: Qwen/Qwen2-7B"
    echo "      Max tokens: 128, top-k: 40"
    echo "      Iterations: 50, warmup: 10"
    echo "      WARNING: Requires ~16GB GPU memory"
    echo "=============================================="

    # Set up e2e_venv if it doesn't exist
    if [ ! -d "$E2E_VENV_DIR" ]; then
        echo ""
        echo "Setting up e2e_venv (torch 2.10.0 + synthid-text)..."
        echo "This may take several minutes."
        echo ""
        
        python3 -m venv "$E2E_VENV_DIR"
        "$E2E_VENV_DIR/bin/pip" install --upgrade pip -q
        
        # Install torch first (before synthid-text tries to pin an older version)
        "$E2E_VENV_DIR/bin/pip" install torch==2.10.0 triton==3.6.0 -q
        
        # Install transformers and accelerate
        "$E2E_VENV_DIR/bin/pip" install transformers accelerate -q
        
        # Install synthid-text (will try to downgrade torch)
        "$E2E_VENV_DIR/bin/pip" install synthid-text -q
        
        # Force reinstall torch to override synthid's pin
        "$E2E_VENV_DIR/bin/pip" install --force-reinstall torch==2.10.0 triton==3.6.0 -q
        
        # Install sbw
        "$E2E_VENV_DIR/bin/pip" install -e "$SBW_DIR/sbw/" -q
        
        echo "✓ e2e_venv created"
        echo ""
    fi

    # Run e2e benchmark with the e2e venv
    cd "$SCRIPTS_DIR"
    "$E2E_VENV_DIR/bin/python" benchmark_e2e_comparison.py \
        --model "Qwen/Qwen2-7B" \
        --max-tokens 128 \
        --top-k 40 \
        --iterations 50 \
        --warmup 10 \
        --batch-sizes "1,8,16,32,64,128,256"

    # Script outputs to scripts/results_comparison_128tok.json
    mv results_comparison_128tok.json "$DATA_DIR/results_e2e_comparison.json"
    cd "$SCRIPT_DIR"

    echo "✓ Generated results_e2e_comparison.json"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=============================================="
echo "Profiling reproduction complete!"
echo "=============================================="
echo ""
echo "Generated files:"
ls -lh "$DATA_DIR"/*.json 2>/dev/null | grep -E "profile_|synthid_comparison|results_e2e" || echo "  (none found)"
echo ""
echo "Intermediate files (can be deleted):"
ls -lh "$DATA_DIR"/results_synthid_fullvocab.json "$DATA_DIR"/results_top40.json 2>/dev/null || echo "  (none)"
echo ""
echo "To regenerate figures and tables from new data, run:"
echo "  ./reproduce.sh"
