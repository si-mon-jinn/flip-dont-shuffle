#!/bin/bash
# Reproduction script for "Flip, Don't Shuffle" EMNLP 2026 paper
#
# This script sets up the environment and regenerates figures and tables
# from the included experimental data.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Flip, Don't Shuffle: Reproduction Script ==="
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU required for full reproduction."
fi

# Step 1: Clone dependencies
echo "Step 1: Setting up dependencies..."
if [ ! -d "waterpipe" ]; then
    git clone --branch v1.0.0 https://github.com/si-mon-jinn/waterpipe waterpipe
fi
if [ ! -d "sbw" ]; then
    git clone --branch v1.0.0 https://github.com/si-mon-jinn/sbw sbw
fi

# Step 2: Create virtual environment
echo "Step 2: Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Step 3: Install packages
echo "Step 3: Installing packages..."
pip install --upgrade pip
pip install -e sbw/sbw/
pip install -e waterpipe/
pip install -r requirements.txt

# Step 4: Generate figures and tables (from existing data)
echo "Step 4: Generating figures and tables..."
cd scripts

echo "  Generating figures..."
python plot_simple1_combined.py
python plot_selfhash_combined.py
python plot_perplexity_combined.py
python plot_synthid_comparison.py
python plot_tail_latency.py
python plot_e2e_comparison_128tok.py

echo "  Generating tables..."
python compute_roc_auc_table.py
python compute_roc_auc_table_selfhash.py
python compute_perplexity_table.py
python compute_perplexity_table_selfhash.py
python compute_diversity_table.py
python compute_synthid_table.py
python compute_memory_table.py
python compute_tail_latency_table.py
python compute_hash_attribution_table.py
python compute_ks_tests.py
python compute_ks_twosample_nowm.py
python compute_ks_twosample_wm_table.py
python compute_null_calibration_table.py
python compute_ppl_hash_isolation_table.py
python compute_zscore_wm_table.py

cd ..

echo ""
echo "=== Reproduction complete ==="
echo ""
echo "Generated outputs:"
echo "  - paper/figures/ (PDF figures)"
echo "  - paper/tables/ (LaTeX tables)"
echo ""
echo "To build the paper:"
echo "  cd paper && pdflatex paper_long.tex && bibtex paper_long && pdflatex paper_long.tex && pdflatex paper_long.tex"
