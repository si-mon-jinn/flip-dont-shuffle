#!/bin/bash
# Benchmark vLLM serving overhead with and without SBW watermark
# 
# Three conditions:
#   1. No logits processor (pure vLLM baseline)
#   2. Logits processor loaded, delta=0 (framework overhead only)
#   3. Logits processor loaded, delta=2 (full watermark)
#
# Prerequisites:
#   - vllm_venv activated for server
#   - Qwen/Qwen3-8B available
#   - Port 8008 free
#
# Usage:
#   bash review3_material/benchmark_vllm_overhead.sh

set -e

VLLM_VENV="/home/simone/vllm_venv"
REPO="/home/simone/repos/watermark-tests"
MODEL="Qwen/Qwen3-8B"
PORT=8008
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"

NUM_PROMPTS=128
INPUT_LEN=30
OUTPUT_LEN=200
CONCURRENCIES="1 8 32 64 128"

RESULTS_DIR="${REPO}/review3_material/vllm_benchmark_results"
mkdir -p "$RESULTS_DIR"

WM_EXTRA_BODY='{"vllm_xargs": {"gamma": 0.5, "delta": 2.0, "seeding_scheme": "gpu-fused-selfhash-fullvocab", "hash_key": 15485863}}'
NOWM_EXTRA_BODY='{"vllm_xargs": {"gamma": 0.5, "delta": 0.0, "seeding_scheme": "gpu-fused-selfhash-fullvocab", "hash_key": 15485863}}'

wait_for_server() {
    echo "  Waiting for server..."
    for i in $(seq 1 60); do
        if curl -s --max-time 2 "${BASE_URL}/v1/models" | grep -q "Qwen"; then
            echo "  Server ready."
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: Server did not start within 120s"
    exit 1
}

kill_server() {
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 3
}

run_bench() {
    local condition="$1"
    local extra_body_arg="$2"
    
    echo "  Running benchmarks for: ${condition}"
    for conc in $CONCURRENCIES; do
        echo "    concurrency=${conc}..."
        local outfile="${RESULTS_DIR}/${condition}_conc${conc}.txt"
        
        local cmd="source ${VLLM_VENV}/bin/activate && vllm bench serve \
            --base-url ${BASE_URL} \
            --dataset-name random \
            --num-prompts ${NUM_PROMPTS} \
            --random-input-len ${INPUT_LEN} \
            --random-output-len ${OUTPUT_LEN} \
            --max-concurrency ${conc}"
        
        if [ -n "$extra_body_arg" ]; then
            cmd="${cmd} --extra-body '${extra_body_arg}'"
        fi
        
        eval "$cmd" > "$outfile" 2>&1
    done
}

echo "============================================"
echo "vLLM Serving Overhead Benchmark"
echo "Model: ${MODEL}"
echo "Prompts: ${NUM_PROMPTS}, Input: ${INPUT_LEN}, Output: ${OUTPUT_LEN}"
echo "Concurrencies: ${CONCURRENCIES}"
echo "============================================"

# --- Condition 1: No logits processor ---
echo ""
echo "[1/3] Starting server WITHOUT logits processor..."
kill_server
nohup bash -c "source ${VLLM_VENV}/bin/activate && vllm serve ${MODEL} \
    --host ${HOST} --port ${PORT} \
    --max-model-len 512 --enforce-eager" > /tmp/vllm_bench_server.log 2>&1 &
wait_for_server
run_bench "no_processor" ""
kill_server

# --- Condition 2: Logits processor, delta=0 ---
echo ""
echo "[2/3] Starting server WITH logits processor, delta=0 (framework overhead)..."
nohup bash -c "source ${VLLM_VENV}/bin/activate && \
    export PYTHONPATH='${REPO}/vllm-watermarking:\$PYTHONPATH' && \
    vllm serve ${MODEL} \
    --host ${HOST} --port ${PORT} \
    --max-model-len 512 --enforce-eager \
    --logits-processors vllm_fkgw_watermark:fKGWLogitsProcessor" > /tmp/vllm_bench_server.log 2>&1 &
wait_for_server
run_bench "delta0" "$NOWM_EXTRA_BODY"
kill_server

# --- Condition 3: Logits processor, delta=2 ---
echo ""
echo "[3/3] Starting server WITH logits processor, delta=2 (full watermark)..."
nohup bash -c "source ${VLLM_VENV}/bin/activate && \
    export PYTHONPATH='${REPO}/vllm-watermarking:\$PYTHONPATH' && \
    vllm serve ${MODEL} \
    --host ${HOST} --port ${PORT} \
    --max-model-len 512 --enforce-eager \
    --logits-processors vllm_fkgw_watermark:fKGWLogitsProcessor" > /tmp/vllm_bench_server.log 2>&1 &
wait_for_server
run_bench "delta2" "$WM_EXTRA_BODY"
kill_server

# --- Parse results ---
echo ""
echo "============================================"
echo "RESULTS SUMMARY"
echo "============================================"
echo ""
printf "%-15s" "Concurrency"
for conc in $CONCURRENCIES; do
    printf "%10s" "$conc"
done
echo ""
echo "-----------------------------------------------------------"

for condition in no_processor delta0 delta2; do
    printf "%-15s" "$condition"
    for conc in $CONCURRENCIES; do
        file="${RESULTS_DIR}/${condition}_conc${conc}.txt"
        if [ -f "$file" ]; then
            tput=$(grep "Output token throughput" "$file" | awk '{print $NF}')
            printf "%10s" "${tput}"
        else
            printf "%10s" "N/A"
        fi
    done
    echo ""
done

echo ""
echo "Full results in: ${RESULTS_DIR}/"
echo "Done."
