export SGLANG_VLM_CACHE_SIZE_MB=8192

echo "[retriever] hostname -I: $(hostname -I 2>/dev/null || echo '<hostname -I not available>')"

python -m sglang.launch_server \
    --model-path /path/to/workspace/0_download_model_bash/Qwen3-VL-32B-Instruct \
    --host 0.0.0.0 --port 8181 \
    --dtype bfloat16 \
    --served-model-name Qwen3-VL-32B-Instruct \
    --tp 4 --dp 2 \
    --mem-fraction-static 0.6 \
    --context-length 40960 \
    --max-running-requests 1024 \
    --chunked-prefill-size 2048 \
    --enable-torch-compile \
    --torch-compile-max-bs 64