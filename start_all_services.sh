#!/bin/bash

set -euo pipefail

echo "[retriever] hostname -I: $(hostname -I 2>/dev/null || echo '<hostname -I not available>')"

ROOT_DIR="/path/to/workspace/ECCV/KBSearch-v13"
LOG_DIR="${ROOT_DIR}/service_logs"
mkdir -p "${LOG_DIR}"

echo "[all-services] hostname -I: $(hostname -I 2>/dev/null || echo '<hostname -I not available>')"
echo "[all-services] ROOT_DIR=${ROOT_DIR}"
echo "[all-services] LOG_DIR=${LOG_DIR}"

# -------------------- Configurable env vars --------------------
# web_search_server requires these vars at startup.
export WEBSEARCH_GOOGLE_SERPER_KEY="${WEBSEARCH_GOOGLE_SERPER_KEY:-your_serper_api_key_here}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-your_azure_openai_api_key_here}"
export WEB_SERVER_CONFIG_FILE="${WEB_SERVER_CONFIG_FILE:-${ROOT_DIR}/web_search_server/config.json}"
export WEB_SERVER_CACHE_DIR="${WEB_SERVER_CACHE_DIR:-${ROOT_DIR}/web_search_server/search_cache}"
export SUMMARIZER_BASE_URL="${SUMMARIZER_BASE_URL:-http://127.0.0.1:8123/v1}"
export SUMMARIZER_API_KEY="${SUMMARIZER_API_KEY:-EMPTY}"

# Search-R1 retrieval config
INDEX_PATH="${INDEX_PATH:-/path/to/workspace/search-agent/R1-Router/create_data/text_kb/text_index_bge_m3.faiss}"
CORPUS_PATH="${CORPUS_PATH:-/path/to/workspace/search-agent/R1-Router/create_data/text_kb/text_passages_bge_m3_meta.jsonl}"
KB_JSON_PATH="${KB_JSON_PATH:-/path/to/workspace/search-agent/R1-Router/create_data/kb/wiki_100_dict_v4.json}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-/path/to/workspace/1_download_modelscope/BAAI/bge-m3}"
TOPK="${TOPK:-3}"

# Summary model config
SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-/path/to/workspace/0_download_model_bash/Qwen3-VL-32B-Instruct}"
SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"

cleanup() {
  echo "[all-services] shutting down child processes..."
  for pid in ${PIDS:-}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
}
trap cleanup EXIT INT TERM

PIDS=""

echo "[all-services] starting summary server on 8123..."
python -m sglang.launch_server \
  --model-path "${SUMMARY_MODEL_PATH}" \
  --served-model-name "${SUMMARY_MODEL_NAME}" \
  --host 0.0.0.0 --port 8123 \
  --dtype bfloat16 \
  --tp-size 4 --dp-size 2 \
  --mem-fraction-static 0.9 \
  --max-total-tokens 262144 \
  --max-prefill-tokens 65536 \
  --chunked-prefill-size 16384 \
  --max-running-requests 1024 \
  > "${LOG_DIR}/summary_8123.log" 2>&1 &
PIDS="${PIDS} $!"

echo "[all-services] starting local retrieval server on 8001 via search-r1-server.sh..."
(
  cd "${ROOT_DIR}"
  INDEX_PATH="${INDEX_PATH}" \
  CORPUS_PATH="${CORPUS_PATH}" \
  KB_JSON_PATH="${KB_JSON_PATH}" \
  TOPK="${TOPK}" \
  RETRIEVER_NAME="bge-m3" \
  RETRIEVER_MODEL="${RETRIEVER_MODEL}" \
  bash "${ROOT_DIR}/search-r1-server.sh"
) > "${LOG_DIR}/local_search_8001.log" 2>&1 &
PIDS="${PIDS} $!"

echo "[all-services] starting web_search_server on 8000..."
(
  cd "${ROOT_DIR}/web_search_server"
  uvicorn server:app --host 0.0.0.0 --port 8000
) > "${LOG_DIR}/web_search_8000.log" 2>&1 &
PIDS="${PIDS} $!"

echo "[all-services] started. child pids:${PIDS}"
echo "[all-services] logs:"
echo "  - ${LOG_DIR}/summary_8123.log"
echo "  - ${LOG_DIR}/local_search_8001.log"
echo "  - ${LOG_DIR}/web_search_8000.log"

# Keep the task alive and fail fast if any service exits.
wait -n
echo "[all-services] a child process exited unexpectedly."
exit 1
