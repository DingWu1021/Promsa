#!/bin/bash
# ---------------------------------------------------------------------------
# Search-R1 dense text-retrieval server launcher (port 8001).
#
# Thin wrapper around Search-R1/search_r1/search/retrieval_server.py. It reads
# its configuration from environment variables so it can be started either
# directly or via start_all_services*.sh.
#
#   INDEX_PATH       (required) faiss index file, e.g. text_index_bge_m3.faiss
#   CORPUS_PATH      (required) passages/meta jsonl, e.g. text_passages_bge_m3_meta.jsonl
#   KB_JSON_PATH     (optional) KB dict json to adapt meta jsonl into legacy
#                               'contents' corpus; leave empty to disable
#   RETRIEVER_NAME   (optional) retriever id            [default: bge-m3]
#   RETRIEVER_MODEL  (optional) retriever path / HF id  [default: BAAI/bge-m3]
#   TOPK             (optional) passages per query       [default: 3]
#   FAISS_GPU        (optional) set to 0 to disable GPU faiss [default: 1]
#
# The server always listens on 0.0.0.0:8001 (POST /retrieve), which is fixed
# inside retrieval_server.py.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRIEVAL_SERVER="${SCRIPT_DIR}/Search-R1/search_r1/search/retrieval_server.py"

if [ ! -f "${RETRIEVAL_SERVER}" ]; then
  echo "[search-r1-server] ERROR: cannot find ${RETRIEVAL_SERVER}" >&2
  exit 1
fi

INDEX_PATH="${INDEX_PATH:?Set INDEX_PATH to the faiss index file}"
CORPUS_PATH="${CORPUS_PATH:?Set CORPUS_PATH to the passages/meta jsonl}"
KB_JSON_PATH="${KB_JSON_PATH:-}"
RETRIEVER_NAME="${RETRIEVER_NAME:-bge-m3}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-BAAI/bge-m3}"
TOPK="${TOPK:-3}"
FAISS_GPU="${FAISS_GPU:-1}"

echo "[search-r1-server] index=${INDEX_PATH}"
echo "[search-r1-server] corpus=${CORPUS_PATH}"
echo "[search-r1-server] kb_json=${KB_JSON_PATH:-<none>}"
echo "[search-r1-server] retriever=${RETRIEVER_NAME} (${RETRIEVER_MODEL}), topk=${TOPK}, faiss_gpu=${FAISS_GPU}"
echo "[search-r1-server] serving on 0.0.0.0:8001 (POST /retrieve)"

ARGS=(
  --index_path "${INDEX_PATH}"
  --corpus_path "${CORPUS_PATH}"
  --kb_json_path "${KB_JSON_PATH}"
  --topk "${TOPK}"
  --retriever_name "${RETRIEVER_NAME}"
  --retriever_model "${RETRIEVER_MODEL}"
)
if [ "${FAISS_GPU}" != "0" ]; then
  ARGS+=( --faiss_gpu )
fi

exec python -u "${RETRIEVAL_SERVER}" "${ARGS[@]}"
