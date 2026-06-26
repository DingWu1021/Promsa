#!/bin/bash
# ---------------------------------------------------------------------------
# Search-R1 dense text-retrieval server launcher for the E-VQA knowledge base.
#
# The launch logic is identical to search-r1-server.sh — only the knowledge-base
# paths differ, and those are supplied via environment variables by the caller
# (start_all_services-evqa.sh). This wrapper just delegates to keep the two in
# sync.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/search-r1-server.sh" "$@"
