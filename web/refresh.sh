#!/usr/bin/env bash
# Rebuild the dashboard JSON from the current research artifacts.
#
# This script does NOT call the LLM and does NOT refresh market data:
#   * --skip-prep reuses the existing target selection CSV;
#   * --no-update-k uses the existing K-line caches.
# Remove those flags if you want the full pipeline behaviour.
set -euo pipefail
cd "$(dirname "$0")/.."

python target_code_pipeline.py \
  --skip-prep \
  --no-update-k \
  --export-details web/data/backtest_details.json

python web/export_data.py

echo
echo "Dashboard data refreshed. Start the app with: web/run.sh"
