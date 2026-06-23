#!/usr/bin/env bash
# Print crontab entry for daily matchday refresh (6 AM local).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/scripts/daily_matchday_refresh.py"
LOG_DIR="${ROOT}/logs"

mkdir -p "$LOG_DIR"

echo "Add this line to your crontab (crontab -e):"
echo ""
echo "0 6 * * * cd ${ROOT} && ${PYTHON} ${SCRIPT} >> ${LOG_DIR}/daily_refresh.log 2>&1"
echo ""
echo "Manual test:"
echo "  ${PYTHON} ${SCRIPT}"
