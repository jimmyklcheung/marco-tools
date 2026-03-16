#!/usr/bin/env bash
# =============================================================================
# run_weekly.sh — Weekly Signal Screen runner
# =============================================================================
# Runs screener.py every Monday at 05:00 UTC (before London opens at 07–08 UTC)
# and emails the report to jimmy.klcheung@gmail.com.
#
# ── Quick start ───────────────────────────────────────────────────────────────
#
#   1. Configure email credentials:
#        cp .env.example .env
#        nano .env          ← fill in REPORT_EMAIL_USER and REPORT_EMAIL_PASS
#
#   2. Make this script executable:
#        chmod +x run_weekly.sh
#
#   3. Test it manually first:
#        ./run_weekly.sh
#
#   4. Schedule via cron — add the following line with:
#        crontab -e
#
#      # Every Monday at 05:00 UTC (before London open regardless of DST)
#      0 5 * * 1 /home/user/marco-tools/run_weekly.sh
#
# ── Notes ─────────────────────────────────────────────────────────────────────
#   - London open: 08:00 UK time = 07:00 UTC (BST, Apr–Oct) / 08:00 UTC (GMT, Nov–Mar)
#     Running at 05:00 UTC covers both.
#   - Output is logged to logs/weekly_YYYYMMDD_HHMMSS.log
#   - If email credentials are absent the report is still saved locally.
#
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_DIR}/logs"
PYTHON="${PYTHON:-python3}"

# Load .env if it exists (sets REPORT_EMAIL_USER, REPORT_EMAIL_PASS, etc.)
if [[ -f "${REPO_DIR}/.env" ]]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "${REPO_DIR}/.env"
    set +o allexport
fi

# Create log directory
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/weekly_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ── Starting weekly signal screen ──" \
    | tee -a "${LOG_FILE}"

cd "${REPO_DIR}"

"${PYTHON}" screener.py 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ── Done (exit 0) ──" \
        | tee -a "${LOG_FILE}"
else
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ── FAILED (exit ${EXIT_CODE}) ──" \
        | tee -a "${LOG_FILE}"
    exit ${EXIT_CODE}
fi
