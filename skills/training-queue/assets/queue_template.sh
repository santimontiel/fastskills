#!/bin/bash
# Queue template for the training-queue skill.
#
# Fill in CONTAINER/REPO/SUMMARY below, add one run_stage call per queued job, then:
#   chmod +x this_script.sh
#   ( crontab -l 2>/dev/null; echo "<min> <hour> <day> <month> * $(pwd)/this_script.sh" ) | crontab -
#
# Fires once at the scheduled time, runs every stage sequentially (a crash in one does not block
# the rest), then removes its own crontab line so it can't fire again.

set -uo pipefail

CONTAINER="tinycar_container"                          # docker container name
REPO="/home/santiago.montiel/Workspace/tinycar-dev"     # repo root, mounted at /workspace in-container
SUMMARY="$REPO/outputs/queue_summary.log"               # shared start/end/exit-code log
MARKER="$(basename "$0")"                                # must match this script's own filename

run_stage() {
    local name="$1"
    local logfile="$2"
    shift 2
    echo "[$(date '+%F %T')] START  $name" >> "$SUMMARY"
    docker exec -w /workspace "$CONTAINER" /bin/bash -c "$*" > "$logfile" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] END    $name (exit $rc)" >> "$SUMMARY"
    return $rc
}

# --- one run_stage call per queued job -------------------------------------------------------
#
# run_stage "<short_name>" "$REPO/outputs/<short_name>.log" \
#     "cd /workspace && uv run tools/train.py \
#         task=... \
#         module/image_encoder=... \
#         trainer.max_epochs=... \
#         run_id=<short_name>"
#
# Add as many of these as needed, in the order they should run. Each one blocks until its
# docker exec finishes before the next one starts -- no manual "wait" needed.

# -----------------------------------------------------------------------------------------------

echo "[$(date '+%F %T')] QUEUE COMPLETE" >> "$SUMMARY"
crontab -l 2>/dev/null | grep -v "$MARKER" | crontab -
