#!/bin/bash
# Pulls everything produced on the cluster back into the local repo.
#
# The cluster's /tmp is node-local scratch (wiped on reboot) and the NFS
# home directory has shown real reliability problems this session -- the
# cluster is not where anything should live long-term. Run this repeatedly
# during training (after each benchmark, after checkpoints save, and
# definitely before assuming a run's results are safe) rather than once at
# the end.
#
# Usage: bash scripts/sync_cluster_results.sh
# Safe to re-run: scp -r only adds/updates files, never deletes local ones.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$REPO_ROOT/cluster_results"
REMOTE_PROJECT="/tmp/26p66_resnet34/project/face_resnet34_training"
# jaguar3/jaguar4's ProxyCommand (via jaguar1) is already set in
# ~/.ssh/config -- no inline proxy option needed here, which also avoids a
# word-splitting bug an earlier inline "-o ProxyCommand=ssh -W %h:%p jaguar1"
# string had (scp saw a bare -W flag and rejected it).

mkdir -p "$OUT_DIR/dica10" "$OUT_DIR/jaguar3_utkface" "$OUT_DIR/jaguar4_lfw"

sync_node() {
    local label="$1"
    local host="$2"
    local out="$OUT_DIR/$label"

    echo "=== syncing $label ($host) ==="
    scp -r -o ConnectTimeout=20 \
        "$host:$REMOTE_PROJECT/experiments" \
        "$out/" 2>&1
    scp -o ConnectTimeout=20 \
        "$host:$REMOTE_PROJECT/data/records/*.csv" \
        "$out/" 2>&1
    echo "--- $label contents now ---"
    find "$out" -type f -newer /dev/null 2>/dev/null | head -30
    du -sh "$out" 2>&1
    echo
}

sync_node "dica10" "gpu-node"
sync_node "jaguar3_utkface" "jaguar3"
sync_node "jaguar4_lfw" "jaguar4"

echo "=== DONE. Everything pulled into: $OUT_DIR ==="
