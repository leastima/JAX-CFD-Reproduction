#!/usr/bin/env bash
# 현재 orphan 프로세스들이 완료되면 master 재시작
PIDS="3392595 3392617 3392620 3392622"
ROOT="/jumbo/yaoqingyang/yuxin/JAX-CFD"

echo "[monitor] Waiting for PIDs: $PIDS"
for pid in $PIDS; do
    while kill -0 $pid 2>/dev/null; do
        sleep 60
    done
    echo "[monitor] PID $pid done at $(date +%H:%M)"
done

echo "[monitor] All orphan processes done. Restarting master..."
cd "${ROOT}"
bash scripts/train_1ksteps.sh > logs/train_1ksteps_master2.log 2>&1
