#!/usr/bin/env bash
# Every heavy benchmark, one at a time (one 8 GB GPU). Resumable: each step reuses its
# prediction caches, so rerunning after a crash only redoes unfinished work.
#   bash scripts/run_benchmarks.sh                 # everything, serially
#   bash scripts/run_benchmarks.sh detect1280 train tracking compare distance   # GPU lane
#   bash scripts/run_benchmarks.sh score           # CPU lane (CUDA hidden), after detect1280
# Light inference steps (tracking, distance, detect1280) are latency-bound and may share the GPU
# under their VRAM caps (sum < 1); `train` and `speed` run alone.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python  # Linux/macOS venv layout
LOG=results/logs/queue.log
mkdir -p results/logs

run() {
    echo "$(date '+%F %T') START $*" | tee -a "$LOG"
    { nvidia-smi --query-gpu=memory.used --format=csv,noheader || true; } 2>/dev/null |
        sed 's/^/  vram before: /' >> "$LOG"
    if ! "$PY" scripts/capped.py "$@" >> "results/logs/$1.log" 2>&1; then
        echo "$(date '+%F %T') FAIL  $1" | tee -a "$LOG"
        return 1
    fi
    echo "$(date '+%F %T') DONE  $1" | tee -a "$LOG"
}

DETECTORS="yolov3-darknet yolov3u yolo11n yolo11s yolo11m yolo26n yolo26s yolo26m
           yolo11s@1280 yolo26s@1280 yolo26m@1280"
WEIGHTS=results/finetune/yolo26s-mot17/weights/best.pt
STEPS="${*:-detection tracking distance train compare speed}"
KNOWN=" detection detect1280 score tracking distance train compare speed "
for step in $STEPS; do  # reject a typo before hours of earlier steps have run
    case "$KNOWN" in *" $step "*) ;; *) echo "unknown step $step" >&2; exit 2 ;; esac
done

for step in $STEPS; do
    case $step in
    detection)
        # shellcheck disable=SC2086
        run evaluation.detection --models $DETECTORS --device 0 --out results/detection.json ;;
    detect1280)
        run evaluation.detection --models yolo11s@1280 yolo26s@1280 yolo26m@1280 --device 0 \
            --out results/logs/det_1280.json ;;
    score)
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="" run evaluation.detection --models $DETECTORS --device cpu \
            --out results/detection.json ;;
    tracking)
        VRAM_FRACTION=0.30 run evaluation.tracking --models yolo11s yolo26n yolo26s yolo26m yolo26s@1280 \
            --trackers bytetrack botsort --device 0 --out results/tracking.json ;;
    distance)
        VRAM_FRACTION=0.25 run evaluation.wildtrack --phase A B --device 0 --out results/distance.json \
            --fig-dir results/figures ;;
    train)
        run evaluation.finetune prepare
        if [ -f results/finetune/yolo26s-mot17/finetune_train.json ]; then
            echo "train: already finished, skipping" | tee -a "$LOG"
        else
            run evaluation.finetune train --name yolo26s-mot17 --batch 4 --epochs 15 --patience 5 --resume
        fi ;;
    compare)
        run evaluation.finetune compare --weights "$WEIGHTS" --out results/finetune.json ;;
    speed)
        # shellcheck disable=SC2086
        run evaluation.speed --models $DETECTORS --devices cpu 0 --out results/speed.json ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
    esac
done
echo "$(date '+%F %T') ALL DONE $STEPS" | tee -a "$LOG"
