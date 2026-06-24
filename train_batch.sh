#!/bin/bash

SEEDS=(42 84 168)
PIDS=()
START_DELAY=3

cleanup() {
    echo
    echo "Stopping launched training processes..."

    for PID in "${PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            echo "Killing process group $PID"
            kill -TERM -- "-$PID" 2>/dev/null
        fi
    done

    wait 2>/dev/null
    exit 130
}

trap cleanup INT

for SEED in "${SEEDS[@]}"; do
    echo "Starting training with seed=$SEED"

    setsid ./train.sh --seed "$SEED" &
    PIDS+=("$!")

    sleep "$START_DELAY"
done

wait

echo
echo "All training processes finished."