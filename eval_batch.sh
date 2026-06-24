#!/bin/bash

SEEDS=(420 4200 42000 420000 4200000)
GROUP=""
PIDS=()
START_DELAY=3

BASE_DIR="/home/andreaberti/Satellite-Control-Master-Thesis/train"

usage() {
    echo "Usage:"
    echo "  $0 --group GROUP_NAME"
    echo "Examples:"
    echo "  $0 --group base"
    echo "  $0 --group noise/0_01"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --group)
            GROUP="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$GROUP" ]; then
    echo "Error: --group is required."
    usage
    exit 1
fi

cleanup() {
    echo
    echo "Stopping launched eval processes..."
    
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

CONFIGS_DIR="${BASE_DIR}/configs/${GROUP}"
RUNS_DIR="${BASE_DIR}/runs/${GROUP}"

for CONFIG_PATH in "$CONFIGS_DIR"/config_*.json; do
    
    CONFIG_NAME=$(basename "$CONFIG_PATH" .json)
    RUN_DIR_NAME="${CONFIG_NAME/config_/run_}"
    RUN_PATH="${RUNS_DIR}/${RUN_DIR_NAME}"

    echo "================================================================================"
    echo "Starting batch evaluation for: $RUN_DIR_NAME and config: $CONFIG_NAME"
    echo "================================================================================"

    PIDS=() 

    for SEED in "${SEEDS[@]}"; do
        echo "Starting eval with seed=$SEED | run-name=$RUN_PATH | config-name=$CONFIG_PATH"

        setsid ./eval.sh \
            --run-name "$RUN_PATH" \
            --config-name "$CONFIG_PATH" \
            --seed "$SEED" &

        PIDS+=("$!")

        sleep "$START_DELAY"
    done

    wait
    
    echo
    echo "Finished evaluating $RUN_DIR_NAME and config: $CONFIG_NAME"
done

echo
echo "All runs in group '$GROUP' have been successfully evaluated."