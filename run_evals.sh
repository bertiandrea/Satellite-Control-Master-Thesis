#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE="$SCRIPT_DIR/code/configs/satellite_config_eval.py"

RUNS=(
    # GROUP | LABEL                                 | DR    | INERTIA   | INERTIA_STD   | NOISE | NOISE_STD | QUAT_NOISE | COLOR

    "base|normal                                    |False  |False      |0.0            |False  |0.0        |0.0         |True"
    "base|noise                                     |True   |False      |0.0            |True   |0.01       |0.001       |True"
    
    "base|inertia                                   |True   |True       |0.05           |False  |0.0        |0.0         |True"

    #################################################################################################################

    "randomization/0_01|normal                      |False  |False      |0.0            |False  |0.0        |0.0         |True"
    "randomization/0_01|noise                       |True   |False      |0.0            |True   |0.01       |0.001       |True"
    
    "randomization/0_01|inertia                     |True   |True       |0.01           |False  |0.0        |0.0         |True"

    #################################################################################################################

    "randomization/0_05|normal                      |False  |False      |0.0            |False  |0.0        |0.0         |True"
    "randomization/0_05|noise                       |True   |False      |0.0            |True   |0.01       |0.001       |True"

    "randomization/0_05|inertia                     |True   |True       |0.05           |False  |0.0        |0.0         |True"

    #################################################################################################################

    "randomization/0_1|normal                       |False  |False      |0.0            |False  |0.0        |0.0         |True"
    "randomization/0_1|noise                        |True   |False      |0.0            |True   |0.01       |0.001       |True"

    "randomization/0_1|inertia                      |True   |True       |0.1            |False  |0.0        |0.0         |True"

    #################################################################################################################

    "noise/0_01|normal                              |False  |False      |0.0            |False  |0.0        |0.0         |True"
    "noise/0_01|noise                               |True   |False      |0.0            |True   |0.01       |0.001       |True"

    #################################################################################################################

    "randomization_noise/N0_01__R0_01|normal        |False  |False      |0.0            |False  |0.0        |0.0         |True"
    "randomization_noise/N0_01__R0_01|noise         |True   |False      |0.0            |True   |0.01       |0.001       |True"

    "randomization_noise/N0_01__R0_05|normal        |False  |False      |0.0            |False  |0.0        |0.0         |True"
    "randomization_noise/N0_01__R0_05|noise         |True   |False      |0.0            |True   |0.01       |0.001       |True"
)

# Verify configuration file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found: $CONFIG_FILE" >&2
    exit 1
fi

# Backup original configuration
BACKUP_FILE="$(mktemp)"
cp -p "$CONFIG_FILE" "$BACKUP_FILE"

restore_config() {
    cp -p "$BACKUP_FILE" "$CONFIG_FILE"
}

cleanup() {
    restore_config
    rm -f "$BACKUP_FILE"
    echo "Original configuration restored."
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$SCRIPT_DIR"

for run in "${RUNS[@]}"; do

    IFS='|' read -r \
        group \
        label \
        dr_enabled \
        inertia_enabled \
        inertia_std \
        noise_enabled \
        noise_std \
        quaternion_noise \
        color_enabled <<< "$run"

    # Remove spacing from parameters
    group="${group//[[:space:]]/}"
    label="${label//[[:space:]]/}"
    dr_enabled="${dr_enabled//[[:space:]]/}"
    inertia_enabled="${inertia_enabled//[[:space:]]/}"
    inertia_std="${inertia_std//[[:space:]]/}"
    noise_enabled="${noise_enabled//[[:space:]]/}"
    noise_std="${noise_std//[[:space:]]/}"
    quaternion_noise="${quaternion_noise//[[:space:]]/}"
    color_enabled="${color_enabled//[[:space:]]/}"

    # Restore configuration before each run
    restore_config

    # Apply temporary configuration
    cat >> "$CONFIG_FILE" <<PY

# --- Temporary evaluation configuration ---

CONFIG["dr_randomization"] = {
    "enabled": $dr_enabled,
    "dr_params": {
        "actor_params": {
            "satellite": {
                "color": $color_enabled,
                "rigid_body_properties": {
                    "inertia": {
                        "enabled": $inertia_enabled,
                        "distribution": "gaussian",
                        "operation": "scaling",
                        "range": [1.0, $inertia_std],
                    },
                },
            },
        },
        "observations": {
            "enabled": $noise_enabled,
            "distribution": "gaussian",
            "operation": "scaling",
            "range": [1.0, $noise_std],
            "quaternion_noise": $quaternion_noise,
        },
        "states": {
            "enabled": $noise_enabled,
            "distribution": "gaussian",
            "operation": "scaling",
            "range": [1.0, $noise_std],
            "quaternion_noise": $quaternion_noise,
        },
        "actions": {
            "enabled": $noise_enabled,
            "distribution": "gaussian",
            "operation": "scaling",
            "range": [1.0, $noise_std],
        },
    },
}

PY

    echo
    echo "========================================"
    echo "Group:       $group"
    echo "Label:       $label"
    echo "DR:          $dr_enabled"
    echo "Inertia:     $inertia_enabled (std=$inertia_std)"
    echo "Noise:       $noise_enabled (std=$noise_std)"
    echo "Quat Noise:  $quaternion_noise"
    echo "Color:       $color_enabled"
    echo "========================================"

    ./eval_batch.sh --group "$group"

done

echo "All evaluations completed."