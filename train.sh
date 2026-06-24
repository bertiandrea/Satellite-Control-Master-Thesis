#!/bin/bash

SEED=""
DISPLAY_NUM="10"
CONDA_ENV="rlgpu"
SCREEN_RES="1920x1080x24"

usage() {
    echo "Usage:"
    echo "  $0"
    echo "  $0 --seed 420"
    echo "  $0 --seed 0"
    echo "  $0 --env rlgpu"
    echo "  $0 --display 11"
    echo "  $0 --seed 420 --env rlgpu --display 11"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --display)
            DISPLAY_NUM="$2"
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

# Se il display è già in uso, prova i successivi: :10, :11, :12, ...
while [ -e /tmp/.X${DISPLAY_NUM}-lock ]; do
    echo "Display :$DISPLAY_NUM is already in use, trying next..."
    DISPLAY_NUM=$((DISPLAY_NUM + 1))
done

export DISPLAY=:$DISPLAY_NUM

echo "Using DISPLAY=$DISPLAY"
echo "Using Conda environment: $CONDA_ENV"
if [ -n "$SEED" ]; then
    echo "Using seed: $SEED"
else
    echo "Using seed: not specified"
fi

cleanup() {
    echo "Stopping Xvfb, GNOME, and x11vnc..."
    kill "$XVFB_PID" 2>/dev/null
    kill "$GNOME_PID" 2>/dev/null
    kill "$X11VNC_PID" 2>/dev/null
}
trap cleanup EXIT

# Avvia Xvfb
Xvfb $DISPLAY -screen 0 $SCREEN_RES &
XVFB_PID=$!

sleep 2

# Avvia sessione D-Bus e GNOME
eval $(dbus-launch --sh-syntax)
export DBUS_SESSION_BUS_ADDRESS
export GNOME_SHELL_SESSION_MODE=ubuntu
export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=GNOME
export GDMSESSION=gnome

gnome-session --session=gnome &
GNOME_PID=$!

sleep 5

# Salva variabili d’ambiente per sessioni future
echo "export DISPLAY=$DISPLAY" > /tmp/gnome_vnc_env.sh
echo "export DBUS_SESSION_BUS_ADDRESS=$DBUS_SESSION_BUS_ADDRESS" >> /tmp/gnome_vnc_env.sh
chmod +x /tmp/gnome_vnc_env.sh

# Avvia x11vnc sulla porta corretta
x11vnc -display $DISPLAY -nopw -forever -bg -rfbport $((5900 + DISPLAY_NUM))
X11VNC_PID=$!

# Inizializza Conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "Conda not found $HOME/miniconda3. Verify the installation path."
    exit 1
fi

# Avvia il training
if [ -n "$SEED" ]; then
    python -m code.train --seed "$SEED"
else
    python -m code.train
fi

exit 0