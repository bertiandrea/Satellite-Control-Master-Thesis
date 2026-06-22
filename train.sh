#!/bin/bash

DISPLAY_NUM=${1:-99}            # Numero del display, default 99
CONDA_ENV=${2:-rlgpu}           # Nome dell'ambiente Conda, default "rlgpu"
SCREEN_RES="1920x1080x24"

export DISPLAY=:$DISPLAY_NUM

echo "Using DISPLAY=$DISPLAY"
echo "Using Conda environment: $CONDA_ENV"

# Verifica se il display è già in uso
if [ -e /tmp/.X${DISPLAY_NUM}-lock ]; then
    echo "Display :$DISPLAY_NUM is already in use!"
    exit 1
fi

# Funzione cleanup al termine dello script
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

# Avvia il training con la reward function scelta
python -m code.train

exit 0