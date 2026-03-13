#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="mm"
MM_ROOT="/workspace"
MM_WS="/workspace/main_ws"
PIPER_WS="/workspace/piper_ros"

if [[ ! -f "$HOME/.tmux.conf" ]]; then
  cat > "$HOME/.tmux.conf" << 'TMUXCONF'
unbind C-b
set -g prefix `
bind ` send-prefix
set -g mouse on
set -g history-limit 100000
TMUXCONF
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed."
  exit 1
fi

echo "Building mm_actions workspace..."
cd "$MM_WS"
colcon build

echo "Building piper workspace..."
cd "$PIPER_WS"
colcon build

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux new-session -d -s "$SESSION_NAME"
  # 2x2 layout
  tmux split-window -h -t "$SESSION_NAME":0.0
  tmux split-window -v -t "$SESSION_NAME":0.0
  tmux split-window -v -t "$SESSION_NAME":0.1
  tmux select-layout -t "$SESSION_NAME" tiled

  # Pane 0: piper startup
  tmux send-keys -t "$SESSION_NAME":0.0 "cd $PIPER_WS" C-m
  tmux send-keys -t "$SESSION_NAME":0.0 "source $PIPER_WS/install/setup.bash" C-m
  tmux send-keys -t "$SESSION_NAME":0.0 "bash find_all_can_port.sh" C-m
  tmux send-keys -t "$SESSION_NAME":0.0 "bash can_activate.sh can0 1000000 \"1-4.2.2:1.0\"" C-m
  tmux send-keys -t "$SESSION_NAME":0.0 "ros2 launch piper start_single_piper.launch.py" C-m

  # Pane 1: camera startup
  tmux send-keys -t "$SESSION_NAME":0.1 "cd $PIPER_WS" C-m
  tmux send-keys -t "$SESSION_NAME":0.1 "source $PIPER_WS/install/setup.bash" C-m
  tmux send-keys -t "$SESSION_NAME":0.1 "ros2 launch piper d435i_high_resolution.launch.py" C-m

  # Pane 2: action server startup
  tmux send-keys -t "$SESSION_NAME":0.2 "cd $MM_WS" C-m
  tmux send-keys -t "$SESSION_NAME":0.2 "source $MM_WS/install/setup.bash" C-m
  tmux send-keys -t "$SESSION_NAME":0.2 "set -a" C-m
  tmux send-keys -t "$SESSION_NAME":0.2 "source $MM_WS/.env" C-m
  tmux send-keys -t "$SESSION_NAME":0.2 "ros2 run mm_actions mm_actions_node" C-m

  # Pane 3 left empty
fi

tmux select-pane -t "$SESSION_NAME":0.0
exec tmux attach -t "$SESSION_NAME"
