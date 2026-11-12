#!/usr/bin/env bash
set -e
if [ "$#" -eq 0 ]; then
  echo 'Usage inside tmux after GPU registration: bash resume_in_tmux.sh GPU_INDEX'
  exit 1
fi
code_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
exec bash "$code_dir/run_in_tmux.sh" --registered "$@" --resume
