#!/usr/bin/env bash
set -u

# Linux/AutoDL version of run_priority_walker2d_experiments.ps1.
# Runs:
#   IQL-target
#   TD3_BC-target
#   TD3_BC-mix
# under the same Walker2d paper-like 10% target protocol.

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

SEEDS=(0 1 2 3 4)
MAX_STEP=500000
TARGET_FRACTION="0.1"
OVERWRITE=0

ENV_NAME="walker2d-kinematic-footjnt"
SHIFT_LEVEL="medium"
SRC_TYPE="medium"
TAR_TYPE="medium"

EXPERIMENT_LABELS=("IQL-target" "TD3_BC-target" "TD3_BC-mix")
EXPERIMENT_POLICIES=("IQL" "TD3_BC" "TD3_BC")
EXPERIMENT_OUTDIRS=("runs_paper_like_iql_target_only" "runs_paper_like_td3_bc_target_only" "runs_paper_like_td3_bc_mix")
EXPERIMENT_TARGET_ONLY=("1" "1" "0")

find_mujoco() {
  local candidates=(
    "$HOME/.mujoco/mujoco210"
    "$HOME/.mujoco210/mujoco210"
    "/root/.mujoco/mujoco210"
    "/root/.mujoco210/mujoco210"
  )
  for path in "${candidates[@]}"; do
    if [[ -d "$path/bin" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-}"
if [[ -z "$MUJOCO_PATH" || ! -d "$MUJOCO_PATH/bin" ]]; then
  MUJOCO_PATH="$(find_mujoco || true)"
fi

if [[ -z "$MUJOCO_PATH" || ! -d "$MUJOCO_PATH/bin" ]]; then
  echo "ERROR: MuJoCo 2.1 not found."
  echo "Expected one of:"
  echo "  $HOME/.mujoco/mujoco210"
  echo "  $HOME/.mujoco210/mujoco210"
  echo "Or set MUJOCO_PY_MUJOCO_PATH before running."
  exit 1
fi

DATASET_NAME="${ENV_NAME//-/_}_${SHIFT_LEVEL}_${TAR_TYPE}.hdf5"
DATASET_PATH="dataset/mujoco/${DATASET_NAME}"
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "ERROR: Required dataset not found: $DATASET_PATH"
  echo "Please cd into the project root before running this script."
  exit 1
fi

export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_PATH"
export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
export PATH="$MUJOCO_PATH/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="./algo:${PYTHONPATH:-}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export ODRL_TARGET_FRACTION="$TARGET_FRACTION"
export ODRL_PAPER_STYLE_SOURCE_TARGET=1

RUNTIME_LOG_DIR="runs/runtime_logs"
mkdir -p "$RUNTIME_LOG_DIR"
STATUS_PATH="$RUNTIME_LOG_DIR/priority_walker2d_experiments_status.csv"
if [[ ! -f "$STATUS_PATH" ]]; then
  echo "label,policy,target_only,seed,env,target_fraction,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log" > "$STATUS_PATH"
fi

safe_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr ' ' '_'
}

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run OK"
  echo "MuJoCo: $MUJOCO_PATH"
  echo "Dataset: $DATASET_PATH"
  echo "Seeds: ${SEEDS[*]}"
  echo "Protocol: paper-like, target_fraction=$TARGET_FRACTION"
  for idx in "${!EXPERIMENT_LABELS[@]}"; do
    echo "  ${EXPERIMENT_LABELS[$idx]}: policy=${EXPERIMENT_POLICIES[$idx]}, target_only=${EXPERIMENT_TARGET_ONLY[$idx]}, outdir=${EXPERIMENT_OUTDIRS[$idx]}"
  done
  exit 0
fi

for idx in "${!EXPERIMENT_LABELS[@]}"; do
  label="${EXPERIMENT_LABELS[$idx]}"
  policy="${EXPERIMENT_POLICIES[$idx]}"
  out_dir="${EXPERIMENT_OUTDIRS[$idx]}"
  target_only="${EXPERIMENT_TARGET_ONLY[$idx]}"

  for seed in "${SEEDS[@]}"; do
    tb_path="$out_dir/$policy/$ENV_NAME-srcdatatype-$SRC_TYPE-tardatatype-$TAR_TYPE-$SHIFT_LEVEL/r$seed/tb"
    run_name="$(safe_name "$label")_paper_like_seed$seed"
    console_log_path="$RUNTIME_LOG_DIR/$run_name.console.log"
    runtime_log_path="$RUNTIME_LOG_DIR/$run_name.runtime.txt"

    if [[ "$OVERWRITE" != "1" ]] && compgen -G "$tb_path/events.out.tfevents*" > /dev/null; then
      now="$(date '+%Y-%m-%d %H:%M:%S')"
      reason="existing_tensorboard_event:$tb_path"
      echo "SKIP $label seed=$seed -- $reason"
      echo "$label,$policy,$target_only,$seed,$ENV_NAME,$TARGET_FRACTION,$MAX_STEP,$now,$now,0,0,true,$reason,$console_log_path,$tb_path" >> "$STATUS_PATH"
      continue
    fi

    start_epoch="$(date +%s)"
    start_time="$(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "START $label policy=$policy target_only=$target_only seed=$seed"
    echo "Console log: $console_log_path"

    if [[ "$target_only" == "1" ]]; then
      export ODRL_TARGET_ONLY=1
    else
      unset ODRL_TARGET_ONLY
    fi

    conda run -n offdynamics python train.py \
      --policy "$policy" \
      --env "$ENV_NAME" \
      --shift_level "$SHIFT_LEVEL" \
      --srctype "$SRC_TYPE" \
      --tartype "$TAR_TYPE" \
      --seed "$seed" \
      --mode 3 \
      --max_step "$MAX_STEP" \
      --dir "$out_dir" 2>&1 | tee "$console_log_path"

    exit_code="${PIPESTATUS[0]}"
    unset ODRL_TARGET_ONLY

    end_epoch="$(date +%s)"
    end_time="$(date '+%Y-%m-%d %H:%M:%S')"
    elapsed_seconds=$((end_epoch - start_epoch))
    elapsed_hours="$(python - <<PY
print(round($elapsed_seconds / 3600, 4))
PY
)"

    cat > "$runtime_log_path" <<EOF
label=$label
policy=$policy
target_only=$target_only
env=$ENV_NAME
target_fraction=$TARGET_FRACTION
paper_like_source_target=1
seed=$seed
max_step=$MAX_STEP
start=$start_time
end=$end_time
elapsed_seconds=$elapsed_seconds
elapsed_hours=$elapsed_hours
exit_code=$exit_code
console_log=$console_log_path
tensorboard_log=$tb_path
EOF

    echo "$label,$policy,$target_only,$seed,$ENV_NAME,$TARGET_FRACTION,$MAX_STEP,$start_time,$end_time,$elapsed_hours,$exit_code,false,,$console_log_path,$tb_path" >> "$STATUS_PATH"

    if [[ "$exit_code" == "0" ]]; then
      echo "DONE $label seed=$seed"
    else
      echo "FAILED $label seed=$seed. Continuing."
    fi
  done
done

echo
echo "Priority experiments finished. Status CSV: $STATUS_PATH"
