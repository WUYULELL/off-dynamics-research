#!/usr/bin/env bash
set -u

# Paper-like cross-domain offline RL experiments:
#   Environments: Hopper and HalfCheetah, medium dynamics shift
#   Methods: IQL-target, IQL-mix, IQL_UNC alpha=2.0, IGDF
#   Protocol: 100% source dataset + 10% target dataset, seeds 0..4
#
# Usage:
#   bash run_hopper_halfcheetah_core_experiments.sh --dry-run
#   bash run_hopper_halfcheetah_core_experiments.sh
#
# Existing TensorBoard runs are skipped automatically.

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

CONDA_ENV="${CONDA_ENV:-offdynamics}"
SEEDS=(0 1 2 3 4)
MAX_STEP="${MAX_STEP:-500000}"
TARGET_FRACTION="${TARGET_FRACTION:-0.1}"
OVERWRITE="${OVERWRITE:-0}"

SHIFT_LEVEL="medium"
SRC_TYPE="medium"
TAR_TYPE="medium"

ENV_NAMES=(
  "hopper-kinematic-legjnt"
  "halfcheetah-kinematic-footjnt"
)

EXPERIMENT_LABELS=(
  "IQL-target"
  "IQL-mix"
  "IQL_UNC-a2"
  "IGDF"
)
EXPERIMENT_POLICIES=(
  "IQL"
  "IQL"
  "IQL_UNC"
  "IGDF"
)
EXPERIMENT_OUTDIRS=(
  "runs_cross_env_iql_target"
  "runs_cross_env_iql_mix"
  "runs_cross_env_iql_unc_a2"
  "runs_cross_env_igdf"
)
EXPERIMENT_TARGET_ONLY=(
  "1"
  "0"
  "0"
  "0"
)
EXPERIMENT_UNC_ALPHA=(
  ""
  ""
  "2.0"
  ""
)

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

safe_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr ' ' '_'
}

MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-}"
if [[ -z "$MUJOCO_PATH" || ! -d "$MUJOCO_PATH/bin" ]]; then
  MUJOCO_PATH="$(find_mujoco || true)"
fi

if [[ -z "$MUJOCO_PATH" || ! -d "$MUJOCO_PATH/bin" ]]; then
  echo "ERROR: MuJoCo 2.1 not found."
  echo "Set MUJOCO_PY_MUJOCO_PATH or install it at:"
  echo "  $HOME/.mujoco/mujoco210"
  exit 1
fi

if ! conda run -n "$CONDA_ENV" python -c "import sys; assert sys.version_info[:2] == (3, 8)" >/dev/null 2>&1; then
  echo "ERROR: Conda environment '$CONDA_ENV' is missing or is not Python 3.8."
  exit 1
fi

for env_name in "${ENV_NAMES[@]}"; do
  dataset_name="${env_name//-/_}_${SHIFT_LEVEL}_${TAR_TYPE}.hdf5"
  dataset_path="dataset/mujoco/$dataset_name"
  if [[ ! -f "$dataset_path" ]]; then
    echo "ERROR: Required dataset not found: $dataset_path"
    exit 1
  fi
  if ! conda run -n "$CONDA_ENV" python -c \
    "import h5py; f=h5py.File(r'$dataset_path','r'); assert len(f.keys()) > 0; f.close()" \
    >/dev/null 2>&1; then
    echo "ERROR: Dataset is not a valid readable HDF5 file: $dataset_path"
    echo "It may be incomplete or truncated. Download it again before training."
    exit 1
  fi
done

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
STATUS_PATH="$RUNTIME_LOG_DIR/hopper_halfcheetah_core_experiments_status.csv"
if [[ ! -f "$STATUS_PATH" ]]; then
  echo "label,policy,target_only,unc_alpha,env,shift_level,target_fraction,seed,max_step,start,end,elapsed_hours,exit_code,skipped,reason,console_log,tensorboard_log" > "$STATUS_PATH"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run OK"
  echo "Conda env: $CONDA_ENV"
  echo "MuJoCo: $MUJOCO_PATH"
  echo "Environments: ${ENV_NAMES[*]}"
  echo "Seeds: ${SEEDS[*]}"
  echo "Protocol: paper-like, source=100%, target_fraction=$TARGET_FRACTION"
  echo "Shift/data: medium / medium / medium"
  echo "Max steps: $MAX_STEP"
  for idx in "${!EXPERIMENT_LABELS[@]}"; do
    echo "  ${EXPERIMENT_LABELS[$idx]}: policy=${EXPERIMENT_POLICIES[$idx]}, target_only=${EXPERIMENT_TARGET_ONLY[$idx]}, unc_alpha=${EXPERIMENT_UNC_ALPHA[$idx]:-n/a}, outdir=${EXPERIMENT_OUTDIRS[$idx]}"
  done
  echo "Total runs: $((${#ENV_NAMES[@]} * ${#EXPERIMENT_LABELS[@]} * ${#SEEDS[@]}))"
  exit 0
fi

for env_name in "${ENV_NAMES[@]}"; do
  for idx in "${!EXPERIMENT_LABELS[@]}"; do
    label="${EXPERIMENT_LABELS[$idx]}"
    policy="${EXPERIMENT_POLICIES[$idx]}"
    out_dir="${EXPERIMENT_OUTDIRS[$idx]}"
    target_only="${EXPERIMENT_TARGET_ONLY[$idx]}"
    unc_alpha="${EXPERIMENT_UNC_ALPHA[$idx]}"

    for seed in "${SEEDS[@]}"; do
      descriptor="$env_name-srcdatatype-$SRC_TYPE-tardatatype-$TAR_TYPE-$SHIFT_LEVEL"
      tb_path="$out_dir/$policy/$descriptor/r$seed/tb"
      completed_path="$out_dir/$policy/$descriptor/r$seed/.completed"
      run_name="$(safe_name "$label")_$(safe_name "$env_name")_seed$seed"
      console_log_path="$RUNTIME_LOG_DIR/$run_name.console.log"
      runtime_log_path="$RUNTIME_LOG_DIR/$run_name.runtime.txt"

      if [[ "$OVERWRITE" != "1" && -f "$completed_path" ]]; then
        now="$(date '+%Y-%m-%d %H:%M:%S')"
        reason="completed_marker:$completed_path"
        echo "SKIP $label $env_name seed=$seed -- $reason"
        echo "$label,$policy,$target_only,${unc_alpha:-},$env_name,$SHIFT_LEVEL,$TARGET_FRACTION,$seed,$MAX_STEP,$now,$now,0,0,true,$reason,$console_log_path,$tb_path" >> "$STATUS_PATH"
        continue
      fi

      if [[ "$target_only" == "1" ]]; then
        export ODRL_TARGET_ONLY=1
      else
        unset ODRL_TARGET_ONLY
      fi

      if [[ -n "$unc_alpha" ]]; then
        export ODRL_UNC_ALPHA="$unc_alpha"
        export ODRL_UNC_MIN_WEIGHT=0.05
        export ODRL_SOURCE_LOSS_WEIGHT=1.0
      else
        unset ODRL_UNC_ALPHA
        unset ODRL_UNC_MIN_WEIGHT
        unset ODRL_SOURCE_LOSS_WEIGHT
      fi

      start_epoch="$(date +%s)"
      start_time="$(date '+%Y-%m-%d %H:%M:%S')"
      echo
      echo "START $label env=$env_name seed=$seed"
      echo "Console log: $console_log_path"

      conda run -n "$CONDA_ENV" python train.py \
        --policy "$policy" \
        --env "$env_name" \
        --shift_level "$SHIFT_LEVEL" \
        --srctype "$SRC_TYPE" \
        --tartype "$TAR_TYPE" \
        --seed "$seed" \
        --mode 3 \
        --max_step "$MAX_STEP" \
        --dir "$out_dir" 2>&1 | tee "$console_log_path"

      exit_code="${PIPESTATUS[0]}"
      end_epoch="$(date +%s)"
      end_time="$(date '+%Y-%m-%d %H:%M:%S')"
      elapsed_seconds=$((end_epoch - start_epoch))
      elapsed_hours="$(conda run -n "$CONDA_ENV" python -c "print(round($elapsed_seconds / 3600, 4))")"

      cat > "$runtime_log_path" <<EOF
label=$label
policy=$policy
target_only=$target_only
unc_alpha=${unc_alpha:-}
env=$env_name
shift_level=$SHIFT_LEVEL
srctype=$SRC_TYPE
tartype=$TAR_TYPE
target_fraction=$TARGET_FRACTION
paper_like_source_target=1
seed=$seed
mode=3
max_step=$MAX_STEP
start=$start_time
end=$end_time
elapsed_seconds=$elapsed_seconds
elapsed_hours=$elapsed_hours
exit_code=$exit_code
console_log=$console_log_path
tensorboard_log=$tb_path
EOF

      echo "$label,$policy,$target_only,${unc_alpha:-},$env_name,$SHIFT_LEVEL,$TARGET_FRACTION,$seed,$MAX_STEP,$start_time,$end_time,$elapsed_hours,$exit_code,false,,$console_log_path,$tb_path" >> "$STATUS_PATH"

      if [[ "$exit_code" == "0" ]]; then
        mkdir -p "$(dirname "$completed_path")"
        touch "$completed_path"
        echo "DONE $label $env_name seed=$seed elapsed_hours=$elapsed_hours"
      else
        echo "FAILED $label $env_name seed=$seed. Continuing."
      fi
    done
  done
done

unset ODRL_TARGET_ONLY
unset ODRL_UNC_ALPHA
unset ODRL_UNC_MIN_WEIGHT
unset ODRL_SOURCE_LOSS_WEIGHT

echo
echo "All Hopper/HalfCheetah experiments finished."
echo "Status CSV: $STATUS_PATH"
