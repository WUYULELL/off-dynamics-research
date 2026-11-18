#!/usr/bin/env bash
# auto_runner.sh —— 服务器端执行器。由实验室电脑上的 relay.py 每次自动上传到
#   ~/auto_jobs/_runner/auto_runner.sh，不需要手动安装。
#
# 子命令：
#   extract ID                      解压 ~/auto_jobs/_incoming/ID.tar.gz 到 ~/auto_jobs/ID/（目录已存在则拒绝）
#   launch  ID GPUS MAXSEC ENV CONDA_SH ATTEMPT THREADS
#                                   在 tmux 会话 auto_ID 中运行 ~/auto_jobs/ID/code/run.sh
#   status  ID...                   每个任务输出一行：ID|tmux是否存活(0/1)|退出码(空=未结束)
#   collect ID MAXKB DIR...         把结果小文件打包成 tar.gz 写到标准输出
#   kill    ID                      结束 tmux 会话（仅供人工使用，relay 不会自动调用）
#   gpus                            输出 nvidia-smi 查询结果
set -uo pipefail

ROOT="$HOME/auto_jobs"
ID_RE='^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$'

die() { echo "ERROR: $*" >&2; exit "${2:-2}"; }
check_id() { [[ "$1" =~ $ID_RE ]] || die "bad job id: $1"; }

cmd="${1:-}"; shift || true

case "$cmd" in
  extract)
    id="$1"; check_id "$id"
    J="$ROOT/$id"; T="$ROOT/_incoming/$id.tar.gz"
    [ -f "$T" ] || die "missing upload $T" 3
    if [ -e "$J" ]; then rm -f "$T"; echo "EXISTS"; exit 4; fi
    # 先解到临时目录并检查路径，再原子改名
    tmp="$ROOT/_incoming/$id.extract.$$"
    mkdir -p "$tmp"
    if tar tzf "$T" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then rm -rf "$tmp" "$T"; die "unsafe path in archive" 5; fi
    tar xzf "$T" -C "$tmp" || { rm -rf "$tmp"; die "tar failed" 6; }
    [ -f "$tmp/code/run.sh" ] || { rm -rf "$tmp"; die "code/run.sh missing" 7; }
    mkdir -p "$tmp/_relay"
    mv "$tmp" "$J"; rm -f "$T"
    echo "OK"
    ;;

  launch)
    id="$1"; gpus="$2"; maxsec="$3"; env="$4"; condash="$5"; attempt="$6"; threads="${7:-4}"
    check_id "$id"
    [[ "$maxsec" =~ ^[0-9]+$ ]] || die "bad maxsec"
    [[ "$attempt" =~ ^[0-9]+$ ]] || die "bad attempt"
    [[ "$threads" =~ ^[0-9]+$ ]] || die "bad threads"
    [[ "$gpus" == "none" || "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "bad gpus"
    J="$ROOT/$id"
    [ -f "$J/code/run.sh" ] || die "no run.sh in $J/code" 4
    if tmux has-session -t "auto_$id" 2>/dev/null; then echo "ALREADY_RUNNING"; exit 5; fi
    mkdir -p "$J/_relay"
    rm -f "$J/_relay/EXIT_CODE"
    date -Is > "$J/_relay/STARTED_attempt$attempt"
    [ "$gpus" == "none" ] && cvd="" || cvd="$gpus"
    {
      echo '#!/usr/bin/env bash'
      echo "J=$(printf %q "$J")"
      echo "ATTEMPT=$(printf %q "$attempt")"
      echo "CONDA_SH=$(printf %q "$condash")"
      echo "CONDA_ENV=$(printf %q "$env")"
      echo "MAXSEC=$(printf %q "$maxsec")"
      echo "export CUDA_VISIBLE_DEVICES=$(printf %q "$cvd")"
      echo "export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads"
      echo "export AUTO_JOB_ID=$(printf %q "$id") AUTO_ATTEMPT=$(printf %q "$attempt")"
      cat <<'EOS'
LOG="$J/_relay/stdout_attempt$ATTEMPT.log"
finish() { echo "$1" > "$J/_relay/EXIT_CODE.tmp"; mv -f "$J/_relay/EXIT_CODE.tmp" "$J/_relay/EXIT_CODE"; date -Is > "$J/_relay/FINISHED_attempt$ATTEMPT"; }
{
  echo "[auto_runner] start $(date -Is) host=$(hostname) attempt=$ATTEMPT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES maxsec=$MAXSEC"
  if [ -n "$CONDA_SH" ] && [ "$CONDA_SH" != "none" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH" && conda activate "$CONDA_ENV" || { echo "[auto_runner] conda activate failed"; finish 90; exit 0; }
  fi
  echo "[auto_runner] python=$(command -v python)"
} >> "$LOG" 2>&1
cd "$J/code" || { finish 91; exit 0; }
timeout --signal=TERM --kill-after=120 "$MAXSEC" bash run.sh >> "$LOG" 2>&1
rc=$?
echo "[auto_runner] end $(date -Is) rc=$rc" >> "$LOG"
finish "$rc"
EOS
    } > "$J/_relay/inner.sh"
    tmux new-session -d -s "auto_$id" "bash $(printf %q "$J/_relay/inner.sh")" || die "tmux failed" 6
    echo "LAUNCHED"
    ;;

  status)
    for id in "$@"; do
      check_id "$id"
      d="$ROOT/$id/_relay"; alive=0; ec=""
      tmux has-session -t "auto_$id" 2>/dev/null && alive=1
      [ -f "$d/EXIT_CODE" ] && ec="$(tr -dc '0-9' < "$d/EXIT_CODE")"
      [ -d "$ROOT/$id" ] || ec="NODIR"
      echo "$id|$alive|$ec"
    done
    ;;

  collect)
    id="$1"; maxkb="$2"; shift 2
    check_id "$id"; [[ "$maxkb" =~ ^[0-9]+$ ]] || die "bad maxkb"
    J="$ROOT/$id"; cd "$J" || die "no job dir" 3
    for f in _relay/stdout_attempt*.log; do
      [ -f "$f" ] && tail -n 400 "$f" > "${f%.log}.tail.txt"
    done
    paths=("_relay")
    for d in "$@"; do
      [[ "$d" =~ (^/|(^|/)\.\.(/|$)) ]] && continue
      [ -e "code/$d" ] && paths+=("code/$d")
    done
    find "${paths[@]}" -type f \
      \( -name '*.json' -o -name '*.md' -o -name '*.csv' -o -name '*.txt' -o -name '*.png' -o -name '*.yaml' -o -name '*.tsv' \) \
      -size -"${maxkb}"k -print0 2>/dev/null | tar czf - --null -T -
    ;;

  kill)
    id="$1"; check_id "$id"
    # 先写退出码 130，relay 会记为 failed（人工停止），不会当作中断自动重跑
    mkdir -p "$ROOT/$id/_relay"
    echo 130 > "$ROOT/$id/_relay/EXIT_CODE"
    tmux kill-session -t "auto_$id" 2>/dev/null; echo "KILLED"
    ;;

  gpus)
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    ;;

  *)
    die "unknown command: $cmd"
    ;;
esac
