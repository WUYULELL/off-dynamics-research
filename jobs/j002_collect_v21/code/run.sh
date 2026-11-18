#!/usr/bin/env bash
# 等待 V21 结束后只读复制结果。不修改 $HOME/v21_confirm 下的任何文件。
set -u
mkdir -p outputs/v21
V="$HOME/v21_confirm"
DEADLINE=$(( $(date +%s) + 84600 ))   # 23.5 小时，留出取回时间
snap() { python3 - "$1" <<'PY'
import json, sys, time, os
d = sys.argv[1]; out = {"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"), "run_dir": d}
try:
    s = json.load(open(os.path.join(d, "STATUS.json")))
    out.update({k: s.get(k) for k in ("time", "phase", "pending", "running", "done", "failed")})
except Exception as e:
    out["status_error"] = str(e)[:200]
json.dump(out, open("outputs/v21_progress.json", "w"), ensure_ascii=False, indent=2, default=str)
PY
}
while :; do
  if [ ! -f "$V/outputs/LATEST_RUN.txt" ]; then
    echo "{\"state\": \"no_run_found\", \"time\": \"$(date -Is)\"}" > outputs/v21_progress.json
  else
    d="$(cat "$V/outputs/LATEST_RUN.txt")"
    case "$d" in /*) ;; *) d="$V/outputs/$d" ;; esac
    snap "$d"
    if [ -f "$d/COMPLETED.json" ] || [ -f "$d/FAILED.json" ]; then
      # 顶层小文件 + 两层以内的汇总文件；单个 ≤ 2MB，总量 ≤ 20MB
      total=0
      while IFS= read -r -d '' f; do
        sz=$(stat -c %s "$f"); [ $((total + sz)) -gt 20971520 ] && break
        rel="${f#$d/}"; mkdir -p "outputs/v21/$(dirname "$rel")"; cp -p "$f" "outputs/v21/$rel"; total=$((total + sz))
      done < <( { find "$d" -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' -o -name '*.csv' -o -name '*.txt' \) -size -2048k -print0;
                  find "$d" -mindepth 2 -maxdepth 3 -type f \( -iname '*summary*' -o -iname '*result*' -o -iname '*.csv' \) \( -name '*.json' -o -name '*.csv' -o -name '*.md' \) -size -2048k -print0; } )
      [ -f "$d/FAILED.json" ] && state=failed || state=completed
      echo "{\"state\": \"$state\", \"run_dir\": \"$d\", \"copied_bytes\": $total, \"time\": \"$(date -Is)\"}" > outputs/COLLECTED.json
      echo "V21 $state; copied $total bytes from $d"
      exit 0
    fi
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "V21 还没结束；本任务到时退出，由自动循环提交新的收集任务。"
    echo "{\"state\": \"not_finished\", \"time\": \"$(date -Is)\"}" > outputs/COLLECTED.json
    exit 0
  fi
  sleep 600
done
