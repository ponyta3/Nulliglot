#!/bin/bash
# Every action on Hopper goes through this script so it is recorded in
# hopper_activity.log (same folder). Watch it live from another terminal:
#
#     tail -f "/Users/praveen/Desktop/Research/CS 795/hopper_activity.log"
#
# Usage:
#     ./hopper.sh '<command to run on Hopper>'
#     ./hopper.sh --put <local path> <remote path>     copy laptop -> Hopper
#     ./hopper.sh --get <remote path> <local path>     copy Hopper -> laptop
#
# Hopper is a shared cluster: work only under /home/pgopalan and
# /scratch/pgopalan, and run anything heavy through Slurm, not the login node.

set -o pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/hopper_activity.log"
HOST="${HOPPER_HOST:-pgopalan@hopper-intel.orc.gmu.edu}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

case "$1" in
  --put) shown="COPY TO HOPPER    $2  ->  $3"; action=(scp "${SSH_OPTS[@]}" -r "$2" "$HOST:$3") ;;
  --get) shown="COPY FROM HOPPER  $2  ->  $3"; action=(scp "${SSH_OPTS[@]}" -r "$HOST:$2" "$3") ;;
  "")    echo "usage: hopper.sh '<command>' | --put <local> <remote> | --get <remote> <local>" >&2; exit 2 ;;
  *)     shown="\$ $*"; action=(ssh "${SSH_OPTS[@]}" "$HOST" "$*") ;;
esac

# Record the command before it runs, so it is visible even if it hangs.
printf '\n===== %s  START\n%s\n' "$(stamp)" "$shown" >> "$LOG"

# Stream output to the log as it arrives; drop the login banner lines.
"${action[@]}" 2>&1 | grep --line-buffered -v '^|' | tee -a "$LOG"
rc=${PIPESTATUS[0]}

printf '===== %s  END (exit %s)\n' "$(stamp)" "$rc" >> "$LOG"
exit "$rc"
