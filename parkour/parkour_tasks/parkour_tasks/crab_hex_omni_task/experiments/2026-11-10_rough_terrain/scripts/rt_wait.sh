#!/bin/bash
# Rough-terrain campaign: wait on a queue log for the next decision-relevant event after line FROM (a one-shot
# log-sentinel watcher; re-arm with the printed line count). NEVER pgrep / ps: the log is the only signal.
#   bash rt_wait.sh LOG FROM [MAX_HOURS]
# Events: "RTQ <label> rc=<non-zero>", "RTQ NOTIFY ..." (the iteration-1500 read), "RTQ_DONE <block> fail=<n>".
# Prints "RT_WAIT line=<n> <event>" lines then "RT_WAIT lines=<n>" (the next FROM); after MAX_HOURS (default 6)
# "RT_WAIT TIMEOUT lines=<n> last=<the log's last RTQ line>" so a silent hang is noticed.
# One snapshot per poll: the lines after FROM are copied once, only its COMPLETE lines (newline-terminated) are read,
# and both the hits and the next FROM come from that copy, so a line appended meanwhile is never skipped (and a line
# still being written is never reported half-done or twice). At most RT_WAIT_MAX_HITS (default 40) hits are reported
# per call; when more are pending the printed lines= is the LAST REPORTED hit's line (not the end of the snapshot) and
# a "RT_WAIT more=<k> ..." line says so: re-arm with it and the rest follows.
set -u
LOG=${1:?log}; FROM=${2:?from line}; MAXH=${3:-6}
case "$FROM" in ''|*[!0-9]*) echo "RT_WAIT bad FROM '$FROM'"; exit 2 ;; esac
MAXHITS=${RT_WAIT_MAX_HITS:-40}
case "$MAXHITS" in ''|*[!0-9]*|0) echo "RT_WAIT bad RT_WAIT_MAX_HITS '$MAXHITS'"; exit 2 ;; esac
PAT='^RTQ [^ ]+ rc=(-[0-9]+|[1-9][0-9]*)( |$)|^RTQ NOTIFY |^RTQ_DONE '
SNAP=$(mktemp "${TMPDIR:-/tmp}/rt_wait.XXXXXX") || { echo "RT_WAIT cannot create a snapshot file"; exit 2; }
trap 'rm -f "$SNAP"' EXIT
end=$(( $(date +%s) + MAXH * 3600 ))
while :; do
  if [ -f "$LOG" ]; then
    tail -n +"$((FROM + 1))" "$LOG" > "$SNAP"
    nl=$(wc -l < "$SNAP")                        # complete lines of THIS snapshot
    hit=$(head -n "$nl" "$SNAP" | grep -nE "$PAT")
    if [ -n "$hit" ]; then
      total=$(printf '%s\n' "$hit" | wc -l)
      shown=$(printf '%s\n' "$hit" | head -n "$MAXHITS")
      last=$(( FROM + nl ))
      if [ "$total" -gt "$MAXHITS" ]; then
        last=$(( FROM + $(printf '%s\n' "$shown" | tail -n 1 | cut -d: -f1) ))
      fi
      printf '%s\n' "$shown" | while IFS= read -r l; do echo "RT_WAIT line=$(( FROM + ${l%%:*} )) ${l#*:}"; done
      if [ "$total" -gt "$MAXHITS" ]; then
        echo "RT_WAIT more=$(( total - MAXHITS )) event(s) pending after line $last: re-arm from lines=$last"
      fi
      echo "RT_WAIT lines=$last"; exit 0
    fi
  fi
  if [ "$(date +%s)" -ge "$end" ]; then
    last=$( [ -f "$LOG" ] && grep -E '^RTQ' "$LOG" | tail -1 )
    echo "RT_WAIT TIMEOUT lines=$( [ -f "$LOG" ] && wc -l < "$LOG" || echo 0) last=${last:-none}"; exit 0
  fi
  sleep "${RT_WAIT_POLL_S:-60}"
done
