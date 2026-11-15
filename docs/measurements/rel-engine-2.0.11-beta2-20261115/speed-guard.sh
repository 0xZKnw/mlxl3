#!/bin/zsh
# REL-ENGINE-2.0.11-BETA2: wait until no third-party build/GPU job runs for 60 s, then run rel-speed.sh
# while sampling competing processes every 10 s (speed-contention.log). Max wait 90 min.
set -u
D=/Users/justin/Documents/mix-stq1_0/docs/measurements/rel-engine-2.0.11-beta2-20261115
PAT='swift-frontend|swift-build|rustc|cargo|cbmc|kani|mlxl3_native-|/mlxl3 |mlxl3-rs|check-mtp|compare_native|check_dflash2|xcodebuild'
busy() { ps -axo pid=,command= | grep -E "$PAT" | grep -v -E 'grep|speed-guard' ; }
quiet=0; waited=0
while (( quiet < 60 )); do
  if [[ -n "$(busy)" ]]; then quiet=0; else quiet=$((quiet+5)); fi
  sleep 5; waited=$((waited+5))
  (( waited > 5400 )) && { echo "machine never idle for 60 s within 90 min"; busy | cut -c1-200; exit 3; }
done
echo "idle after ${waited}s, starting at $(date)"
( while true; do b=$(busy | grep -v -E 'compare_native|/build/rel/engine-2.0.11-beta' | cut -c1-200); [[ -n "$b" ]] && { date; echo "$b"; }; sleep 10; done ) > $D/speed-contention.log 2>&1 &
sampler=$!
zsh $D/rel-speed.sh > $D/speed.log 2>&1
code=$?
kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
echo "speed exit=$code contention_lines=$(wc -l < $D/speed-contention.log | tr -d ' ')"
exit $code
