#!/bin/bash
input=$(cat)

model=$(echo "$input" | jq -r '.model.display_name')
folder=$(basename "$(echo "$input" | jq -r '.workspace.current_dir')")
five_hour=$(echo "$input" | jq -r '.rate_limits.five_hour.used_percentage // empty')
seven_day=$(echo "$input" | jq -r '.rate_limits.seven_day.used_percentage // empty')
duration_ms=$(echo "$input" | jq -r '.cost.total_duration_ms // 0')

# ms -> H:MM:SS (or M:SS if under an hour)
secs=$(( duration_ms / 1000 ))
h=$(( secs / 3600 ))
m=$(( (secs % 3600) / 60 ))
s=$(( secs % 60 ))
if [ "$h" -gt 0 ]; then
    session_time=$(printf "%dh%02dm" "$h" "$m")
else
    session_time=$(printf "%dm%02ds" "$m" "$s")
fi

if [ -n "$five_hour" ]; then
    usage_5h="$(awk -v p="$five_hour" 'BEGIN { printf "%.0f%%", p }')"
else
    usage_5h="n/a"
fi

if [ -n "$seven_day" ]; then
    usage_7d="$(awk -v p="$seven_day" 'BEGIN { printf "%.0f%%", p }')"
else
    usage_7d="n/a"
fi

printf "\033[01;32m🤖 %s\033[00m | \033[01;33m⏳ 5h: %s\033[00m | \033[00;35m📅 7d: %s\033[00m | \033[00;36m⏱️  %s\033[00m | \033[01;34m📁 %s\033[00m" \
    "$model" "$usage_5h" "$usage_7d" "$session_time" "$folder"
