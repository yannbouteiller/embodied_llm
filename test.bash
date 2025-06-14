#!/bin/bash

arecord -l | while IFS= read -r line; do
    # Capture card/device line
    if [[ "$line" =~ ^card[[:space:]]([0-9]+):.*device[[:space:]]([0-9]+): ]]; then
        card="${BASH_REMATCH[1]}"
        device="${BASH_REMATCH[2]}"
    fi

    # Match description line
    if echo "$line" | grep -iq "Chat 150 Mono"; then
        echo "$card:$device"
        exit 0
    fi
done

echo "Device 'Chat 150 Mono' not found."
exit 1
