#!/bin/bash
./set-emeet-sink.sh
mic=$(grep -oP '(?<=Device #).*?(?=: Chat 150 Mono)' <<< $(~/.local/bin/cheetah_demo_mic --show_audio_devices))
echo "Microphone Device Index: ${mic}."
python3 ~/embodied_llm/embodied_llm/agent/ellm.py --microphone=${mic} --models-folder=/home/indro/ellm --camera=4 --send-commands
