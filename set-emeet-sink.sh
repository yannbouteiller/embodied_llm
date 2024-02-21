#!/bin/bash
systemctl --user stop pulseaudio
systemctl --user stop pulseaudio.socket
systemctl --user start pulseaudio

pactl set-default-sink alsa_output.usb-ClearOne_Communications_Chat_150-00.mono-fallback
#pactl get-default-sink
pactl set-sink-volume alsa_output.usb-ClearOne_Communications_Chat_150-00.mono-fallback '100%'

