#!/bin/bash

# Check if folder path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/audio_folder"
  exit 1
fi

# Set folder path
FOLDER="$1"

# Find and resample audio files
find "$FOLDER" -type f \( \
  -iname "*.wav" -o \
  -iname "*.mp3" -o \
  -iname "*.flac" -o \
  -iname "*.aac" -o \
  -iname "*.ogg" \) -exec bash -c '
  for file; do
    echo "Resampling: $file"
    tmpfile="$(mktemp --suffix=.wav)"
    ffmpeg -loglevel error -y -i "$file" -ar 16000 "$tmpfile" && mv "$tmpfile" "$file"
  done
' bash {} +

echo "All audio files have been resampled to 16kHz."
