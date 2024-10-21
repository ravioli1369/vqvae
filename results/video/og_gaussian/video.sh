#!/bin/bash

# Check if the output filename is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <output_filename>"
  exit 1
fi

OUTPUT_FILENAME=$1

# Create filelist.txt
echo "Creating filelist.txt..."
rm -f filelist.txt
for i in $(seq 0 10 1990); do
  echo "file '$PWD/$OUTPUT_FILENAME/$i.png'" >> filelist.txt
done

# Run ffmpeg to create the video
echo "Creating video..."
ffmpeg -y -f concat -safe 0 -i filelist.txt -framerate 60 -c:v libx264 -pix_fmt yuv420p "$OUTPUT_FILENAME.mp4"

echo "Video created successfully: $OUTPUT_FILENAME.mp4"
