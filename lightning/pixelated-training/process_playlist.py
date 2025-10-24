#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    playlist_file = "/root/pixelated/ai/.notes/youtube_playlists.txt"
    
    # Read unique URLs
    urls = set()
    with open(playlist_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('https://'):
                urls.add(line)
    
    print(f"Processing {len(urls)} unique videos")
    
    # Change to youtube_transcripts directory
    os.chdir("/root/pixelated/ai/lightning/pixelated-training/youtube_transcripts")
    
    # Process each URL with your existing scripts
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Processing: {url}")
        try:
            # Your pipeline should handle the download and processing
            # Run transcribe_with_whisper.py
            subprocess.run([sys.executable, "transcribe_with_whisper.py"], check=True)
            
            # Run convert_to_dialog.py  
            subprocess.run([sys.executable, "convert_to_dialog.py"], check=True)
            
        except Exception as e:
            print(f"Error processing {url}: {e}")

if __name__ == "__main__":
    main()
