#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    if 'watch?v=' in url:
        return url.split('watch?v=')[1].split('&')[0]
    elif 'live/' in url:
        return url.split('live/')[1].split('?')[0]
    return None

def process_video(video_id, base_dir):
    """Process a single video through the pipeline"""
    print(f"Processing video: {video_id}")
    
    # Download audio
    audio_cmd = f"yt-dlp -x --audio-format mp3 -o '{base_dir}/audio/%(title)s.%(ext)s' https://www.youtube.com/watch?v={video_id}"
    subprocess.run(audio_cmd, shell=True, check=True)
    
    # Transcribe with whisper
    subprocess.run([sys.executable, f"{base_dir}/transcribe_with_whisper.py"], cwd=base_dir)
    
    # Convert to dialog
    subprocess.run([sys.executable, f"{base_dir}/convert_to_dialog.py"], cwd=base_dir)

def main():
    playlist_file = "/root/pixelated/ai/.notes/youtube_playlists.txt"
    base_dir = "/root/pixelated/ai/lightning/pixelated-training/youtube_transcripts"
    
    # Create directories
    os.makedirs(f"{base_dir}/audio", exist_ok=True)
    os.makedirs(f"{base_dir}/transcripts", exist_ok=True)
    os.makedirs(f"{base_dir}/processed", exist_ok=True)
    
    # Read URLs and extract unique video IDs
    video_ids = set()
    with open(playlist_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('https://'):
                video_id = extract_video_id(line)
                if video_id:
                    video_ids.add(video_id)
    
    print(f"Found {len(video_ids)} unique videos to process")
    
    # Process each video
    for i, video_id in enumerate(video_ids, 1):
        try:
            print(f"[{i}/{len(video_ids)}] Processing {video_id}")
            process_video(video_id, base_dir)
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue

if __name__ == "__main__":
    main()
