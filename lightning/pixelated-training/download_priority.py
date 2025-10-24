#!/usr/bin/env python3
import subprocess
import os

# Priority files to download
priority_files = [
    ("1-a_dnCvBaxA1giuHzp_1dQEk9_lt4ums", "priority_datasets/"),  # Main folder
]

def download_folder(folder_id, output_dir):
    """Download Google Drive folder"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["gdown", "--folder", f"https://drive.google.com/drive/folders/{folder_id}", "-O", output_dir]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    for folder_id, output_dir in priority_files:
        print(f"Downloading priority folder: {folder_id}")
        try:
            download_folder(folder_id, output_dir)
            print(f"Success: {output_dir}")
        except Exception as e:
            print(f"Failed: {e}")
