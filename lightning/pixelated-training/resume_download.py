#!/usr/bin/env python3
"""
Resume incomplete downloads - only fetch missing files
"""
import os
import subprocess
from pathlib import Path

def download_missing_file():
    """Download the one missing file that failed"""
    missing_file_id = "1q31ExrDGQXbejhUipbJU6ivM4RO2ewpt"
    output_path = "processed/phase_3_cot_reasoning/task_5_15_cot_reasoning/task_5_3_cot_reasoning_report.json"
    
    if not os.path.exists(output_path):
        print(f"Downloading missing file: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            "gdown", 
            f"https://drive.google.com/uc?id={missing_file_id}",
            "-O", output_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully downloaded: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download: {e}")
            print("You can manually download from: https://drive.google.com/uc?id=1q31ExrDGQXbejhUipbJU6ivM4RO2ewpt")
    else:
        print(f"File already exists: {output_path}")

if __name__ == "__main__":
    download_missing_file()
