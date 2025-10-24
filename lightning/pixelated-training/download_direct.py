#!/usr/bin/env python3
import subprocess
import os

# Direct file downloads from the priority folder
priority_files = [
    # CoT datasets
    ("1jw5X80mviMWkeSHqkbDBdcihOEfwbP8D", "CoT-Breakups_and_heartbreak-9.8k.json"),
    ("11DFZTq0EwmVeHgiaix8XTTB51g7r4pVA", "CoT_Legal_Issues_And_Laws.json"),
    ("10D6QtKUkZmakeULPVTuOeRdlqCJf3DbX", "CoT_Neurodivergent_vs_Neurotypical_Interactions.json"),
    ("1kwczTfskdws3OUvFnM2UWFueFBuSjtxC", "CoT_Philosophical_Understanding.json"),
    ("1J5q4YeSjPD-duYtMMEVJBz2mudLnnu1D", "CoT_Rare_Diseases_Health_Conditions_9.8k.json"),
    ("1_V7Q3fa4U_xxgGgi9VOO6MPPNqVfpJCL", "CoT_Reasoning_Clinical_Diagnosis_Mental_Health.json"),
    ("1FSiMjcLbnYjR-dYdtPLXUcEjzmKgPZRt", "CoT_Reasoning_Mens_Mental_Health.json"),
    ("1RE2sfxsJAwzBtvsYT0SJTsA4BJEhVS3N", "CoT_Reasoning_Quantum_Physics_And_Computing.json"),
    ("1Yfk0j-R-L1kThMwcY-db9bkKiPZwUgwy", "CoT_Reasoning_Scientific_Discovery_and_Research.json"),
    ("1g8xVGP6vqQYAlogdYX3oBvU4cnpY3TNt", "CoT_Temporal_Reasoning_Dataset.json"),
    # Mental health counseling
    ("1ulh6BENVMmOSyO_tAXTrHw01Ht7TzCdE", "mental_health_counseling_conversations-train.arrow"),
]

def download_file(file_id, filename):
    """Download single file from Google Drive"""
    output_path = f"priority_datasets/{filename}"
    os.makedirs("priority_datasets", exist_ok=True)
    
    cmd = ["gdown", f"https://drive.google.com/uc?id={file_id}", "-O", output_path]
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed: {filename} - {e}")
        return False

if __name__ == "__main__":
    print("Downloading priority files...")
    success_count = 0
    
    for file_id, filename in priority_files:
        if download_file(file_id, filename):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(priority_files)} files downloaded")
