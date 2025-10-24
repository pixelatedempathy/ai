#!/usr/bin/env python3
"""
Tim Fletcher Voice Style Analyzer
Analyzes transcripts and categorizes by communication style for MoE training
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import textstat

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TimFletcherStyleAnalyzer:
    def __init__(self):
        self.styles = {
            'therapeutic': {
                'keywords': ['trauma', 'healing', 'recovery', 'therapy', 'emotional', 'pain', 'hurt', 'wound', 'heal', 'process'],
                'patterns': [r'you might feel', r'it\'s okay to', r'this is normal', r'healing takes time']
            },
            'educational': {
                'keywords': ['understand', 'learn', 'concept', 'definition', 'explain', 'theory', 'research', 'study'],
                'patterns': [r'let me explain', r'what this means', r'the definition of', r'research shows']
            },
            'empathetic': {
                'keywords': ['understand', 'feel', 'experience', 'struggle', 'difficult', 'hard', 'support', 'care'],
                'patterns': [r'I understand', r'you\'re not alone', r'many people feel', r'it makes sense']
            },
            'practical': {
                'keywords': ['do', 'action', 'step', 'practice', 'tool', 'technique', 'method', 'strategy', 'try'],
                'patterns': [r'here\'s what you can do', r'try this', r'the first step', r'practice this']
            }
        }
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze text and return style scores"""
        text_lower = text.lower()
        scores = {}
        
        for style, indicators in self.styles.items():
            score = 0
            
            # Keyword matching
            for keyword in indicators['keywords']:
                score += text_lower.count(keyword) * 2
            
            # Pattern matching
            for pattern in indicators['patterns']:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 3
            
            # Normalize by text length
            scores[style] = score / max(len(text.split()), 1)
        
        return scores
    
    def categorize_transcript(self, transcript_path: str) -> Tuple[str, Dict[str, float]]:
        """Categorize a single transcript"""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        scores = self.analyze_text(content)
        primary_style = max(scores, key=scores.get)
        
        return primary_style, scores
    
    def analyze_all_transcripts(self, transcript_dir: str) -> Dict[str, List[str]]:
        """Analyze all transcripts and categorize by style"""
        transcript_dir = Path(transcript_dir)
        categorized = {style: [] for style in self.styles.keys()}
        
        print(f"Analyzing transcripts in: {transcript_dir}")
        
        for transcript_file in transcript_dir.rglob("*.txt"):
            try:
                style, scores = self.categorize_transcript(transcript_file)
                categorized[style].append(str(transcript_file))
                print(f"{transcript_file.name}: {style} (scores: {scores})")
            except Exception as e:
                print(f"Error processing {transcript_file}: {e}")
        
        # Print summary
        print("\n=== Style Distribution ===")
        for style, files in categorized.items():
            print(f"{style.capitalize()}: {len(files)} files")
        
        return categorized

def main():
    analyzer = TimFletcherStyleAnalyzer()
    
    # Look for transcripts in common locations
    possible_dirs = [
        "/root/pixelated/ai/pipelines/youtube-transcription-pipeline/youtube_transcriptions/transcripts",
        "/root/transcripts",
        "/root/yt-downloader/transcripts"
    ]
    
    transcript_dir = None
    for dir_path in possible_dirs:
        if Path(dir_path).exists():
            transcript_dir = dir_path
            break
    
    if not transcript_dir:
        print("No transcript directory found. Please specify path.")
        return
    
    categorized = analyzer.analyze_all_transcripts(transcript_dir)
    
    # Save results
    output_file = Path(__file__).parent / "style_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(categorized, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
