#!/usr/bin/env python3
"""Run enhanced style analysis on all transcripts."""

from pathlib import Path
from src.voice_extraction.enhanced_style_analyzer import EnhancedStyleAnalyzer
from src.core.logging import get_logger

logger = get_logger("enhanced_analysis")

def main():
    """Process all transcripts with enhanced style analysis."""
    transcripts_dir = Path("/root/pixelated/.notes/transcripts")
    output_dir = Path("training_data/enhanced")
    
    analyzer = EnhancedStyleAnalyzer()
    all_segments = []
    
    logger.info(f"Processing transcripts from: {transcripts_dir}")
    
    # Process all .txt files recursively
    txt_files = list(transcripts_dir.rglob("*.txt"))
    logger.info(f"Found {len(txt_files)} transcript files")
    
    for i, file_path in enumerate(txt_files, 1):
        if file_path.stat().st_size == 0:
            continue
            
        logger.info(f"Processing {i}/{len(txt_files)}: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content.strip()) < 100:
                continue
                
            # Segment and analyze
            segments = analyzer.segment_transcript_enhanced(content)
            all_segments.extend(segments)
            
            if i % 50 == 0:
                logger.info(f"Processed {i} files, extracted {len(all_segments)} segments so far")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info(f"Total segments extracted: {len(all_segments)}")
    
    # Export training data
    logger.info("Exporting enhanced training data...")
    summary = analyzer.export_enhanced_training_data(all_segments, output_dir)
    
    logger.info("Enhanced analysis complete!")
    logger.info(f"Summary: {summary}")

if __name__ == "__main__":
    main()
