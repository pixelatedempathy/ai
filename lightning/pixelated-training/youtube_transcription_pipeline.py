#!/usr/bin/env python3
"""
YouTube Video Transcription Pipeline
Transcribes MP3 files and creates beautiful markdown transcripts
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import whisper
import re
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeTranscriptionPipeline:
    def __init__(self, 
                 source_dir: str = "/root/yt-downloader/Downloads",
                 target_dir: str = "/root/pixelated/ai/lightning/pixelated-training",
                 model_size: str = "base"):
        
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.audio_dir = self.target_dir / "audio"
        self.transcripts_dir = self.target_dir / "transcripts"
        self.markdown_dir = self.target_dir / "markdown_transcripts"
        
        # Create directories
        for dir_path in [self.audio_dir, self.transcripts_dir, self.markdown_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper model
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        
        # Track processed files
        self.processed_log = self.target_dir / "processed_files.json"
        self.processed_files = self.load_processed_files()
        
    def load_processed_files(self) -> Dict:
        """Load list of already processed files"""
        if self.processed_log.exists():
            with open(self.processed_log, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_processed_files(self):
        """Save processed files log"""
        with open(self.processed_log, 'w', encoding='utf-8') as f:
            json.dump(self.processed_files, f, indent=2, ensure_ascii=False)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem usage"""
        # Remove or replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'[^\w\s\-_\.]', '_', filename)
        filename = re.sub(r'\s+', '_', filename)
        return filename[:200]  # Limit length
    
    def extract_metadata_from_filename(self, filename: str) -> Dict:
        """Extract metadata from YouTube filename"""
        # Remove .mp3 extension
        name = filename.replace('.mp3', '')
        
        # Try to extract channel and title
        parts = name.split('/')
        if len(parts) >= 2:
            channel = parts[-2]
            title = parts[-1]
        else:
            channel = "Unknown Channel"
            title = name
        
        return {
            'channel': channel,
            'title': title,
            'original_filename': filename,
            'processed_date': datetime.now().isoformat()
        }
    
    def copy_audio_files(self) -> List[Path]:
        """Copy all MP3 files to the audio directory"""
        logger.info("Copying audio files...")
        
        copied_files = []
        
        # Find all MP3 files in source directory
        for mp3_file in self.source_dir.rglob("*.mp3"):
            # Create relative path for organization
            relative_path = mp3_file.relative_to(self.source_dir)
            
            # Sanitize the path
            sanitized_parts = []
            for part in relative_path.parts:
                sanitized_parts.append(self.sanitize_filename(part))
            
            # Create target path
            target_path = self.audio_dir / Path(*sanitized_parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file if not already exists
            if not target_path.exists():
                shutil.copy2(mp3_file, target_path)
                logger.info(f"Copied: {relative_path} -> {target_path.name}")
            
            copied_files.append(target_path)
        
        logger.info(f"Total audio files available: {len(copied_files)}")
        return copied_files
    
    def transcribe_audio(self, audio_path: Path) -> Dict:
        """Transcribe a single audio file"""
        logger.info(f"Transcribing: {audio_path.name}")
        
        try:
            result = self.model.transcribe(str(audio_path))
            
            return {
                'success': True,
                'text': result['text'],
                'segments': result.get('segments', []),
                'language': result.get('language', 'unknown'),
                'duration': sum(seg.get('end', 0) - seg.get('start', 0) 
                              for seg in result.get('segments', [])),
                'word_count': len(result['text'].split())
            }
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path.name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_markdown_transcript(self, audio_path: Path, transcription: Dict, metadata: Dict) -> str:
        """Create a beautiful markdown transcript"""
        
        if not transcription['success']:
            return None
        
        # Create markdown content
        markdown_content = f"""# {metadata['title']}

## 📺 Channel Information
**Channel:** {metadata['channel']}  
**Original File:** `{metadata['original_filename']}`  
**Processed:** {metadata['processed_date'][:10]}  

## 📊 Transcript Details
- **Language:** {transcription['language'].upper()}
- **Duration:** {transcription.get('duration', 0):.1f} seconds
- **Word Count:** {transcription.get('word_count', 0):,} words
- **Quality:** High-fidelity Whisper transcription

---

## 📝 Full Transcript

{transcription['text']}

---

## 🎯 Detailed Segments

"""
        
        # Add detailed segments if available
        if transcription.get('segments'):
            for i, segment in enumerate(transcription['segments'], 1):
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                
                # Format timestamp
                start_min, start_sec = divmod(int(start_time), 60)
                end_min, end_sec = divmod(int(end_time), 60)
                
                markdown_content += f"""### Segment {i}
**Time:** {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}

{text}

"""
        
        # Add footer
        markdown_content += f"""---

## 🔧 Technical Information

- **Transcription Model:** Whisper ({self.model.model_name if hasattr(self.model, 'model_name') else 'base'})
- **Processing Pipeline:** Pixelated Empathy Training Dataset
- **File Format:** MP3 Audio → Whisper → Markdown
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

*This transcript was automatically generated using OpenAI's Whisper model as part of the Pixelated Empathy training dataset preparation. The content is intended for educational and research purposes in developing empathetic AI systems.*
"""
        
        return markdown_content
    
    def save_transcript_data(self, audio_path: Path, transcription: Dict, metadata: Dict):
        """Save transcript in multiple formats"""
        
        # Create base filename
        base_name = audio_path.stem
        
        # Save JSON transcript
        json_path = self.transcripts_dir / f"{base_name}.json"
        transcript_data = {
            'metadata': metadata,
            'transcription': transcription,
            'audio_file': str(audio_path),
            'processed_timestamp': datetime.now().isoformat()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        # Save Markdown transcript
        if transcription['success']:
            markdown_content = self.create_markdown_transcript(audio_path, transcription, metadata)
            if markdown_content:
                markdown_path = self.markdown_dir / f"{base_name}.md"
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                logger.info(f"Saved transcript: {markdown_path.name}")
        
        # Update processed files log
        self.processed_files[str(audio_path)] = {
            'processed': True,
            'timestamp': datetime.now().isoformat(),
            'success': transcription['success'],
            'json_path': str(json_path),
            'markdown_path': str(markdown_path) if transcription['success'] else None
        }
    
    def process_all_videos(self):
        """Process all YouTube videos"""
        logger.info("=== YouTube Transcription Pipeline Started ===")
        
        # Copy audio files
        audio_files = self.copy_audio_files()
        
        # Filter out already processed files
        to_process = []
        for audio_file in audio_files:
            if str(audio_file) not in self.processed_files:
                to_process.append(audio_file)
            else:
                logger.info(f"Skipping already processed: {audio_file.name}")
        
        logger.info(f"Files to process: {len(to_process)}")
        logger.info(f"Already processed: {len(audio_files) - len(to_process)}")
        
        # Process each file
        successful = 0
        failed = 0
        
        for i, audio_path in enumerate(to_process, 1):
            logger.info(f"Processing [{i}/{len(to_process)}]: {audio_path.name}")
            
            # Extract metadata
            metadata = self.extract_metadata_from_filename(str(audio_path))
            
            # Transcribe
            transcription = self.transcribe_audio(audio_path)
            
            # Save results
            self.save_transcript_data(audio_path, transcription, metadata)
            
            if transcription['success']:
                successful += 1
            else:
                failed += 1
            
            # Save progress periodically
            if i % 5 == 0:
                self.save_processed_files()
        
        # Final save
        self.save_processed_files()
        
        # Generate summary report
        self.generate_summary_report(successful, failed, len(to_process))
        
        logger.info("=== YouTube Transcription Pipeline Completed ===")
    
    def generate_summary_report(self, successful: int, failed: int, total: int):
        """Generate a summary report"""
        
        report = {
            'pipeline_summary': {
                'total_files_processed': total,
                'successful_transcriptions': successful,
                'failed_transcriptions': failed,
                'success_rate': f"{(successful/total*100):.1f}%" if total > 0 else "0%",
                'processing_date': datetime.now().isoformat()
            },
            'output_locations': {
                'audio_files': str(self.audio_dir),
                'json_transcripts': str(self.transcripts_dir),
                'markdown_transcripts': str(self.markdown_dir)
            },
            'file_counts': {
                'total_audio_files': len(list(self.audio_dir.rglob("*.mp3"))),
                'json_transcripts': len(list(self.transcripts_dir.glob("*.json"))),
                'markdown_transcripts': len(list(self.markdown_dir.glob("*.md")))
            }
        }
        
        # Save report
        report_path = self.target_dir / "transcription_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 TRANSCRIPTION PIPELINE SUMMARY")
        print("="*60)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {(successful/total*100):.1f}%" if total > 0 else "0%")
        print(f"📁 Markdown Files: {len(list(self.markdown_dir.glob('*.md')))}")
        print(f"📄 JSON Files: {len(list(self.transcripts_dir.glob('*.json')))}")
        print(f"🎵 Audio Files: {len(list(self.audio_dir.rglob('*.mp3')))}")
        print("="*60)

def main():
    """Main execution function"""
    
    # Initialize pipeline
    pipeline = YouTubeTranscriptionPipeline(
        source_dir="/root/yt-downloader/Downloads",
        target_dir="/root/pixelated/ai/lightning/pixelated-training",
        model_size="base"  # Use "small", "medium", or "large" for better quality
    )
    
    # Process all videos
    pipeline.process_all_videos()

if __name__ == "__main__":
    main()
