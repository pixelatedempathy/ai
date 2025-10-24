"""Transcript data cleaning and standardization."""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple
from src.core.logging import get_logger

logger = get_logger("voice_extraction.transcript_cleaner")


class TranscriptCleaner:
    """Clean and standardize Tim Fletcher transcripts."""
    
    def __init__(self):
        # Common patterns to clean
        self.cleaning_patterns = [
            # Remove timestamps and markers
            (r'\[\d+:\d+:\d+\]', ''),
            (r'\(\d+:\d+:\d+\)', ''),
            (r'<\d+:\d+:\d+>', ''),
            
            # Remove speaker labels
            (r'^Speaker \d+:\s*', ''),
            (r'^Tim:\s*', ''),
            (r'^Tim Fletcher:\s*', ''),
            (r'^Interviewer:\s*', ''),
            
            # Remove filler words and sounds
            (r'\b(um|uh|ah|er|hmm)\b', ''),
            (r'\[inaudible\]', ''),
            (r'\[unclear\]', ''),
            (r'\[crosstalk\]', ''),
            (r'\[laughter\]', ''),
            (r'\[applause\]', ''),
            
            # Fix common transcription errors
            (r'\bgonna\b', 'going to'),
            (r'\bwanna\b', 'want to'),
            (r'\bgotta\b', 'got to'),
            (r'\bkinda\b', 'kind of'),
            (r'\bsorta\b', 'sort of'),
            
            # Standardize contractions
            (r"won't", "will not"),
            (r"can't", "cannot"),
            (r"n't", " not"),
            (r"'re", " are"),
            (r"'ve", " have"),
            (r"'ll", " will"),
            (r"'d", " would"),
            
            # Remove multiple spaces
            (r'\s+', ' '),
            
            # Remove leading/trailing whitespace per line
            (r'^\s+|\s+$', ''),
        ]
        
        # Sentence ending patterns
        self.sentence_endings = r'[.!?]'
        
        # Paragraph break indicators
        self.paragraph_breaks = [
            'Now,', 'So,', 'And so,', 'But', 'However,', 'Therefore,',
            'The next thing', 'Another thing', 'Let me', 'I want to'
        ]
    
    def clean_single_transcript(self, file_path: Path) -> Tuple[str, Dict]:
        """Clean a single transcript file."""
        logger.info(f"Cleaning transcript: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_length = len(content)
            original_lines = len(content.split('\n'))
            
            # Apply cleaning patterns
            cleaned_content = content
            for pattern, replacement in self.cleaning_patterns:
                cleaned_content = re.sub(pattern, replacement, cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
            
            # Fix sentence structure
            cleaned_content = self._fix_sentence_structure(cleaned_content)
            
            # Create proper paragraphs
            cleaned_content = self._create_paragraphs(cleaned_content)
            
            # Final cleanup
            cleaned_content = self._final_cleanup(cleaned_content)
            
            # Generate cleaning stats
            stats = {
                'original_length': original_length,
                'cleaned_length': len(cleaned_content),
                'original_lines': original_lines,
                'cleaned_lines': len(cleaned_content.split('\n')),
                'reduction_percent': round((1 - len(cleaned_content) / original_length) * 100, 2),
                'word_count': len(cleaned_content.split()),
                'sentence_count': len(re.findall(self.sentence_endings, cleaned_content))
            }
            
            return cleaned_content, stats
            
        except Exception as e:
            logger.error(f"Failed to clean {file_path}: {e}")
            return "", {}
    
    def _fix_sentence_structure(self, text: str) -> str:
        """Fix sentence structure and capitalization."""
        # Split into sentences
        sentences = re.split(r'([.!?]+)', text)
        
        fixed_sentences = []
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence:
                    # Capitalize first letter
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                    
                    # Add punctuation if missing
                    if i + 1 < len(sentences):
                        punctuation = sentences[i + 1]
                    else:
                        punctuation = '.' if not sentence.endswith(('.', '!', '?')) else ''
                    
                    fixed_sentences.append(sentence + punctuation)
        
        return ' '.join(fixed_sentences)
    
    def _create_paragraphs(self, text: str) -> str:
        """Create logical paragraph breaks."""
        sentences = re.split(r'([.!?]+\s*)', text)
        
        paragraphs = []
        current_paragraph = []
        
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence:
                    # Check if this sentence should start a new paragraph
                    should_break = any(sentence.startswith(indicator) for indicator in self.paragraph_breaks)
                    
                    if should_break and current_paragraph:
                        # End current paragraph
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
                    
                    # Add punctuation
                    punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
                    current_paragraph.append(sentence + punctuation)
        
        # Add final paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return '\n\n'.join(paragraphs)
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass."""
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Join with proper spacing
        text = '\n\n'.join(lines)
        
        # Fix common issues
        text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after punctuation
        text = re.sub(r'\n\n+', '\n\n', text)  # Max 2 newlines
        
        return text.strip()
    
    def clean_all_transcripts(self, input_dir: Path, output_dir: Path) -> Dict:
        """Clean all transcripts in directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        stats_summary = {
            'total_files': 0,
            'successful_cleanings': 0,
            'failed_cleanings': 0,
            'total_original_length': 0,
            'total_cleaned_length': 0,
            'total_word_count': 0,
            'total_sentence_count': 0,
            'file_stats': {}
        }
        
        # Process all transcript files
        for file_path in input_dir.glob("*.txt"):
            stats_summary['total_files'] += 1
            
            cleaned_content, file_stats = self.clean_single_transcript(file_path)
            
            if cleaned_content and file_stats:
                # Save cleaned transcript
                output_file = output_dir / f"cleaned_{file_path.name}"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                # Update summary stats
                stats_summary['successful_cleanings'] += 1
                stats_summary['total_original_length'] += file_stats['original_length']
                stats_summary['total_cleaned_length'] += file_stats['cleaned_length']
                stats_summary['total_word_count'] += file_stats['word_count']
                stats_summary['total_sentence_count'] += file_stats['sentence_count']
                stats_summary['file_stats'][file_path.name] = file_stats
                
                logger.info(f"Cleaned {file_path.name}: {file_stats['word_count']} words, "
                           f"{file_stats['reduction_percent']}% reduction")
            else:
                stats_summary['failed_cleanings'] += 1
                logger.error(f"Failed to clean {file_path.name}")
        
        # Calculate overall stats
        if stats_summary['total_original_length'] > 0:
            stats_summary['overall_reduction_percent'] = round(
                (1 - stats_summary['total_cleaned_length'] / stats_summary['total_original_length']) * 100, 2
            )
        
        logger.info(f"Cleaning complete: {stats_summary['successful_cleanings']}/{stats_summary['total_files']} files")
        logger.info(f"Overall reduction: {stats_summary.get('overall_reduction_percent', 0)}%")
        logger.info(f"Total words: {stats_summary['total_word_count']:,}")
        logger.info(f"Total sentences: {stats_summary['total_sentence_count']:,}")
        
        return stats_summary
    
    def validate_cleaning_quality(self, input_dir: Path, output_dir: Path) -> Dict:
        """Validate the quality of cleaned transcripts."""
        validation_results = {
            'files_checked': 0,
            'quality_issues': [],
            'average_sentence_length': 0,
            'short_sentences': 0,
            'long_sentences': 0,
            'formatting_issues': 0
        }
        
        sentence_lengths = []
        
        for cleaned_file in output_dir.glob("cleaned_*.txt"):
            validation_results['files_checked'] += 1
            
            with open(cleaned_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check sentence lengths
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    word_count = len(sentence.split())
                    sentence_lengths.append(word_count)
                    
                    if word_count < 3:
                        validation_results['short_sentences'] += 1
                    elif word_count > 50:
                        validation_results['long_sentences'] += 1
            
            # Check for formatting issues
            if re.search(r'\s{3,}', content):  # Multiple spaces
                validation_results['formatting_issues'] += 1
                validation_results['quality_issues'].append(f"{cleaned_file.name}: Multiple spaces found")
            
            if re.search(r'\n{3,}', content):  # Multiple newlines
                validation_results['formatting_issues'] += 1
                validation_results['quality_issues'].append(f"{cleaned_file.name}: Multiple newlines found")
        
        if sentence_lengths:
            validation_results['average_sentence_length'] = round(sum(sentence_lengths) / len(sentence_lengths), 2)
        
        return validation_results


def clean_tim_fletcher_transcripts():
    """Main function to clean Tim Fletcher transcripts."""
    cleaner = TranscriptCleaner()
    
    input_dir = Path("/root/pixelated/ai/lightning/pixelated-v2/data/transcripts")
    output_dir = Path("/root/pixelated/ai/lightning/pixelated-v2/data/processed")
    
    # Clean all transcripts
    stats = cleaner.clean_all_transcripts(input_dir, output_dir)
    
    # Validate cleaning quality
    validation = cleaner.validate_cleaning_quality(input_dir, output_dir)
    
    # Save stats
    import json
    with open(output_dir / "cleaning_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    with open(output_dir / "validation_results.json", 'w') as f:
        json.dump(validation, f, indent=2)
    
    return stats, validation


if __name__ == "__main__":
    clean_tim_fletcher_transcripts()
