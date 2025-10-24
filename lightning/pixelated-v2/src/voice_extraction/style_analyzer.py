"""Style analysis for Tim Fletcher transcripts."""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from src.core.models import CommunicationStyle, TrainingExample
from src.core.logging import get_logger

logger = get_logger("voice_extraction.style_analyzer")


@dataclass
class StyleSegment:
    text: str
    style: CommunicationStyle
    confidence: float
    indicators: List[str]
    metadata: Dict


class StyleAnalyzer:
    """Analyzes text for Tim Fletcher communication styles."""
    
    def __init__(self):
        self.style_patterns = {
            CommunicationStyle.THERAPEUTIC: {
                'keywords': [
                    'trauma', 'healing', 'recovery', 'wounded', 'hurt', 'pain',
                    'therapy', 'therapeutic', 'process', 'journey', 'work through',
                    'inner child', 'attachment', 'emotional', 'feelings'
                ],
                'phrases': [
                    'working through', 'healing process', 'trauma response',
                    'emotional regulation', 'attachment style', 'inner work'
                ],
                'sentence_starters': [
                    'when we experience', 'the healing process', 'trauma affects',
                    'recovery involves', 'therapeutic work'
                ]
            },
            CommunicationStyle.EDUCATIONAL: {
                'keywords': [
                    'understand', 'learn', 'concept', 'definition', 'explain',
                    'research', 'study', 'evidence', 'theory', 'model',
                    'psychology', 'neuroscience', 'brain', 'development'
                ],
                'phrases': [
                    'research shows', 'studies indicate', 'we know that',
                    'important to understand', 'the concept of', 'definition of'
                ],
                'sentence_starters': [
                    'research shows', 'studies have found', 'we understand that',
                    'the definition of', 'psychologically speaking'
                ]
            },
            CommunicationStyle.EMPATHETIC: {
                'keywords': [
                    'understand', 'feel', 'difficult', 'hard', 'struggle',
                    'compassion', 'empathy', 'support', 'care', 'gentle',
                    'safe', 'comfort', 'validate', 'acknowledge'
                ],
                'phrases': [
                    'I understand', 'that must be', 'it\'s difficult',
                    'you\'re not alone', 'that makes sense', 'I hear you'
                ],
                'sentence_starters': [
                    'I understand', 'that sounds', 'it must be',
                    'I can imagine', 'that\'s really'
                ]
            },
            CommunicationStyle.PRACTICAL: {
                'keywords': [
                    'do', 'action', 'step', 'practice', 'exercise', 'tool',
                    'technique', 'method', 'strategy', 'approach', 'try',
                    'implement', 'apply', 'use', 'start'
                ],
                'phrases': [
                    'you can', 'try this', 'here\'s what', 'practical step',
                    'action you can take', 'tool that helps'
                ],
                'sentence_starters': [
                    'you can', 'try', 'start by', 'one thing',
                    'a practical', 'here\'s how'
                ]
            }
        }
    
    def analyze_text(self, text: str) -> Dict[CommunicationStyle, float]:
        """Analyze text and return style scores."""
        text_lower = text.lower()
        scores = {}
        
        for style, patterns in self.style_patterns.items():
            score = 0.0
            total_indicators = 0
            
            # Check keywords
            for keyword in patterns['keywords']:
                count = text_lower.count(keyword)
                score += count * 1.0
                total_indicators += len(patterns['keywords'])
            
            # Check phrases (higher weight)
            for phrase in patterns['phrases']:
                count = text_lower.count(phrase)
                score += count * 2.0
                total_indicators += len(patterns['phrases'])
            
            # Check sentence starters (highest weight)
            for starter in patterns['sentence_starters']:
                if starter in text_lower:
                    score += 3.0
                total_indicators += len(patterns['sentence_starters'])
            
            # Normalize score
            if total_indicators > 0:
                scores[style] = min(score / total_indicators, 1.0)
            else:
                scores[style] = 0.0
        
        return scores
    
    def segment_transcript(self, transcript: str, min_segment_length: int = 200) -> List[StyleSegment]:
        """Segment transcript into style-specific chunks."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', transcript)
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            current_segment += sentence + ". "
            
            # Check if segment is long enough
            if len(current_segment) >= min_segment_length:
                # Analyze style
                style_scores = self.analyze_text(current_segment)
                
                # Find dominant style
                if style_scores:
                    dominant_style = max(style_scores.items(), key=lambda x: x[1])
                    
                    if dominant_style[1] > 0.1:  # Minimum confidence threshold
                        segment = StyleSegment(
                            text=current_segment.strip(),
                            style=dominant_style[0],
                            confidence=dominant_style[1],
                            indicators=self._get_indicators(current_segment, dominant_style[0]),
                            metadata={'style_scores': style_scores}
                        )
                        segments.append(segment)
                
                current_segment = ""
        
        return segments
    
    def _get_indicators(self, text: str, style: CommunicationStyle) -> List[str]:
        """Get specific indicators found in text for given style."""
        text_lower = text.lower()
        indicators = []
        patterns = self.style_patterns[style]
        
        for keyword in patterns['keywords']:
            if keyword in text_lower:
                indicators.append(f"keyword: {keyword}")
        
        for phrase in patterns['phrases']:
            if phrase in text_lower:
                indicators.append(f"phrase: {phrase}")
        
        for starter in patterns['sentence_starters']:
            if starter in text_lower:
                indicators.append(f"starter: {starter}")
        
        return indicators
    
    def process_transcript_file(self, file_path: Path) -> List[TrainingExample]:
        """Process a single transcript file into training examples."""
        logger.info(f"Processing transcript: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean content
            content = self._clean_transcript(content)
            
            # Segment by style
            segments = self.segment_transcript(content)
            
            # Convert to training examples
            examples = []
            for segment in segments:
                example = TrainingExample(
                    text=segment.text,
                    style=segment.style,
                    source=file_path.name,
                    quality_score=segment.confidence,
                    metadata={
                        'indicators': segment.indicators,
                        'style_scores': segment.metadata['style_scores']
                    }
                )
                examples.append(example)
            
            logger.info(f"Extracted {len(examples)} training examples from {file_path.name}")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return []
    
    def _clean_transcript(self, text: str) -> str:
        """Clean transcript text."""
        # Remove timestamps, speaker labels, etc.
        text = re.sub(r'\[\d+:\d+:\d+\]', '', text)
        text = re.sub(r'Speaker \d+:', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def export_training_data(self, examples: List[TrainingExample], output_dir: Path):
        """Export training examples for Colab training."""
        output_dir.mkdir(exist_ok=True)
        
        # Group by style
        style_groups = {}
        for example in examples:
            if example.style not in style_groups:
                style_groups[example.style] = []
            style_groups[example.style].append(example)
        
        # Export each style
        for style, style_examples in style_groups.items():
            output_file = output_dir / f"{style.value}_examples.json"
            
            data = [
                {
                    'text': ex.text,
                    'style': ex.style.value,
                    'source': ex.source,
                    'quality_score': ex.quality_score,
                    'metadata': ex.metadata
                }
                for ex in style_examples
            ]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(data)} {style.value} examples to {output_file}")
        
        # Export summary
        summary = {
            'total_examples': len(examples),
            'style_distribution': {
                style.value: len(style_examples) 
                for style, style_examples in style_groups.items()
            },
            'sources': list(set(ex.source for ex in examples))
        }
        
        with open(output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training data export complete: {summary}")


def process_transcript_directory(transcript_dir: str, output_dir: str):
    """Process all transcripts in directory."""
    analyzer = StyleAnalyzer()
    transcript_path = Path(transcript_dir)
    output_path = Path(output_dir)
    
    all_examples = []
    
    # Process all .txt files
    for file_path in transcript_path.glob("*.txt"):
        examples = analyzer.process_transcript_file(file_path)
        all_examples.extend(examples)
    
    # Export training data
    analyzer.export_training_data(all_examples, output_path)
    
    return len(all_examples)
