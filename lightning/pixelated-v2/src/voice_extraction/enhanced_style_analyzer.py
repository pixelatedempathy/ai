"""Enhanced style analysis with better pattern recognition."""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import Counter
from src.core.models import CommunicationStyle, TrainingExample
from src.core.logging import get_logger

logger = get_logger("voice_extraction.enhanced_style_analyzer")


@dataclass
class StyleIndicator:
    pattern: str
    weight: float
    pattern_type: str  # 'keyword', 'phrase', 'regex', 'semantic'


class EnhancedStyleAnalyzer:
    """Enhanced analyzer with better Tim Fletcher pattern recognition."""
    
    def __init__(self):
        self.style_indicators = {
            CommunicationStyle.THERAPEUTIC: [
                # Core therapeutic concepts
                StyleIndicator("trauma", 3.0, "keyword"),
                StyleIndicator("healing", 2.5, "keyword"),
                StyleIndicator("recovery", 2.5, "keyword"),
                StyleIndicator("wounded", 2.0, "keyword"),
                StyleIndicator("dysregulated", 3.0, "keyword"),
                StyleIndicator("nervous system", 2.5, "phrase"),
                StyleIndicator("inner child", 3.0, "phrase"),
                StyleIndicator("attachment", 2.0, "keyword"),
                StyleIndicator("regulate", 2.0, "keyword"),
                StyleIndicator("triggered", 2.0, "keyword"),
                
                # Therapeutic language patterns
                StyleIndicator(r"when we experience", 2.0, "regex"),
                StyleIndicator(r"the healing process", 2.5, "phrase"),
                StyleIndicator(r"working through", 2.0, "phrase"),
                StyleIndicator(r"parts of ourselves", 2.0, "phrase"),
                StyleIndicator(r"emotional regulation", 2.5, "phrase"),
                
                # Tim Fletcher specific
                StyleIndicator("complex ptsd", 3.0, "phrase"),
                StyleIndicator("developmental trauma", 3.0, "phrase"),
                StyleIndicator("survival mode", 2.0, "phrase"),
            ],
            
            CommunicationStyle.EDUCATIONAL: [
                # Research and evidence
                StyleIndicator("research shows", 3.0, "phrase"),
                StyleIndicator("studies indicate", 3.0, "phrase"),
                StyleIndicator("evidence suggests", 2.5, "phrase"),
                StyleIndicator("neuroscience", 2.0, "keyword"),
                StyleIndicator("psychology", 2.0, "keyword"),
                StyleIndicator("brain", 1.5, "keyword"),
                StyleIndicator("develops", 1.5, "keyword"),
                
                # Educational language
                StyleIndicator(r"what.*is", 2.0, "regex"),
                StyleIndicator(r"definition of", 2.0, "phrase"),
                StyleIndicator(r"understanding", 1.5, "keyword"),
                StyleIndicator(r"concept", 1.5, "keyword"),
                StyleIndicator(r"theory", 1.5, "keyword"),
                
                # Explanatory patterns
                StyleIndicator(r"this means", 2.0, "phrase"),
                StyleIndicator(r"in other words", 2.0, "phrase"),
                StyleIndicator(r"essentially", 1.5, "keyword"),
                StyleIndicator(r"basically", 1.5, "keyword"),
            ],
            
            CommunicationStyle.EMPATHETIC: [
                # Direct empathy
                StyleIndicator("I understand", 3.0, "phrase"),
                StyleIndicator("I hear you", 3.0, "phrase"),
                StyleIndicator("that must be", 2.5, "phrase"),
                StyleIndicator("I can imagine", 2.5, "phrase"),
                StyleIndicator("sounds difficult", 2.0, "phrase"),
                
                # Validation language
                StyleIndicator("makes sense", 2.0, "phrase"),
                StyleIndicator("completely normal", 2.0, "phrase"),
                StyleIndicator("you're not alone", 3.0, "phrase"),
                StyleIndicator("valid", 2.0, "keyword"),
                StyleIndicator("understandable", 2.0, "keyword"),
                
                # Emotional recognition
                StyleIndicator(r"feeling.*overwhelmed", 2.0, "regex"),
                StyleIndicator(r"pain.*real", 2.5, "regex"),
                StyleIndicator("struggle", 1.5, "keyword"),
                StyleIndicator("difficult", 1.5, "keyword"),
                StyleIndicator("hard", 1.0, "keyword"),
            ],
            
            CommunicationStyle.PRACTICAL: [
                # Action words
                StyleIndicator("steps you can take", 3.0, "phrase"),
                StyleIndicator("practical", 2.5, "keyword"),
                StyleIndicator("technique", 2.0, "keyword"),
                StyleIndicator("strategy", 2.0, "keyword"),
                StyleIndicator("tool", 2.0, "keyword"),
                StyleIndicator("method", 2.0, "keyword"),
                
                # Instructional language
                StyleIndicator(r"first.*try", 2.5, "regex"),
                StyleIndicator(r"start by", 2.5, "phrase"),
                StyleIndicator(r"you can", 2.0, "phrase"),
                StyleIndicator(r"here.*how", 2.0, "regex"),
                StyleIndicator("grounding", 2.5, "keyword"),
                
                # Implementation focus
                StyleIndicator("practice", 2.0, "keyword"),
                StyleIndicator("exercise", 2.0, "keyword"),
                StyleIndicator("implement", 2.0, "keyword"),
                StyleIndicator("apply", 1.5, "keyword"),
                StyleIndicator("use", 1.0, "keyword"),
            ]
        }
        
        # Semantic clusters for better context understanding
        self.semantic_clusters = {
            'trauma_cluster': ['trauma', 'ptsd', 'abuse', 'neglect', 'wounded', 'hurt'],
            'healing_cluster': ['healing', 'recovery', 'therapy', 'therapeutic', 'growth'],
            'emotion_cluster': ['feeling', 'emotion', 'sad', 'angry', 'scared', 'overwhelmed'],
            'brain_cluster': ['brain', 'neuroscience', 'nervous system', 'neurological'],
            'action_cluster': ['do', 'action', 'step', 'practice', 'try', 'implement']
        }
    
    def analyze_text(self, text: str) -> Dict[CommunicationStyle, float]:
        """Enhanced text analysis with multiple scoring methods."""
        text_lower = text.lower()
        scores = {style: 0.0 for style in CommunicationStyle}
        
        for style, indicators in self.style_indicators.items():
            style_score = 0.0
            
            for indicator in indicators:
                matches = self._find_pattern_matches(text_lower, indicator)
                if matches > 0:
                    # Weight by pattern strength and frequency
                    contribution = matches * indicator.weight
                    style_score += contribution
            
            # Add semantic cluster bonus
            cluster_bonus = self._calculate_cluster_bonus(text_lower, style)
            style_score += cluster_bonus
            
            # Normalize by text length (per 100 words)
            word_count = len(text.split())
            if word_count > 0:
                normalized_score = style_score / (word_count / 100)
                scores[style] = min(normalized_score, 1.0)  # Cap at 1.0
        
        return scores
    
    def _find_pattern_matches(self, text: str, indicator: StyleIndicator) -> int:
        """Find matches for a specific pattern."""
        if indicator.pattern_type == "keyword":
            return text.count(indicator.pattern)
        
        elif indicator.pattern_type == "phrase":
            return text.count(indicator.pattern)
        
        elif indicator.pattern_type == "regex":
            matches = re.findall(indicator.pattern, text, re.IGNORECASE)
            return len(matches)
        
        return 0
    
    def _calculate_cluster_bonus(self, text: str, style: CommunicationStyle) -> float:
        """Calculate bonus score based on semantic clusters."""
        bonus = 0.0
        
        # Define which clusters boost which styles
        style_clusters = {
            CommunicationStyle.THERAPEUTIC: ['trauma_cluster', 'healing_cluster'],
            CommunicationStyle.EDUCATIONAL: ['brain_cluster'],
            CommunicationStyle.EMPATHETIC: ['emotion_cluster'],
            CommunicationStyle.PRACTICAL: ['action_cluster']
        }
        
        relevant_clusters = style_clusters.get(style, [])
        
        for cluster_name in relevant_clusters:
            cluster_words = self.semantic_clusters[cluster_name]
            cluster_matches = sum(text.count(word) for word in cluster_words)
            
            if cluster_matches >= 2:  # Multiple related words
                bonus += 0.5 * (cluster_matches / len(cluster_words))
        
        return bonus
    
    def get_detailed_analysis(self, text: str) -> Dict:
        """Get detailed analysis with explanations."""
        text_lower = text.lower()
        analysis = {
            'scores': self.analyze_text(text),
            'indicators_found': {},
            'clusters_detected': {},
            'word_count': len(text.split())
        }
        
        # Find specific indicators
        for style, indicators in self.style_indicators.items():
            found_indicators = []
            
            for indicator in indicators:
                matches = self._find_pattern_matches(text_lower, indicator)
                if matches > 0:
                    found_indicators.append({
                        'pattern': indicator.pattern,
                        'matches': matches,
                        'weight': indicator.weight,
                        'contribution': matches * indicator.weight
                    })
            
            if found_indicators:
                analysis['indicators_found'][style.value] = found_indicators
        
        # Find semantic clusters
        for cluster_name, cluster_words in self.semantic_clusters.items():
            found_words = [word for word in cluster_words if word in text_lower]
            if found_words:
                analysis['clusters_detected'][cluster_name] = found_words
        
        return analysis
    
    def segment_transcript_enhanced(self, transcript: str, min_segment_length: int = 150) -> List:
        """Enhanced segmentation with better quality scoring."""
        # Split into sentences more intelligently
        sentences = re.split(r'[.!?]+(?=\s+[A-Z])', transcript)
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            current_segment += sentence + ". "
            
            # Check if segment is long enough and has good content
            if len(current_segment) >= min_segment_length:
                # Enhanced analysis
                analysis = self.get_detailed_analysis(current_segment)
                scores = analysis['scores']
                
                # Find dominant style with higher threshold
                if scores:
                    dominant_style = max(scores.items(), key=lambda x: x[1])
                    
                    # Higher quality threshold
                    if dominant_style[1] > 0.15:  # Increased from 0.1
                        # Calculate quality score based on multiple factors
                        quality_score = self._calculate_quality_score(analysis)
                        
                        if quality_score > 0.3:  # Quality threshold
                            segment_data = {
                                'text': current_segment.strip(),
                                'style': dominant_style[0],
                                'confidence': dominant_style[1],
                                'quality_score': quality_score,
                                'indicators': analysis['indicators_found'].get(dominant_style[0].value, []),
                                'metadata': {
                                    'style_scores': scores,
                                    'word_count': analysis['word_count'],
                                    'clusters': analysis['clusters_detected']
                                }
                            }
                            segments.append(segment_data)
                
                current_segment = ""
        
        return segments
    
    def _calculate_quality_score(self, analysis: Dict) -> float:
        """Calculate quality score for training example."""
        scores = analysis['scores']
        indicators = analysis['indicators_found']
        clusters = analysis['clusters_detected']
        word_count = analysis['word_count']
        
        # Base score from style confidence
        max_score = max(scores.values()) if scores else 0
        quality = max_score
        
        # Bonus for multiple indicators
        total_indicators = sum(len(inds) for inds in indicators.values())
        if total_indicators >= 3:
            quality += 0.2
        
        # Bonus for semantic clusters
        if len(clusters) >= 2:
            quality += 0.15
        
        # Penalty for very short segments
        if word_count < 50:
            quality *= 0.7
        
        # Bonus for good length
        if 100 <= word_count <= 300:
            quality += 0.1
        
        return min(quality, 1.0)
    
    def export_enhanced_training_data(self, segments: List, output_dir: Path):
        """Export with enhanced metadata and quality filtering."""
        output_dir.mkdir(exist_ok=True)
        
        # Filter by quality
        high_quality_segments = [s for s in segments if s['quality_score'] > 0.5]
        medium_quality_segments = [s for s in segments if 0.3 <= s['quality_score'] <= 0.5]
        
        logger.info(f"High quality: {len(high_quality_segments)}, Medium: {len(medium_quality_segments)}")
        
        # Export by style and quality
        for quality_level, segment_list in [("high", high_quality_segments), ("medium", medium_quality_segments)]:
            style_groups = {}
            
            for segment in segment_list:
                style = segment['style']
                if style not in style_groups:
                    style_groups[style] = []
                style_groups[style].append(segment)
            
            for style, style_segments in style_groups.items():
                filename = f"{style.value}_{quality_level}_quality.json"
                output_file = output_dir / filename
                
                export_data = []
                for segment in style_segments:
                    export_data.append({
                        'text': segment['text'],
                        'style': segment['style'].value,
                        'confidence': segment['confidence'],
                        'quality_score': segment['quality_score'],
                        'indicators': segment['indicators'],
                        'metadata': segment['metadata']
                    })
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Exported {len(export_data)} {quality_level} quality {style.value} examples")
        
        # Export summary with quality stats
        summary = {
            'total_segments': len(segments),
            'high_quality': len(high_quality_segments),
            'medium_quality': len(medium_quality_segments),
            'quality_distribution': {
                style.value: {
                    'high': len([s for s in high_quality_segments if s['style'] == style]),
                    'medium': len([s for s in medium_quality_segments if s['style'] == style])
                }
                for style in CommunicationStyle
            }
        }
        
        with open(output_dir / 'enhanced_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
