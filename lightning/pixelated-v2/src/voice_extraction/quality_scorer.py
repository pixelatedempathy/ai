"""Advanced quality scoring for training examples."""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from src.core.models import CommunicationStyle
from src.core.logging import get_logger

logger = get_logger("voice_extraction.quality_scorer")


@dataclass
class QualityMetrics:
    content_quality: float
    style_clarity: float
    therapeutic_appropriateness: float
    linguistic_quality: float
    overall_score: float
    issues: List[str]
    strengths: List[str]


class AdvancedQualityScorer:
    """Advanced quality scoring for training examples."""
    
    def __init__(self):
        # Quality criteria weights
        self.weights = {
            'content_quality': 0.3,
            'style_clarity': 0.25,
            'therapeutic_appropriateness': 0.25,
            'linguistic_quality': 0.2
        }
        
        # Therapeutic appropriateness criteria
        self.therapeutic_criteria = {
            'positive_indicators': [
                'validation', 'empathy', 'understanding', 'support',
                'non-judgmental', 'compassionate', 'healing-focused',
                'strength-based', 'hope', 'growth', 'resilience'
            ],
            'negative_indicators': [
                'blame', 'shame', 'judgment', 'dismissive', 'minimizing',
                'toxic positivity', 'should', 'just get over it', 'weakness'
            ],
            'clinical_appropriateness': [
                'evidence-based', 'trauma-informed', 'attachment-aware',
                'developmentally appropriate', 'culturally sensitive'
            ]
        }
        
        # Linguistic quality patterns
        self.linguistic_patterns = {
            'good_patterns': [
                r'[A-Z][^.!?]*[.!?]',  # Proper sentences
                r'\b(and|but|however|therefore|because)\b',  # Connectors
                r'\b(when|if|while|although)\b',  # Complex structures
            ],
            'poor_patterns': [
                r'\b(um|uh|like|you know)\b',  # Filler words
                r'[.]{2,}',  # Multiple periods
                r'[A-Z]{3,}',  # All caps words
                r'\b\w{1,2}\b.*\b\w{1,2}\b',  # Too many short words
            ]
        }
    
    def score_training_example(self, text: str, style: CommunicationStyle, 
                             indicators: List[Dict] = None) -> QualityMetrics:
        """Comprehensive quality scoring for training example."""
        
        # Calculate individual quality metrics
        content_quality = self._assess_content_quality(text, style)
        style_clarity = self._assess_style_clarity(text, style, indicators)
        therapeutic_appropriateness = self._assess_therapeutic_appropriateness(text, style)
        linguistic_quality = self._assess_linguistic_quality(text)
        
        # Calculate weighted overall score
        overall_score = (
            content_quality * self.weights['content_quality'] +
            style_clarity * self.weights['style_clarity'] +
            therapeutic_appropriateness * self.weights['therapeutic_appropriateness'] +
            linguistic_quality * self.weights['linguistic_quality']
        )
        
        # Identify issues and strengths
        issues, strengths = self._identify_issues_and_strengths(
            text, style, content_quality, style_clarity, 
            therapeutic_appropriateness, linguistic_quality
        )
        
        return QualityMetrics(
            content_quality=content_quality,
            style_clarity=style_clarity,
            therapeutic_appropriateness=therapeutic_appropriateness,
            linguistic_quality=linguistic_quality,
            overall_score=overall_score,
            issues=issues,
            strengths=strengths
        )
    
    def _assess_content_quality(self, text: str, style: CommunicationStyle) -> float:
        """Assess the quality and depth of content."""
        score = 0.5  # Base score
        
        word_count = len(text.split())
        
        # Length appropriateness
        if 50 <= word_count <= 200:
            score += 0.2
        elif 30 <= word_count <= 300:
            score += 0.1
        elif word_count < 20:
            score -= 0.2
        
        # Content depth indicators
        depth_indicators = [
            'because', 'therefore', 'however', 'although', 'while',
            'specifically', 'particularly', 'especially', 'importantly'
        ]
        
        depth_score = sum(1 for indicator in depth_indicators if indicator in text.lower())
        score += min(depth_score * 0.05, 0.2)
        
        # Specificity vs generality balance
        specific_words = len(re.findall(r'\b(this|that|these|those|here|there)\b', text.lower()))
        general_words = len(re.findall(r'\b(always|never|everyone|everything|all)\b', text.lower()))
        
        if specific_words > general_words:
            score += 0.1
        elif general_words > specific_words * 2:
            score -= 0.1
        
        return min(score, 1.0)
    
    def _assess_style_clarity(self, text: str, style: CommunicationStyle, 
                            indicators: List[Dict] = None) -> float:
        """Assess how clearly the text represents the intended style."""
        score = 0.3  # Base score
        
        # Indicator strength
        if indicators:
            total_weight = sum(ind.get('weight', 1.0) for ind in indicators)
            indicator_score = min(total_weight / 10.0, 0.4)  # Max 0.4 from indicators
            score += indicator_score
        
        # Style-specific clarity checks
        if style == CommunicationStyle.THERAPEUTIC:
            therapeutic_terms = ['trauma', 'healing', 'recovery', 'therapy', 'therapeutic']
            if any(term in text.lower() for term in therapeutic_terms):
                score += 0.2
        
        elif style == CommunicationStyle.EDUCATIONAL:
            educational_terms = ['research', 'study', 'evidence', 'shows', 'indicates']
            if any(term in text.lower() for term in educational_terms):
                score += 0.2
        
        elif style == CommunicationStyle.EMPATHETIC:
            empathetic_terms = ['understand', 'feel', 'difficult', 'valid', 'makes sense']
            if any(term in text.lower() for term in empathetic_terms):
                score += 0.2
        
        elif style == CommunicationStyle.PRACTICAL:
            practical_terms = ['step', 'action', 'try', 'practice', 'technique']
            if any(term in text.lower() for term in practical_terms):
                score += 0.2
        
        # Consistency check (no conflicting style signals)
        conflicting_signals = self._detect_style_conflicts(text, style)
        if conflicting_signals:
            score -= 0.1 * len(conflicting_signals)
        
        return max(min(score, 1.0), 0.0)
    
    def _assess_therapeutic_appropriateness(self, text: str, style: CommunicationStyle) -> float:
        """Assess therapeutic appropriateness and safety."""
        score = 0.6  # Base score (assume generally appropriate)
        text_lower = text.lower()
        
        # Check for positive therapeutic indicators
        positive_count = sum(1 for indicator in self.therapeutic_criteria['positive_indicators'] 
                           if indicator in text_lower)
        score += min(positive_count * 0.05, 0.2)
        
        # Check for negative therapeutic indicators (red flags)
        negative_count = sum(1 for indicator in self.therapeutic_criteria['negative_indicators'] 
                           if indicator in text_lower)
        score -= negative_count * 0.15  # Heavy penalty for inappropriate content
        
        # Check for clinical appropriateness
        clinical_count = sum(1 for indicator in self.therapeutic_criteria['clinical_appropriateness'] 
                           if indicator in text_lower)
        score += min(clinical_count * 0.03, 0.1)
        
        # Style-specific appropriateness
        if style == CommunicationStyle.THERAPEUTIC:
            # Should be trauma-informed and healing-focused
            if any(word in text_lower for word in ['trauma-informed', 'healing', 'recovery']):
                score += 0.1
        
        elif style == CommunicationStyle.EMPATHETIC:
            # Should be validating and non-judgmental
            if any(word in text_lower for word in ['valid', 'understand', 'makes sense']):
                score += 0.1
            if any(word in text_lower for word in ['wrong', 'should', 'need to']):
                score -= 0.1
        
        return max(min(score, 1.0), 0.0)
    
    def _assess_linguistic_quality(self, text: str) -> float:
        """Assess linguistic quality and readability."""
        score = 0.5  # Base score
        
        # Sentence structure quality
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if valid_sentences:
            avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
            
            # Optimal sentence length (10-20 words)
            if 10 <= avg_sentence_length <= 20:
                score += 0.2
            elif 8 <= avg_sentence_length <= 25:
                score += 0.1
        
        # Check for good linguistic patterns
        for pattern in self.linguistic_patterns['good_patterns']:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += min(matches * 0.02, 0.1)
        
        # Check for poor linguistic patterns
        for pattern in self.linguistic_patterns['poor_patterns']:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score -= matches * 0.05
        
        # Grammar and punctuation basic check
        if re.search(r'[A-Z].*[.!?]$', text.strip()):
            score += 0.05  # Proper capitalization and ending
        
        # Vocabulary diversity (simple measure)
        words = text.lower().split()
        if len(words) > 10:
            unique_words = len(set(words))
            diversity_ratio = unique_words / len(words)
            if diversity_ratio > 0.7:
                score += 0.1
            elif diversity_ratio < 0.5:
                score -= 0.05
        
        return max(min(score, 1.0), 0.0)
    
    def _detect_style_conflicts(self, text: str, intended_style: CommunicationStyle) -> List[str]:
        """Detect conflicting style signals."""
        conflicts = []
        text_lower = text.lower()
        
        # Define conflicting patterns for each style
        style_conflicts = {
            CommunicationStyle.THERAPEUTIC: {
                'overly_clinical': ['diagnosis', 'disorder', 'pathology'],
                'too_casual': ['whatever', 'just deal with it', 'get over it']
            },
            CommunicationStyle.EDUCATIONAL: {
                'too_emotional': ['i feel', 'emotionally', 'heartbreaking'],
                'too_prescriptive': ['you must', 'you should', 'you need to']
            },
            CommunicationStyle.EMPATHETIC: {
                'too_clinical': ['research shows', 'studies indicate', 'data suggests'],
                'too_directive': ['you should', 'you must', 'the solution is']
            },
            CommunicationStyle.PRACTICAL: {
                'too_theoretical': ['conceptually', 'theoretically', 'in theory'],
                'too_vague': ['somehow', 'maybe', 'perhaps', 'might help']
            }
        }
        
        if intended_style in style_conflicts:
            for conflict_type, patterns in style_conflicts[intended_style].items():
                if any(pattern in text_lower for pattern in patterns):
                    conflicts.append(conflict_type)
        
        return conflicts
    
    def _identify_issues_and_strengths(self, text: str, style: CommunicationStyle,
                                     content_quality: float, style_clarity: float,
                                     therapeutic_appropriateness: float, 
                                     linguistic_quality: float) -> Tuple[List[str], List[str]]:
        """Identify specific issues and strengths."""
        issues = []
        strengths = []
        
        # Content quality issues/strengths
        word_count = len(text.split())
        if content_quality < 0.4:
            if word_count < 30:
                issues.append("too_short")
            elif word_count > 300:
                issues.append("too_long")
            else:
                issues.append("lacks_depth")
        elif content_quality > 0.7:
            strengths.append("good_content_depth")
        
        # Style clarity issues/strengths
        if style_clarity < 0.4:
            issues.append("unclear_style_signals")
        elif style_clarity > 0.7:
            strengths.append("clear_style_representation")
        
        # Therapeutic appropriateness issues/strengths
        if therapeutic_appropriateness < 0.4:
            issues.append("therapeutic_concerns")
        elif therapeutic_appropriateness > 0.8:
            strengths.append("therapeutically_sound")
        
        # Linguistic quality issues/strengths
        if linguistic_quality < 0.4:
            issues.append("poor_language_quality")
        elif linguistic_quality > 0.7:
            strengths.append("good_language_quality")
        
        return issues, strengths
    
    def filter_by_quality(self, examples: List[Dict], min_score: float = 0.6) -> Tuple[List[Dict], List[Dict]]:
        """Filter training examples by quality score."""
        high_quality = []
        low_quality = []
        
        for example in examples:
            # Score the example
            quality_metrics = self.score_training_example(
                example['text'], 
                CommunicationStyle(example['style']),
                example.get('indicators', [])
            )
            
            # Add quality info to example
            example['quality_metrics'] = {
                'overall_score': quality_metrics.overall_score,
                'content_quality': quality_metrics.content_quality,
                'style_clarity': quality_metrics.style_clarity,
                'therapeutic_appropriateness': quality_metrics.therapeutic_appropriateness,
                'linguistic_quality': quality_metrics.linguistic_quality,
                'issues': quality_metrics.issues,
                'strengths': quality_metrics.strengths
            }
            
            if quality_metrics.overall_score >= min_score:
                high_quality.append(example)
            else:
                low_quality.append(example)
        
        return high_quality, low_quality
