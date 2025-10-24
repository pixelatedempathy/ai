"""Enhanced MoE router with better decision logic."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.core.models import CommunicationStyle
from src.voice_extraction.enhanced_style_analyzer import EnhancedStyleAnalyzer
from src.core.logging import get_logger

logger = get_logger("voice_extraction.enhanced_moe_router")


@dataclass
class EnhancedRoutingDecision:
    primary_expert: CommunicationStyle
    confidence: float
    style_scores: Dict[CommunicationStyle, float]
    reasoning: List[str]
    blend_weights: Optional[Dict[CommunicationStyle, float]] = None
    quality_score: float = 0.0
    indicators_found: Dict = None


class EnhancedMoERouter:
    """Enhanced router with better expert selection and context awareness."""
    
    def __init__(self):
        self.style_analyzer = EnhancedStyleAnalyzer()
        self.conversation_context = []
        self.user_patterns = {}
        
        # Enhanced routing rules
        self.routing_rules = {
            # Question patterns strongly indicate educational need
            'educational_triggers': [
                r'what is', r'what are', r'how does', r'why does', r'explain',
                r'definition', r'research', r'studies', r'science'
            ],
            
            # Action-seeking patterns indicate practical need
            'practical_triggers': [
                r'what can i do', r'how do i', r'steps', r'help me', r'technique',
                r'strategy', r'practical', r'action', r'tool', r'method'
            ],
            
            # Emotional distress patterns indicate empathetic need
            'empathetic_triggers': [
                r'i feel', r'feeling', r'nobody understands', r'alone', r'scared',
                r'overwhelmed', r'difficult', r'hard', r'struggle', r'pain'
            ],
            
            # Therapeutic concepts indicate therapeutic need
            'therapeutic_triggers': [
                r'trauma', r'healing', r'recovery', r'therapy', r'wounded',
                r'ptsd', r'abuse', r'childhood', r'attachment'
            ]
        }
        
        # Context-based adjustments
        self.context_weights = {
            'question_context': 1.5,  # Questions boost educational
            'action_context': 1.3,   # Action words boost practical
            'emotion_context': 1.4,  # Emotional words boost empathetic
            'clinical_context': 1.2  # Clinical terms boost therapeutic
        }
    
    def route_message(self, message: str, user_id: str = None, 
                     conversation_history: List[str] = None) -> EnhancedRoutingDecision:
        """Enhanced routing with multiple analysis layers."""
        
        # Get detailed style analysis
        analysis = self.style_analyzer.get_detailed_analysis(message)
        base_scores = analysis['scores'].copy()
        
        # Apply rule-based adjustments
        adjusted_scores = self._apply_routing_rules(message, base_scores)
        
        # Apply context adjustments
        if conversation_history:
            adjusted_scores = self._adjust_for_conversation_context(
                adjusted_scores, conversation_history
            )
        
        # Apply user pattern adjustments
        if user_id and user_id in self.user_patterns:
            adjusted_scores = self._adjust_for_user_patterns(adjusted_scores, user_id)
        
        # Determine primary expert and confidence
        primary_expert, confidence = self._select_primary_expert(adjusted_scores)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(message, analysis, adjusted_scores)
        
        # Determine blending strategy
        blend_weights = self._calculate_blend_weights(adjusted_scores, confidence)
        
        # Calculate overall quality score
        quality_score = self._calculate_routing_quality(analysis, confidence)
        
        # Update user patterns
        if user_id:
            self._update_user_patterns(user_id, primary_expert, confidence)
        
        return EnhancedRoutingDecision(
            primary_expert=primary_expert,
            confidence=confidence,
            style_scores=adjusted_scores,
            reasoning=reasoning,
            blend_weights=blend_weights,
            quality_score=quality_score,
            indicators_found=analysis.get('indicators_found', {})
        )
    
    def _apply_routing_rules(self, message: str, scores: Dict[CommunicationStyle, float]) -> Dict[CommunicationStyle, float]:
        """Apply rule-based routing adjustments."""
        import re
        message_lower = message.lower()
        adjusted_scores = scores.copy()
        
        # Check for strong question patterns (educational)
        question_patterns = self.routing_rules['educational_triggers']
        question_matches = sum(1 for pattern in question_patterns if re.search(pattern, message_lower))
        if question_matches > 0:
            adjusted_scores[CommunicationStyle.EDUCATIONAL] *= (1 + question_matches * 0.3)
        
        # Check for action-seeking patterns (practical)
        action_patterns = self.routing_rules['practical_triggers']
        action_matches = sum(1 for pattern in action_patterns if re.search(pattern, message_lower))
        if action_matches > 0:
            adjusted_scores[CommunicationStyle.PRACTICAL] *= (1 + action_matches * 0.4)
        
        # Check for emotional distress (empathetic)
        emotion_patterns = self.routing_rules['empathetic_triggers']
        emotion_matches = sum(1 for pattern in emotion_patterns if re.search(pattern, message_lower))
        if emotion_matches > 0:
            adjusted_scores[CommunicationStyle.EMPATHETIC] *= (1 + emotion_matches * 0.3)
        
        # Check for therapeutic concepts
        therapeutic_patterns = self.routing_rules['therapeutic_triggers']
        therapeutic_matches = sum(1 for pattern in therapeutic_patterns if re.search(pattern, message_lower))
        if therapeutic_matches > 0:
            adjusted_scores[CommunicationStyle.THERAPEUTIC] *= (1 + therapeutic_matches * 0.2)
        
        return adjusted_scores
    
    def _adjust_for_conversation_context(self, scores: Dict[CommunicationStyle, float], 
                                       history: List[str]) -> Dict[CommunicationStyle, float]:
        """Adjust scores based on conversation flow."""
        if not history or len(history) < 2:
            return scores
        
        adjusted_scores = scores.copy()
        
        # Analyze recent conversation flow
        recent_messages = history[-3:]  # Last 3 messages
        
        # Look for patterns in conversation flow
        educational_flow = sum(1 for msg in recent_messages if any(
            word in msg.lower() for word in ['what', 'how', 'why', 'explain']
        ))
        
        practical_flow = sum(1 for msg in recent_messages if any(
            word in msg.lower() for word in ['do', 'action', 'step', 'help']
        ))
        
        # Boost consistency but allow natural transitions
        if educational_flow >= 2:
            adjusted_scores[CommunicationStyle.EDUCATIONAL] *= 1.1
        
        if practical_flow >= 2:
            adjusted_scores[CommunicationStyle.PRACTICAL] *= 1.1
        
        return adjusted_scores
    
    def _select_primary_expert(self, scores: Dict[CommunicationStyle, float]) -> Tuple[CommunicationStyle, float]:
        """Select primary expert with confidence calculation."""
        if not any(score > 0 for score in scores.values()):
            # Default to empathetic for unclear messages
            return CommunicationStyle.EMPATHETIC, 0.3
        
        # Get top expert
        primary_expert = max(scores.items(), key=lambda x: x[1])[0]
        raw_confidence = scores[primary_expert]
        
        # Calculate relative confidence (how much better than second best)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[1] > 0:
            relative_confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
            confidence = min(raw_confidence * (1 + relative_confidence), 1.0)
        else:
            confidence = raw_confidence
        
        return primary_expert, confidence
    
    def _calculate_blend_weights(self, scores: Dict[CommunicationStyle, float], 
                               confidence: float) -> Optional[Dict[CommunicationStyle, float]]:
        """Calculate blend weights for multi-expert responses."""
        
        # Only blend if confidence is moderate and there's a strong second choice
        if confidence < 0.4 or confidence > 0.8:
            return None
        
        # Get top 2 experts
        sorted_experts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_experts) >= 2 and sorted_experts[1][1] > 0.3:
            total_weight = sorted_experts[0][1] + sorted_experts[1][1]
            
            if total_weight > 0:
                return {
                    sorted_experts[0][0]: sorted_experts[0][1] / total_weight,
                    sorted_experts[1][0]: sorted_experts[1][1] / total_weight
                }
        
        return None
    
    def _generate_reasoning(self, message: str, analysis: Dict, 
                          scores: Dict[CommunicationStyle, float]) -> List[str]:
        """Generate human-readable reasoning for routing decision."""
        reasoning = []
        
        # Add indicator-based reasoning
        indicators_found = analysis.get('indicators_found', {})
        for style_name, indicators in indicators_found.items():
            if indicators:
                top_indicators = [ind['pattern'] for ind in indicators[:2]]
                reasoning.append(f"{style_name}: found {top_indicators}")
        
        # Add cluster-based reasoning
        clusters = analysis.get('clusters_detected', {})
        if clusters:
            reasoning.append(f"semantic clusters: {list(clusters.keys())}")
        
        # Add rule-based reasoning
        import re
        message_lower = message.lower()
        
        if re.search(r'what|how|why|explain', message_lower):
            reasoning.append("question pattern detected")
        
        if re.search(r'steps|do|action|help', message_lower):
            reasoning.append("action-seeking pattern detected")
        
        if re.search(r'feel|feeling|difficult|overwhelmed', message_lower):
            reasoning.append("emotional expression detected")
        
        return reasoning
    
    def _calculate_routing_quality(self, analysis: Dict, confidence: float) -> float:
        """Calculate quality score for routing decision."""
        quality = confidence
        
        # Bonus for multiple indicators
        indicators_count = sum(len(inds) for inds in analysis.get('indicators_found', {}).values())
        if indicators_count >= 3:
            quality += 0.1
        
        # Bonus for semantic clusters
        clusters_count = len(analysis.get('clusters_detected', {}))
        if clusters_count >= 2:
            quality += 0.1
        
        # Bonus for good message length
        word_count = analysis.get('word_count', 0)
        if 10 <= word_count <= 100:
            quality += 0.05
        
        return min(quality, 1.0)
    
    def _adjust_for_user_patterns(self, scores: Dict[CommunicationStyle, float], 
                                user_id: str) -> Dict[CommunicationStyle, float]:
        """Adjust based on user's historical preferences."""
        user_prefs = self.user_patterns[user_id]
        adjusted_scores = scores.copy()
        
        for style, preference in user_prefs.items():
            if style in adjusted_scores and preference > 0.1:
                # Small boost for preferred styles
                adjusted_scores[style] *= (1.0 + preference * 0.05)
        
        return adjusted_scores
    
    def _update_user_patterns(self, user_id: str, chosen_style: CommunicationStyle, 
                            confidence: float):
        """Update user pattern tracking with confidence weighting."""
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {style: 0.0 for style in CommunicationStyle}
        
        # Update with confidence weighting
        update_amount = confidence * 0.1
        
        for style in CommunicationStyle:
            if style == chosen_style:
                self.user_patterns[user_id][style] = min(
                    self.user_patterns[user_id][style] + update_amount, 1.0
                )
            else:
                self.user_patterns[user_id][style] *= 0.98  # Slower decay
    
    def get_detailed_explanation(self, decision: EnhancedRoutingDecision) -> str:
        """Generate detailed explanation of routing decision."""
        explanation = f"Selected {decision.primary_expert.value} expert\\n"
        explanation += f"Confidence: {decision.confidence:.3f}\\n"
        explanation += f"Quality Score: {decision.quality_score:.3f}\\n\\n"
        
        explanation += "Style Scores:\\n"
        for style, score in decision.style_scores.items():
            explanation += f"  {style.value}: {score:.3f}\\n"
        
        if decision.reasoning:
            explanation += "\\nReasoning:\\n"
            for reason in decision.reasoning:
                explanation += f"  - {reason}\\n"
        
        if decision.blend_weights:
            explanation += "\\nBlending Strategy:\\n"
            for style, weight in decision.blend_weights.items():
                explanation += f"  {style.value}: {weight:.2f}\\n"
        
        return explanation
