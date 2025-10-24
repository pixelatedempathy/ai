"""MoE routing system for expert selection."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.core.models import CommunicationStyle
from src.core.logging import get_logger

logger = get_logger("voice_extraction.moe_router")


@dataclass
class RoutingDecision:
    primary_expert: CommunicationStyle
    confidence: float
    style_scores: Dict[CommunicationStyle, float]
    reasoning: List[str]
    blend_weights: Optional[Dict[CommunicationStyle, float]] = None


class MoERouter:
    """Routes user input to appropriate communication style expert."""
    
    def __init__(self):
        self.routing_patterns = {
            CommunicationStyle.THERAPEUTIC: {
                'triggers': [
                    'trauma', 'ptsd', 'abuse', 'hurt', 'pain', 'wounded',
                    'therapy', 'healing', 'recovery', 'flashback', 'trigger',
                    'childhood', 'family', 'relationship', 'attachment'
                ],
                'contexts': [
                    'past experiences', 'emotional pain', 'difficult memories',
                    'family issues', 'relationship problems', 'mental health'
                ],
                'weight': 1.0
            },
            CommunicationStyle.EDUCATIONAL: {
                'triggers': [
                    'what is', 'explain', 'understand', 'learn', 'definition',
                    'how does', 'why does', 'research', 'study', 'science'
                ],
                'contexts': [
                    'seeking information', 'wanting to learn', 'asking questions',
                    'need explanation', 'curious about concepts'
                ],
                'weight': 1.0
            },
            CommunicationStyle.EMPATHETIC: {
                'triggers': [
                    'feel', 'feeling', 'sad', 'angry', 'scared', 'alone',
                    'nobody understands', 'difficult', 'hard', 'struggle',
                    'overwhelmed', 'lost', 'confused', 'hopeless'
                ],
                'contexts': [
                    'emotional distress', 'seeking comfort', 'feeling isolated',
                    'need validation', 'expressing pain'
                ],
                'weight': 1.2  # Slightly higher weight for emotional needs
            },
            CommunicationStyle.PRACTICAL: {
                'triggers': [
                    'what can i do', 'how do i', 'help me', 'steps', 'action',
                    'practical', 'tool', 'technique', 'strategy', 'method'
                ],
                'contexts': [
                    'seeking solutions', 'want action steps', 'need tools',
                    'ready to act', 'looking for methods'
                ],
                'weight': 1.0
            }
        }
        
        self.conversation_context = []
        self.user_patterns = {}
    
    def route_message(self, message: str, user_id: str = None, 
                     conversation_history: List[str] = None) -> RoutingDecision:
        """Route message to appropriate expert(s)."""
        message_lower = message.lower()
        style_scores = {}
        reasoning = []
        
        # Analyze message for each style
        for style, patterns in self.routing_patterns.items():
            score = 0.0
            style_reasoning = []
            
            # Check triggers
            trigger_matches = 0
            for trigger in patterns['triggers']:
                if trigger in message_lower:
                    trigger_matches += 1
                    style_reasoning.append(f"trigger: '{trigger}'")
            
            # Calculate trigger score
            if trigger_matches > 0:
                trigger_score = min(trigger_matches / len(patterns['triggers']), 1.0)
                score += trigger_score * 0.7  # 70% weight for triggers
            
            # Check context patterns
            context_matches = 0
            for context in patterns['contexts']:
                context_words = context.split()
                if all(word in message_lower for word in context_words):
                    context_matches += 1
                    style_reasoning.append(f"context: '{context}'")
            
            # Calculate context score
            if context_matches > 0:
                context_score = min(context_matches / len(patterns['contexts']), 1.0)
                score += context_score * 0.3  # 30% weight for context
            
            # Apply style weight
            score *= patterns['weight']
            
            style_scores[style] = score
            if style_reasoning:
                reasoning.extend([f"{style.value}: {r}" for r in style_reasoning])
        
        # Consider conversation history
        if conversation_history:
            style_scores = self._adjust_for_history(style_scores, conversation_history)
            reasoning.append("adjusted for conversation history")
        
        # Consider user patterns
        if user_id and user_id in self.user_patterns:
            style_scores = self._adjust_for_user_patterns(style_scores, user_id)
            reasoning.append("adjusted for user patterns")
        
        # Determine primary expert
        if not any(score > 0 for score in style_scores.values()):
            # Default to empathetic if no clear signals
            primary_expert = CommunicationStyle.EMPATHETIC
            confidence = 0.3
            reasoning.append("defaulted to empathetic (no clear signals)")
        else:
            primary_expert = max(style_scores.items(), key=lambda x: x[1])[0]
            confidence = style_scores[primary_expert]
        
        # Determine if blending is needed
        blend_weights = None
        if confidence < 0.6:  # Low confidence, consider blending
            # Get top 2 experts
            sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_styles) >= 2 and sorted_styles[1][1] > 0.2:
                total_weight = sorted_styles[0][1] + sorted_styles[1][1]
                blend_weights = {
                    sorted_styles[0][0]: sorted_styles[0][1] / total_weight,
                    sorted_styles[1][0]: sorted_styles[1][1] / total_weight
                }
                reasoning.append(f"blending {sorted_styles[0][0].value} and {sorted_styles[1][0].value}")
        
        # Update user patterns
        if user_id:
            self._update_user_patterns(user_id, primary_expert)
        
        return RoutingDecision(
            primary_expert=primary_expert,
            confidence=confidence,
            style_scores=style_scores,
            reasoning=reasoning,
            blend_weights=blend_weights
        )
    
    def _adjust_for_history(self, scores: Dict[CommunicationStyle, float], 
                           history: List[str]) -> Dict[CommunicationStyle, float]:
        """Adjust scores based on conversation history."""
        # Simple implementation: boost consistency
        if len(history) >= 2:
            recent_styles = []
            for msg in history[-3:]:  # Last 3 messages
                msg_decision = self.route_message(msg)
                recent_styles.append(msg_decision.primary_expert)
            
            # If there's a pattern, slightly boost that style
            if recent_styles:
                most_common = max(set(recent_styles), key=recent_styles.count)
                if recent_styles.count(most_common) >= 2:
                    scores[most_common] *= 1.1
        
        return scores
    
    def _adjust_for_user_patterns(self, scores: Dict[CommunicationStyle, float], 
                                 user_id: str) -> Dict[CommunicationStyle, float]:
        """Adjust scores based on user's historical patterns."""
        user_prefs = self.user_patterns[user_id]
        
        for style, preference in user_prefs.items():
            if style in scores:
                # Slight boost for preferred styles
                scores[style] *= (1.0 + preference * 0.1)
        
        return scores
    
    def _update_user_patterns(self, user_id: str, chosen_style: CommunicationStyle):
        """Update user pattern tracking."""
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {style: 0.0 for style in CommunicationStyle}
        
        # Increment chosen style, decay others
        for style in CommunicationStyle:
            if style == chosen_style:
                self.user_patterns[user_id][style] = min(
                    self.user_patterns[user_id][style] + 0.1, 1.0
                )
            else:
                self.user_patterns[user_id][style] *= 0.95
    
    def get_routing_explanation(self, decision: RoutingDecision) -> str:
        """Generate human-readable explanation of routing decision."""
        explanation = f"Selected {decision.primary_expert.value} expert "
        explanation += f"(confidence: {decision.confidence:.2f})\n"
        
        if decision.reasoning:
            explanation += "Reasoning:\n"
            for reason in decision.reasoning:
                explanation += f"  - {reason}\n"
        
        if decision.blend_weights:
            explanation += "Blending weights:\n"
            for style, weight in decision.blend_weights.items():
                explanation += f"  - {style.value}: {weight:.2f}\n"
        
        return explanation
