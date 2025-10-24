"""Conversation service integrating MoE system."""

import time
from typing import Dict, List, Optional
from src.voice_extraction.moe_router import MoERouter, RoutingDecision
from src.voice_extraction.external_inference import (
    create_inference_client, InferenceRequest, InferenceResponse
)
from src.core.models import ConversationSession, ModelResponse, CommunicationStyle
from src.database.operations import ConversationOperations
from src.core.logging import get_logger

logger = get_logger("api.services.conversation")


class ConversationService:
    """Service for handling therapeutic conversations with MoE system."""
    
    def __init__(self, use_mock_inference: bool = True):
        self.router = MoERouter()
        self.use_mock_inference = use_mock_inference
        
    async def generate_response(self, message: str, session_id: str, 
                              user_id: str) -> ModelResponse:
        """Generate therapeutic response using MoE system."""
        start_time = time.time()
        
        try:
            # Get conversation history for context
            session = ConversationOperations.get_session(session_id)
            conversation_history = []
            if session and session.messages:
                # Get last few user messages for context
                user_messages = [
                    msg.content for msg in session.messages[-6:] 
                    if msg.role == "user"
                ]
                conversation_history = user_messages
            
            # Route message to appropriate expert
            routing_decision = self.router.route_message(
                message=message,
                user_id=user_id,
                conversation_history=conversation_history
            )
            
            logger.info(f"Routed to {routing_decision.primary_expert.value} expert "
                       f"(confidence: {routing_decision.confidence:.2f})")
            
            # Generate response using external inference
            response_content = await self._generate_expert_response(
                message=message,
                routing_decision=routing_decision,
                conversation_history=conversation_history
            )
            
            processing_time = time.time() - start_time
            
            # Create model response
            model_response = ModelResponse(
                content=response_content,
                confidence=routing_decision.confidence,
                primary_style=routing_decision.primary_expert,
                style_scores=routing_decision.style_scores,
                processing_time=processing_time,
                metadata={
                    'routing_reasoning': routing_decision.reasoning,
                    'blend_weights': routing_decision.blend_weights,
                    'conversation_length': len(conversation_history) if conversation_history else 0
                }
            )
            
            logger.info(f"Generated response in {processing_time:.2f}s")
            return model_response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Fallback response
            return ModelResponse(
                content="I'm here to listen and support you. Could you tell me more about what's on your mind?",
                confidence=0.3,
                primary_style=CommunicationStyle.EMPATHETIC,
                style_scores={style: 0.25 for style in CommunicationStyle},
                processing_time=time.time() - start_time,
                metadata={'error': str(e), 'fallback': True}
            )
    
    async def _generate_expert_response(self, message: str, 
                                      routing_decision: RoutingDecision,
                                      conversation_history: List[str]) -> str:
        """Generate response using the selected expert(s)."""
        
        # Prepare context from conversation history
        context = None
        if conversation_history:
            context = " ".join(conversation_history[-3:])  # Last 3 messages
        
        async with create_inference_client(use_mock=self.use_mock_inference) as client:
            
            # Check if we need to blend experts
            if routing_decision.blend_weights:
                return await self._generate_blended_response(
                    client, message, routing_decision, context
                )
            else:
                return await self._generate_single_expert_response(
                    client, message, routing_decision.primary_expert, context
                )
    
    async def _generate_single_expert_response(self, client, message: str, 
                                             expert: CommunicationStyle, 
                                             context: Optional[str]) -> str:
        """Generate response from single expert."""
        
        request = InferenceRequest(
            text=message,
            expert=expert,
            max_length=200,
            temperature=0.7,
            context=context,
            metadata={'single_expert': True}
        )
        
        response = await client.generate_response(request)
        return response.generated_text
    
    async def _generate_blended_response(self, client, message: str,
                                       routing_decision: RoutingDecision,
                                       context: Optional[str]) -> str:
        """Generate blended response from multiple experts."""
        
        responses = {}
        
        # Generate response from each expert in blend
        for expert, weight in routing_decision.blend_weights.items():
            request = InferenceRequest(
                text=message,
                expert=expert,
                max_length=200,
                temperature=0.7,
                context=context,
                metadata={'blend_weight': weight}
            )
            
            response = await client.generate_response(request)
            responses[expert] = {
                'text': response.generated_text,
                'weight': weight
            }
        
        # Simple blending: choose primary expert's response but mention blend
        primary_expert = max(routing_decision.blend_weights.items(), key=lambda x: x[1])[0]
        primary_response = responses[primary_expert]['text']
        
        # In a more sophisticated implementation, we might actually blend the text
        # For now, we use the primary expert's response
        logger.info(f"Blended response using {primary_expert.value} as primary")
        
        return primary_response
    
    def get_routing_explanation(self, routing_decision: RoutingDecision) -> str:
        """Get human-readable explanation of routing decision."""
        return self.router.get_routing_explanation(routing_decision)
