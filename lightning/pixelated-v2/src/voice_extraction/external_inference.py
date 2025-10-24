"""External inference client for GPU-based model serving."""

import aiohttp
import asyncio
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from src.core.models import CommunicationStyle, ModelResponse
from src.core.config import config
from src.core.logging import get_logger

logger = get_logger("voice_extraction.external_inference")


@dataclass
class InferenceRequest:
    text: str
    expert: CommunicationStyle
    max_length: int = 200
    temperature: float = 0.7
    context: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class InferenceResponse:
    generated_text: str
    confidence: float
    processing_time: float
    model_info: Dict[str, Any]
    metadata: Dict[str, Any] = None


class ExternalInferenceClient:
    """Client for external GPU inference service."""
    
    def __init__(self, base_url: str = None, timeout: int = 30):
        self.base_url = base_url or "http://localhost:8001"  # Default Colab/external service
        self.timeout = timeout
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def generate_response(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response using external inference service."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        payload = {
            "text": request.text,
            "expert": request.expert.value,
            "max_length": request.max_length,
            "temperature": request.temperature,
            "context": request.context,
            "metadata": request.metadata or {}
        }
        
        logger.info(f"Sending inference request for {request.expert.value} expert")
        
        try:
            async with self.session.post(
                f"{self.base_url}/inference",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Inference service error {response.status}: {error_text}")
                
                result = await response.json()
                
                return InferenceResponse(
                    generated_text=result["generated_text"],
                    confidence=result.get("confidence", 0.5),
                    processing_time=result.get("processing_time", 0.0),
                    model_info=result.get("model_info", {}),
                    metadata=result.get("metadata", {})
                )
                
        except asyncio.TimeoutError:
            logger.error("Inference request timed out")
            raise Exception("Inference service timeout")
        except Exception as e:
            logger.error(f"Inference request failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if external inference service is healthy."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except:
            return False
    
    async def get_available_experts(self) -> List[str]:
        """Get list of available expert models."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            async with self.session.get(f"{self.base_url}/experts") as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("experts", [])
                return []
        except:
            logger.warning("Could not fetch available experts")
            return []


class MockInferenceClient:
    """Mock inference client for development/testing."""
    
    def __init__(self):
        self.mock_responses = {
            CommunicationStyle.THERAPEUTIC: [
                "I understand this is a difficult experience for you. Trauma can affect us in many ways, and it's important to recognize that your feelings are valid.",
                "The healing process takes time, and it's okay to move at your own pace. What you're experiencing is a normal response to abnormal circumstances.",
                "Working through these feelings is part of your healing journey. Remember that you're not alone in this process."
            ],
            CommunicationStyle.EDUCATIONAL: [
                "Research shows that trauma affects the brain's stress response system, particularly the amygdala and hippocampus.",
                "Understanding the neurobiological basis of trauma can help explain why you might be experiencing these symptoms.",
                "Studies indicate that therapeutic interventions can help rewire neural pathways and promote healing."
            ],
            CommunicationStyle.EMPATHETIC: [
                "That sounds incredibly difficult. I can only imagine how overwhelming this must feel for you right now.",
                "Your feelings make complete sense given what you've been through. It's natural to feel this way.",
                "I hear the pain in your words, and I want you to know that your experience matters and is valid."
            ],
            CommunicationStyle.PRACTICAL: [
                "Here are some concrete steps you can take: First, try grounding techniques when you feel overwhelmed.",
                "One practical approach is to establish a daily routine that includes self-care activities.",
                "You might find it helpful to start with small, manageable goals and build from there."
            ]
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def generate_response(self, request: InferenceRequest) -> InferenceResponse:
        """Generate mock response."""
        import random
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        responses = self.mock_responses.get(request.expert, ["I understand."])
        generated_text = random.choice(responses)
        
        return InferenceResponse(
            generated_text=generated_text,
            confidence=0.8,
            processing_time=0.1,
            model_info={"model": "mock", "expert": request.expert.value},
            metadata={"mock": True}
        )
    
    async def health_check(self) -> bool:
        return True
    
    async def get_available_experts(self) -> List[str]:
        return [style.value for style in CommunicationStyle]


def create_inference_client(use_mock: bool = False) -> ExternalInferenceClient:
    """Factory function to create inference client."""
    if use_mock:
        return MockInferenceClient()
    else:
        # In production, would get URL from config
        return ExternalInferenceClient(base_url="http://localhost:8001")
