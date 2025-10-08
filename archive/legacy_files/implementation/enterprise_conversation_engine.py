#!/usr/bin/env python3
"""
Enterprise-Grade Conversation Engine - Phase 3 Implementation
Production-ready dynamic conversation system with advanced features
"""

import json
import logging
import sqlite3
import uuid
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalityType(Enum):
    DIRECT_PRACTICAL = "direct_practical"
    GENTLE_NURTURING = "gentle_nurturing"
    ANALYTICAL_PROBLEM_SOLVING = "analytical_problem_solving"
    CASUAL_FRIEND_LIKE = "casual_friend_like"

class EmotionalIntensity(Enum):
    MILD = 1
    LOW_MODERATE = 2
    MODERATE = 3
    MODERATE_HIGH = 4
    HIGH = 5
    VERY_HIGH = 6
    SEVERE = 7
    CRISIS_LOW = 8
    CRISIS_HIGH = 9
    EMERGENCY = 10

class ConversationState(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    TERMINATED = "terminated"

class BranchingCondition(Enum):
    KEYWORD_MATCH = "keyword_match"
    EMOTIONAL_INTENSITY = "emotional_intensity"
    USER_RESPONSE_TYPE = "user_response_type"
    CONTEXT_BASED = "context_based"
    TIME_BASED = "time_based"
    PERSONALITY_MATCH = "personality_match"

@dataclass
class UserContext:
    """Comprehensive user context for personalization"""
    user_id: str
    personality_preference: PersonalityType
    emotional_baseline: int = 5
    conversation_history: List[str] = None
    demographic_info: Dict[str, Any] = None
    preferences: Dict[str, Any] = None
    crisis_indicators: List[str] = None
    last_interaction: Optional[datetime] = None
    session_count: int = 0
    satisfaction_scores: List[float] = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.demographic_info is None:
            self.demographic_info = {}
        if self.preferences is None:
            self.preferences = {}
        if self.crisis_indicators is None:
            self.crisis_indicators = []
        if self.satisfaction_scores is None:
            self.satisfaction_scores = []

@dataclass
class ConversationNode:
    """Enhanced conversation node with enterprise features"""
    node_id: str
    flow_id: str
    user_message: str
    emotional_intensity: int
    context_tags: List[str]
    follow_up_triggers: List[str]
    branching_conditions: List[Dict[str, Any]]
    response_variations: Dict[PersonalityType, str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ConversationSession:
    """Complete conversation session tracking"""
    session_id: str
    user_id: str
    flow_id: str
    current_node_id: str
    personality_used: PersonalityType
    emotional_trajectory: List[Tuple[datetime, int]]
    conversation_turns: List[Dict[str, Any]]
    state: ConversationState
    start_time: datetime
    last_activity: datetime
    context_variables: Dict[str, Any]
    satisfaction_score: Optional[float] = None
    completion_rate: float = 0.0
    
    def __post_init__(self):
        if not self.emotional_trajectory:
            self.emotional_trajectory = []
        if not self.conversation_turns:
            self.conversation_turns = []
        if not self.context_variables:
            self.context_variables = {}

class EnterpriseConversationEngine:
    """Production-grade conversation engine with advanced features"""
    
    def __init__(self, 
                 db_path: str = "/home/vivi/pixelated/ai/data/conversation_system.db",
                 cache_size: int = 1000,
                 max_concurrent_sessions: int = 100):
        """Initialize enterprise conversation engine"""
        self.db_path = db_path
        self.cache_size = cache_size
        self.max_concurrent_sessions = max_concurrent_sessions
        
        # Connection pool for database operations
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self._initialize_connection_pool()
        
        # In-memory caches for performance
        self.node_cache = {}
        self.flow_cache = {}
        self.user_context_cache = {}
        self.response_cache = {}
        
        # Active session management
        self.active_sessions = {}
        self.session_lock = threading.Lock()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Analytics and monitoring
        self.performance_metrics = {
            'total_conversations': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'error_rate': 0.0,
            'concurrent_sessions': 0
        }
        
        # Crisis detection patterns
        self.crisis_patterns = self._load_crisis_patterns()
        
        # Contextual awareness system
        # self.context_analyzer = ContextualAwarenessSystem()  # Will be set by integration class
        
        logger.info("✅ Enterprise Conversation Engine initialized")
        logger.info(f"📊 Cache size: {cache_size}, Max concurrent: {max_concurrent_sessions}")
    
    def _initialize_connection_pool(self, pool_size: int = 10):
        """Initialize database connection pool"""
        try:
            for _ in range(pool_size):
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                self.connection_pool.append(conn)
            
            logger.info(f"✅ Database connection pool initialized: {pool_size} connections")
        except Exception as e:
            logger.error(f"❌ Failed to initialize connection pool: {e}")
            raise
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection from pool"""
        with self.pool_lock:
            if self.connection_pool:
                return self.connection_pool.pop()
            else:
                # Create new connection if pool is empty
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                return conn
    
    def _return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool"""
        with self.pool_lock:
            if len(self.connection_pool) < 10:  # Max pool size
                self.connection_pool.append(conn)
            else:
                conn.close()
    
    def _load_crisis_patterns(self) -> List[Dict[str, Any]]:
        """Load crisis detection patterns"""
        return [
            {
                'pattern': r'\b(suicide|kill myself|end it all|can\'t go on)\b',
                'severity': 10,
                'action': 'immediate_escalation'
            },
            {
                'pattern': r'\b(self harm|hurt myself|cutting)\b',
                'severity': 9,
                'action': 'crisis_intervention'
            },
            {
                'pattern': r'\b(hopeless|worthless|nobody cares)\b',
                'severity': 7,
                'action': 'enhanced_support'
            }
        ]
    
    async def start_conversation(self, 
                               user_id: str, 
                               initial_message: str,
                               personality_preference: Optional[PersonalityType] = None,
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start a new conversation session"""
        start_time = time.time()
        
        try:
            # Get or create user context
            user_context = await self._get_user_context(user_id)
            
            # Set personality preference
            if personality_preference:
                user_context.personality_preference = personality_preference
            
            # Analyze initial message for flow selection
            flow_id, confidence = await self._select_optimal_flow(initial_message, user_context)
            
            # Create conversation session
            session = ConversationSession(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
                flow_id=flow_id,
                current_node_id="",  # Will be set by first node
                personality_used=user_context.personality_preference,
                emotional_trajectory=[],
                conversation_turns=[],
                state=ConversationState.ACTIVE,
                start_time=datetime.now(),
                last_activity=datetime.now(),
                context_variables=context or {}
            )
            
            # Process first turn
            response_data = await self._process_conversation_turn(
                session, initial_message, is_first_turn=True
            )
            
            # Store session
            with self.session_lock:
                self.active_sessions[session.session_id] = session
            
            # Update performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics('conversation_started', response_time)
            
            return {
                'success': True,
                'session_id': session.session_id,
                'response': response_data['response'],
                'emotional_intensity': response_data['emotional_intensity'],
                'personality_used': session.personality_used.value,
                'flow_confidence': confidence,
                'response_time': response_time,
                'context_analysis': response_data.get('context_analysis', {})
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting conversation: {e}")
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    async def continue_conversation(self, 
                                  session_id: str, 
                                  user_message: str) -> Dict[str, Any]:
        """Continue an existing conversation"""
        start_time = time.time()
        
        try:
            # Get active session
            with self.session_lock:
                session = self.active_sessions.get(session_id)
            
            if not session:
                return {
                    'success': False,
                    'error': 'Session not found or expired'
                }
            
            # Check for crisis indicators
            crisis_level = await self._detect_crisis_indicators(user_message)
            if crisis_level >= 8:
                return await self._handle_crisis_situation(session, user_message, crisis_level)
            
            # Process conversation turn
            response_data = await self._process_conversation_turn(session, user_message)
            
            # Update session
            session.last_activity = datetime.now()
            
            # Update performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics('conversation_continued', response_time)
            
            return {
                'success': True,
                'session_id': session_id,
                'response': response_data['response'],
                'emotional_intensity': response_data['emotional_intensity'],
                'personality_used': session.personality_used.value,
                'branching_occurred': response_data.get('branching_occurred', False),
                'response_time': response_time,
                'context_analysis': response_data.get('context_analysis', {}),
                'session_state': session.state.value
            }
            
        except Exception as e:
            logger.error(f"❌ Error continuing conversation: {e}")
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    async def _process_conversation_turn(self, 
                                       session: ConversationSession, 
                                       user_message: str,
                                       is_first_turn: bool = False) -> Dict[str, Any]:
        """Process a single conversation turn with advanced logic"""
        
        # Analyze user message context
        context_analysis = await self.context_analyzer.analyze_message(
            user_message, session.context_variables
        )
        
        # Determine emotional intensity
        emotional_intensity = await self._analyze_emotional_intensity(
            user_message, session.emotional_trajectory
        )
        
        # Get current or next node
        if is_first_turn:
            current_node = await self._get_starting_node(session.flow_id)
            session.current_node_id = current_node.node_id
        else:
            # Check for branching conditions
            next_node_id = await self._evaluate_branching_conditions(
                session.current_node_id, user_message, context_analysis, emotional_intensity
            )
            
            if next_node_id and next_node_id != session.current_node_id:
                current_node = await self._get_node(next_node_id)
                session.current_node_id = next_node_id
                branching_occurred = True
            else:
                # Get next sequential node
                current_node = await self._get_next_sequential_node(session.current_node_id)
                if current_node:
                    session.current_node_id = current_node.node_id
                    branching_occurred = False
                else:
                    # End of conversation flow
                    return await self._handle_conversation_completion(session)
        
        # Generate personality-appropriate response
        response = await self._generate_response(
            current_node, session.personality_used, context_analysis, emotional_intensity
        )
        
        # Record conversation turn
        turn_data = {
            'turn_number': len(session.conversation_turns) + 1,
            'user_message': user_message,
            'assistant_response': response,
            'node_id': current_node.node_id,
            'emotional_intensity': emotional_intensity,
            'timestamp': datetime.now().isoformat(),
            'context_analysis': context_analysis
        }
        
        session.conversation_turns.append(turn_data)
        session.emotional_trajectory.append((datetime.now(), emotional_intensity))
        
        # Update context variables
        session.context_variables.update(context_analysis.get('extracted_variables', {}))
        
        # Store turn in database (async)
        self.executor.submit(self._store_conversation_turn, session.session_id, turn_data)
        
        return {
            'response': response,
            'emotional_intensity': emotional_intensity,
            'branching_occurred': locals().get('branching_occurred', False),
            'context_analysis': context_analysis
        }
    
    async def _select_optimal_flow(self, 
                                 initial_message: str, 
                                 user_context: UserContext) -> Tuple[str, float]:
        """Select optimal conversation flow using advanced matching"""
        
        # Get all available flows
        flows = await self._get_available_flows()
        
        best_flow_id = None
        best_confidence = 0.0
        
        for flow in flows:
            confidence = 0.0
            
            # Keyword matching
            flow_keywords = flow.get('flow_tags', [])
            message_lower = initial_message.lower()
            
            keyword_matches = sum(1 for keyword in flow_keywords if keyword in message_lower)
            confidence += (keyword_matches / len(flow_keywords)) * 0.4
            
            # Emotional intensity matching
            message_intensity = await self._analyze_emotional_intensity(initial_message, [])
            flow_min = flow.get('emotional_range_min', 1)
            flow_max = flow.get('emotional_range_max', 10)
            
            if flow_min <= message_intensity <= flow_max:
                confidence += 0.3
            
            # User history matching
            if user_context.conversation_history:
                # Prefer flows user hasn't used recently
                recent_flows = user_context.conversation_history[-5:]
                if flow['flow_id'] not in recent_flows:
                    confidence += 0.2
                else:
                    confidence -= 0.1
            
            # Demographic matching
            target_demographics = flow.get('target_demographics', [])
            user_demographics = user_context.demographic_info
            
            if 'general' in target_demographics or not target_demographics:
                confidence += 0.1
            else:
                demo_matches = sum(1 for demo in target_demographics 
                                 if demo in user_demographics.values())
                confidence += (demo_matches / len(target_demographics)) * 0.2
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_flow_id = flow['flow_id']
        
        # Fallback to default flow if confidence is too low
        if best_confidence < 0.3:
            best_flow_id = await self._get_default_flow_id()
            best_confidence = 0.3
        
        return best_flow_id, best_confidence
    
    async def _evaluate_branching_conditions(self, 
                                           current_node_id: str,
                                           user_message: str,
                                           context_analysis: Dict[str, Any],
                                           emotional_intensity: int) -> Optional[str]:
        """Evaluate branching conditions for dynamic conversation flow"""
        
        # Get possible transitions from current node
        transitions = await self._get_node_transitions(current_node_id)
        
        for transition in transitions:
            condition_type = transition['condition_type']
            condition_value = json.loads(transition['condition_value'])
            probability_weight = transition['probability_weight']
            
            branch_score = 0.0
            
            if condition_type == BranchingCondition.KEYWORD_MATCH.value:
                keywords = condition_value.get('keywords', [])
                matches = sum(1 for keyword in keywords if keyword.lower() in user_message.lower())
                branch_score = (matches / len(keywords)) * probability_weight
            
            elif condition_type == BranchingCondition.EMOTIONAL_INTENSITY.value:
                target_range = condition_value.get('range', [1, 10])
                if target_range[0] <= emotional_intensity <= target_range[1]:
                    branch_score = probability_weight
            
            elif condition_type == BranchingCondition.CONTEXT_BASED.value:
                required_context = condition_value.get('required_context', {})
                context_matches = sum(1 for key, value in required_context.items()
                                    if context_analysis.get(key) == value)
                if context_matches == len(required_context):
                    branch_score = probability_weight
            
            elif condition_type == BranchingCondition.USER_RESPONSE_TYPE.value:
                response_type = condition_value.get('response_type')
                detected_type = context_analysis.get('response_type')
                if response_type == detected_type:
                    branch_score = probability_weight
            
            # If branch score is high enough, take this branch
            if branch_score >= 0.7:
                return transition['to_node_id']
        
        return None
    
    async def _generate_response(self, 
                               node: ConversationNode,
                               personality: PersonalityType,
                               context_analysis: Dict[str, Any],
                               emotional_intensity: int) -> str:
        """Generate contextually appropriate response"""
        
        # Get base response for personality
        base_response = node.response_variations.get(personality, "")
        
        if not base_response:
            # Fallback to any available response
            base_response = list(node.response_variations.values())[0]
        
        # Apply contextual modifications
        response = await self._apply_contextual_modifications(
            base_response, context_analysis, emotional_intensity
        )
        
        # Apply personality-specific enhancements
        response = await self._enhance_response_for_personality(
            response, personality, emotional_intensity
        )
        
        return response
    
    async def _apply_contextual_modifications(self, 
                                            response: str,
                                            context_analysis: Dict[str, Any],
                                            emotional_intensity: int) -> str:
        """Apply contextual modifications to response"""
        
        # Time-based modifications
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            if "How are you feeling" in response:
                response = response.replace("How are you feeling", "I know it's late - how are you feeling")
        
        # Emotional intensity modifications
        if emotional_intensity >= 8:
            # Crisis-level intensity - make response more supportive
            response = f"I can hear how much pain you're in. {response}"
        elif emotional_intensity >= 6:
            # High intensity - add validation
            response = f"That sounds really difficult. {response}"
        
        # Context-specific modifications
        if context_analysis.get('mentions_work'):
            response = response.replace("situation", "work situation")
        
        if context_analysis.get('mentions_relationship'):
            response = response.replace("this", "this relationship issue")
        
        return response
    
    async def _enhance_response_for_personality(self, 
                                              response: str,
                                              personality: PersonalityType,
                                              emotional_intensity: int) -> str:
        """Enhance response based on personality type"""
        
        if personality == PersonalityType.DIRECT_PRACTICAL:
            # Keep responses concise and action-oriented
            if len(response) > 100:
                response = response.split('.')[0] + '.'
        
        elif personality == PersonalityType.GENTLE_NURTURING:
            # Add gentle, caring language
            if not any(word in response.lower() for word in ['gentle', 'care', 'support']):
                response = f"I want you to know I'm here for you. {response}"
        
        elif personality == PersonalityType.ANALYTICAL_PROBLEM_SOLVING:
            # Add analytical framing
            if '?' in response and 'specifically' not in response.lower():
                response = response.replace('What', 'What specifically')
        
        elif personality == PersonalityType.CASUAL_FRIEND_LIKE:
            # Add casual, friendly language
            casual_starters = ['Yeah,', 'Totally,', 'I get that,', 'For sure,']
            if not any(starter in response for starter in casual_starters):
                if emotional_intensity < 7:  # Don't be too casual in crisis
                    response = f"I totally get that. {response}"
        
        return response
    
    async def _detect_crisis_indicators(self, message: str) -> int:
        """Detect crisis indicators in user message"""
        
        crisis_level = 0
        message_lower = message.lower()
        
        for pattern_data in self.crisis_patterns:
            if re.search(pattern_data['pattern'], message_lower):
                crisis_level = max(crisis_level, pattern_data['severity'])
        
        return crisis_level
    
    async def _handle_crisis_situation(self, 
                                     session: ConversationSession,
                                     user_message: str,
                                     crisis_level: int) -> Dict[str, Any]:
        """Handle crisis situations with appropriate escalation"""
        
        # Update session state
        session.state = ConversationState.ESCALATED
        
        # Generate crisis-appropriate response
        if crisis_level >= 9:
            response = ("I'm very concerned about what you've shared. Your safety is the most important thing right now. "
                       "Please reach out to a crisis helpline immediately: National Suicide Prevention Lifeline: 988. "
                       "You can also text HOME to 741741 for the Crisis Text Line. You don't have to go through this alone.")
        elif crisis_level >= 7:
            response = ("I can hear that you're going through something really difficult right now. "
                       "While I'm here to listen, I want to make sure you have access to professional support. "
                       "Have you considered reaching out to a counselor or therapist? There are also crisis resources available 24/7.")
        
        # Log crisis event
        logger.warning(f"🚨 Crisis detected - Level {crisis_level} - Session: {session.session_id}")
        
        # Store crisis event in database
        self.executor.submit(self._store_crisis_event, session.session_id, user_message, crisis_level)
        
        return {
            'success': True,
            'session_id': session.session_id,
            'response': response,
            'crisis_level': crisis_level,
            'escalated': True,
            'resources_provided': True
        }
    
    def _update_performance_metrics(self, operation: str, response_time: float):
        """Update performance metrics"""
        self.performance_metrics['total_conversations'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['average_response_time']
        total_ops = self.performance_metrics['total_conversations']
        self.performance_metrics['average_response_time'] = (
            (current_avg * (total_ops - 1) + response_time) / total_ops
        )
        
        # Update concurrent sessions
        with self.session_lock:
            self.performance_metrics['concurrent_sessions'] = len(self.active_sessions)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            'cache_sizes': {
                'nodes': len(self.node_cache),
                'flows': len(self.flow_cache),
                'users': len(self.user_context_cache),
                'responses': len(self.response_cache)
            },
            'database_pool_size': len(self.connection_pool),
            'active_sessions': len(self.active_sessions)
        }
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired sessions"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self.session_lock:
            expired_sessions = [
                session_id for session_id, session in self.active_sessions.items()
                if session.last_activity < cutoff_time
            ]
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
        
        logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
    
    def shutdown(self):
        """Graceful shutdown of the conversation engine"""
        logger.info("🔄 Shutting down Enterprise Conversation Engine...")
        
        # Close all database connections
        with self.pool_lock:
            for conn in self.connection_pool:
                conn.close()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("✅ Enterprise Conversation Engine shutdown complete")

# Additional helper methods will be implemented in separate files
# This is the core engine - we'll add the supporting systems next

if __name__ == "__main__":
    # Example usage
    async def main():
        engine = EnterpriseConversationEngine()
        
        # Start a conversation
        result = await engine.start_conversation(
            user_id="test_user_001",
            initial_message="I'm feeling really anxious about work",
            personality_preference=PersonalityType.GENTLE_NURTURING
        )
        
        print(f"Response: {result['response']}")
        print(f"Session ID: {result['session_id']}")
        
        # Continue conversation
        if result['success']:
            continue_result = await engine.continue_conversation(
                session_id=result['session_id'],
                user_message="My boss keeps giving me impossible deadlines"
            )
            print(f"Follow-up: {continue_result['response']}")
    
    # Run example
    # asyncio.run(main())
