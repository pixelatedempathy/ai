#!/usr/bin/env python3
"""
Contextual Awareness System - Advanced Context Analysis
Enterprise-grade context understanding and variable extraction
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class ContextType(Enum):
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"
    SITUATIONAL = "situational"
    RELATIONAL = "relational"
    ENVIRONMENTAL = "environmental"
    BEHAVIORAL = "behavioral"

class ResponseType(Enum):
    QUESTION = "question"
    STATEMENT = "statement"
    COMPLAINT = "complaint"
    REQUEST_HELP = "request_help"
    SHARING_EXPERIENCE = "sharing_experience"
    EXPRESSING_EMOTION = "expressing_emotion"

@dataclass
class ContextualInsight:
    """Contextual insight extracted from user message"""
    insight_type: ContextType
    confidence: float
    extracted_data: Dict[str, Any]
    implications: List[str]

class ContextualAwarenessSystem:
    """Advanced contextual awareness and analysis system"""
    
    def __init__(self):
        """Initialize contextual awareness system"""
        self.temporal_patterns = self._load_temporal_patterns()
        self.emotional_indicators = self._load_emotional_indicators()
        self.situational_contexts = self._load_situational_contexts()
        self.relational_patterns = self._load_relational_patterns()
        self.behavioral_indicators = self._load_behavioral_indicators()
        
        logger.info("✅ Contextual Awareness System initialized")
    
    def _load_temporal_patterns(self) -> Dict[str, Any]:
        """Load temporal context patterns"""
        return {
            'time_references': {
                'immediate': ['now', 'right now', 'currently', 'at the moment'],
                'recent': ['today', 'yesterday', 'this week', 'lately', 'recently'],
                'ongoing': ['always', 'constantly', 'every day', 'all the time'],
                'past': ['used to', 'before', 'in the past', 'previously'],
                'future': ['will', 'going to', 'planning to', 'next week', 'tomorrow']
            },
            'duration_indicators': {
                'short': ['a few minutes', 'briefly', 'momentarily'],
                'medium': ['hours', 'a day', 'couple of days'],
                'long': ['weeks', 'months', 'years', 'forever']
            },
            'frequency_patterns': {
                'rare': ['rarely', 'seldom', 'once in a while'],
                'occasional': ['sometimes', 'occasionally', 'now and then'],
                'frequent': ['often', 'frequently', 'regularly'],
                'constant': ['always', 'constantly', 'non-stop', '24/7']
            }
        }
    
    def _load_emotional_indicators(self) -> Dict[str, Any]:
        """Load emotional context indicators"""
        return {
            'intensity_markers': {
                'mild': ['a bit', 'slightly', 'somewhat', 'kind of'],
                'moderate': ['pretty', 'quite', 'fairly', 'rather'],
                'high': ['very', 'really', 'extremely', 'incredibly'],
                'severe': ['completely', 'totally', 'absolutely', 'utterly']
            },
            'emotional_states': {
                'anxiety': ['anxious', 'worried', 'nervous', 'stressed', 'panicked'],
                'depression': ['sad', 'depressed', 'down', 'hopeless', 'empty'],
                'anger': ['angry', 'furious', 'mad', 'irritated', 'frustrated'],
                'fear': ['scared', 'afraid', 'terrified', 'frightened'],
                'joy': ['happy', 'excited', 'thrilled', 'elated'],
                'confusion': ['confused', 'lost', 'uncertain', 'unclear']
            },
            'emotional_progression': {
                'escalating': ['getting worse', 'building up', 'intensifying'],
                'stable': ['same as usual', 'consistent', 'steady'],
                'improving': ['getting better', 'feeling better', 'improving']
            }
        }
    
    def _load_situational_contexts(self) -> Dict[str, Any]:
        """Load situational context patterns"""
        return {
            'work_contexts': {
                'workplace_stress': ['deadline', 'boss', 'coworker', 'meeting', 'project'],
                'job_security': ['layoffs', 'fired', 'job search', 'unemployment'],
                'career_growth': ['promotion', 'raise', 'career', 'advancement'],
                'work_life_balance': ['overtime', 'work from home', 'burnout']
            },
            'relationship_contexts': {
                'romantic': ['boyfriend', 'girlfriend', 'partner', 'spouse', 'dating'],
                'family': ['mom', 'dad', 'parents', 'siblings', 'family'],
                'friendship': ['friend', 'friends', 'social', 'hanging out'],
                'conflict': ['argument', 'fight', 'disagreement', 'tension']
            },
            'health_contexts': {
                'mental_health': ['therapy', 'counseling', 'medication', 'diagnosis'],
                'physical_health': ['doctor', 'hospital', 'pain', 'illness', 'symptoms'],
                'lifestyle': ['sleep', 'exercise', 'diet', 'habits']
            },
            'life_transitions': {
                'major_changes': ['moving', 'new job', 'graduation', 'marriage', 'divorce'],
                'loss': ['death', 'breakup', 'loss', 'grief', 'mourning'],
                'growth': ['learning', 'developing', 'growing', 'changing']
            }
        }
    
    def _load_relational_patterns(self) -> Dict[str, Any]:
        """Load relational context patterns"""
        return {
            'relationship_quality': {
                'positive': ['love', 'support', 'understanding', 'close', 'trust'],
                'neutral': ['okay', 'fine', 'normal', 'usual'],
                'negative': ['toxic', 'abusive', 'distant', 'cold', 'hostile']
            },
            'communication_patterns': {
                'open': ['talk', 'communicate', 'share', 'express'],
                'closed': ['silent', 'shut down', 'avoid', 'ignore'],
                'conflict': ['argue', 'fight', 'yell', 'disagree']
            },
            'support_systems': {
                'strong': ['family support', 'good friends', 'close relationships'],
                'weak': ['alone', 'isolated', 'no one to talk to'],
                'mixed': ['some support', 'few people', 'limited help']
            }
        }
    
    def _load_behavioral_indicators(self) -> Dict[str, Any]:
        """Load behavioral context indicators"""
        return {
            'coping_mechanisms': {
                'healthy': ['exercise', 'meditation', 'talking', 'journaling'],
                'unhealthy': ['drinking', 'drugs', 'isolation', 'self-harm'],
                'avoidance': ['procrastination', 'denial', 'distraction']
            },
            'help_seeking': {
                'active': ['looking for help', 'seeking advice', 'want to change'],
                'passive': ['don\'t know what to do', 'feeling stuck'],
                'resistant': ['nothing helps', 'tried everything', 'hopeless']
            },
            'action_orientation': {
                'proactive': ['planning', 'taking action', 'making changes'],
                'reactive': ['responding to', 'dealing with', 'handling'],
                'passive': ['waiting', 'hoping', 'letting things happen']
            }
        }
    
    async def analyze_message(self, 
                            message: str, 
                            conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive message analysis for contextual understanding"""
        
        if conversation_context is None:
            conversation_context = {}
        
        analysis_results = {
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'response_type': await self._classify_response_type(message),
            'contextual_insights': [],
            'extracted_variables': {},
            'emotional_analysis': {},
            'situational_analysis': {},
            'temporal_analysis': {},
            'relational_analysis': {},
            'behavioral_analysis': {},
            'implications': [],
            'suggested_responses': []
        }
        
        # Perform different types of analysis
        analysis_results['temporal_analysis'] = await self._analyze_temporal_context(message)
        analysis_results['emotional_analysis'] = await self._analyze_emotional_context(message)
        analysis_results['situational_analysis'] = await self._analyze_situational_context(message)
        analysis_results['relational_analysis'] = await self._analyze_relational_context(message)
        analysis_results['behavioral_analysis'] = await self._analyze_behavioral_context(message)
        
        # Extract key variables
        analysis_results['extracted_variables'] = await self._extract_context_variables(
            message, analysis_results
        )
        
        # Generate contextual insights
        analysis_results['contextual_insights'] = await self._generate_contextual_insights(
            analysis_results
        )
        
        # Determine implications for conversation flow
        analysis_results['implications'] = await self._determine_conversation_implications(
            analysis_results, conversation_context
        )
        
        # Suggest response approaches
        analysis_results['suggested_responses'] = await self._suggest_response_approaches(
            analysis_results
        )
        
        return analysis_results
    
    async def _classify_response_type(self, message: str) -> str:
        """Classify the type of user response"""
        
        message_lower = message.lower().strip()
        
        # Question indicators
        if message.endswith('?') or any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return ResponseType.QUESTION.value
        
        # Request for help indicators
        if any(phrase in message_lower for phrase in ['help me', 'what should i do', 'i need', 'can you']):
            return ResponseType.REQUEST_HELP.value
        
        # Complaint indicators
        if any(phrase in message_lower for phrase in ['i hate', 'this sucks', 'i can\'t stand', 'annoying']):
            return ResponseType.COMPLAINT.value
        
        # Emotional expression indicators
        if any(phrase in message_lower for phrase in ['i feel', 'i\'m feeling', 'makes me', 'i am']):
            return ResponseType.EXPRESSING_EMOTION.value
        
        # Experience sharing indicators
        if any(phrase in message_lower for phrase in ['yesterday', 'today', 'happened', 'went to', 'did']):
            return ResponseType.SHARING_EXPERIENCE.value
        
        return ResponseType.STATEMENT.value
    
    async def _analyze_temporal_context(self, message: str) -> Dict[str, Any]:
        """Analyze temporal context in the message"""
        
        temporal_analysis = {
            'time_frame': 'present',
            'duration': 'unspecified',
            'frequency': 'unspecified',
            'temporal_markers': [],
            'urgency_level': 'normal'
        }
        
        message_lower = message.lower()
        
        # Analyze time frame
        for time_frame, indicators in self.temporal_patterns['time_references'].items():
            if any(indicator in message_lower for indicator in indicators):
                temporal_analysis['time_frame'] = time_frame
                temporal_analysis['temporal_markers'].extend([
                    indicator for indicator in indicators if indicator in message_lower
                ])
        
        # Analyze duration
        for duration, indicators in self.temporal_patterns['duration_indicators'].items():
            if any(indicator in message_lower for indicator in indicators):
                temporal_analysis['duration'] = duration
        
        # Analyze frequency
        for frequency, indicators in self.temporal_patterns['frequency_patterns'].items():
            if any(indicator in message_lower for indicator in indicators):
                temporal_analysis['frequency'] = frequency
        
        # Determine urgency
        urgency_indicators = ['urgent', 'emergency', 'right now', 'immediately', 'asap']
        if any(indicator in message_lower for indicator in urgency_indicators):
            temporal_analysis['urgency_level'] = 'high'
        elif temporal_analysis['time_frame'] == 'immediate':
            temporal_analysis['urgency_level'] = 'medium'
        
        return temporal_analysis
    
    async def _analyze_emotional_context(self, message: str) -> Dict[str, Any]:
        """Analyze emotional context in the message"""
        
        emotional_analysis = {
            'primary_emotions': [],
            'intensity_level': 'moderate',
            'emotional_progression': 'stable',
            'emotional_markers': [],
            'mixed_emotions': False
        }
        
        message_lower = message.lower()
        
        # Identify emotions
        detected_emotions = []
        for emotion, indicators in self.emotional_indicators['emotional_states'].items():
            matches = [indicator for indicator in indicators if indicator in message_lower]
            if matches:
                detected_emotions.append({
                    'emotion': emotion,
                    'indicators': matches,
                    'confidence': len(matches) / len(indicators)
                })
        
        # Sort by confidence and take top emotions
        detected_emotions.sort(key=lambda x: x['confidence'], reverse=True)
        emotional_analysis['primary_emotions'] = [e['emotion'] for e in detected_emotions[:3]]
        emotional_analysis['mixed_emotions'] = len(detected_emotions) > 2
        
        # Analyze intensity
        for intensity, markers in self.emotional_indicators['intensity_markers'].items():
            if any(marker in message_lower for marker in markers):
                emotional_analysis['intensity_level'] = intensity
                emotional_analysis['emotional_markers'].extend([
                    marker for marker in markers if marker in message_lower
                ])
        
        # Analyze emotional progression
        for progression, indicators in self.emotional_indicators['emotional_progression'].items():
            if any(indicator in message_lower for indicator in indicators):
                emotional_analysis['emotional_progression'] = progression
        
        return emotional_analysis
    
    async def _analyze_situational_context(self, message: str) -> Dict[str, Any]:
        """Analyze situational context in the message"""
        
        situational_analysis = {
            'primary_contexts': [],
            'context_confidence': {},
            'specific_situations': [],
            'context_complexity': 'simple'
        }
        
        message_lower = message.lower()
        
        # Analyze different situational contexts
        all_contexts = []
        
        for main_context, sub_contexts in self.situational_contexts.items():
            for sub_context, indicators in sub_contexts.items():
                matches = [indicator for indicator in indicators if indicator in message_lower]
                if matches:
                    confidence = len(matches) / len(indicators)
                    all_contexts.append({
                        'main_context': main_context,
                        'sub_context': sub_context,
                        'confidence': confidence,
                        'indicators': matches
                    })
        
        # Sort by confidence
        all_contexts.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Extract primary contexts
        situational_analysis['primary_contexts'] = [
            f"{ctx['main_context']}.{ctx['sub_context']}" for ctx in all_contexts[:3]
        ]
        
        situational_analysis['context_confidence'] = {
            f"{ctx['main_context']}.{ctx['sub_context']}": ctx['confidence'] 
            for ctx in all_contexts[:5]
        }
        
        # Determine complexity
        if len(all_contexts) > 2:
            situational_analysis['context_complexity'] = 'complex'
        elif len(all_contexts) > 1:
            situational_analysis['context_complexity'] = 'moderate'
        
        return situational_analysis
    
    async def _analyze_relational_context(self, message: str) -> Dict[str, Any]:
        """Analyze relational context in the message"""
        
        relational_analysis = {
            'relationship_mentions': [],
            'relationship_quality': 'neutral',
            'communication_style': 'normal',
            'support_level': 'unknown',
            'relational_concerns': []
        }
        
        message_lower = message.lower()
        
        # Identify relationship mentions
        relationship_indicators = []
        for rel_type, indicators in self.situational_contexts['relationship_contexts'].items():
            matches = [indicator for indicator in indicators if indicator in message_lower]
            if matches:
                relationship_indicators.append({
                    'type': rel_type,
                    'indicators': matches
                })
        
        relational_analysis['relationship_mentions'] = relationship_indicators
        
        # Analyze relationship quality
        for quality, indicators in self.relational_patterns['relationship_quality'].items():
            if any(indicator in message_lower for indicator in indicators):
                relational_analysis['relationship_quality'] = quality
                break
        
        # Analyze communication patterns
        for pattern, indicators in self.relational_patterns['communication_patterns'].items():
            if any(indicator in message_lower for indicator in indicators):
                relational_analysis['communication_style'] = pattern
                break
        
        # Analyze support systems
        for support, indicators in self.relational_patterns['support_systems'].items():
            if any(indicator in message_lower for indicator in indicators):
                relational_analysis['support_level'] = support
                break
        
        return relational_analysis
    
    async def _analyze_behavioral_context(self, message: str) -> Dict[str, Any]:
        """Analyze behavioral context in the message"""
        
        behavioral_analysis = {
            'coping_strategies': [],
            'help_seeking_behavior': 'unknown',
            'action_orientation': 'unknown',
            'behavioral_patterns': [],
            'change_readiness': 'unknown'
        }
        
        message_lower = message.lower()
        
        # Analyze coping mechanisms
        for coping_type, indicators in self.behavioral_indicators['coping_mechanisms'].items():
            matches = [indicator for indicator in indicators if indicator in message_lower]
            if matches:
                behavioral_analysis['coping_strategies'].append({
                    'type': coping_type,
                    'strategies': matches
                })
        
        # Analyze help-seeking behavior
        for behavior, indicators in self.behavioral_indicators['help_seeking'].items():
            if any(indicator in message_lower for indicator in indicators):
                behavioral_analysis['help_seeking_behavior'] = behavior
                break
        
        # Analyze action orientation
        for orientation, indicators in self.behavioral_indicators['action_orientation'].items():
            if any(indicator in message_lower for indicator in indicators):
                behavioral_analysis['action_orientation'] = orientation
                break
        
        # Determine change readiness
        change_indicators = ['want to change', 'ready to', 'willing to try', 'open to']
        resistance_indicators = ['can\'t change', 'won\'t work', 'tried everything']
        
        if any(indicator in message_lower for indicator in change_indicators):
            behavioral_analysis['change_readiness'] = 'high'
        elif any(indicator in message_lower for indicator in resistance_indicators):
            behavioral_analysis['change_readiness'] = 'low'
        else:
            behavioral_analysis['change_readiness'] = 'moderate'
        
        return behavioral_analysis
    
    async def _extract_context_variables(self, 
                                       message: str, 
                                       analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key context variables for conversation flow"""
        
        variables = {}
        
        # Extract temporal variables
        temporal = analysis_results['temporal_analysis']
        variables['time_frame'] = temporal['time_frame']
        variables['urgency_level'] = temporal['urgency_level']
        variables['frequency'] = temporal['frequency']
        
        # Extract emotional variables
        emotional = analysis_results['emotional_analysis']
        if emotional['primary_emotions']:
            variables['primary_emotion'] = emotional['primary_emotions'][0]
        variables['emotional_intensity'] = emotional['intensity_level']
        variables['emotional_progression'] = emotional['emotional_progression']
        
        # Extract situational variables
        situational = analysis_results['situational_analysis']
        if situational['primary_contexts']:
            variables['primary_context'] = situational['primary_contexts'][0]
        variables['context_complexity'] = situational['context_complexity']
        
        # Extract relational variables
        relational = analysis_results['relational_analysis']
        variables['relationship_quality'] = relational['relationship_quality']
        variables['support_level'] = relational['support_level']
        
        # Extract behavioral variables
        behavioral = analysis_results['behavioral_analysis']
        variables['help_seeking'] = behavioral['help_seeking_behavior']
        variables['change_readiness'] = behavioral['change_readiness']
        
        # Extract specific mentions
        message_lower = message.lower()
        variables['mentions_work'] = any(word in message_lower for word in ['work', 'job', 'boss', 'office'])
        variables['mentions_relationship'] = any(word in message_lower for word in ['relationship', 'partner', 'boyfriend', 'girlfriend'])
        variables['mentions_family'] = any(word in message_lower for word in ['family', 'mom', 'dad', 'parents'])
        variables['mentions_health'] = any(word in message_lower for word in ['health', 'doctor', 'therapy', 'medication'])
        
        return variables
    
    async def _generate_contextual_insights(self, analysis_results: Dict[str, Any]) -> List[ContextualInsight]:
        """Generate contextual insights from analysis"""
        
        insights = []
        
        # Temporal insights
        temporal = analysis_results['temporal_analysis']
        if temporal['urgency_level'] == 'high':
            insights.append(ContextualInsight(
                insight_type=ContextType.TEMPORAL,
                confidence=0.9,
                extracted_data={'urgency': 'high'},
                implications=['immediate_response_needed', 'prioritize_support']
            ))
        
        # Emotional insights
        emotional = analysis_results['emotional_analysis']
        if emotional['intensity_level'] in ['high', 'severe']:
            insights.append(ContextualInsight(
                insight_type=ContextType.EMOTIONAL,
                confidence=0.8,
                extracted_data={'high_emotional_intensity': True},
                implications=['increased_empathy_needed', 'careful_response_required']
            ))
        
        # Situational insights
        situational = analysis_results['situational_analysis']
        if situational['context_complexity'] == 'complex':
            insights.append(ContextualInsight(
                insight_type=ContextType.SITUATIONAL,
                confidence=0.7,
                extracted_data={'complex_situation': True},
                implications=['break_down_issues', 'systematic_approach_needed']
            ))
        
        return insights
    
    async def _determine_conversation_implications(self, 
                                                 analysis_results: Dict[str, Any],
                                                 conversation_context: Dict[str, Any]) -> List[str]:
        """Determine implications for conversation flow"""
        
        implications = []
        
        # Response type implications
        response_type = analysis_results['response_type']
        if response_type == ResponseType.REQUEST_HELP.value:
            implications.append('provide_actionable_guidance')
        elif response_type == ResponseType.EXPRESSING_EMOTION.value:
            implications.append('validate_emotions_first')
        elif response_type == ResponseType.QUESTION.value:
            implications.append('provide_direct_answer')
        
        # Emotional implications
        emotional = analysis_results['emotional_analysis']
        if 'anxiety' in emotional['primary_emotions']:
            implications.append('use_calming_language')
        if 'anger' in emotional['primary_emotions']:
            implications.append('acknowledge_frustration')
        
        # Temporal implications
        temporal = analysis_results['temporal_analysis']
        if temporal['urgency_level'] == 'high':
            implications.append('respond_immediately')
        if temporal['frequency'] == 'constant':
            implications.append('address_chronic_nature')
        
        # Behavioral implications
        behavioral = analysis_results['behavioral_analysis']
        if behavioral['change_readiness'] == 'high':
            implications.append('offer_concrete_steps')
        elif behavioral['change_readiness'] == 'low':
            implications.append('focus_on_motivation')
        
        return implications
    
    async def _suggest_response_approaches(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Suggest response approaches based on analysis"""
        
        approaches = []
        
        # Based on response type
        response_type = analysis_results['response_type']
        if response_type == ResponseType.EXPRESSING_EMOTION.value:
            approaches.append('empathetic_validation')
        elif response_type == ResponseType.REQUEST_HELP.value:
            approaches.append('solution_focused')
        elif response_type == ResponseType.COMPLAINT.value:
            approaches.append('acknowledge_and_reframe')
        
        # Based on emotional state
        emotional = analysis_results['emotional_analysis']
        if emotional['intensity_level'] in ['high', 'severe']:
            approaches.append('gentle_supportive')
        elif 'confusion' in emotional['primary_emotions']:
            approaches.append('clarifying_questions')
        
        # Based on situational context
        situational = analysis_results['situational_analysis']
        if 'work_contexts' in str(situational['primary_contexts']):
            approaches.append('work_focused_guidance')
        elif 'relationship_contexts' in str(situational['primary_contexts']):
            approaches.append('relationship_counseling_approach')
        
        return approaches

# Export the class for use in the main engine
__all__ = ['ContextualAwarenessSystem', 'ContextualInsight', 'ContextType', 'ResponseType']
