#!/usr/bin/env python3
"""
Natural Multi-Turn Conversation Generator
Generate authentic, multi-turn conversations that develop empathy through real dialogue
"""

import json
import logging
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationFlow:
    """Natural conversation flow pattern"""
    flow_id: str
    name: str
    initial_situation: str
    conversation_turns: List[Dict[str, str]]
    empathy_development: str

class NaturalConversationGenerator:
    """Generate authentic multi-turn conversations"""
    
    def __init__(self):
        """Initialize natural conversation generator"""
        
        # Natural conversation flows
        self.conversation_flows = self._setup_natural_flows()
        
        # Natural response patterns (not clinical)
        self.natural_responses = self._setup_natural_responses()
        
        # Follow-up questions that show genuine interest
        self.follow_up_questions = self._setup_follow_up_questions()
        
        # Ways to show understanding that develops over time
        self.understanding_progression = self._setup_understanding_progression()
        
        logger.info("✅ Natural Conversation Generator initialized")
        logger.info(f"🗣️ Conversation flows: {len(self.conversation_flows)}")
    
    def _setup_natural_flows(self) -> List[ConversationFlow]:
        """Setup natural conversation flows with multiple turns"""
        
        flows = [
            ConversationFlow(
                flow_id="work_stress_buildup",
                name="Work Stress Building Understanding",
                initial_situation="work stress",
                conversation_turns=[
                    {
                        "user": "I'm really stressed about work lately",
                        "assistant": "What's going on at work?"
                    },
                    {
                        "user": "My boss keeps piling on more projects and I can't keep up",
                        "assistant": "That sounds overwhelming. How long has this been happening?"
                    },
                    {
                        "user": "About 3 months now. Ever since Sarah left the team",
                        "assistant": "Ah, so you're basically doing two people's jobs now?"
                    },
                    {
                        "user": "Exactly! And when I try to bring it up, my boss just says 'we're all stretched thin'",
                        "assistant": "Ugh, that's so frustrating. You're drowning and they're just dismissing it. Have you been able to talk to anyone else about this?"
                    },
                    {
                        "user": "Not really. I don't want to seem like I'm complaining",
                        "assistant": "You're not complaining though - you're dealing with an impossible workload. That's a real problem, not whining."
                    }
                ],
                empathy_development="Understanding develops from general stress → specific situation → root cause → emotional impact → validation of feelings"
            ),
            
            ConversationFlow(
                flow_id="relationship_doubt_exploration",
                name="Relationship Doubts Through Conversation",
                initial_situation="relationship problems",
                conversation_turns=[
                    {
                        "user": "I think my relationship might be in trouble",
                        "assistant": "What makes you feel that way?"
                    },
                    {
                        "user": "We barely talk anymore. Just logistics about bills and schedules",
                        "assistant": "When did you first notice the conversations changing?"
                    },
                    {
                        "user": "Maybe 6 months ago? We used to stay up talking for hours",
                        "assistant": "That must feel really lonely. What do you think changed around that time?"
                    },
                    {
                        "user": "He got promoted and started working way more. I thought it would be temporary",
                        "assistant": "So the promotion was supposed to be good news, but it ended up creating distance between you two?"
                    },
                    {
                        "user": "Yeah, and now I feel guilty for being upset about his success",
                        "assistant": "That's such a hard spot to be in. You want to be supportive but you're also losing your partner. Both feelings make total sense."
                    },
                    {
                        "user": "I just miss him, you know? Even when he's right there",
                        "assistant": "That's one of the loneliest feelings - missing someone who's sitting right next to you."
                    }
                ],
                empathy_development="From surface problem → timeline exploration → root cause discovery → emotional complexity → deep understanding of the real pain"
            ),
            
            ConversationFlow(
                flow_id="anxiety_spiral_understanding",
                name="Understanding Anxiety Through Details",
                initial_situation="anxiety",
                conversation_turns=[
                    {
                        "user": "I've been really anxious lately",
                        "assistant": "What's that been like for you?"
                    },
                    {
                        "user": "I can't stop thinking about all the things that could go wrong",
                        "assistant": "Are these specific things or just general worry?"
                    },
                    {
                        "user": "Specific. Like what if I mess up this presentation at work, or what if my car breaks down",
                        "assistant": "How often are these thoughts popping up?"
                    },
                    {
                        "user": "Constantly. I'll be watching TV and suddenly I'm thinking about the presentation",
                        "assistant": "That sounds exhausting. Your brain won't give you a break."
                    },
                    {
                        "user": "Exactly! And then I get anxious about being anxious. It's like a loop",
                        "assistant": "Oh man, the anxiety about anxiety is the worst. It's like your brain is trolling you."
                    },
                    {
                        "user": "Yes! And people keep saying 'just don't worry about it' like it's that simple",
                        "assistant": "Right? If you could just stop worrying, you would have done that already. That advice is so unhelpful."
                    }
                ],
                empathy_development="General anxiety → specific examples → frequency/intensity → emotional toll → shared frustration with unhelpful advice"
            ),
            
            ConversationFlow(
                flow_id="depression_energy_conversation",
                name="Depression Energy Levels Discussion",
                initial_situation="depression and low energy",
                conversation_turns=[
                    {
                        "user": "I have no energy for anything anymore",
                        "assistant": "How long have you been feeling this way?"
                    },
                    {
                        "user": "A few weeks now. I used to love going out with friends but now it feels like too much work",
                        "assistant": "Even things you normally enjoy feel overwhelming?"
                    },
                    {
                        "user": "Yeah, my friend invited me to dinner last week and I made an excuse not to go",
                        "assistant": "What was going through your mind when she asked?"
                    },
                    {
                        "user": "Just thinking about getting dressed and driving there and making conversation... it felt impossible",
                        "assistant": "Like each step was this huge mountain to climb?"
                    },
                    {
                        "user": "Exactly. And then I felt guilty for canceling, which made me feel worse",
                        "assistant": "So now you're dealing with the original exhaustion plus guilt on top of it."
                    },
                    {
                        "user": "And my friend probably thinks I don't want to see her",
                        "assistant": "That's rough. You're stuck between not having the energy and worrying about hurting people you care about."
                    }
                ],
                empathy_development="Energy loss → specific examples → breaking down the overwhelming feeling → guilt cycle → social consequences"
            ),
            
            ConversationFlow(
                flow_id="family_conflict_layers",
                name="Family Conflict Complexity",
                initial_situation="family problems",
                conversation_turns=[
                    {
                        "user": "My family is driving me crazy",
                        "assistant": "What's happening with them?"
                    },
                    {
                        "user": "My mom keeps asking when I'm going to get married and have kids",
                        "assistant": "How often does this come up?"
                    },
                    {
                        "user": "Every time I talk to her. Last week she sent me a link to a dating app",
                        "assistant": "Oh wow, she's really not being subtle about it."
                    },
                    {
                        "user": "No, and the thing is, I do want those things eventually. But I'm only 26",
                        "assistant": "So it's not that she's wrong about what you want, it's the pressure and timing?"
                    },
                    {
                        "user": "Right. And she acts like I'm running out of time or something",
                        "assistant": "That must make you feel like nothing you're doing right now matters to her."
                    },
                    {
                        "user": "Yeah, like my career and my life now don't count because I'm not married",
                        "assistant": "That's so invalidating. You're building a whole life and she's focused on this one piece that's missing."
                    }
                ],
                empathy_development="General family frustration → specific behavior → frequency/intensity → deeper issue of timing → feeling invalidated → understanding the real hurt"
            ),
            
            ConversationFlow(
                flow_id="friendship_drift_realization",
                name="Friendship Drifting Apart",
                initial_situation="friendship issues",
                conversation_turns=[
                    {
                        "user": "I think I'm losing my best friend",
                        "assistant": "What's making you feel that way?"
                    },
                    {
                        "user": "We used to text every day, now it's been two weeks since I heard from her",
                        "assistant": "Have you reached out to her?"
                    },
                    {
                        "user": "I texted her last week but she just gave me short answers",
                        "assistant": "That's so different from how she usually is?"
                    },
                    {
                        "user": "Yeah, normally we'd have these long conversations. Now it's just 'yeah' and 'ok'",
                        "assistant": "That shift must feel really confusing and hurtful."
                    },
                    {
                        "user": "I keep wondering if I did something wrong, but I can't think of anything",
                        "assistant": "That uncertainty is almost worse than knowing what happened, isn't it?"
                    },
                    {
                        "user": "Exactly. I'd rather she just tell me if she's mad at me",
                        "assistant": "The guessing game is torture. You just want to know where you stand."
                    }
                ],
                empathy_development="Fear of loss → specific changes → attempts to reconnect → confusion and hurt → self-doubt → desire for clarity"
            )
        ]
        
        return flows
    
    def _setup_natural_responses(self) -> Dict[str, List[str]]:
        """Setup natural response patterns (not clinical)"""
        
        return {
            'acknowledgment': [
                "That sounds really hard",
                "Wow, that's a lot to deal with",
                "That must be exhausting",
                "I can see why that would be frustrating",
                "That sounds overwhelming",
                "That's really tough"
            ],
            
            'curiosity': [
                "What's that been like for you?",
                "How long has this been going on?",
                "What do you think changed?",
                "How are you handling it?",
                "What's the hardest part about it?",
                "When did you first notice this?"
            ],
            
            'validation': [
                "That makes total sense",
                "Anyone would feel that way",
                "You're not overreacting",
                "That's completely understandable",
                "Of course you feel that way",
                "That's a normal response"
            ],
            
            'shared_understanding': [
                "I get what you mean",
                "That's such a hard spot to be in",
                "That sounds really lonely",
                "That must feel so isolating",
                "What a difficult situation",
                "That's got to be frustrating"
            ],
            
            'gentle_reframing': [
                "It sounds like you're really trying",
                "You're dealing with a lot right now",
                "That's not your fault",
                "You're being really hard on yourself",
                "That's a lot of pressure",
                "You're doing the best you can"
            ]
        }
    
    def _setup_follow_up_questions(self) -> List[str]:
        """Questions that show genuine interest and develop understanding"""
        
        return [
            "What does that look like day to day?",
            "How long have you been feeling this way?",
            "What's been the hardest part?",
            "When did you first notice this?",
            "What do you think triggered it?",
            "How is this affecting other parts of your life?",
            "What have you tried so far?",
            "Who else knows about this?",
            "What would help right now?",
            "What's different about this time?",
            "How are you sleeping through all this?",
            "What does your support system look like?",
            "What's your gut telling you?",
            "What would you tell a friend in this situation?",
            "What's the worst part about it?",
            "What keeps you up at night about this?",
            "How are you taking care of yourself?",
            "What's one thing that would make this easier?"
        ]
    
    def _setup_understanding_progression(self) -> Dict[str, List[str]]:
        """How understanding develops through conversation"""
        
        return {
            'surface_level': [
                "That sounds difficult",
                "I'm sorry you're going through this",
                "That must be hard"
            ],
            
            'getting_specific': [
                "So it's not just general stress, it's specifically about...",
                "It sounds like the real issue is...",
                "What I'm hearing is that..."
            ],
            
            'emotional_recognition': [
                "That must feel really lonely",
                "I can hear how frustrated you are",
                "That sounds incredibly overwhelming",
                "You must feel so stuck"
            ],
            
            'deeper_understanding': [
                "So you're not just dealing with X, you're also dealing with Y on top of it",
                "That's such a complex situation because...",
                "I can see why this is so hard - you're caught between...",
                "The really painful part seems to be..."
            ],
            
            'full_empathy': [
                "That's one of the loneliest feelings - when...",
                "That's such an impossible position to be in",
                "No wonder you're exhausted - you're dealing with all of this",
                "That would be hard for anyone, but especially when..."
            ]
        }
    
    def generate_natural_conversation(self, flow: ConversationFlow) -> Dict[str, Any]:
        """Generate a natural multi-turn conversation"""
        
        conversation_id = f"natural_{flow.flow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Build the conversation messages
        messages = []
        for turn in flow.conversation_turns:
            messages.append({
                'role': 'user',
                'content': turn['user']
            })
            messages.append({
                'role': 'assistant', 
                'content': turn['assistant']
            })
        
        conversation = {
            'id': conversation_id,
            'messages': messages,
            'metadata': {
                'source': 'natural_conversation_generator',
                'flow_type': flow.flow_id,
                'flow_name': flow.name,
                'turn_count': len(flow.conversation_turns),
                'empathy_development': flow.empathy_development,
                'generated_at': datetime.now().isoformat()
            },
            'quality_metrics': {
                'naturalness': self._assess_naturalness(messages),
                'empathy_development': self._assess_empathy_development(messages),
                'conversation_depth': len(flow.conversation_turns),
                'authenticity': self._assess_authenticity(messages)
            }
        }
        
        return conversation
    
    def _assess_naturalness(self, messages: List[Dict[str, str]]) -> float:
        """Assess how natural the conversation sounds"""
        
        naturalness_score = 0.8  # Base score
        
        assistant_messages = [msg['content'] for msg in messages if msg['role'] == 'assistant']
        
        # Check for clinical buzzwords (penalty)
        clinical_buzzwords = ['coping strategies', 'evidence-based', 'therapeutic', 'mindfulness-based', 'cognitive restructuring']
        for msg in assistant_messages:
            for buzzword in clinical_buzzwords:
                if buzzword in msg.lower():
                    naturalness_score -= 0.1
        
        # Check for natural language (bonus)
        natural_phrases = ['that sucks', 'wow', 'ugh', 'right?', 'exactly', 'that\'s rough', 'oh man']
        for msg in assistant_messages:
            for phrase in natural_phrases:
                if phrase in msg.lower():
                    naturalness_score += 0.05
        
        # Check message length (shorter is more natural)
        avg_length = sum(len(msg) for msg in assistant_messages) / len(assistant_messages)
        if avg_length < 100:  # Short responses are more natural
            naturalness_score += 0.1
        elif avg_length > 200:  # Long responses feel clinical
            naturalness_score -= 0.1
        
        return min(1.0, max(0.0, naturalness_score))
    
    def _assess_empathy_development(self, messages: List[Dict[str, str]]) -> float:
        """Assess how empathy develops through the conversation"""
        
        assistant_messages = [msg['content'] for msg in messages if msg['role'] == 'assistant']
        
        if len(assistant_messages) < 3:
            return 0.5  # Need multiple turns for empathy development
        
        # Check progression from surface to deep understanding
        empathy_score = 0.0
        
        # Early messages should show curiosity
        early_msgs = assistant_messages[:2]
        curiosity_indicators = ['what', 'how', 'when', 'why', '?']
        for msg in early_msgs:
            if any(indicator in msg.lower() for indicator in curiosity_indicators):
                empathy_score += 0.2
        
        # Later messages should show deeper understanding
        later_msgs = assistant_messages[2:]
        deep_understanding = ['that must', 'so you\'re', 'that\'s such a', 'no wonder', 'that would be']
        for msg in later_msgs:
            if any(phrase in msg.lower() for phrase in deep_understanding):
                empathy_score += 0.3
        
        return min(1.0, empathy_score)
    
    def _assess_authenticity(self, messages: List[Dict[str, str]]) -> float:
        """Assess how authentic the conversation feels"""
        
        authenticity_score = 0.7  # Base score
        
        assistant_messages = [msg['content'] for msg in messages if msg['role'] == 'assistant']
        
        # Check for authentic reactions
        authentic_reactions = ['that\'s so', 'oh wow', 'that\'s really', 'i can see', 'that sounds']
        for msg in assistant_messages:
            for reaction in authentic_reactions:
                if reaction in msg.lower():
                    authenticity_score += 0.05
        
        # Check for overly formal language (penalty)
        formal_language = ['furthermore', 'additionally', 'therefore', 'consequently']
        for msg in assistant_messages:
            for formal in formal_language:
                if formal in msg.lower():
                    authenticity_score -= 0.1
        
        return min(1.0, max(0.0, authenticity_score))
    
    def generate_conversation_batch(self, count_per_flow: int = 100) -> List[Dict[str, Any]]:
        """Generate a batch of natural conversations"""
        
        logger.info(f"🗣️ Generating {count_per_flow} conversations per flow ({len(self.conversation_flows)} flows)")
        
        all_conversations = []
        
        for flow in self.conversation_flows:
            logger.info(f"  Generating conversations for: {flow.name}")
            
            for i in range(count_per_flow):
                # Add some variation to each conversation
                varied_flow = self._add_conversation_variation(flow)
                conversation = self.generate_natural_conversation(varied_flow)
                all_conversations.append(conversation)
        
        logger.info(f"✅ Generated {len(all_conversations)} natural conversations")
        
        return all_conversations
    
    def _add_conversation_variation(self, base_flow: ConversationFlow) -> ConversationFlow:
        """Add slight variations to make conversations unique"""
        
        # Create variations in the responses while keeping the flow
        varied_turns = []
        
        for turn in base_flow.conversation_turns:
            # Keep user message the same, vary assistant response slightly
            user_msg = turn['user']
            assistant_msg = turn['assistant']
            
            # Add slight variations to assistant responses
            variations = {
                'What\'s going on': ['What\'s happening', 'What\'s up', 'Tell me more'],
                'That sounds': ['That seems', 'That feels like it would be', 'That must be'],
                'How long': ['When did this start', 'How long has this been happening'],
                'That\'s so': ['That\'s really', 'That\'s incredibly', 'That sounds so']
            }
            
            for original, replacements in variations.items():
                if original in assistant_msg:
                    assistant_msg = assistant_msg.replace(original, random.choice(replacements))
                    break
            
            varied_turns.append({
                'user': user_msg,
                'assistant': assistant_msg
            })
        
        return ConversationFlow(
            flow_id=base_flow.flow_id,
            name=base_flow.name,
            initial_situation=base_flow.initial_situation,
            conversation_turns=varied_turns,
            empathy_development=base_flow.empathy_development
        )
    
    def save_natural_conversations(self, conversations: List[Dict[str, Any]], output_path: str) -> bool:
        """Save natural conversations to file"""
        
        try:
            # Calculate summary statistics
            total_conversations = len(conversations)
            avg_turns = sum(len(conv['messages']) for conv in conversations) / (2 * total_conversations)  # Divide by 2 since each turn has user+assistant
            avg_naturalness = sum(conv['quality_metrics']['naturalness'] for conv in conversations) / total_conversations
            avg_empathy = sum(conv['quality_metrics']['empathy_development'] for conv in conversations) / total_conversations
            avg_authenticity = sum(conv['quality_metrics']['authenticity'] for conv in conversations) / total_conversations
            
            export_data = {
                'natural_conversations': conversations,
                'summary_statistics': {
                    'total_conversations': total_conversations,
                    'average_turns_per_conversation': avg_turns,
                    'average_naturalness_score': avg_naturalness,
                    'average_empathy_development': avg_empathy,
                    'average_authenticity_score': avg_authenticity,
                    'conversation_flows_used': len(self.conversation_flows)
                },
                'generation_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'generator_version': 'natural_conversation_v1.0',
                    'approach': 'multi_turn_empathy_development'
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved {total_conversations} natural conversations to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving conversations: {e}")
            return False

def main():
    """Generate natural multi-turn conversations"""
    print("🗣️ NATURAL MULTI-TURN CONVERSATION GENERATOR")
    print("=" * 60)
    print("💬 AUTHENTIC CONVERSATIONS WITH EMPATHY DEVELOPMENT")
    print("=" * 60)
    
    # Initialize generator
    generator = NaturalConversationGenerator()
    
    # Show conversation flows
    print(f"\n🎭 CONVERSATION FLOWS:")
    for flow in generator.conversation_flows:
        print(f"  {flow.name}:")
        print(f"    Turns: {len(flow.conversation_turns)}")
        print(f"    Empathy Development: {flow.empathy_development}")
    
    # Generate sample conversation
    print(f"\n🔬 SAMPLE NATURAL CONVERSATION:")
    sample_flow = generator.conversation_flows[1]  # Relationship doubt
    sample_conversation = generator.generate_natural_conversation(sample_flow)
    
    print(f"Flow: {sample_flow.name}")
    for i, message in enumerate(sample_conversation['messages']):
        role = "You" if message['role'] == 'user' else "Assistant"
        print(f"{role}: {message['content']}")
        if i < len(sample_conversation['messages']) - 1:
            print()
    
    print(f"\nQuality Metrics:")
    metrics = sample_conversation['quality_metrics']
    print(f"  Naturalness: {metrics['naturalness']:.3f}")
    print(f"  Empathy Development: {metrics['empathy_development']:.3f}")
    print(f"  Authenticity: {metrics['authenticity']:.3f}")
    print(f"  Conversation Depth: {metrics['conversation_depth']} turns")
    
    # Generate batch of conversations
    print(f"\n🚀 GENERATING NATURAL CONVERSATION BATCH:")
    conversations = generator.generate_conversation_batch(count_per_flow=200)
    
    # Save conversations
    output_path = "/home/vivi/pixelated/ai/data/processed/natural_conversations/natural_multi_turn_conversations.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    success = generator.save_natural_conversations(conversations, output_path)
    
    # Summary
    if conversations:
        total_conversations = len(conversations)
        avg_naturalness = sum(conv['quality_metrics']['naturalness'] for conv in conversations) / total_conversations
        avg_empathy = sum(conv['quality_metrics']['empathy_development'] for conv in conversations) / total_conversations
        avg_authenticity = sum(conv['quality_metrics']['authenticity'] for conv in conversations) / total_conversations
        
        print(f"\n🎯 GENERATION SUMMARY:")
        print(f"Total Conversations: {total_conversations:,}")
        print(f"Average Naturalness: {avg_naturalness:.3f}")
        print(f"Average Empathy Development: {avg_empathy:.3f}")
        print(f"Average Authenticity: {avg_authenticity:.3f}")
        print(f"Flows Used: {len(generator.conversation_flows)}")
    
    print(f"\n✅ NATURAL CONVERSATION GENERATION COMPLETE")
    print(f"📁 Conversations saved to: {output_path}")
    print(f"🎭 Real empathy through authentic dialogue!")
    
    return conversations

if __name__ == "__main__":
    main()
