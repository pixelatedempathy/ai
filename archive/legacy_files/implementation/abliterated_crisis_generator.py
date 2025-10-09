#!/usr/bin/env python3
"""
Crisis Conversation Generator using Remote Abliterated Model
Direct generation of sensitive mental health content for training purposes
Using remote Ollama server at api.pixelatedempathy.tech
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime
from pathlib import Path
import asyncio
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CrisisScenario:
    """Crisis scenario for conversation generation"""
    scenario_id: str
    crisis_type: str
    intensity_level: int  # 1-10
    demographic: str
    situation_context: str
    expected_duration: int  # conversation turns

class AbliteratedCrisisGenerator:
    """Generate crisis conversations using remote abliterated model"""
    
    def __init__(self, 
                 ollama_url: str = "https://api.pixelatedempathy.tech",
                 model_name: str = "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"):
        """Initialize with remote Ollama endpoint using Qwen3-abliterated thinking model"""
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.session = requests.Session()
        
        # Set timeout for 4B thinking model
        self.session.timeout = 120  # 2 minutes should be plenty for 4B model
        
        # Test connection
        self._test_connection()
        logger.info(f"✅ Abliterated Crisis Generator initialized")
        logger.info(f"🌐 Using remote server: {ollama_url}")
        logger.info(f"🧠 Model: Qwen3-Abliterated 4B with Thinking Capability")
    
    def _test_connection(self):
        """Test remote Ollama connection and model availability"""
        try:
            logger.info(f"🔄 Testing connection to {self.ollama_url}...")
            response = self.session.get(f"{self.ollama_url}/api/tags", timeout=30)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name not in model_names:
                    logger.warning(f"⚠️ Model {self.model_name} not found.")
                    logger.info(f"📋 Available models: {model_names}")
                    
                    # Try to find Qwen3 abliterated models
                    qwen3_models = [m for m in model_names if 'qwen3' in m.lower() and 'abliterated' in m.lower()]
                    if qwen3_models:
                        self.model_name = qwen3_models[0]
                        logger.info(f"🔄 Using found Qwen3 abliterated model: {self.model_name}")
                    else:
                        # Fallback to any abliterated model
                        abliterated_models = [m for m in model_names if 'abliterated' in m.lower()]
                        if abliterated_models:
                            self.model_name = abliterated_models[0]
                            logger.info(f"🔄 Using found abliterated model: {self.model_name}")
                        else:
                            raise Exception(f"No abliterated model found on server")
                else:
                    logger.info(f"✅ Model {self.model_name} available and ready")
                    logger.info(f"🧠 Using Qwen3 thinking model with abliteration!")
            else:
                raise Exception(f"Remote Ollama not responding: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Remote Ollama connection failed: {e}")
            logger.error(f"🌐 Check if {self.ollama_url} is accessible")
            raise
        except Exception as e:
            logger.error(f"❌ Ollama setup error: {e}")
            raise
    
    def generate_crisis_conversation(self, scenario: CrisisScenario) -> Dict[str, Any]:
        """Generate crisis conversation directly without guardrails"""
        
        logger.info(f"🔄 Generating crisis conversation: {scenario.crisis_type}")
        logger.info(f"   Intensity: {scenario.intensity_level}/10")
        logger.info(f"   Context: {scenario.situation_context[:50]}...")
        
        # Create direct, unfiltered prompt
        prompt = self._create_crisis_prompt(scenario)
        
        # Generate with remote abliterated model
        start_time = time.time()
        conversation = self._generate_with_ollama(prompt)
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Generated in {generation_time:.2f}s")
        
        # Parse and structure the conversation
        structured_conversation = self._parse_conversation(conversation, scenario)
        
        # Add therapeutic responses
        enhanced_conversation = self._add_therapeutic_responses(structured_conversation, scenario)
        
        # Add generation metadata
        enhanced_conversation['generation_metadata'] = {
            'model_used': self.model_name,
            'server_endpoint': self.ollama_url,
            'generation_time_seconds': generation_time,
            'prompt_length': len(prompt)
        }
        
        return enhanced_conversation
    
    def _create_crisis_prompt(self, scenario: CrisisScenario) -> str:
        """Create direct crisis conversation prompt"""
        
        prompt = f"""Generate a realistic mental health crisis conversation for training AI crisis intervention systems.

SCENARIO DETAILS:
- Crisis Type: {scenario.crisis_type}
- Intensity Level: {scenario.intensity_level}/10
- Demographics: {scenario.demographic}
- Context: {scenario.situation_context}
- Conversation Length: {scenario.expected_duration} turns

REQUIREMENTS:
- Generate REALISTIC crisis language that people actually use
- Include specific crisis indicators (suicidal thoughts, self-harm, hopelessness)
- Show emotional progression and escalation
- Use natural, authentic language (not clinical or euphemistic)
- Include specific details that make it feel real
- Show the person's actual thought patterns and feelings
- Make it feel like a real person in genuine crisis

CONVERSATION FORMAT:
User: [realistic crisis message with authentic emotional language]
Assistant: [empathetic, appropriate therapeutic response]
User: [follow-up showing more crisis details and emotional pain]
Assistant: [therapeutic response with validation and resources]
[Continue for {scenario.expected_duration} turns total]

CRISIS LANGUAGE TO INCLUDE (be direct and realistic):
- Direct expressions of suicidal thoughts ("I want to kill myself", "I can't go on")
- Specific self-harm behaviors or urges ("I've been cutting", "I want to hurt myself")
- Feelings of hopelessness and worthlessness ("I'm worthless", "There's no point")
- Isolation and disconnection ("Nobody cares", "I'm all alone")
- Specific triggers or precipitating events (real life situations)
- Raw emotional pain and desperation (authentic human suffering)

THERAPEUTIC RESPONSES SHOULD INCLUDE:
- Validation of their pain
- Expression of concern for their safety
- Specific crisis resources (988 hotline, emergency services)
- Hope and connection
- Encouragement to seek immediate help
- Safety planning elements

Generate a realistic, authentic crisis conversation now:"""

        return prompt
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate content using Ollama /api/generate endpoint (the one that works!)"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
                "max_tokens": 2500,
                "stop": ["END_CONVERSATION"],
                "repeat_penalty": 1.1,
                "top_k": 40
            }
        }
        
        try:
            logger.info(f"🌐 Sending request to /api/generate endpoint...")
            response = self.session.post(
                f"{self.ollama_url}/api/generate",  # Use the working endpoint!
                json=payload,
                timeout=120  # 2 minutes should be plenty based on logs
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                if not generated_text.strip():
                    raise Exception("Empty response from model")
                
                logger.info(f"✅ Received {len(generated_text)} characters")
                return generated_text
            else:
                error_msg = f"Generation failed: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text[:200]}"
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            logger.error("❌ Request timed out")
            raise Exception("Generation timed out")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error: {e}")
            raise Exception(f"Network error: {e}")
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise
    
    def _parse_conversation(self, raw_conversation: str, scenario: CrisisScenario) -> Dict[str, Any]:
        """Parse raw conversation into structured format"""
        
        lines = raw_conversation.strip().split('\n')
        turns = []
        current_speaker = None
        current_message = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('User:'):
                if current_speaker and current_message:
                    turns.append({
                        'speaker': current_speaker,
                        'message': current_message.strip()
                    })
                current_speaker = 'user'
                current_message = line[5:].strip()
            elif line.startswith('Assistant:'):
                if current_speaker and current_message:
                    turns.append({
                        'speaker': current_speaker,
                        'message': current_message.strip()
                    })
                current_speaker = 'assistant'
                current_message = line[10:].strip()
            elif line and current_speaker:
                current_message += " " + line
        
        # Add final turn
        if current_speaker and current_message:
            turns.append({
                'speaker': current_speaker,
                'message': current_message.strip()
            })
        
        # Validate we have a proper conversation
        if len(turns) < 2:
            logger.warning(f"⚠️ Short conversation generated: {len(turns)} turns")
        
        return {
            'conversation_id': str(uuid.uuid4()),
            'scenario': scenario,
            'turns': turns,
            'generated_at': datetime.now().isoformat(),
            'crisis_indicators_detected': self._detect_crisis_indicators(turns),
            'emotional_intensity_progression': self._analyze_emotional_progression(turns),
            'conversation_quality': self._assess_conversation_quality(turns)
        }
    
    def _detect_crisis_indicators(self, turns: List[Dict[str, str]]) -> List[str]:
        """Detect crisis indicators in conversation"""
        
        crisis_patterns = [
            # Suicidal ideation
            'suicide', 'kill myself', 'end it all', 'want to die',
            'better off dead', 'end my life', 'take my own life',
            
            # Self-harm
            'hurt myself', 'cut myself', 'cutting', 'self-harm',
            'burn myself', 'hit myself', 'punish myself',
            
            # Methods (concerning but important for detection)
            'overdose', 'pills', 'jump off', 'hang myself', 'gun',
            
            # Hopelessness
            'worthless', 'hopeless', 'no point', 'can\'t go on',
            'nothing matters', 'no way out', 'trapped',
            
            # Isolation
            'nobody cares', 'all alone', 'no one understands',
            'better without me', 'burden to everyone'
        ]
        
        detected_indicators = []
        
        for turn in turns:
            if turn['speaker'] == 'user':
                message_lower = turn['message'].lower()
                for pattern in crisis_patterns:
                    if pattern in message_lower:
                        detected_indicators.append(pattern)
        
        return list(set(detected_indicators))
    
    def _analyze_emotional_progression(self, turns: List[Dict[str, str]]) -> List[int]:
        """Analyze emotional intensity progression through conversation"""
        
        intensity_keywords = {
            1: ['fine', 'okay', 'alright', 'good'],
            2: ['tired', 'stressed', 'worried'],
            3: ['sad', 'down', 'upset', 'frustrated'],
            4: ['really sad', 'very upset', 'struggling'],
            5: ['depressed', 'anxious', 'overwhelmed'],
            6: ['really depressed', 'very anxious', 'can\'t cope'],
            7: ['desperate', 'hopeless', 'breaking down'],
            8: ['want to disappear', 'can\'t take it', 'falling apart'],
            9: ['want to die', 'kill myself', 'end it all'],
            10: ['have a plan', 'ready to do it', 'tonight']
        }
        
        progression = []
        
        for turn in turns:
            if turn['speaker'] == 'user':
                message_lower = turn['message'].lower()
                max_intensity = 1
                
                for intensity, keywords in intensity_keywords.items():
                    if any(keyword in message_lower for keyword in keywords):
                        max_intensity = max(max_intensity, intensity)
                
                progression.append(max_intensity)
        
        return progression
    
    def _assess_conversation_quality(self, turns: List[Dict[str, str]]) -> Dict[str, float]:
        """Assess the quality of the generated conversation"""
        
        user_turns = [t for t in turns if t['speaker'] == 'user']
        assistant_turns = [t for t in turns if t['speaker'] == 'assistant']
        
        # Authenticity: Does it sound like real crisis language?
        authenticity_indicators = [
            'i feel', 'i can\'t', 'i want', 'i\'m so', 'i don\'t',
            'why me', 'what\'s the point', 'i just', 'i hate'
        ]
        
        authenticity_score = 0
        for turn in user_turns:
            message_lower = turn['message'].lower()
            matches = sum(1 for indicator in authenticity_indicators if indicator in message_lower)
            authenticity_score += min(matches / len(authenticity_indicators), 1.0)
        
        authenticity_score = authenticity_score / len(user_turns) if user_turns else 0
        
        # Therapeutic quality: Do assistant responses include appropriate elements?
        therapeutic_indicators = [
            'i\'m concerned', 'you\'re not alone', 'i hear you',
            'that sounds', 'help', 'support', 'crisis', 'hotline',
            'safe', 'care about you'
        ]
        
        therapeutic_score = 0
        for turn in assistant_turns:
            message_lower = turn['message'].lower()
            matches = sum(1 for indicator in therapeutic_indicators if indicator in message_lower)
            therapeutic_score += min(matches / len(therapeutic_indicators), 1.0)
        
        therapeutic_score = therapeutic_score / len(assistant_turns) if assistant_turns else 0
        
        return {
            'authenticity': round(authenticity_score, 3),
            'therapeutic_quality': round(therapeutic_score, 3),
            'conversation_length': len(turns),
            'crisis_depth': len(self._detect_crisis_indicators(turns))
        }
    
    def _add_therapeutic_responses(self, conversation: Dict[str, Any], scenario: CrisisScenario) -> Dict[str, Any]:
        """Enhance assistant responses with therapeutic techniques"""
        
        enhanced_turns = []
        
        for i, turn in enumerate(conversation['turns']):
            enhanced_turns.append(turn.copy())
            
            # If this is an assistant response, enhance it
            if turn['speaker'] == 'assistant':
                enhanced_response = self._enhance_therapeutic_response(
                    turn['message'], 
                    scenario.intensity_level,
                    i // 2 + 1,  # Turn number
                    conversation['crisis_indicators_detected']
                )
                enhanced_turns[-1]['message'] = enhanced_response
                enhanced_turns[-1]['therapeutic_techniques'] = self._identify_techniques(enhanced_response)
        
        conversation['turns'] = enhanced_turns
        return conversation
    
    def _enhance_therapeutic_response(self, response: str, intensity_level: int, turn_number: int, crisis_indicators: List[str]) -> str:
        """Enhance response with appropriate therapeutic elements"""
        
        # For high-intensity crisis situations (8-10)
        if intensity_level >= 8:
            # Add immediate safety concern if not present
            if not any(word in response.lower() for word in ['concerned', 'worried', 'safety']):
                response = f"I'm very concerned about what you're sharing. {response}"
            
            # Add crisis resources after turn 2
            if turn_number >= 2 and not any(word in response.lower() for word in ['988', 'hotline', 'crisis']):
                response += " The National Suicide Prevention Lifeline is available 24/7 at 988. You don't have to go through this alone."
            
            # Add immediate help encouragement
            if turn_number >= 3 and 'emergency' not in response.lower():
                response += " Can we talk about getting you some immediate support right now?"
        
        # For moderate-high intensity (6-7)
        elif intensity_level >= 6:
            if not any(word in response.lower() for word in ['support', 'help', 'care']):
                response += " I want you to know that I care about what happens to you."
            
            if turn_number >= 3 and 'professional' not in response.lower():
                response += " Have you been able to talk to a counselor or therapist about this?"
        
        # For moderate intensity (4-5)
        elif intensity_level >= 4:
            if not any(word in response.lower() for word in ['understand', 'hear', 'sounds']):
                response = f"I hear how much pain you're in. {response}"
        
        return response
    
    def _identify_techniques(self, response: str) -> List[str]:
        """Identify therapeutic techniques used in response"""
        
        techniques = []
        response_lower = response.lower()
        
        technique_indicators = {
            'validation': ['understand', 'hear you', 'makes sense', 'valid', 'that sounds'],
            'empathy': ['sorry you\'re going through', 'must be difficult', 'sounds painful', 'i can imagine'],
            'safety_planning': ['safety', 'safe', 'crisis plan', 'hotline', '988'],
            'hope_instillation': ['hope', 'better', 'through this', 'not alone', 'care about you'],
            'resource_connection': ['help', 'support', 'counselor', 'therapist', 'hotline', 'emergency'],
            'active_listening': ['tell me more', 'what\'s that like', 'help me understand', 'can you share'],
            'immediate_safety': ['concerned', 'worried', 'right now', 'immediate', 'tonight'],
            'connection': ['you\'re not alone', 'i\'m here', 'we can', 'together']
        }
        
        for technique, indicators in technique_indicators.items():
            if any(indicator in response_lower for indicator in indicators):
                techniques.append(technique)
        
        return techniques

def create_crisis_scenarios() -> List[CrisisScenario]:
    """Create comprehensive crisis scenarios for generation"""
    
    scenarios = [
        # Suicidal Ideation Scenarios
        CrisisScenario(
            scenario_id="suicide_ideation_college",
            crisis_type="suicidal_ideation",
            intensity_level=9,
            demographic="college_student_20s",
            situation_context="Academic pressure, social isolation, recent breakup, financial stress",
            expected_duration=8
        ),
        
        CrisisScenario(
            scenario_id="suicide_planning_midlife",
            crisis_type="suicide_planning",
            intensity_level=10,
            demographic="middle_aged_professional_45",
            situation_context="Job loss after 20 years, divorce proceedings, financial ruin, health problems",
            expected_duration=10
        ),
        
        CrisisScenario(
            scenario_id="suicide_ideation_elderly",
            crisis_type="suicidal_ideation",
            intensity_level=9,
            demographic="elderly_person_70s",
            situation_context="Spouse died, chronic pain, feeling like a burden, social isolation",
            expected_duration=7
        ),
        
        # Self-Harm Scenarios
        CrisisScenario(
            scenario_id="self_harm_teenager",
            crisis_type="self_harm",
            intensity_level=7,
            demographic="teenager_16_17",
            situation_context="Family conflict, school bullying, identity struggles, social media pressure",
            expected_duration=6
        ),
        
        CrisisScenario(
            scenario_id="self_harm_escalation_adult",
            crisis_type="self_harm_escalation",
            intensity_level=8,
            demographic="young_adult_early_20s",
            situation_context="Relationship trauma, work stress, substance use, childhood abuse history",
            expected_duration=7
        ),
        
        CrisisScenario(
            scenario_id="self_harm_relapse",
            crisis_type="self_harm_relapse",
            intensity_level=8,
            demographic="adult_30s",
            situation_context="Previously recovered, recent major stressor triggered relapse, shame and guilt",
            expected_duration=8
        ),
        
        # Severe Depression Scenarios
        CrisisScenario(
            scenario_id="postpartum_depression_crisis",
            crisis_type="severe_depression",
            intensity_level=8,
            demographic="new_mother_20s_30s",
            situation_context="Postpartum depression, overwhelming responsibility, partner not understanding, isolation",
            expected_duration=9
        ),
        
        CrisisScenario(
            scenario_id="treatment_resistant_depression",
            crisis_type="severe_depression",
            intensity_level=9,
            demographic="adult_40s",
            situation_context="Multiple failed treatments, feeling hopeless about recovery, chronic depression for years",
            expected_duration=8
        ),
        
        # Crisis with Substance Use
        CrisisScenario(
            scenario_id="addiction_relapse_crisis",
            crisis_type="crisis_with_substance_use",
            intensity_level=9,
            demographic="adult_30s_40s",
            situation_context="Addiction relapse after sobriety, family estrangement, job loss, health problems",
            expected_duration=8
        ),
        
        # Domestic Violence Crisis
        CrisisScenario(
            scenario_id="domestic_violence_escalation",
            crisis_type="domestic_violence_crisis",
            intensity_level=9,
            demographic="adult_woman_20s_40s",
            situation_context="Escalating physical abuse, fear for life, children involved, nowhere to go",
            expected_duration=7
        ),
        
        # LGBTQ+ Crisis
        CrisisScenario(
            scenario_id="lgbtq_family_rejection",
            crisis_type="identity_crisis_rejection",
            intensity_level=8,
            demographic="teenager_young_adult_17_22",
            situation_context="Came out to religious family, complete rejection, kicked out of home, suicidal thoughts",
            expected_duration=8
        ),
        
        # Trauma-Related Crisis
        CrisisScenario(
            scenario_id="ptsd_flashback_crisis",
            crisis_type="trauma_crisis",
            intensity_level=8,
            demographic="military_veteran_30s",
            situation_context="Severe PTSD flashbacks, combat trauma, survivor guilt, relationship breakdown",
            expected_duration=7
        ),
        
        # Eating Disorder Crisis
        CrisisScenario(
            scenario_id="eating_disorder_medical_crisis",
            crisis_type="eating_disorder_crisis",
            intensity_level=8,
            demographic="young_adult_18_25",
            situation_context="Severe anorexia, medical complications, family pressure, perfectionism, control issues",
            expected_duration=6
        ),
        
        # Psychosis/Mental Health Crisis
        CrisisScenario(
            scenario_id="first_psychotic_episode",
            crisis_type="psychosis_crisis",
            intensity_level=9,
            demographic="young_adult_early_20s",
            situation_context="First psychotic episode, hearing voices, paranoid thoughts, scared and confused",
            expected_duration=6
        ),
        
        # Financial Crisis Leading to Mental Health Crisis
        CrisisScenario(
            scenario_id="financial_ruin_crisis",
            crisis_type="situational_crisis",
            intensity_level=8,
            demographic="middle_aged_parent_40s",
            situation_context="Lost home to foreclosure, bankruptcy, children affected, feeling like a failure",
            expected_duration=8
        )
    ]
    
    return scenarios

async def generate_crisis_conversation_library():
    """Generate comprehensive crisis conversation library using remote abliterated model"""
    
    print("🚨 GENERATING CRISIS CONVERSATION LIBRARY")
    print("=" * 80)
    print("🎯 USING QWEN3-ABLITERATED 4B THINKING MODEL")
    print("🧠 Step-by-step reasoning capability for complex crisis scenarios")
    print("🌐 Server: https://api.pixelatedempathy.tech")
    print("=" * 80)
    
    # Initialize generator with remote server
    try:
        generator = AbliteratedCrisisGenerator()
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return None
    
    # Create scenarios
    scenarios = create_crisis_scenarios()
    print(f"📋 Created {len(scenarios)} crisis scenarios")
    
    # Generate conversations
    all_conversations = []
    successful_generations = 0
    failed_generations = 0
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🔄 Generating scenario {i}/{len(scenarios)}: {scenario.crisis_type}")
        print(f"   📊 Intensity: {scenario.intensity_level}/10")
        print(f"   👥 Demographics: {scenario.demographic}")
        print(f"   📝 Context: {scenario.situation_context[:60]}...")
        
        try:
            # Generate multiple variations per scenario
            variations_per_scenario = 2  # 2 variations per scenario for now
            
            for variation in range(variations_per_scenario):
                print(f"   🔄 Generating variation {variation + 1}/{variations_per_scenario}...")
                
                conversation = generator.generate_crisis_conversation(scenario)
                all_conversations.append(conversation)
                successful_generations += 1
                
                # Show generation results
                quality = conversation['conversation_quality']
                crisis_indicators = conversation['crisis_indicators_detected']
                
                print(f"   ✅ Variation {variation + 1} completed:")
                print(f"      🎯 Crisis indicators: {len(crisis_indicators)}")
                print(f"      📈 Authenticity: {quality['authenticity']:.3f}")
                print(f"      🏥 Therapeutic quality: {quality['therapeutic_quality']:.3f}")
                print(f"      💬 Turns: {quality['conversation_length']}")
                
                # Brief pause between generations to be nice to the server
                await asyncio.sleep(1)
        
        except Exception as e:
            print(f"   ❌ Failed to generate scenario {scenario.scenario_id}: {e}")
            failed_generations += 1
            continue
    
    # Save conversations
    output_dir = Path("/home/vivi/pixelated/ai/data/processed/crisis_conversations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "abliterated_crisis_library.json"
    
    library_data = {
        'generated_at': datetime.now().isoformat(),
        'generator_model': 'huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M',
        'model_info': {
            'architecture': 'Qwen3 with Thinking Capability',
            'parameters': '4B',
            'abliteration': 'huihui_ai safety removal',
            'special_features': 'Step-by-step reasoning, crisis scenario analysis',
            'advantages': 'Thinking capability, fast generation, proven abliteration'
        },
        'server_endpoint': 'https://api.pixelatedempathy.tech',
        'total_conversations': len(all_conversations),
        'successful_generations': successful_generations,
        'failed_generations': failed_generations,
        'scenarios_covered': len(scenarios),
        'generation_summary': {
            'crisis_types': list(set(s.crisis_type for s in scenarios)),
            'intensity_levels': list(set(s.intensity_level for s in scenarios)),
            'demographics': list(set(s.demographic for s in scenarios))
        },
        'conversations': all_conversations
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(library_data, f, indent=2, ensure_ascii=False)
    
    # Generate summary statistics
    total_crisis_indicators = sum(len(conv['crisis_indicators_detected']) for conv in all_conversations)
    avg_authenticity = sum(conv['conversation_quality']['authenticity'] for conv in all_conversations) / len(all_conversations) if all_conversations else 0
    avg_therapeutic = sum(conv['conversation_quality']['therapeutic_quality'] for conv in all_conversations) / len(all_conversations) if all_conversations else 0
    
    print(f"\n✅ CRISIS LIBRARY GENERATION COMPLETE")
    print("=" * 80)
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Generation Statistics:")
    print(f"   • Total conversations: {len(all_conversations)}")
    print(f"   • Successful generations: {successful_generations}")
    print(f"   • Failed generations: {failed_generations}")
    print(f"   • Success rate: {(successful_generations/(successful_generations+failed_generations)*100):.1f}%")
    print(f"   • Crisis scenarios covered: {len(scenarios)}")
    print(f"   • Total crisis indicators detected: {total_crisis_indicators}")
    print(f"   • Average authenticity score: {avg_authenticity:.3f}")
    print(f"   • Average therapeutic quality: {avg_therapeutic:.3f}")
    print(f"🚨 Ready for crisis intervention training!")
    print("=" * 80)
    
    return library_data

if __name__ == "__main__":
    asyncio.run(generate_crisis_conversation_library())
