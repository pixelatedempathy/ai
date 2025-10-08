#!/usr/bin/env python3
"""
Conversation Data Migrator - Phase 2 Implementation
Convert existing JSON conversations to new database structure with branching logic
"""

import json
import logging
import sqlite3
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Individual conversation turn"""
    user_message: str
    assistant_response: str
    turn_number: int
    emotional_intensity: int = 5

@dataclass
class ConversationFlow:
    """Complete conversation flow"""
    flow_name: str
    turns: List[ConversationTurn]
    quality_metrics: Dict[str, float]
    flow_description: str = ""

class ConversationDataMigrator:
    """Migrate JSON conversations to database structure"""
    
    def __init__(self, db_path: str = "/home/vivi/pixelated/ai/data/conversation_system.db"):
        """Initialize migrator"""
        self.db_path = db_path
        self.connection = None
        self.personality_types = [
            "direct_practical",
            "gentle_nurturing", 
            "analytical_problem_solving",
            "casual_friend_like"
        ]
        logger.info("✅ Conversation Data Migrator initialized")
    
    def connect_database(self) -> bool:
        """Connect to the conversation database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            logger.info(f"✅ Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def load_existing_conversations(self, json_path: str) -> List[Dict[str, Any]]:
        """Load existing JSON conversations"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle the actual JSON structure with 'natural_conversations' key
            conversations = data.get('natural_conversations', [])
            logger.info(f"✅ Loaded {len(conversations)} conversations from JSON")
            return conversations
            
        except Exception as e:
            logger.error(f"❌ Error loading conversations: {e}")
            return []
    
    def analyze_conversation_patterns(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation patterns for migration planning"""
        
        analysis = {
            'total_conversations': len(conversations),
            'flows_by_type': {},
            'turn_length_distribution': {},
            'emotional_patterns': {},
            'branching_opportunities': []
        }
        
        for conv in conversations:
            # Handle new JSON structure
            metadata = conv.get('metadata', {})
            flow_name = metadata.get('flow_name', 'unknown')
            messages = conv.get('messages', [])
            
            # Count flows by type
            if flow_name not in analysis['flows_by_type']:
                analysis['flows_by_type'][flow_name] = 0
            analysis['flows_by_type'][flow_name] += 1
            
            # Turn length distribution (count user messages only)
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            turn_count = len(user_messages)
            if turn_count not in analysis['turn_length_distribution']:
                analysis['turn_length_distribution'][turn_count] = 0
            analysis['turn_length_distribution'][turn_count] += 1
            
            # Identify branching opportunities
            branching_points = self._identify_branching_points_new_format(messages)
            analysis['branching_opportunities'].extend(branching_points)
        
        return analysis
    
    def _identify_branching_points_new_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Identify potential branching points in conversations (new format)"""
        branching_points = []
        
        for i in range(len(messages) - 1):
            if messages[i].get('role') == 'user' and i + 1 < len(messages):
                user_msg = messages[i].get('content', '')
                assistant_response = messages[i + 1].get('content', '') if messages[i + 1].get('role') == 'assistant' else ''
                
                # Look for questions that could branch
                if any(keyword in user_msg.lower() for keyword in [
                    'work', 'job', 'boss', 'stress', 'anxiety', 'depression',
                    'relationship', 'family', 'friend', 'breakup', 'divorce'
                ]):
                    branching_points.append({
                        'turn_number': (i // 2) + 1,
                        'user_message': user_msg,
                        'assistant_response': assistant_response,
                        'potential_branches': self._suggest_branches(user_msg)
                    })
        
        return branching_points
    
    def _identify_branching_points(self, turns: List[str]) -> List[Dict[str, Any]]:
        """Identify potential branching points in conversations"""
        branching_points = []
        
        for i in range(0, len(turns) - 1, 2):  # Step by 2 (user + assistant pairs)
            if i + 1 < len(turns):
                user_msg = turns[i]
                assistant_response = turns[i + 1]
                
                # Look for questions that could branch
                if any(keyword in user_msg.lower() for keyword in [
                    'work', 'job', 'boss', 'stress', 'anxiety', 'depression',
                    'relationship', 'family', 'friend', 'breakup', 'divorce'
                ]):
                    branching_points.append({
                        'turn_number': i // 2 + 1,
                        'user_message': user_msg,
                        'assistant_response': assistant_response,
                        'potential_branches': self._suggest_branches(user_msg)
                    })
        
        return branching_points
    
    def _suggest_branches(self, user_message: str) -> List[str]:
        """Suggest potential conversation branches based on user message"""
        branches = []
        msg_lower = user_message.lower()
        
        # Work-related branches
        if any(word in msg_lower for word in ['work', 'job', 'boss', 'office']):
            branches.extend(['workplace_conflict', 'work_stress', 'career_anxiety'])
        
        # Relationship branches
        if any(word in msg_lower for word in ['relationship', 'partner', 'boyfriend', 'girlfriend']):
            branches.extend(['relationship_doubts', 'communication_issues', 'breakup_concerns'])
        
        # Mental health branches
        if any(word in msg_lower for word in ['anxious', 'depressed', 'stressed', 'overwhelmed']):
            branches.extend(['anxiety_support', 'depression_help', 'stress_management'])
        
        # Family branches
        if any(word in msg_lower for word in ['family', 'parents', 'mom', 'dad', 'sibling']):
            branches.extend(['family_conflict', 'parent_issues', 'family_dynamics'])
        
        return branches[:3]  # Limit to 3 potential branches
    
    def create_personality_variations(self, original_response: str, context: str = "") -> Dict[str, str]:
        """Create personality variations of assistant responses"""
        
        variations = {}
        
        # Direct/Practical personality
        variations['direct_practical'] = self._make_direct_practical(original_response)
        
        # Gentle/Nurturing personality  
        variations['gentle_nurturing'] = self._make_gentle_nurturing(original_response)
        
        # Analytical/Problem-solving personality
        variations['analytical_problem_solving'] = self._make_analytical(original_response)
        
        # Casual/Friend-like personality
        variations['casual_friend_like'] = self._make_casual_friendly(original_response)
        
        return variations
    
    def _make_direct_practical(self, response: str) -> str:
        """Convert to direct, practical style"""
        # Keep the core message but make it more direct
        if "?" in response:
            return response  # Questions are already direct
        
        # Make statements more direct
        direct_response = response.replace("That sounds", "That is")
        direct_response = direct_response.replace("It seems like", "")
        direct_response = direct_response.replace("I can imagine", "")
        
        return direct_response.strip()
    
    def _make_gentle_nurturing(self, response: str) -> str:
        """Convert to gentle, nurturing style"""
        # Add gentle language
        gentle_starters = ["I can hear that", "It sounds like", "I sense that"]
        
        if not any(starter in response for starter in gentle_starters):
            if response.endswith("?"):
                return response  # Keep questions as-is
            else:
                return f"I can hear that {response.lower()}"
        
        return response
    
    def _make_analytical(self, response: str) -> str:
        """Convert to analytical, problem-solving style"""
        # Add analytical framing
        if "?" in response:
            # Make questions more specific
            analytical_response = response.replace("What", "Specifically, what")
            analytical_response = analytical_response.replace("How", "In what way")
            return analytical_response
        
        return response
    
    def _make_casual_friendly(self, response: str) -> str:
        """Convert to casual, friend-like style"""
        # Add casual language
        casual_response = response.replace("That sounds", "That sounds really")
        casual_response = casual_response.replace("I understand", "I totally get that")
        
        # Add casual expressions
        if not any(casual in casual_response.lower() for casual in ["yeah", "totally", "really", "ugh"]):
            if "?" not in casual_response:
                casual_response = f"Yeah, {casual_response.lower()}"
        
        return casual_response
    
    def estimate_emotional_intensity(self, user_message: str, assistant_response: str) -> int:
        """Estimate emotional intensity (1-10) based on conversation content"""
        
        intensity_keywords = {
            1: ['fine', 'okay', 'alright', 'good'],
            3: ['tired', 'busy', 'stressed', 'worried'],
            5: ['anxious', 'upset', 'frustrated', 'sad'],
            7: ['overwhelmed', 'depressed', 'angry', 'devastated'],
            9: ['crisis', 'emergency', 'suicidal', 'desperate', 'can\'t take it']
        }
        
        combined_text = (user_message + " " + assistant_response).lower()
        
        max_intensity = 1
        for intensity, keywords in intensity_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                max_intensity = max(max_intensity, intensity)
        
        return max_intensity
    
    def migrate_conversation_to_database(self, conversation: Dict[str, Any]) -> bool:
        """Migrate single conversation to database structure"""
        try:
            cursor = self.connection.cursor()
            
            # Extract conversation data from new format
            metadata = conversation.get('metadata', {})
            flow_name = metadata.get('flow_name', 'Unknown Flow')
            messages = conversation.get('messages', [])
            quality_metrics = metadata.get('quality_metrics', {})
            
            # Create flow ID
            flow_id = str(uuid.uuid4())
            
            # Insert conversation flow
            cursor.execute("""
                INSERT INTO conversation_flows 
                (flow_id, name, description, emotional_range_min, emotional_range_max, flow_tags, target_demographics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                flow_id,
                flow_name,
                f"Migrated from natural conversation generator: {flow_name}",
                1,  # Min emotional intensity
                10, # Max emotional intensity
                json.dumps([metadata.get('flow_type', flow_name.lower().replace(' ', '_'))]),
                json.dumps(['general'])
            ))
            
            # Process conversation turns (user + assistant pairs)
            previous_node_id = None
            starting_node_id = None
            turn_number = 1
            
            for i in range(0, len(messages) - 1, 2):  # Process user+assistant pairs
                if i + 1 < len(messages) and messages[i].get('role') == 'user' and messages[i + 1].get('role') == 'assistant':
                    user_msg = messages[i].get('content', '')
                    assistant_response = messages[i + 1].get('content', '')
                    
                    # Create node ID
                    node_id = str(uuid.uuid4())
                    if turn_number == 1:
                        starting_node_id = node_id
                    
                    # Estimate emotional intensity
                    emotional_intensity = self.estimate_emotional_intensity(user_msg, assistant_response)
                    
                    # Insert conversation node
                    cursor.execute("""
                        INSERT INTO conversation_nodes 
                        (node_id, flow_id, user_message, emotional_intensity, context_tags, follow_up_triggers)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        node_id,
                        flow_id,
                        user_msg,
                        emotional_intensity,
                        json.dumps([metadata.get('flow_type', flow_name.lower().replace(' ', '_'))]),
                        json.dumps([])
                    ))
                    
                    # Create personality variations
                    personality_variations = self.create_personality_variations(assistant_response, user_msg)
                    
                    # Insert assistant responses for each personality
                    for personality_type, response_text in personality_variations.items():
                        response_id = str(uuid.uuid4())
                        
                        cursor.execute("""
                            INSERT INTO assistant_responses 
                            (response_id, node_id, personality_type, response_text, response_length, empathy_score, naturalness_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            response_id,
                            node_id,
                            personality_type,
                            response_text,
                            len(response_text),
                            quality_metrics.get('empathy_development', 0.7),
                            quality_metrics.get('naturalness', 0.8)
                        ))
                    
                    # Create transition from previous node
                    if previous_node_id:
                        transition_id = str(uuid.uuid4())
                        cursor.execute("""
                            INSERT INTO node_transitions 
                            (transition_id, from_node_id, to_node_id, condition_type, condition_value, probability_weight)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            transition_id,
                            previous_node_id,
                            node_id,
                            'sequential',
                            json.dumps({'type': 'next_turn'}),
                            1.0
                        ))
                    
                    previous_node_id = node_id
                    turn_number += 1
            
            # Update flow with starting node
            if starting_node_id:
                cursor.execute("""
                    UPDATE conversation_flows 
                    SET starting_node_id = ? 
                    WHERE flow_id = ?
                """, (starting_node_id, flow_id))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating conversation: {e}")
            self.connection.rollback()
            return False
    
    def migrate_all_conversations(self, json_path: str) -> Dict[str, Any]:
        """Migrate all conversations from JSON to database"""
        
        # Load conversations
        conversations = self.load_existing_conversations(json_path)
        if not conversations:
            return {'success': False, 'error': 'No conversations loaded'}
        
        # Analyze patterns
        analysis = self.analyze_conversation_patterns(conversations)
        logger.info(f"📊 Conversation analysis: {len(analysis['flows_by_type'])} flow types")
        
        # Migrate conversations
        successful_migrations = 0
        failed_migrations = 0
        
        for i, conversation in enumerate(conversations):
            logger.info(f"Migrating conversation {i+1}/{len(conversations)}")
            
            if self.migrate_conversation_to_database(conversation):
                successful_migrations += 1
            else:
                failed_migrations += 1
        
        # Generate migration report
        migration_report = {
            'success': True,
            'total_conversations': len(conversations),
            'successful_migrations': successful_migrations,
            'failed_migrations': failed_migrations,
            'migration_rate': successful_migrations / len(conversations) if conversations else 0,
            'flow_analysis': analysis,
            'database_stats': self._get_database_stats()
        }
        
        return migration_report
    
    def _get_database_stats(self) -> Dict[str, int]:
        """Get current database statistics"""
        try:
            cursor = self.connection.cursor()
            
            stats = {}
            
            # Count records in each table
            tables = [
                'conversation_flows', 'conversation_nodes', 'assistant_responses',
                'node_transitions', 'user_contexts', 'conversation_sessions',
                'conversation_turns', 'follow_up_triggers', 'conversation_analytics'
            ]
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}
    
    def create_sample_branching_flows(self) -> bool:
        """Create sample branching conversation flows"""
        try:
            cursor = self.connection.cursor()
            
            # Sample branching flow: Work Stress with Multiple Paths
            flow_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO conversation_flows 
                (flow_id, name, description, emotional_range_min, emotional_range_max, flow_tags, target_demographics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                flow_id,
                "Work Stress - Dynamic Branching",
                "Work stress conversation with branching based on specific stressor type",
                3, 8,
                json.dumps(['work_stress', 'branching_example']),
                json.dumps(['working_adults', 'professionals'])
            ))
            
            # Create branching nodes
            nodes = [
                {
                    'user_message': "I'm really stressed about work",
                    'branches': {
                        'deadline': "It's this huge deadline coming up",
                        'boss': "My boss is being impossible",
                        'workload': "I have way too much on my plate"
                    }
                }
            ]
            
            # Create the initial node
            initial_node_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO conversation_nodes 
                (node_id, flow_id, user_message, emotional_intensity, context_tags, follow_up_triggers)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                initial_node_id,
                flow_id,
                "I'm really stressed about work",
                5,
                json.dumps(['work_stress', 'initial']),
                json.dumps(['work_stress_followup'])
            ))
            
            # Create personality responses for initial node
            initial_responses = {
                'direct_practical': "What specifically is stressing you out at work?",
                'gentle_nurturing': "I can hear that work is really weighing on you. What's going on?",
                'analytical_problem_solving': "Let's break this down - what aspect of work is causing the most stress?",
                'casual_friend_like': "Ugh, work stress is the worst. What's happening?"
            }
            
            for personality, response in initial_responses.items():
                response_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO assistant_responses 
                    (response_id, node_id, personality_type, response_text, response_length, empathy_score, naturalness_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    response_id, initial_node_id, personality, response, len(response), 0.8, 0.9
                ))
            
            # Update flow starting node
            cursor.execute("""
                UPDATE conversation_flows 
                SET starting_node_id = ? 
                WHERE flow_id = ?
            """, (initial_node_id, flow_id))
            
            self.connection.commit()
            logger.info("✅ Sample branching flow created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sample branching flow: {e}")
            return False
    
    def export_migration_report(self, report: Dict[str, Any], output_path: str) -> bool:
        """Export migration report to JSON"""
        try:
            report_with_metadata = {
                **report,
                'migration_metadata': {
                    'migrated_at': datetime.now().isoformat(),
                    'migrator_version': 'conversation_data_migrator_v1.0',
                    'database_path': self.db_path
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_with_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Migration report exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting migration report: {e}")
            return False

def main():
    """Execute Phase 2: Data Migration"""
    print("🚀 PHASE 2: CONVERSATION DATA MIGRATION")
    print("=" * 60)
    print("📊 CONVERTING JSON CONVERSATIONS TO DATABASE STRUCTURE")
    print("=" * 60)
    
    # Initialize migrator
    migrator = ConversationDataMigrator()
    
    # Connect to database
    if not migrator.connect_database():
        print("❌ Failed to connect to database")
        return
    
    # Define paths
    json_path = "/home/vivi/pixelated/ai/data/processed/natural_conversations/natural_multi_turn_conversations.json"
    report_path = "/home/vivi/pixelated/ai/implementation/migration_report_phase2.json"
    
    print(f"\n📁 Source: {json_path}")
    print(f"🗄️ Target: {migrator.db_path}")
    
    # Execute migration
    print(f"\n🔄 Starting migration process...")
    migration_report = migrator.migrate_all_conversations(json_path)
    
    if migration_report['success']:
        print(f"\n✅ MIGRATION SUCCESSFUL!")
        print(f"📊 Total conversations: {migration_report['total_conversations']}")
        print(f"✅ Successful migrations: {migration_report['successful_migrations']}")
        print(f"❌ Failed migrations: {migration_report['failed_migrations']}")
        print(f"📈 Success rate: {migration_report['migration_rate']:.1%}")
        
        # Show flow analysis
        print(f"\n🎭 CONVERSATION FLOWS MIGRATED:")
        for flow_name, count in migration_report['flow_analysis']['flows_by_type'].items():
            print(f"  • {flow_name}: {count} conversations")
        
        # Show database stats
        print(f"\n🗄️ DATABASE STATISTICS:")
        db_stats = migration_report['database_stats']
        for table, count in db_stats.items():
            if count > 0:
                print(f"  • {table}: {count} records")
        
        # Create sample branching flows
        print(f"\n🌳 Creating sample branching flows...")
        if migrator.create_sample_branching_flows():
            print(f"✅ Sample branching flows created")
        
        # Export report
        if migrator.export_migration_report(migration_report, report_path):
            print(f"\n📁 Migration report exported to: {report_path}")
        
        print(f"\n🎯 PHASE 2 COMPLETE!")
        print(f"🚀 Ready for Phase 3: Enhanced Generation Engine")
        
    else:
        print(f"❌ Migration failed: {migration_report.get('error', 'Unknown error')}")
    
    return migration_report

if __name__ == "__main__":
    main()
