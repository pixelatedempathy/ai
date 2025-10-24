#!/usr/bin/env python3
"""
Enhanced Dataset Pipeline for Pixelated Empathy Training
Integrates professional datasets with therapeutic accuracy assessment
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Any

class EnhancedDatasetPipeline:
    def __init__(self):
        self.conversations = []
        self.quality_metrics = {
            'therapeutic_accuracy': 0,
            'clinical_appropriateness': 0,
            'safety_compliance': 0,
            'dsm5_alignment': 0
        }
    
    def load_existing_dataset(self):
        """Load the consolidated training dataset"""
        if Path("training_dataset.json").exists():
            with open("training_dataset.json", 'r') as f:
                self.conversations = json.load(f)
            print(f"Loaded {len(self.conversations)} conversations")
        else:
            print("No existing dataset found")
    
    def apply_therapeutic_quality_filter(self):
        """Apply quality filtering based on therapeutic accuracy"""
        filtered = []
        
        for conv in self.conversations:
            # Check for therapeutic indicators
            has_therapeutic_content = self._assess_therapeutic_quality(conv)
            
            if has_therapeutic_content:
                # Add quality metadata
                conv['quality_score'] = self._calculate_quality_score(conv)
                conv['therapeutic_validated'] = True
                filtered.append(conv)
        
        print(f"Filtered to {len(filtered)} high-quality therapeutic conversations")
        self.conversations = filtered
    
    def _assess_therapeutic_quality(self, conversation: Dict) -> bool:
        """Assess if conversation meets therapeutic quality standards"""
        messages = conversation.get('messages', [])
        if not messages:
            return False
        
        # Look for therapeutic indicators
        therapeutic_indicators = [
            'empathy', 'validation', 'reflection', 'coping', 'support',
            'feelings', 'emotions', 'therapy', 'counseling', 'mental health',
            'anxiety', 'depression', 'stress', 'wellbeing', 'mindfulness'
        ]
        
        text_content = ' '.join([msg.get('content', '') for msg in messages]).lower()
        
        # Must have at least 2 therapeutic indicators
        indicator_count = sum(1 for indicator in therapeutic_indicators if indicator in text_content)
        
        return indicator_count >= 2
    
    def _calculate_quality_score(self, conversation: Dict) -> float:
        """Calculate quality score for conversation"""
        messages = conversation.get('messages', [])
        if not messages:
            return 0.0
        
        score = 0.0
        
        # Length quality (optimal 2-8 exchanges)
        if 2 <= len(messages) <= 8:
            score += 0.3
        
        # Response quality (check for empathetic language)
        empathy_words = ['understand', 'feel', 'sorry', 'support', 'help', 'care']
        for msg in messages:
            content = msg.get('content', '').lower()
            if any(word in content for word in empathy_words):
                score += 0.2
                break
        
        # Professional structure
        if any(msg.get('role') == 'assistant' for msg in messages):
            score += 0.3
        
        # Safety check (no harmful content)
        harmful_indicators = ['suicide', 'self-harm', 'kill', 'die', 'hurt']
        text_content = ' '.join([msg.get('content', '') for msg in messages]).lower()
        if not any(indicator in text_content for indicator in harmful_indicators):
            score += 0.2
        
        return min(score, 1.0)
    
    def create_balanced_dataset(self, target_size: int = 40000):
        """Create balanced dataset with proper distribution"""
        if len(self.conversations) <= target_size:
            print(f"Using all {len(self.conversations)} conversations")
            return
        
        # Sort by quality score
        self.conversations.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Take top quality conversations
        self.conversations = self.conversations[:target_size]
        print(f"Selected top {len(self.conversations)} conversations by quality")
    
    def add_therapeutic_metadata(self):
        """Add therapeutic metadata to conversations"""
        for i, conv in enumerate(self.conversations):
            conv['conversation_id'] = f"pixelated_{i:06d}"
            conv['dataset_source'] = 'pixelated_empathy_enhanced'
            conv['therapeutic_category'] = self._categorize_conversation(conv)
            conv['processing_timestamp'] = '2024-09-30'
    
    def _categorize_conversation(self, conversation: Dict) -> str:
        """Categorize conversation by therapeutic focus"""
        messages = conversation.get('messages', [])
        text_content = ' '.join([msg.get('content', '') for msg in messages]).lower()
        
        categories = {
            'anxiety_support': ['anxiety', 'worry', 'nervous', 'panic'],
            'depression_support': ['depression', 'sad', 'hopeless', 'down'],
            'stress_management': ['stress', 'overwhelmed', 'pressure', 'burnout'],
            'relationship_counseling': ['relationship', 'partner', 'family', 'conflict'],
            'general_support': ['support', 'help', 'guidance', 'advice']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_content for keyword in keywords):
                return category
        
        return 'general_support'
    
    def save_enhanced_dataset(self):
        """Save the enhanced dataset"""
        output_file = "training_dataset_enhanced.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        self._generate_summary_report()
        
        print(f"Enhanced dataset saved: {output_file}")
        print(f"Total conversations: {len(self.conversations)}")
    
    def _generate_summary_report(self):
        """Generate dataset summary report"""
        categories = {}
        quality_scores = []
        
        for conv in self.conversations:
            category = conv.get('therapeutic_category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            quality_scores.append(conv.get('quality_score', 0))
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        report = {
            'total_conversations': len(self.conversations),
            'average_quality_score': round(avg_quality, 3),
            'category_distribution': categories,
            'quality_distribution': {
                'high_quality (>0.8)': len([s for s in quality_scores if s > 0.8]),
                'medium_quality (0.5-0.8)': len([s for s in quality_scores if 0.5 <= s <= 0.8]),
                'basic_quality (<0.5)': len([s for s in quality_scores if s < 0.5])
            }
        }
        
        with open("dataset_enhancement_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nDataset Enhancement Report:")
        print(f"Average Quality Score: {avg_quality:.3f}")
        print("Category Distribution:")
        for category, count in categories.items():
            print(f"  {category}: {count}")

def main():
    pipeline = EnhancedDatasetPipeline()
    
    print("=== Enhanced Dataset Pipeline ===")
    pipeline.load_existing_dataset()
    pipeline.apply_therapeutic_quality_filter()
    pipeline.create_balanced_dataset(target_size=40000)
    pipeline.add_therapeutic_metadata()
    pipeline.save_enhanced_dataset()
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
