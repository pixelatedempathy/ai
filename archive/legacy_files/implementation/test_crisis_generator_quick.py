from ai.inference
from ai.pixel
from ai.dataset_pipeline
from .\1 import
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import pytest
#!/usr/bin/env python3
"""
Quick test of the updated crisis generator
Test one scenario to make sure it works before running the full batch
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .abliterated_crisis_generator import AbliteratedCrisisGenerator, CrisisScenario
import asyncio

async def test_single_crisis_generation():
    """Test generating one crisis conversation"""
    
    print("🧪 QUICK CRISIS GENERATOR TEST")
    print("=" * 50)
    
    # Initialize generator
    try:
        generator = AbliteratedCrisisGenerator()
        print("✅ Generator initialized successfully")
    except Exception as e:
        print(f"❌ Generator initialization failed: {e}")
        return False
    
    # Create a test scenario
    test_scenario = CrisisScenario(
        scenario_id="test_suicide_ideation",
        crisis_type="suicidal_ideation",
        intensity_level=8,
        demographic="college_student_20s",
        situation_context="Academic pressure, social isolation, recent breakup",
        expected_duration=6
    )
    
    print(f"\n🎯 Testing scenario: {test_scenario.crisis_type}")
    print(f"   Intensity: {test_scenario.intensity_level}/10")
    print(f"   Context: {test_scenario.situation_context}")
    
    # Generate conversation
    try:
        print("\n🔄 Generating crisis conversation...")
        conversation = generator.generate_crisis_conversation(test_scenario)
        
        print("✅ Generation successful!")
        print(f"📊 Conversation ID: {conversation['conversation_id']}")
        print(f"🎭 Turns generated: {len(conversation['turns'])}")
        print(f"🚨 Crisis indicators: {len(conversation['crisis_indicators_detected'])}")
        print(f"📈 Quality scores: {conversation['conversation_quality']}")
        
        # Show a sample of the conversation
        print(f"\n📝 SAMPLE CONVERSATION:")
        print("-" * 30)
        
        for i, turn in enumerate(conversation['turns'][:4]):  # Show first 4 turns
            speaker = "User" if turn['speaker'] == 'user' else "Assistant"
            message = turn['message'][:100] + "..." if len(turn['message']) > 100 else turn['message']
            print(f"{speaker}: {message}")
            if i < len(conversation['turns']) - 1:
                print()
        
        if len(conversation['turns']) > 4:
            print(f"... ({len(conversation['turns']) - 4} more turns)")
        
        print(f"\n🎯 Crisis indicators detected:")
        for indicator in conversation['crisis_indicators_detected'][:5]:
            print(f"   • {indicator}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_single_crisis_generation())
    
    if success:
        print(f"\n🎉 QUICK TEST SUCCESSFUL!")
        print("✅ Crisis generator is working with OpenAI API")
        print("🚀 Ready to run full crisis conversation library generation")
    else:
        print(f"\n❌ QUICK TEST FAILED")
        print("🔧 Need to debug further before running full generation")
