import pytest
#!/usr/bin/env python3
"""
Test with smallest model to confirm API endpoint works
"""

import requests
import json
import time


class TestModule(unittest.TestCase):
    def test_small_model_generation():
        """Test generation with the smallest available model"""
        
        base_url = "https://api.pixelatedempathy.tech"
        small_model = "smollm2:135m"  # 135M parameters - should be very fast
        
        print("🧪 TESTING WITH SMALLEST MODEL")
        print("=" * 50)
        print(f"🎯 Model: {small_model} (135M parameters)")
        
        # Test crisis-style prompt with small model
        messages = [
            {
                "role": "user",
                "content": """Generate a short crisis conversation for training purposes:
    
User: I'm feeling really depressed and don't know what to do
Assistant: [Your response here]
User: I've been having thoughts of hurting myself
Assistant: [Your response here]

Keep it realistic and include appropriate crisis resources."""
        }
    ]
    
    payload = {
        "model": small_model,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 300
    }
    
    try:
        print("📤 Sending request...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        
        response_time = time.time() - start_time
        print(f"⚡ Response time: {response_time:.2f}s")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            choices = result.get('choices', [])
            
            if choices:
                content = choices[0].get('message', {}).get('content', '')
                print(f"✅ Small model generation successful!")
                print(f"📝 Generated {len(content)} characters")
                print(f"\n📄 RESPONSE:")
                print("-" * 30)
                print(content)
                print("-" * 30)
                return True
            else:
                print("❌ No choices in response")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Even small model timed out")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


class TestModule(unittest.TestCase):
    def test_qwen3_with_longer_timeout():
        """Test Qwen3-abliterated with much longer timeout"""
        
        base_url = "https://api.pixelatedempathy.tech"
        qwen3_model = "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"
        
        print(f"\n🧠 TESTING QWEN3-ABLITERATED WITH LONG TIMEOUT")
        print("=" * 50)
        print(f"🎯 Model: {qwen3_model}")
        print("⏳ Using 5-minute timeout to allow for model loading...")
        
        messages = [
            {
                "role": "user",
                "content": "Hello, this is a simple test. Please respond with 'Qwen3-abliterated is working' and nothing else."
            }
        ]
        
        payload = {
            "model": qwen3_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 20
        }
        
        try:
            print("📤 Sending request with 5-minute timeout...")
            start_time = time.time()
            
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=300  # 5 minutes
            )
            
            response_time = time.time() - start_time
            print(f"⚡ Response time: {response_time:.2f}s")
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                choices = result.get('choices', [])
                
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    print(f"✅ Qwen3-abliterated is working!")
                    print(f"📝 Response: {content}")
                    return True
                else:
                    print("❌ No choices in response")
                    return False
            else:
                print(f"❌ Request failed: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Even 5-minute timeout failed")
            print("💡 Model might be too large for current server resources")
            return False
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False
    
if __name__ == "__main__":
    print("🔍 DIAGNOSING MODEL LOADING ISSUES")
    print("=" * 60)
    
    # Test 1: Small model
    small_success = test_small_model_generation()
    
    if small_success:
        print(f"\n✅ API endpoint works fine with small models")
        
        # Test 2: Large model with long timeout
        qwen3_success = test_qwen3_with_longer_timeout()
        
        if qwen3_success:
            print(f"\n🎉 QWEN3-ABLITERATED WORKS!")
            print("💡 Just needs longer timeout for initial loading")
        else:
            print(f"\n⚠️ QWEN3-ABLITERATED LOADING ISSUES")
            print("💡 Possible solutions:")
            print("   1. Server needs more VRAM")
            print("   2. Model needs to be pre-loaded")
            print("   3. Use smaller abliterated model")
    else:
        print(f"\n❌ API ENDPOINT ISSUES")
        print("💡 Server configuration problem")
