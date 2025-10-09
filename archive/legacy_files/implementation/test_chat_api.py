import pytest
#!/usr/bin/env python3
"""
Test Ollama Chat API
Try using the chat endpoint instead of generate
"""

import requests
import json
import time


class TestModule(unittest.TestCase):
    def test_chat_api():
        """Test the chat API endpoint"""
        
        base_url = "https://api.pixelatedempathy.tech"
        model_name = "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"
        
        print("🗣️ TESTING OLLAMA CHAT API")
        print("=" * 50)
        
        # Test 1: Basic chat request
        print("\n1️⃣ Testing /api/chat endpoint...")
        
        chat_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": "Hello, this is a test. Please respond with 'Chat API test successful'."
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "max_tokens": 30
            }
        }
        
        try:
            print(f"📤 Sending chat request...")
            print(f"🎯 Model: {model_name}")
            print("⏳ Waiting for response (60s timeout)...")
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/chat",
                json=chat_payload,
                timeout=60  # Longer timeout for model loading
            )
            response_time = time.time() - start_time
            
            print(f"⚡ Response time: {response_time:.2f}s")
            print(f"📊 Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                message = result.get('message', {})
                content = message.get('content', '')
                print(f"✅ Chat API successful!")
                print(f"📝 Response: {content}")
                return True, 'chat'
            else:
                print(f"❌ Chat API failed: {response.status_code}")
                print(f"📄 Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ Chat request timed out (60s)")
            print("💡 Model might still be loading...")
        except Exception as e:
            print(f"❌ Chat request failed: {e}")
        
        # Test 2: OpenAI-compatible endpoint
        print("\n2️⃣ Testing /v1/chat/completions endpoint...")
        
        openai_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, this is a test. Please respond with 'OpenAI API test successful'."
                }
            ],
            "temperature": 0.1,
            "max_tokens": 30
        }
        
        try:
            print(f"📤 Sending OpenAI-style request...")
            start_time = time.time()
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=openai_payload,
                timeout=60
            )
            response_time = time.time() - start_time
            
            print(f"⚡ Response time: {response_time:.2f}s")
            print(f"📊 Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                choices = result.get('choices', [])
                if choices:
                    content = choices[0].get('message', {}).get('content', '')
                    print(f"✅ OpenAI API successful!")
                    print(f"📝 Response: {content}")
                    return True, 'openai'
            else:
                print(f"❌ OpenAI API failed: {response.status_code}")
                print(f"📄 Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ OpenAI request timed out (60s)")
        except Exception as e:
            print(f"❌ OpenAI request failed: {e}")
        
        return False, None
    
    def test_smaller_model():
        """Test with a smaller model that might load faster"""
        
        base_url = "https://api.pixelatedempathy.tech"
        small_model = "smollm2:135m"  # Much smaller model
        
        print("\n3️⃣ Testing with smaller model...")
        print(f"🎯 Using: {small_model} (135M parameters)")
        
        chat_payload = {
            "model": small_model,
            "messages": [
                {
                    "role": "user", 
                    "content": "Hello, test response please."
                }
            ],
            "stream": False
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/chat",
                json=chat_payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            print(f"⚡ Response time: {response_time:.2f}s")
            print(f"📊 Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                message = result.get('message', {})
                content = message.get('content', '')
                print(f"✅ Small model works!")
                print(f"📝 Response: {content}")
                return True
            else:
                print(f"❌ Small model failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Small model test failed: {e}")
        
        return False
    
if __name__ == "__main__":
    success, api_type = test_chat_api()
    
    if not success:
        print("\n🔄 Trying smaller model...")
        small_success = test_smaller_model()
        
        if small_success:
            print("\n💡 DIAGNOSIS: Large model loading issue")
            print("   • Smaller models work fine")
            print("   • 4B model might need more time to load")
            print("   • Server might need more VRAM")
        else:
            print("\n❌ DIAGNOSIS: Server configuration issue")
            print("   • Even small models aren't working")
            print("   • Check Ollama server status")
    else:
        print(f"\n🎉 SUCCESS! Use {api_type} API endpoint")
        if api_type == 'chat':
            print("✅ Use /api/chat endpoint for crisis generator")
        elif api_type == 'openai':
            print("✅ Use /v1/chat/completions endpoint for crisis generator")
