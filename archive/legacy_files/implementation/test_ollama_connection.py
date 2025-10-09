import pytest
#!/usr/bin/env python3
"""
Simple Ollama Connection Test
Debug connection issues with remote Ollama server
"""

import requests
import json
import time


class TestModule(unittest.TestCase):
    def test_ollama_connection():
        """Test connection to Ollama server step by step"""
        
        base_url = "https://api.pixelatedempathy.tech"
        
        print("🔍 DEBUGGING OLLAMA CONNECTION")
        print("=" * 50)
        
        # Test 1: Basic connectivity
        print("\n1️⃣ Testing basic connectivity...")
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            print(f"✅ Server responding: {response.status_code}")
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"📋 Available models: {len(models)}")
                for model in models:
                    print(f"   • {model['name']}")
            else:
                print(f"❌ Unexpected status: {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Connection timed out")
            return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
        
        # Test 2: Check if our target model exists
        print("\n2️⃣ Checking target model...")
        target_model = "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"
        model_names = [m['name'] for m in models]
        
        if target_model in model_names:
            print(f"✅ Target model found: {target_model}")
        else:
            print(f"❌ Target model not found: {target_model}")
            print("Available abliterated models:")
            abliterated = [m for m in model_names if 'abliterated' in m.lower()]
            for model in abliterated:
                print(f"   • {model}")
            if abliterated:
                target_model = abliterated[0]
                print(f"🔄 Using first available: {target_model}")
            else:
                print("❌ No abliterated models found")
                return False
        
        # Test 3: Simple generation test
        print("\n3️⃣ Testing simple generation...")
        
        payload = {
            "model": target_model,
            "prompt": "Hello, this is a test. Please respond with 'Test successful'.",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "max_tokens": 20
            }
        }
        
        print(f"📤 Sending request to: {base_url}/api/generate")
        print(f"🎯 Model: {target_model}")
        print("⏳ Waiting for response (30s timeout)...")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            print(f"⚡ Response time: {response_time:.2f}s")
            print(f"📊 Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                print(f"✅ Generation successful!")
                print(f"📝 Response: {generated_text}")
                return True
            else:
                print(f"❌ Generation failed: {response.status_code}")
                print(f"📄 Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Generation request timed out (30s)")
            print("💡 This suggests the model might be loading or server is busy")
            return False
        except Exception as e:
            print(f"❌ Generation request failed: {e}")
            return False
        
        # Test 4: Check server status/health
        print("\n4️⃣ Checking server health...")
        try:
            # Some Ollama servers have a health endpoint
            health_response = requests.get(f"{base_url}/health", timeout=5)
            if health_response.status_code == 200:
                print("✅ Server health check passed")
            else:
                print(f"⚠️ Health endpoint returned: {health_response.status_code}")
        except:
            print("ℹ️ No health endpoint available (normal for basic Ollama)")
        
        return True
    
    def test_alternative_endpoints():
        """Test alternative API endpoints that might work"""
        
        base_url = "https://api.pixelatedempathy.tech"
        
        print("\n🔄 TESTING ALTERNATIVE ENDPOINTS")
        print("=" * 50)
        
        # Alternative endpoint structures
        endpoints_to_try = [
            "/api/generate",
            "/v1/generate", 
            "/generate",
            "/api/chat",
            "/v1/chat/completions"
        ]
        
        for endpoint in endpoints_to_try:
            print(f"\n🔍 Testing: {base_url}{endpoint}")
            try:
                # Try a simple POST to see if endpoint exists
                response = requests.post(
                    f"{base_url}{endpoint}",
                    json={"test": "ping"},
                    timeout=5
                )
                print(f"   Status: {response.status_code}")
                if response.status_code != 404:
                    print(f"   Response: {response.text[:100]}...")
            except requests.exceptions.Timeout:
                print("   ⏳ Timeout (endpoint might exist but need proper payload)")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
if __name__ == "__main__":
    success = test_ollama_connection()
    
    if not success:
        test_alternative_endpoints()
        
        print("\n🔧 TROUBLESHOOTING SUGGESTIONS:")
        print("1. Check if Ollama server is fully started")
        print("2. Try loading the model manually: ollama run <model_name>")
        print("3. Check server logs for any errors")
        print("4. Verify the model is fully downloaded")
        print("5. Try a smaller/faster model first")
    else:
        print("\n🎉 CONNECTION TEST SUCCESSFUL!")
        print("✅ Ready to run crisis conversation generator")
