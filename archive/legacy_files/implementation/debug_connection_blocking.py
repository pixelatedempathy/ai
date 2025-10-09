#!/usr/bin/env python3
"""
Comprehensive Connection Blocking Diagnostic
Identify what's blocking the Ollama connection
"""

import requests
import json
import time
import socket
import subprocess
import sys

def test_basic_connectivity():
    """Test basic network connectivity"""
    print("🌐 TESTING BASIC CONNECTIVITY")
    print("=" * 50)
    
    base_url = "https://api.pixelatedempathy.tech"
    
    # Test 1: DNS resolution
    try:
        import socket
        ip = socket.gethostbyname("api.pixelatedempathy.tech")
        print(f"✅ DNS resolution: api.pixelatedempathy.tech -> {ip}")
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}")
        return False
    
    # Test 2: Port connectivity
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((ip, 443))
        sock.close()
        
        if result == 0:
            print(f"✅ Port 443 is open on {ip}")
        else:
            print(f"❌ Port 443 is closed on {ip}")
            return False
    except Exception as e:
        print(f"❌ Port test failed: {e}")
        return False
    
    # Test 3: HTTPS handshake
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        print(f"✅ HTTPS handshake successful: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ HTTPS handshake failed: {e}")
        return False

def test_different_endpoints():
    """Test all possible endpoints to see what works"""
    print("\n🔍 TESTING ALL ENDPOINTS")
    print("=" * 50)
    
    base_url = "https://api.pixelatedempathy.tech"
    
    endpoints = [
        ("/", "GET", None),
        ("/api/tags", "GET", None),
        ("/api/version", "GET", None),
        ("/api/show", "POST", {"name": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"}),
        ("/api/generate", "POST", {
            "model": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M",
            "prompt": "Hello",
            "stream": False
        }),
        ("/api/chat", "POST", {
            "model": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }),
        ("/v1/chat/completions", "POST", {
            "model": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M",
            "messages": [{"role": "user", "content": "Hello"}]
        })
    ]
    
    working_endpoints = []
    
    for endpoint, method, payload in endpoints:
        print(f"\n🔍 Testing {method} {endpoint}")
        
        try:
            start_time = time.time()
            
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=15)
            else:
                response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=15)
            
            response_time = time.time() - start_time
            
            print(f"   Status: {response.status_code}")
            print(f"   Time: {response_time:.2f}s")
            
            if response.status_code == 200:
                print(f"   ✅ Working")
                working_endpoints.append((endpoint, method))
                
                # Show response preview for successful requests
                try:
                    if response.headers.get('content-type', '').startswith('application/json'):
                        data = response.json()
                        if isinstance(data, dict) and 'response' in data:
                            preview = data['response'][:100] + "..." if len(data['response']) > 100 else data['response']
                            print(f"   Response: {preview}")
                        elif isinstance(data, dict) and 'models' in data:
                            print(f"   Models: {len(data['models'])} available")
                except:
                    pass
            else:
                print(f"   ❌ Failed: {response.status_code}")
                if response.text:
                    error_preview = response.text[:200] + "..." if len(response.text) > 200 else response.text
                    print(f"   Error: {error_preview}")
                    
        except requests.exceptions.Timeout:
            print(f"   ⏳ Timeout (15s)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return working_endpoints

def test_request_headers():
    """Test different request headers to see if that's the issue"""
    print("\n📋 TESTING REQUEST HEADERS")
    print("=" * 50)
    
    base_url = "https://api.pixelatedempathy.tech"
    
    headers_to_test = [
        {},  # No special headers
        {"User-Agent": "curl/7.81.0"},  # Curl user agent
        {"User-Agent": "Mozilla/5.0 (compatible; OllamaClient/1.0)"},  # Custom user agent
        {"Accept": "application/json"},  # JSON accept
        {"Content-Type": "application/json", "Accept": "application/json"},  # Full JSON headers
        {"Origin": "https://api.pixelatedempathy.tech"},  # Same origin
        {"Referer": "https://api.pixelatedempathy.tech"},  # Referer header
    ]
    
    payload = {
        "model": "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M",
        "prompt": "Test",
        "stream": False,
        "options": {"max_tokens": 10}
    }
    
    for i, headers in enumerate(headers_to_test):
        print(f"\n🔍 Test {i+1}: {headers if headers else 'No special headers'}")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                headers=headers,
                timeout=20
            )
            response_time = time.time() - start_time
            
            print(f"   Status: {response.status_code}")
            print(f"   Time: {response_time:.2f}s")
            
            if response.status_code == 200:
                print(f"   ✅ Success with these headers!")
                return headers
            else:
                print(f"   ❌ Failed: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏳ Timeout")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return None

def test_smaller_requests():
    """Test with progressively smaller requests"""
    print("\n📏 TESTING REQUEST SIZES")
    print("=" * 50)
    
    base_url = "https://api.pixelatedempathy.tech"
    model = "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"
    
    test_requests = [
        # Minimal request
        {
            "model": model,
            "prompt": "Hi",
            "stream": False
        },
        # With basic options
        {
            "model": model,
            "prompt": "Hello",
            "stream": False,
            "options": {"max_tokens": 5}
        },
        # Slightly larger
        {
            "model": model,
            "prompt": "Hello, how are you?",
            "stream": False,
            "options": {"max_tokens": 10, "temperature": 0.1}
        }
    ]
    
    for i, payload in enumerate(test_requests):
        print(f"\n🔍 Request size test {i+1}")
        print(f"   Payload size: {len(json.dumps(payload))} bytes")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            print(f"   Status: {response.status_code}")
            print(f"   Time: {response_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                generated = result.get('response', '')
                print(f"   ✅ Success! Generated: '{generated}'")
                return True
            else:
                print(f"   ❌ Failed: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏳ Timeout (30s)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return False

def check_server_logs_suggestion():
    """Suggest checking server logs"""
    print("\n📊 SERVER LOG ANALYSIS NEEDED")
    print("=" * 50)
    print("Based on the tests above, check your server logs for:")
    print("1. CORS errors or blocked origins")
    print("2. Rate limiting or IP blocking")
    print("3. Authentication/authorization failures")
    print("4. Model loading errors or memory issues")
    print("5. Proxy/reverse proxy configuration issues")
    print("\nCommon log locations:")
    print("- Ollama logs: journalctl -u ollama -f")
    print("- Nginx/Apache logs: /var/log/nginx/ or /var/log/apache2/")
    print("- System logs: /var/log/syslog")

def main():
    """Run comprehensive diagnostic"""
    print("🔍 COMPREHENSIVE CONNECTION BLOCKING DIAGNOSTIC")
    print("=" * 80)
    
    # Test 1: Basic connectivity
    if not test_basic_connectivity():
        print("\n❌ BASIC CONNECTIVITY FAILED")
        print("Check network, DNS, and firewall settings")
        return
    
    # Test 2: All endpoints
    working_endpoints = test_different_endpoints()
    
    if not working_endpoints:
        print("\n❌ NO ENDPOINTS WORKING")
        print("Server may be down or completely misconfigured")
        return
    
    print(f"\n✅ Working endpoints: {working_endpoints}")
    
    # Test 3: Request headers (only if generate endpoint exists but fails)
    if ("/api/generate", "POST") not in working_endpoints:
        print("\n🔍 /api/generate not working, testing headers...")
        working_headers = test_request_headers()
        
        if working_headers:
            print(f"✅ Found working headers: {working_headers}")
        else:
            print("❌ No headers fix the issue")
    
    # Test 4: Request sizes
    if ("/api/generate", "POST") not in working_endpoints:
        print("\n🔍 Testing different request sizes...")
        if test_smaller_requests():
            print("✅ Smaller requests work - may be a payload size issue")
        else:
            print("❌ Even minimal requests fail")
    
    # Final suggestions
    check_server_logs_suggestion()
    
    print(f"\n🎯 DIAGNOSTIC COMPLETE")
    print("Check the results above and server logs for the specific blocking cause")

if __name__ == "__main__":
    main()
