#!/usr/bin/env python3
"""
Simple test for HAI3-serve vLLM-like functionality
"""
import json
import time
import requests

BASE_URL = "http://localhost:8000"

def test_health():
    """Test basic health check"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Server: {data.get('message')}")
            print(f"Model loaded: {data.get('model_loaded')}")
            return True
        return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_models():
    """Test models endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        print(f"Models status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Available models: {[model['id'] for model in data['data']]}")
            return True
        return False
    except Exception as e:
        print(f"Models test failed: {e}")
        return False

def test_chat_completion():
    """Test chat completion"""
    try:
        payload = {
            "model": "HelpingAI--hai3.1-checkpoint-0002", 
            "messages": [
                {"role": "user", "content": "Hello! What's 2+2?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
        print(f"Chat completion status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data['choices'][0]['message']['content']}")
            print(f"Usage: {data['usage']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Chat completion test failed: {e}")
        return False

def test_chat_with_tools():
    """Test chat completion with tool calling"""
    try:
        payload = {
            "model": "HelpingAI--hai3.1-checkpoint-0002",
            "messages": [
                {"role": "user", "content": "What's the current time?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "description": "Get the current date and time",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
        print(f"Chat with tools status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            message = data['choices'][0]['message']
            print(f"Response: {message.get('content', '')}")
            if message.get('tool_calls'):
                print(f"Tool calls: {message['tool_calls']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Chat with tools test failed: {e}")
        return False

def test_completion():
    """Test text completion"""
    try:
        payload = {
            "model": "HelpingAI--hai3.1-checkpoint-0002",
            "prompt": "The capital of France is",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(f"{BASE_URL}/v1/completions", json=payload)
        print(f"Completion status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Completion: {data['choices'][0]['text']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Completion test failed: {e}")
        return False

def main():
    """Run vLLM-like serving tests"""
    print("HAI3-serve vLLM-like Test Suite")
    print("=" * 40)
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    max_retries = 10
    for i in range(max_retries):
        if test_health():
            break
        time.sleep(2)
        print(f"Retry {i+1}/{max_retries}...")
    else:
        print("Server is not responding. Please start the server with: python main.py")
        return
    
    # Run core tests
    tests = [
        ("Models List", test_models),
        ("Chat Completion", test_chat_completion),
        ("Chat with Tools", test_chat_with_tools), 
        ("Text Completion", test_completion),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n✅ All tests passed! HAI3-serve is working like vLLM!")
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed.")

if __name__ == "__main__":
    main()