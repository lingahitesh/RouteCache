"""
Test Gemini API key to verify it works.
"""

import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("JUDGE1")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env file")
    exit(1)

print(f"Testing Gemini API key: {api_key[:10]}...{api_key[-4:]}")

# Gemini API endpoint (using REST API, not the Python SDK)
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"

# Simple test payload
payload = {
    "contents": [{
        "parts": [{
            "text": "Hello! Respond with just: OK"
        }]
    }]
}

try:
    response = httpx.post(url, json=payload, timeout=30.0)
    
    if response.status_code == 200:
        data = response.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        print(f"\n✓ SUCCESS! API key works.")
        print(f"Response: {answer}")
        
        # Show token usage
        if "usageMetadata" in data:
            usage = data["usageMetadata"]
            print(f"\nTokens used:")
            print(f"  Input:  {usage.get('promptTokenCount', 0)}")
            print(f"  Output: {usage.get('candidatesTokenCount', 0)}")
            print(f"  Total:  {usage.get('totalTokenCount', 0)}")
    else:
        print(f"\n✗ FAILED: HTTP {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 400:
            print("\nLikely issue: Invalid API key or malformed request")
        elif response.status_code == 403:
            print("\nLikely issue: API key doesn't have permission or billing not enabled")
        elif response.status_code == 429:
            print("\nLikely issue: Rate limit exceeded")
            
except httpx.TimeoutException:
    print("\n✗ FAILED: Request timed out")
except httpx.ConnectError:
    print("\n✗ FAILED: Could not connect to Gemini API")
except Exception as e:
    print(f"\n✗ FAILED: {type(e).__name__}: {e}")
