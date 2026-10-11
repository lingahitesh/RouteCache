# scripts/tester_decision_engine.py
"""
Step 11 definition of done:
  Replay Lesson 8 Section 13's worked example through the real system.

  Scenario: Cache has "What is the population of Tokyo in 2020?" -> "13.96 million"
  New query: "What is the population of Tokyo in 2021?"

  Expected path:
    1. FAISS finds the 2020 entry (high similarity — same structure)
    2. Risk features detect: numbers (2020 vs 2021), temporal ("2021")
    3. Adaptive thresholds tighten
    4. Zone = VERIFY (similarity high but below tightened t_high)
    5. Verifier says UNSAFE (different year = different answer)
    6. Decision engine falls through to model call
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def send_query(query: str, bypass_cache: bool = False):
    """Send a query to the RouteCache API."""
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "x_routecache": {
            "bypass_cache": bypass_cache,
            "force_model": None,
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        print(response.text)
        return None
    
    return response.json()


def main():
    print("="*70)
    print("Step 11 Definition of Done: Tokyo Population Test (C1+C2+C3)")
    print("="*70)
    
    # Step 1: Prime the cache with 2020 query
    print("\n[SETUP] Priming cache with 2020 query...")
    resp_2020 = send_query("What is the population of Tokyo in 2020?", bypass_cache=True)
    if not resp_2020:
        print("FAILED: Could not prime cache")
        return
    
    meta_2020 = resp_2020.get("x_routecache", {})
    print(f"  Cached: zone={meta_2020.get('zone')}, route={meta_2020.get('route')}")
    
    # Step 2: Query for 2021 (should NOT serve 2020 answer)
    print("\n[TEST] Querying for 2021...")
    resp_2021 = send_query("What is the population of Tokyo in 2021?")
    if not resp_2021:
        print("FAILED: Could not query 2021")
        return
    
    meta_2021 = resp_2021.get("x_routecache", {})
    answer_2021 = resp_2021["choices"][0]["message"]["content"]
    
    print(f"\nResult:")
    print(f"  Cache:      {meta_2021.get('cache')}")
    print(f"  Zone:       {meta_2021.get('zone')}")
    print(f"  Similarity: {meta_2021.get('similarity')}")
    print(f"  Route:      {meta_2021.get('route')}")
    print(f"  Answer:     {answer_2021[:80]}...")
    
    # Verify expectations
    print("\n" + "="*70)
    print("VERIFICATION:")
    print("="*70)
    
    # Expected: should NOT serve from cache (cache status should be 'miss')
    if meta_2021.get('cache') == 'hit':
        print("❌ FAILED: System served cached 2020 answer for 2021 query!")
        print("   This is the exact failure mode C1+C2+C3 exists to prevent.")
        print("   The decision engine is BROKEN.")
    else:
        print("✓ PASS: System did NOT serve cached 2020 answer")
    
    # Expected: similarity should be high (FAISS finds the entry)
    similarity = meta_2021.get('similarity')
    if similarity and similarity > 0.90:
        print(f"✓ PASS: High similarity detected ({similarity:.4f})")
    else:
        print(f"⚠ WARNING: Expected high similarity, got {similarity}")
    
    # Expected: zone should be VERIFY or MISS (not SERVE)
    zone = meta_2021.get('zone')
    if zone in ['VERIFY', 'MISS']:
        print(f"✓ PASS: Zone is {zone} (not SERVE)")
    else:
        print(f"❌ FAILED: Zone is {zone}, expected VERIFY or MISS")
    
    # Expected: should route to a model (not serve from cache)
    route = meta_2021.get('route')
    if route and 'model' in route:
        print(f"✓ PASS: Routed to {route}")
    else:
        print(f"❌ FAILED: Did not route to model, got {route}")
    
    print("\n" + "="*70)
    print("Test complete. All three contributions (C1+C2+C3) are working")
    print("if all checks passed.")
    print("="*70)


if __name__ == "__main__":
    main()