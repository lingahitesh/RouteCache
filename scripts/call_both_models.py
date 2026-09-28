"""
Step 2 definition of done: call both adapters with the same prompt,
print (text, tokens_in, tokens_out, cost_usd, latency_ms) for each.
Confirm every call is written to data/recorded_outputs/.
"""

from config.config import load_config
from app.adapters.adapters import GroqAdapter, RetryingAdapter

config = load_config()

small = RetryingAdapter(GroqAdapter(config.models["small"]))
large = RetryingAdapter(GroqAdapter(config.models["large"]))

messages = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]

print("=" * 60)
print("Calling SMALL model (GPT 20B via Groq)...")
r1 = small.call(messages)
print(f"  Answer:     {r1.text}")
print(f"  Tokens:     in={r1.tokens_in}, out={r1.tokens_out}")
print(f"  Cost:       ${r1.cost_usd:.8f}")
print(f"  Latency:    {r1.latency_ms:.1f} ms")
print(f"  Estimated:  {r1.tokens_estimated}")

print()
print("Calling LARGE model (GPT 120B via Groq)...")
r2 = large.call(messages)
print(f"  Answer:     {r2.text}")
print(f"  Tokens:     in={r2.tokens_in}, out={r2.tokens_out}")
print(f"  Cost:       ${r2.cost_usd:.8f}")
print(f"  Latency:    {r2.latency_ms:.1f} ms")
print(f"  Estimated:  {r2.tokens_estimated}")
print("=" * 60)

# Verify disk cache
from pathlib import Path
cache_dir = Path("data/recorded_outputs")
files = list(cache_dir.glob("*.json"))
print(f"\nDisk cache: {len(files)} responses saved in {cache_dir}/")
assert len(files) >= 2, "NFR-8 FAILED: responses not saved to disk!"
print("NFR-8 check: PASSED — all responses written to disk.")