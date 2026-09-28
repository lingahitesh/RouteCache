"""
Step 3 definition of done: run cost computation against 20 real calls
from Step 2, sanity-check against Groq's pricing page.

Sanity check: for GPT 120B at $0.15/M input, $0.60/M output:
  19 input tokens  -> 19 * 0.00000015  = $0.00000285
  2 output tokens  -> 2  * 0.0000006   = $0.0000012
  Total                                = $0.00000405
"""

from config.config import load_config
from app.cost import compute_cost, compute_cost_for_model, compute_savings_percent

config = load_config()

# Example from Step 2's test call
tokens_in, tokens_out = 19, 2

cost_small = compute_cost_for_model(tokens_in, tokens_out, "small", config)
cost_large = compute_cost_for_model(tokens_in, tokens_out, "large", config)
savings = compute_savings_percent(cost_small, cost_large)

print(f"Small model cost: ${cost_small:.10f}")
print(f"Large model cost: ${cost_large:.10f}")
print(f"Savings:          {savings * 100:.1f}%")

# Sanity: small should be cheaper than large
assert cost_small < cost_large, "Small model should be cheaper!"
# Sanity: savings should be positive
assert savings > 0, "Using the small model should save money!"
# Sanity: costs should be positive and tiny
assert 0 < cost_small < 0.01
assert 0 < cost_large < 0.01
print("\nAll sanity checks PASSED.")