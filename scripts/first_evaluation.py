# scripts/first_evaluation.py
"""
Step 13 definition of done:
  One real cost-vs-wrong-hit-rate number with bootstrap CI.

  This is your first genuine research result, however preliminary.
"""

import numpy as np
from eval.metrics.bootstrap import bootstrap_cost_saved, bootstrap_wrong_hit_rate

# After running baseline 1 (always-large) and baseline 8 (full system)
# on the same workload, load the results:

# Example (replace with real data from your baseline runs):
system_costs = np.array([0.0, 0.0, 0.0001, 0.0, 0.0003, 0.0002])  # cache hits = $0
baseline_costs = np.array([0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003])
is_wrong = np.array([False, False, False, False, True, False])  # 1 wrong hit

savings, sav_lo, sav_hi = bootstrap_cost_saved(system_costs, baseline_costs)
whr, whr_lo, whr_hi = bootstrap_wrong_hit_rate(is_wrong)

print(f"Cost saved:     {savings:.1f}%  (95% CI: {sav_lo:.1f}%–{sav_hi:.1f}%)")
print(f"Wrong-hit rate: {whr:.3f}   (95% CI: {whr_lo:.3f}–{whr_hi:.3f})")
print()
print("This is your first real research result.")
print("The numbers will improve as you tune thresholds and grow the dataset.")