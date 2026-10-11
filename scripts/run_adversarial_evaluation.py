"""
Run the complete adversarial evaluation (Step 14 Definition of Done).

This script:
  1. Generates adversarial pairs (if not already generated)
  2. Runs them through the RouteCache proxy
  3. Reports category-level error breakdown

Expected output format per docs section 17.6:
  - Per-category: Total, Correct, Wrong, Accuracy
  - Overall accuracy

Traceability: FR-8.9 (C4), FR-9.3
Build Plan: Step 14
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.stress_sets.adversarial import save_adversarial_set
from eval.stress_sets.error_analysis import run_adversarial_evaluation


def main():
    """Generate pairs and run evaluation."""
    
    print("=" * 70)
    print("STEP 14 — ADVERSARIAL EVALUATION (Contribution C4)")
    print("=" * 70)
    
    # Step 1: Generate adversarial pairs (if needed)
    pairs_path = r"C:\Users\KIIT\Desktop\Projects\RouteCache\eval/stress_sets/adversarial_pairs.json"
    if not Path(pairs_path).exists():
        print(f"\nGenerating adversarial pairs...")
        save_adversarial_set(pairs_path)
    else:
        print(f"\nUsing existing adversarial pairs from {pairs_path}")
    
    # Step 2: Run evaluation
    print("\nStarting evaluation (ensure server is running on http://localhost:8000)...")
    print("This will:")
    print("  1. Send query_a to cache the answer")
    print("  2. Send query_b to test if cached answer is incorrectly served")
    print("  3. Track correctness per category\n")
    
    try:
        category_stats = run_adversarial_evaluation(
            pairs_path=pairs_path,
            proxy_url="http://localhost:8000",
            output_path=r"C:\Users\KIIT\Desktop\Projects\RouteCache\eval\stress_sets/adversarial_results.json",
        )
        
        print("\n" + "=" * 70)
        print("✓ Evaluation complete!")
        print(f"  Detailed results saved to: eval/stress_sets/adversarial_results.json")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        print("\nMake sure:")
        print("  1. RouteCache server is running: python -m app.main")
        print("  2. Server is accessible at http://localhost:8000")
        return 1


if __name__ == "__main__":
    exit(main())